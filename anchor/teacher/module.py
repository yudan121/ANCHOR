from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from torch.nn.functional import one_hot

from scvi import REGISTRY_KEYS, settings
from scvi.module._classifier import Classifier
from scvi.module._totalvae import TOTALVAE
from scvi.module.base import LossOutput, auto_move_data
from scvi.train import SemiSupervisedTrainingPlan

def set_scvi_training_seed(seed: int = 2026) -> None:
    """Seed scvi-tools and local RNGs before model construction/training."""
    seed = int(seed)
    settings.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_protein_teacher_stats(
    protein: pd.DataFrame | np.ndarray,
    protein_names: Sequence[str],
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Fit ANCHOR's robust arcsinh scaling for protein inputs."""
    protein_raw = np.asarray(protein, dtype=np.float32)
    n_proteins = protein_raw.shape[1]
    kappa = np.ones(n_proteins, dtype=np.float32)
    median = np.zeros(n_proteins, dtype=np.float32)
    mad = np.ones(n_proteins, dtype=np.float32)
    for idx in range(n_proteins):
        raw = protein_raw[:, idx]
        nonzero = raw[raw > 0]
        if nonzero.size:
            kappa[idx] = max(float(np.median(nonzero)), 1.0)
        transformed = np.arcsinh(raw / kappa[idx]).astype(np.float32)
        observed = transformed[raw > 0]
        if observed.size:
            median[idx] = float(np.median(observed))
            mad[idx] = float(np.median(np.abs(observed - median[idx])) * 1.4826 + eps)
    return {
        "protein_names": list(protein_names),
        "kappa": kappa,
        "median": median,
        "mad": mad,
    }


class _AnchorTeacherModule(TOTALVAE):
    """TOTALVAE backbone with a semi-supervised classifier head.

    The latent regularizer uses the standard VAE prior, \(N(0, I)\), rather
    than a label-conditioned latent prior.
    """

    def __init__(
        self,
        *args,
        n_labels: int,
        prior_spec: dict[str, Any],
        protein_names: Sequence[str],
        protein_teacher_stats: dict[str, Any],
        standard_normal_prior_enable: bool = True,
        standard_normal_prior_loss_weight: float = 1.0,
        standard_normal_prior_warmup_steps: int = 0,
        standard_normal_prior_ramp_steps: int = 0,
        standard_normal_prior_safe_mode: bool = False,
        standard_normal_prior_detach_outliers: bool = False,
        standard_normal_prior_skip_extreme_z1: bool = False,
        standard_normal_prior_extreme_z1_threshold: float = 30.0,
        standard_normal_prior_min_scale: float = 1e-4,
        standard_normal_prior_max_scale: float = 1e3,
        standard_normal_prior_max_abs_loc: float = 50.0,
        standard_normal_prior_max_reconstruction: float = 1e3,
        standard_normal_prior_max_kl: float = 1e4,
        standard_normal_prior_detach_scale_multiplier: float = 1.0,
        standard_normal_prior_detach_loss_multiplier: float = 1.0,
        n_hidden: int = 256,
        n_latent: int = 20,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 1,
        dropout_rate_encoder: float = 0.2,
        dropout_rate_decoder: float = 0.2,
        use_batch_norm: str = "both",
        use_layer_norm: str = "none",
        **kwargs,
    ):
        super().__init__(
            *args,
            n_labels=n_labels,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            dropout_rate_encoder=dropout_rate_encoder,
            dropout_rate_decoder=dropout_rate_decoder,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )
        self.n_labels = n_labels
        self.prior_spec = prior_spec
        self.protein_to_index = {name: idx for idx, name in enumerate(protein_names)}
        self._loss_call_counter = 0
        self.standard_normal_prior_enable = bool(standard_normal_prior_enable)
        self.standard_normal_prior_loss_weight = float(standard_normal_prior_loss_weight)
        self.standard_normal_prior_warmup_steps = int(standard_normal_prior_warmup_steps)
        self.standard_normal_prior_ramp_steps = int(standard_normal_prior_ramp_steps)
        self.standard_normal_prior_safe_mode = bool(standard_normal_prior_safe_mode)
        self.standard_normal_prior_detach_outliers = bool(standard_normal_prior_detach_outliers)
        self.standard_normal_prior_skip_extreme_z1 = bool(standard_normal_prior_skip_extreme_z1)
        self.standard_normal_prior_extreme_z1_threshold = float(standard_normal_prior_extreme_z1_threshold)
        self.standard_normal_prior_min_scale = float(standard_normal_prior_min_scale)
        self.standard_normal_prior_max_scale = float(standard_normal_prior_max_scale)
        self.standard_normal_prior_max_abs_loc = float(standard_normal_prior_max_abs_loc)
        self.standard_normal_prior_max_reconstruction = float(standard_normal_prior_max_reconstruction)
        self.standard_normal_prior_max_kl = float(standard_normal_prior_max_kl)
        self.standard_normal_prior_detach_scale_multiplier = float(standard_normal_prior_detach_scale_multiplier)
        self.standard_normal_prior_detach_loss_multiplier = float(standard_normal_prior_detach_loss_multiplier)

        use_batch_norm_encoder = use_batch_norm in {"encoder", "both"}
        use_layer_norm_encoder = use_layer_norm in {"encoder", "both"}

        self.classifier = Classifier(
            n_latent,
            n_hidden=n_hidden,
            n_labels=n_labels,
            n_layers=n_layers_encoder,
            dropout_rate=dropout_rate_encoder,
            logits=True,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.latent_prior_variant = "standard_normal_prior"
        self.label_conditioned_prior_enabled = False
        self.z_prior_conditioned_on_y = False
        self.z_prior_distribution = "standard_normal"

        for key in ["kappa", "median", "mad"]:
            self.register_buffer(
                f"protein_teacher_{key}",
                torch.as_tensor(protein_teacher_stats[key], dtype=torch.float32),
            )

    def _standard_normal_prior_weight(self) -> float:
        if not self.standard_normal_prior_enable:
            return 0.0
        target = max(float(self.standard_normal_prior_loss_weight), 0.0)
        if target <= 0:
            return 0.0
        warmup = max(int(self.standard_normal_prior_warmup_steps), 0)
        ramp = max(int(self.standard_normal_prior_ramp_steps), 0)
        step = int(self._loss_call_counter)
        if warmup <= 0 and ramp <= 0:
            return target
        if step < warmup:
            return 0.0
        if ramp <= 0:
            return target
        return target * float(min(1.0, max(0.0, (step - warmup + 1) / ramp)))

    def _protein_teacher_values(self, y: torch.Tensor) -> torch.Tensor:
        expected = int(self.protein_teacher_kappa.shape[0])
        actual = int(y.shape[-1])
        if actual != expected:
            raise RuntimeError(
                "Protein teacher shape mismatch: minibatch protein tensor has "
                f"{actual} columns but teacher stats expect {expected}. "
                "This usually means `protein_expression_heldout`, `protein_names`, "
                "or the fitted teacher stats were built from different protein panels."
            )
        transformed = torch.asinh(y / self.protein_teacher_kappa.clamp_min(1e-6))
        teacher = (transformed - self.protein_teacher_median) / self.protein_teacher_mad.clamp_min(1e-6)
        return teacher.clamp(-3.0, 3.0)

    def _protein_mask_for_minibatch(self, y: torch.Tensor, panel_index: torch.Tensor) -> torch.Tensor | None:
        if self.protein_batch_mask is None:
            return None
        mask = torch.zeros_like(y)
        for panel in torch.unique(panel_index):
            panel_rows = (panel_index == panel).reshape(-1)
            panel_mask = self.protein_batch_mask[str(int(panel.item()))].astype(np.float32)
            if int(panel_mask.shape[0]) != int(y.shape[-1]):
                raise RuntimeError(
                    "Protein panel mask shape mismatch: panel "
                    f"{int(panel.item())} has mask width {int(panel_mask.shape[0])}, "
                    f"but minibatch protein tensor has width {int(y.shape[-1])}."
                )
            mask[panel_rows] = torch.tensor(panel_mask, device=y.device)
        return mask

    def _standard_normal_prior_raw_terms(self, qz1: Normal, z1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros(z1.shape[0], dtype=z1.dtype, device=z1.device)
        standard = Normal(torch.zeros_like(qz1.loc), torch.ones_like(qz1.scale))
        standard_normal_kl = kl(qz1, standard).sum(dim=-1)
        return zero, standard_normal_kl

    def _standard_normal_prior_terms(
        self,
        qz1: Normal,
        z1: torch.Tensor,
        *,
        tensors: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        standard_normal_z_reconstruction, standard_normal_kl = self._standard_normal_prior_raw_terms(qz1, z1)
        z1_max_abs = z1.detach().abs().amax(dim=1)
        if self.standard_normal_prior_skip_extreme_z1:
            threshold = max(float(self.standard_normal_prior_extreme_z1_threshold), 0.0)
            skip_extreme_z1 = z1_max_abs > threshold
            if bool(skip_extreme_z1.any().detach().cpu().item()):
                standard_normal_z_reconstruction = torch.where(
                    skip_extreme_z1,
                    torch.zeros_like(standard_normal_z_reconstruction).detach(),
                    standard_normal_z_reconstruction,
                )
                standard_normal_kl = torch.where(skip_extreme_z1, torch.zeros_like(standard_normal_kl).detach(), standard_normal_kl)
        if self.standard_normal_prior_safe_mode:
            max_kl = max(float(self.standard_normal_prior_max_kl), 1.0)
            standard_normal_kl_safe = torch.nan_to_num(
                standard_normal_kl,
                nan=max_kl,
                posinf=max_kl,
                neginf=0.0,
            ).clamp(0.0, max_kl)
            if self.standard_normal_prior_detach_outliers:
                loss_buffer = max(float(self.standard_normal_prior_detach_loss_multiplier), 1.0)
                standard_normal_kl_bad = (~torch.isfinite(standard_normal_kl)) | (standard_normal_kl < 0) | (
                    standard_normal_kl > (max_kl * loss_buffer)
                )
                standard_normal_kl = torch.where(standard_normal_kl_bad, standard_normal_kl_safe.detach(), standard_normal_kl_safe)
            else:
                standard_normal_kl = standard_normal_kl_safe
        return standard_normal_z_reconstruction, standard_normal_kl

    def _standard_normal_prior_terms_restricted(
        self,
        qz1: Normal,
        z1: torch.Tensor,
        allowed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._standard_normal_prior_terms(qz1, z1)

    def classification_loss(self, tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.classify_tensors(tensors)
        true_labels = tensors[REGISTRY_KEYS.LABELS_KEY].long().reshape(-1)
        if logits.ndim > 2:
            # scvi's labelled sampler can add a singleton/grouping axis when
            # n_samples_per_label is large.  Flatten before returning logits so
            # SemiSupervisedTrainingPlan/torchmetrics see standard class logits.
            logits = logits.reshape(-1, logits.shape[-1])
        if logits.shape[0] != true_labels.shape[0]:
            raise RuntimeError(
                "Labelled CE shape mismatch after flattening: "
                f"logits={tuple(logits.shape)}, labels={tuple(true_labels.shape)}"
            )
        return F.cross_entropy(logits, true_labels), true_labels, logits

    def classify_tensors(self, tensors: dict[str, torch.Tensor], use_posterior_mean: bool = True) -> torch.Tensor:
        inference_inputs = self._get_inference_input(tensors)
        inference_outputs = self.inference(**inference_inputs)
        z = inference_outputs["qz"].loc if use_posterior_mean else inference_outputs["z"]
        return self.classifier(z)

    @auto_move_data
    def classify(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_index: torch.Tensor | None = None,
        panel_index: torch.Tensor | None = None,
        cont_covs=None,
        cat_covs=None,
        use_posterior_mean: bool = True,
    ) -> torch.Tensor:
        if panel_index is None:
            panel_index = batch_index
        inference_outputs = self.inference(
            x=x,
            y=y,
            batch_index=batch_index,
            panel_index=panel_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        z = inference_outputs["qz"].loc if use_posterior_mean else inference_outputs["z"]
        return self.classifier(z)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        pro_recons_weight=1.0,
        kl_weight=1.0,
        labelled_tensors: dict[str, torch.Tensor] | None = None,
        classification_ratio: float | None = None,
        epoch: int | None = None,
    ):
        qz = inference_outputs["qz"]
        ql = inference_outputs["ql"]
        z = inference_outputs["z"]
        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]
        per_batch_efficiency = generative_outputs["per_batch_efficiency"]

        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        panel_index = tensors[self.panel_key]

        protein_mask = self._protein_mask_for_minibatch(y, panel_index)
        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x,
            y,
            px_,
            py_,
            protein_mask,
            per_batch_efficiency,
        )

        standard_normal_prior_weight = self._standard_normal_prior_weight()
        if standard_normal_prior_weight > 0:
            standard_normal_z_reconstruction, standard_normal_kl = self._standard_normal_prior_terms(qz, z, tensors=tensors)
        else:
            standard_normal_z_reconstruction = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            standard_normal_kl = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                one_hot(batch_index.squeeze(-1), n_batch).float(),
                self.library_log_means,
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index.squeeze(-1), n_batch).float(),
                self.library_log_vars,
            )
            kl_div_l_gene = kl(ql, Normal(local_library_log_means, torch.sqrt(local_library_log_vars))).sum(dim=1)
        else:
            kl_div_l_gene = torch.zeros_like(standard_normal_kl)

        kl_div_back_pro_full = kl(
            Normal(py_["back_alpha"], py_["back_beta"]),
            inference_outputs["back_mean_prior"],
        )
        lkl_back_pro_full = -torch.distributions.LogNormal(
            torch.tensor([0.0], device=x.device),
            torch.tensor([1.0], device=x.device),
        ).log_prob(per_batch_efficiency)
        lkl_protein_expressed = -1e-3 * torch.distributions.Bernoulli(
            logits=py_["mixing"]
        ).log_prob(torch.ones_like(py_["mixing"]))
        if protein_mask is not None:
            kl_div_back_pro = protein_mask.bool() * kl_div_back_pro_full
            kl_div_back_pro = (
                kl_div_back_pro.sum(dim=1)
                + lkl_back_pro_full.sum(dim=1)
                + lkl_protein_expressed.sum(dim=1)
            )
        else:
            kl_div_back_pro = (
                kl_div_back_pro_full.sum(dim=1)
                + lkl_back_pro_full.sum(dim=1)
                + lkl_protein_expressed.sum(dim=1)
            )

        reconstruction_prior_loss = torch.mean(
            reconst_loss_gene
            + kl_weight * pro_recons_weight * reconst_loss_protein
            + float(standard_normal_prior_weight) * standard_normal_z_reconstruction
            + kl_weight * float(standard_normal_prior_weight) * standard_normal_kl
            + kl_div_l_gene
            + kl_weight * kl_div_back_pro
        )
        loss = reconstruction_prior_loss

        # Compatibility classifier pass: this extra forward preserves the
        # BatchNorm/dropout side effects used by the historical training path.
        # It has zero objective contribution and is kept only for reproducible
        # teacher refinement behavior.
        q_c = F.softmax(self.classifier(z), dim=-1)
        compatibility_loss = torch.zeros((), device=q_c.device)
        loss = loss + compatibility_loss

        ce_loss = true_labels = logits = None
        if labelled_tensors is not None:
            ce_loss, true_labels, logits = self.classification_loss(labelled_tensors)
            loss = loss + float(classification_ratio or 0.0) * ce_loss

        self._loss_call_counter += 1

        return LossOutput(
            loss=loss,
            reconstruction_loss={
                "reconst_loss_gene": reconst_loss_gene,
                "reconst_loss_protein": reconst_loss_protein,
                "standard_normal_z_reconstruction": standard_normal_z_reconstruction,
            },
            kl_local={
                "standard_normal_kl": standard_normal_kl,
                "kl_div_l_gene": kl_div_l_gene,
                "kl_div_back_pro": kl_div_back_pro,
            },
            classification_loss=ce_loss,
            true_labels=true_labels,
            logits=logits,
            extra_metrics={
                "reconstruction_prior_loss": reconstruction_prior_loss.detach(),
            },
        )


class _AnchorTeacherTrainingPlan(SemiSupervisedTrainingPlan):
    def forward(self, *args, **kwargs):
        self.loss_kwargs["epoch"] = int(self.current_epoch)
        return super().forward(*args, **kwargs)

    def compute_and_log_metrics(self, loss_output: LossOutput, metrics: dict, mode: str):
        from scvi.train._trainingplans import TrainingPlan

        TrainingPlan.compute_and_log_metrics(self, loss_output, metrics, mode)
        self._compute_and_log_classification_metrics_compat(loss_output, mode)
        if loss_output.extra_metrics is None:
            return
        for key in ["reconstruction_prior_loss"]:
            if key in loss_output.extra_metrics:
                self.log_with_mode(
                    key,
                    loss_output.extra_metrics[key],
                    mode,
                    on_step=self.on_step,
                    on_epoch=self.on_epoch,
                    batch_size=loss_output.n_obs_minibatch,
                )

    def _compute_and_log_classification_metrics_compat(self, loss_output: LossOutput, mode: str):
        if loss_output.classification_loss is None:
            return
        if loss_output.true_labels is None or loss_output.logits is None:
            return

        classification_loss = loss_output.classification_loss
        logits = loss_output.logits
        true_labels = loss_output.true_labels.reshape(-1).long()

        logits_2d = logits.reshape(-1, logits.shape[-1])
        predicted_labels = torch.argmax(logits_2d, dim=-1).reshape(-1).long()
        if predicted_labels.numel() != true_labels.numel():
            true_labels = true_labels.reshape(predicted_labels.shape)

        valid = (true_labels >= 0) & (true_labels < int(self.n_classes))
        if not torch.any(valid):
            return

        predicted_labels = predicted_labels[valid]
        true_labels = true_labels[valid]
        logits_2d = logits_2d[valid]

        accuracy = (predicted_labels == true_labels).to(logits_2d.dtype).mean()
        f1 = accuracy
        try:
            import torchmetrics.functional as tmf

            ce = tmf.classification.multiclass_calibration_error(
                logits_2d.float(),
                true_labels,
                int(self.n_classes),
            )
        except Exception:
            ce = torch.tensor(float("nan"), device=logits_2d.device)

        for key, value in [
            ("classification_loss", classification_loss),
            ("accuracy", accuracy),
            ("f1_score", f1),
            ("calibration_error", ce),
        ]:
            self.log_with_mode(
                key,
                value,
                mode,
                on_step=self.on_step,
                on_epoch=self.on_epoch,
                batch_size=loss_output.n_obs_minibatch,
            )
