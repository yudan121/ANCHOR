from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping

import anndata as ad
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager, fields
from scvi.data.fields import LabelsWithUnlabeledObsField
from scvi.model._totalvi import TOTALVI
from scvi.model.base._training_mixin import SemisupervisedTrainingMixin
from scvi.module.base import LossOutput

from .module import _AnchorTeacherModule, _AnchorTeacherTrainingPlan


# scvi-tools separates a high-level model wrapper from the PyTorch module.
# This base wrapper registers the AnnData fields used by the initial teacher:
# RNA counts, labels, batches, protein counts, and optional covariates.
class _AnchorTeacherBaseModel(SemisupervisedTrainingMixin, TOTALVI):
    _module_cls = _AnchorTeacherModule
    _training_plan_cls = _AnchorTeacherTrainingPlan

    def __init__(
        self,
        adata: ad.AnnData,
        *,
        n_labels: int,
        prior_spec: dict[str, Any],
        protein_names: Sequence[str],
        protein_teacher_stats: dict[str, Any],
        n_latent: int = 20,
        **model_kwargs,
    ):
        super().__init__(
            adata,
            n_latent=n_latent,
            n_labels=n_labels,
            prior_spec=prior_spec,
            protein_names=protein_names,
            protein_teacher_stats=protein_teacher_stats,
            **model_kwargs,
        )
        self._set_indices_and_labels()
        self.n_labels = n_labels
        self.was_pretrained = True

    @classmethod
    def setup_anndata(
        cls,
        adata: ad.AnnData,
        protein_expression_obsm_key: str,
        labels_key: str,
        unlabeled_category: str,
        protein_names_uns_key: str | None = None,
        batch_key: str | None = None,
        panel_key: str | None = None,
        layer: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        if panel_key is not None:
            batch_field = fields.CategoricalObsField("panel", panel_key)
        else:
            batch_field = fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            LabelsWithUnlabeledObsField(REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category),
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            fields.NumericalObsField(ANCHOR_ADATA_INDEX_KEY, ANCHOR_ADATA_INDEX_KEY, required=False),
            fields.NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            fields.CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            fields.NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            fields.ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=True,
                batch_field=batch_field,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=True,
            ),
        ]
        if panel_key is not None:
            anndata_fields.insert(0, fields.CategoricalObsField("panel", panel_key))
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)


class AnchorTeacherModel(_AnchorTeacherBaseModel):
    """Initial teacher used before query pseudo-label supervision is added."""

    _module_cls = _AnchorTeacherModule
    _training_plan_cls = _AnchorTeacherTrainingPlan


PAIR_QUERY_PSEUDO_SELECTED_KEY = "pair_query_pseudolabel_selected"
PAIR_QUERY_PSEUDO_TARGET_KEY = "pair_query_pseudolabel_target"
PAIR_QUERY_PSEUDO_SOURCE_PAIR_KEY = "pair_query_pseudolabel_source_pair"
PAIR_QUERY_PSEUDO_STRATEGY_KEY = "pair_query_pseudolabel_strategy"
ANCHOR_ADATA_INDEX_KEY = "anchor_adata_index"


@dataclass
class PairQueryPseudoLabelBundle:
    cell_level: pd.DataFrame
    summary: pd.DataFrame
    conflicts: pd.DataFrame
    counts_by_label: pd.DataFrame


# Adds query pseudo-label cross-entropy on top of the base teacher loss.
# The selected cells and target labels are read from AnnData fields registered
# by _AnchorPseudoLabelBaseModel below.
class _AnchorPseudoLabelModule(_AnchorTeacherModule):
    def __init__(self, *args, query_pseudolabel_classification_ratio: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_pseudolabel_classification_ratio = float(query_pseudolabel_classification_ratio)

    def _query_pseudolabel_loss(self, logits_c: torch.Tensor, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        zero = torch.zeros((), device=logits_c.device)
        if self.query_pseudolabel_classification_ratio <= 0:
            return zero
        selected = tensors[PAIR_QUERY_PSEUDO_SELECTED_KEY].reshape(-1) > 0.5
        pseudo_targets = tensors[PAIR_QUERY_PSEUDO_TARGET_KEY].reshape(-1).long()
        valid = selected & pseudo_targets.ge(0) & pseudo_targets.lt(int(self.n_labels))
        if not valid.any():
            return zero
        pseudo_ce = F.cross_entropy(logits_c[valid], pseudo_targets[valid], reduction="mean")
        return float(self.query_pseudolabel_classification_ratio) * pseudo_ce

    def loss(self, tensors, *args, **kwargs):
        loss_output = super().loss(tensors, *args, **kwargs)
        inference_outputs = args[0] if len(args) > 0 else kwargs.get("inference_outputs")
        if inference_outputs is None:
            if loss_output.extra_metrics is None:
                loss_output.extra_metrics = {}
            loss_output.extra_metrics["query_pseudo_ce_loss"] = torch.zeros((), device=loss_output.loss.device)
            return loss_output
        logits_c = self.classifier(inference_outputs["z"])
        query_pseudo_ce_loss = self._query_pseudolabel_loss(logits_c, tensors)
        loss_output.loss = loss_output.loss + query_pseudo_ce_loss
        if loss_output.extra_metrics is None:
            loss_output.extra_metrics = {}
        loss_output.extra_metrics["query_pseudo_ce_loss"] = query_pseudo_ce_loss.detach()
        return loss_output


# Extends the base teacher training plan only by logging query pseudo-label CE.
class _AnchorPseudoLabelTrainingPlan(_AnchorTeacherTrainingPlan):
    def compute_and_log_metrics(self, loss_output: LossOutput, metrics: dict, mode: str):
        super().compute_and_log_metrics(loss_output, metrics, mode)
        if loss_output.extra_metrics is None:
            return
        if "query_pseudo_ce_loss" in loss_output.extra_metrics:
            self.log_with_mode(
                "query_pseudo_ce_loss",
                loss_output.extra_metrics["query_pseudo_ce_loss"],
                mode,
                on_step=self.on_step,
                on_epoch=self.on_epoch,
                batch_size=loss_output.n_obs_minibatch,
            )


# Model wrapper for modules that consume query pseudo-label fields.  Compared
# with _AnchorTeacherBaseModel, it registers two extra per-cell fields:
# whether a query cell is selected and its pseudo-label target index.
class _AnchorPseudoLabelBaseModel(_AnchorTeacherBaseModel):
    _module_cls = _AnchorPseudoLabelModule
    _training_plan_cls = _AnchorPseudoLabelTrainingPlan

    @classmethod
    def setup_anndata(
        cls,
        adata: ad.AnnData,
        protein_expression_obsm_key: str,
        labels_key: str,
        unlabeled_category: str,
        pseudo_selected_key: str = PAIR_QUERY_PSEUDO_SELECTED_KEY,
        pseudo_target_key: str = PAIR_QUERY_PSEUDO_TARGET_KEY,
        protein_names_uns_key: str | None = None,
        batch_key: str | None = None,
        panel_key: str | None = None,
        layer: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        if panel_key is not None:
            batch_field = fields.CategoricalObsField("panel", panel_key)
        else:
            batch_field = fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            LabelsWithUnlabeledObsField(REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category),
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            fields.NumericalObsField(ANCHOR_ADATA_INDEX_KEY, ANCHOR_ADATA_INDEX_KEY, required=False),
            fields.NumericalObsField(PAIR_QUERY_PSEUDO_SELECTED_KEY, pseudo_selected_key),
            fields.NumericalObsField(PAIR_QUERY_PSEUDO_TARGET_KEY, pseudo_target_key),
            fields.NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            fields.CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            fields.NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            fields.ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=True,
                batch_field=batch_field,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=True,
            ),
        ]
        if panel_key is not None:
            anndata_fields.insert(0, fields.CategoricalObsField("panel", panel_key))
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)


def apply_pair_query_pseudolabels(
    adata: ad.AnnData,
    pseudolabel_bundle: PairQueryPseudoLabelBundle,
    *,
    selected_key: str = PAIR_QUERY_PSEUDO_SELECTED_KEY,
    target_key: str = PAIR_QUERY_PSEUDO_TARGET_KEY,
    source_pair_key: str = PAIR_QUERY_PSEUDO_SOURCE_PAIR_KEY,
    strategy_key: str = PAIR_QUERY_PSEUDO_STRATEGY_KEY,
) -> None:
    """Write selected query pseudo-labels into AnnData for the next teacher round."""

    obs_index = pd.Index(adata.obs_names.astype(str))
    adata.obs[selected_key] = np.zeros(obs_index.shape[0], dtype=np.float32)
    adata.obs[target_key] = np.full(obs_index.shape[0], -1, dtype=np.int64)
    adata.obs[source_pair_key] = ""
    adata.obs[strategy_key] = ""
    if pseudolabel_bundle.cell_level.empty:
        return
    used = pseudolabel_bundle.cell_level.loc[pseudolabel_bundle.cell_level["used_for_training"].astype(bool)].copy()
    if used.empty:
        return
    used = used.drop_duplicates(subset=["obs_name"], keep="first").copy()
    used = used.set_index("obs_name").reindex(obs_index.intersection(pd.Index(used["obs_name"].astype(str)))).dropna(how="all")
    if used.empty:
        return
    adata.obs.loc[used.index, selected_key] = 1.0
    adata.obs.loc[used.index, target_key] = used["pseudo_target_index"].astype(int).to_numpy()
    adata.obs.loc[used.index, source_pair_key] = used["pair_key"].astype(str).to_numpy()
    adata.obs.loc[used.index, strategy_key] = used["strategy"].astype(str).to_numpy()


SMALLCLASS_CE_MODE_OFF = "off"
SMALLCLASS_CE_MODE_OVERSAMPLE = "oversample"
DEFAULT_SMALLCLASS_MIN_EFFECTIVE_SELECTED = 50
DEFAULT_SMALLCLASS_MAX_REPEATS_PER_CELL = 9


@dataclass
class SmallClassAugmentationConfig:
    repeat_by_label: list[int]
    positive_protein_indices_by_label: dict[int, list[int]]
    negative_protein_indices_by_label: dict[int, list[int]]
    positive_gene_indices_by_label: dict[int, list[int]]
    negative_gene_indices_by_label: dict[int, list[int]]


def build_smallclass_repeat_table(
    by_class: pd.DataFrame,
    *,
    label_categories: Sequence[str],
    min_effective_selected_per_class: int = DEFAULT_SMALLCLASS_MIN_EFFECTIVE_SELECTED,
    max_repeats_per_cell: int = DEFAULT_SMALLCLASS_MAX_REPEATS_PER_CELL,
) -> pd.DataFrame:
    """Compute how many extra CE-weighted views are needed for small pseudo-label classes."""

    rows: list[dict[str, Any]] = []
    by_label = by_class.set_index(by_class["target_label"].astype(str)) if not by_class.empty else pd.DataFrame()
    for label_index, label in enumerate([str(label) for label in label_categories]):
        n_selected = 0
        if not by_label.empty and label in by_label.index and "n_selected" in by_label.columns:
            value = by_label.loc[label, "n_selected"]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            n_selected = int(value) if pd.notna(value) else 0
        repeat = 0
        if 0 < n_selected < int(min_effective_selected_per_class):
            repeat = int(math.ceil(float(min_effective_selected_per_class) / float(n_selected))) - 1
            repeat = max(0, min(int(max_repeats_per_cell), repeat))
        rows.append(
            {
                "label_index": int(label_index),
                "target_label": label,
                "n_selected": int(n_selected),
                "aug_repeats_per_cell": int(repeat),
                "n_augmented_views": int(n_selected * repeat),
                "effective_training_count": int(n_selected * (1 + repeat)),
            }
        )
    return pd.DataFrame(rows)


def _marker_indices(markers: Sequence[str], feature_names: Sequence[str]) -> list[int]:
    feature_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    return [feature_to_idx[str(marker)] for marker in markers if str(marker) in feature_to_idx]


def build_smallclass_augmentation_config(
    repeat_table: pd.DataFrame,
    *,
    label_categories: Sequence[str],
    leaf_marker_specs: Mapping[str, Mapping[str, Sequence[str]]],
    protein_names: Sequence[str],
    gene_names: Sequence[str],
) -> SmallClassAugmentationConfig:
    """Convert the repeat table and marker specs into module-ready augmentation metadata."""

    repeat_by_label = [0 for _ in label_categories]
    if not repeat_table.empty:
        for row in repeat_table.itertuples(index=False):
            repeat_by_label[int(row.label_index)] = int(row.aug_repeats_per_cell)
    pos_protein: dict[int, list[int]] = {}
    neg_protein: dict[int, list[int]] = {}
    pos_gene: dict[int, list[int]] = {}
    neg_gene: dict[int, list[int]] = {}
    for label_index, label in enumerate([str(label) for label in label_categories]):
        spec = leaf_marker_specs.get(label, {})
        pos_protein[label_index] = _marker_indices(spec.get("positive", []), protein_names)
        neg_protein[label_index] = _marker_indices(spec.get("negative", []), protein_names)
        pos_gene[label_index] = []
        neg_gene[label_index] = []
    return SmallClassAugmentationConfig(repeat_by_label, pos_protein, neg_protein, pos_gene, neg_gene)


# Adds small-class pseudo-label CE on top of the generic query pseudo-label CE.
# The regular pseudo-label CE treats selected query anchors equally; this layer
# gives under-selected classes extra effective weight when oversampling is on.
class _AnchorPseudoTeacherModule(_AnchorPseudoLabelModule):
    def __init__(
        self,
        *args,
        smallclass_ce_mode: str = SMALLCLASS_CE_MODE_OFF,
        smallclass_repeat_by_label: Sequence[int] | None = None,
        smallclass_positive_protein_indices_by_label: Mapping[int, Sequence[int]] | None = None,
        smallclass_negative_protein_indices_by_label: Mapping[int, Sequence[int]] | None = None,
        smallclass_positive_gene_indices_by_label: Mapping[int, Sequence[int]] | None = None,
        smallclass_negative_gene_indices_by_label: Mapping[int, Sequence[int]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        mode = str(smallclass_ce_mode)
        if mode not in {SMALLCLASS_CE_MODE_OFF, SMALLCLASS_CE_MODE_OVERSAMPLE}:
            raise ValueError(f"Unknown smallclass_ce_mode={mode!r}")
        self.smallclass_ce_mode = mode
        repeat = list(smallclass_repeat_by_label or [0] * int(self.n_labels))
        if len(repeat) != int(self.n_labels):
            raise ValueError("smallclass_repeat_by_label length must match n_labels")
        self.smallclass_repeat_by_label = torch.as_tensor(repeat, dtype=torch.long)
        self.smallclass_positive_protein_indices_by_label = {
            int(k): [int(v) for v in values]
            for k, values in (smallclass_positive_protein_indices_by_label or {}).items()
        }
        self.smallclass_negative_protein_indices_by_label = {
            int(k): [int(v) for v in values]
            for k, values in (smallclass_negative_protein_indices_by_label or {}).items()
        }
        self.smallclass_positive_gene_indices_by_label = {
            int(k): [int(v) for v in values]
            for k, values in (smallclass_positive_gene_indices_by_label or {}).items()
        }
        self.smallclass_negative_gene_indices_by_label = {
            int(k): [int(v) for v in values]
            for k, values in (smallclass_negative_gene_indices_by_label or {}).items()
        }

    def _query_pseudolabel_smallclass_loss(self, logits_c: torch.Tensor, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        zero = torch.zeros((), device=logits_c.device)
        if self.query_pseudolabel_classification_ratio <= 0 or self.smallclass_ce_mode != SMALLCLASS_CE_MODE_OVERSAMPLE:
            return zero
        selected = tensors[PAIR_QUERY_PSEUDO_SELECTED_KEY].reshape(-1) > 0.5
        pseudo_targets = tensors[PAIR_QUERY_PSEUDO_TARGET_KEY].reshape(-1).long()
        valid = selected & pseudo_targets.ge(0) & pseudo_targets.lt(int(self.n_labels))
        if not valid.any():
            return zero
        repeat_by_label = self.smallclass_repeat_by_label.to(device=logits_c.device)
        repeats = repeat_by_label[pseudo_targets.clamp(0, int(self.n_labels) - 1)]
        valid = valid & repeats.gt(0)
        if not valid.any():
            return zero
        ce = F.cross_entropy(logits_c[valid], pseudo_targets[valid], reduction="none")
        weights = repeats[valid].to(dtype=ce.dtype)
        return float(self.query_pseudolabel_classification_ratio) * (ce * weights).sum() / weights.sum().clamp_min(1.0)

    def loss(self, tensors, *args, **kwargs):
        loss_output = super().loss(tensors, *args, **kwargs)
        inference_outputs = args[0] if len(args) > 0 else kwargs.get("inference_outputs")
        if inference_outputs is None:
            if loss_output.extra_metrics is None:
                loss_output.extra_metrics = {}
            loss_output.extra_metrics["query_pseudo_smallclass_ce_loss"] = torch.zeros((), device=loss_output.loss.device)
            return loss_output
        smallclass_loss = self._query_pseudolabel_smallclass_loss(self.classifier(inference_outputs["z"]), tensors)
        loss_output.loss = loss_output.loss + smallclass_loss
        if loss_output.extra_metrics is None:
            loss_output.extra_metrics = {}
        loss_output.extra_metrics["query_pseudo_smallclass_ce_loss"] = smallclass_loss.detach()
        return loss_output


# Extends pseudo-label logging with the small-class CE term.
class _AnchorPseudoTeacherTrainingPlan(_AnchorPseudoLabelTrainingPlan):
    def compute_and_log_metrics(self, loss_output: LossOutput, metrics: dict, mode: str):
        super().compute_and_log_metrics(loss_output, metrics, mode)
        if loss_output.extra_metrics is None:
            return
        if "query_pseudo_smallclass_ce_loss" in loss_output.extra_metrics:
            self.log_with_mode(
                "query_pseudo_smallclass_ce_loss",
                loss_output.extra_metrics["query_pseudo_smallclass_ce_loss"],
                mode,
                on_step=self.on_step,
                on_epoch=self.on_epoch,
                batch_size=loss_output.n_obs_minibatch,
            )


class AnchorPseudoTeacherModel(_AnchorPseudoLabelBaseModel):
    """Teacher used in refinement rounds with query pseudo-label supervision."""

    _module_cls = _AnchorPseudoTeacherModule
    _training_plan_cls = _AnchorPseudoTeacherTrainingPlan
