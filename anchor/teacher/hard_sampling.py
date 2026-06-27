from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.dataloaders import AnnDataLoader, SemiSupervisedDataLoader, SemiSupervisedDataSplitter

def _move_tensor_tree_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_tensor_tree_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_tensor_tree_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree_to_device(v, device) for v in value)
    return value


class HardWeightedSemiSupervisedDataLoader(SemiSupervisedDataLoader):
    """Class-balanced labelled sampler with mutable per-cell weights."""

    def __init__(
        self,
        *args: Any,
        hard_ref_sampling_wrong_weight: float = 10.0,
        hard_ref_sampling_correct_weight: float = 1.0,
        hard_ref_sampling_max_wrong_fraction: float | None = None,
        hard_ref_sampling_min_wrong_per_label: int | None = None,
        hard_ref_sampling_seed: int = 0,
        **kwargs: Any,
    ):
        # Be defensive: scvi stores splitter extras in data_loader_kwargs, so
        # custom hard-sampling keys must never fall through to PyTorch DataLoader.
        hard_ref_sampling_wrong_weight = kwargs.pop(
            "hard_ref_sampling_wrong_weight", hard_ref_sampling_wrong_weight
        )
        hard_ref_sampling_correct_weight = kwargs.pop(
            "hard_ref_sampling_correct_weight", hard_ref_sampling_correct_weight
        )
        hard_ref_sampling_max_wrong_fraction = kwargs.pop(
            "hard_ref_sampling_max_wrong_fraction", hard_ref_sampling_max_wrong_fraction
        )
        hard_ref_sampling_min_wrong_per_label = kwargs.pop(
            "hard_ref_sampling_min_wrong_per_label", hard_ref_sampling_min_wrong_per_label
        )
        hard_ref_sampling_seed = kwargs.pop("hard_ref_sampling_seed", hard_ref_sampling_seed)
        self.hard_ref_sampling_wrong_weight = float(hard_ref_sampling_wrong_weight)
        self.hard_ref_sampling_correct_weight = float(hard_ref_sampling_correct_weight)
        self.hard_ref_sampling_max_wrong_fraction = (
            None if hard_ref_sampling_max_wrong_fraction is None else float(hard_ref_sampling_max_wrong_fraction)
        )
        self.hard_ref_sampling_min_wrong_per_label = (
            None if hard_ref_sampling_min_wrong_per_label is None else int(hard_ref_sampling_min_wrong_per_label)
        )
        self.hard_ref_sampling_rng = np.random.default_rng(int(hard_ref_sampling_seed))
        self.hard_ref_sampling_weights: np.ndarray | None = None
        self.hard_ref_sampling_last_sampled: np.ndarray = np.zeros(0, dtype=np.int64)
        self.hard_ref_sampling_epoch = -1
        super().__init__(*args, **kwargs)
        n_obs = int(self.adata_manager.adata.n_obs)
        if self.hard_ref_sampling_weights is None:
            self.hard_ref_sampling_weights = np.ones(n_obs, dtype=np.float64)

    def subsample_labels(self):
        """Subsample each label class using current per-cell weights."""
        if self.n_samples_per_label is None:
            sampled = np.concatenate(self.labeled_locs)
            self.hard_ref_sampling_last_sampled = sampled.astype(np.int64, copy=False)
            return sampled

        sample_idx = []
        for loc in self.labeled_locs:
            loc = np.asarray(loc, dtype=np.int64)
            if loc.size == 0:
                continue
            n_draw = int(self.n_samples_per_label)
            weights = None
            if self.hard_ref_sampling_weights is not None:
                weights = np.asarray(self.hard_ref_sampling_weights[loc], dtype=np.float64)
                weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
                if float(weights.sum()) <= 0:
                    weights = None
                else:
                    weights = weights / float(weights.sum())
            if (
                weights is not None
                and self.hard_ref_sampling_max_wrong_fraction is not None
                and self.hard_ref_sampling_max_wrong_fraction < 1.0
            ):
                raw_weights = np.asarray(self.hard_ref_sampling_weights[loc], dtype=np.float64)
                hard_mask = raw_weights > (self.hard_ref_sampling_correct_weight + 1e-12)
                hard_loc = loc[hard_mask]
                easy_loc = loc[~hard_mask]
                if hard_loc.size and easy_loc.size:
                    max_hard = int(np.floor(n_draw * max(0.0, self.hard_ref_sampling_max_wrong_fraction)))
                    min_hard = 0
                    if self.hard_ref_sampling_min_wrong_per_label is not None and max_hard > 0:
                        min_hard = min(int(self.hard_ref_sampling_min_wrong_per_label), max_hard)
                    hard_mass = float(np.clip(weights[hard_mask].sum(), 0.0, 1.0))
                    n_hard = min(max_hard, max(min_hard, int(np.rint(n_draw * hard_mass))))
                    n_easy = n_draw - n_hard
                    parts = []
                    if n_hard:
                        hard_weights = raw_weights[hard_mask]
                        hard_weights = hard_weights / float(hard_weights.sum())
                        parts.append(
                            self.hard_ref_sampling_rng.choice(
                                hard_loc,
                                n_hard,
                                replace=hard_loc.size < n_hard,
                                p=hard_weights,
                            )
                        )
                    if n_easy:
                        easy_weights = raw_weights[~hard_mask]
                        easy_weights = np.where(np.isfinite(easy_weights) & (easy_weights > 0), easy_weights, 0.0)
                        easy_probs = None if float(easy_weights.sum()) <= 0 else easy_weights / float(easy_weights.sum())
                        parts.append(
                            self.hard_ref_sampling_rng.choice(
                                easy_loc,
                                n_easy,
                                replace=easy_loc.size < n_easy,
                                p=easy_probs,
                            )
                        )
                    label_subset = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
                    self.hard_ref_sampling_rng.shuffle(label_subset)
                else:
                    replace = loc.size < n_draw
                    label_subset = self.hard_ref_sampling_rng.choice(loc, n_draw, replace=replace, p=weights)
            else:
                replace = loc.size < n_draw
                label_subset = self.hard_ref_sampling_rng.choice(loc, n_draw, replace=replace, p=weights)
            sample_idx.append(label_subset.astype(np.int64, copy=False))
        sampled = np.concatenate(sample_idx) if sample_idx else np.zeros(0, dtype=np.int64)
        self.hard_ref_sampling_last_sampled = sampled.astype(np.int64, copy=False)
        return sampled


class HardWeightedSemiSupervisedDataSplitter(SemiSupervisedDataSplitter):
    """Semi-supervised splitter that keeps hard-sampling kwargs out of base loaders."""

    data_loader_class = SemiSupervisedDataLoader

    def __init__(
        self,
        *args: Any,
        hard_ref_sampling_wrong_weight: float = 10.0,
        hard_ref_sampling_correct_weight: float = 1.0,
        hard_ref_sampling_max_wrong_fraction: float | None = None,
        hard_ref_sampling_min_wrong_per_label: int | None = None,
        hard_ref_sampling_seed: int = 0,
        **kwargs: Any,
    ):
        self.hard_ref_sampling_wrong_weight = float(
            kwargs.pop("hard_ref_sampling_wrong_weight", hard_ref_sampling_wrong_weight)
        )
        self.hard_ref_sampling_correct_weight = float(
            kwargs.pop("hard_ref_sampling_correct_weight", hard_ref_sampling_correct_weight)
        )
        self.hard_ref_sampling_max_wrong_fraction = kwargs.pop(
            "hard_ref_sampling_max_wrong_fraction", hard_ref_sampling_max_wrong_fraction
        )
        if self.hard_ref_sampling_max_wrong_fraction is not None:
            self.hard_ref_sampling_max_wrong_fraction = float(self.hard_ref_sampling_max_wrong_fraction)
        self.hard_ref_sampling_min_wrong_per_label = kwargs.pop(
            "hard_ref_sampling_min_wrong_per_label", hard_ref_sampling_min_wrong_per_label
        )
        if self.hard_ref_sampling_min_wrong_per_label is not None:
            self.hard_ref_sampling_min_wrong_per_label = int(self.hard_ref_sampling_min_wrong_per_label)
        self.hard_ref_sampling_seed = int(kwargs.pop("hard_ref_sampling_seed", hard_ref_sampling_seed))
        super().__init__(*args, **kwargs)

    def train_dataloader(self):
        """Create the hard-weighted train data loader."""
        return HardWeightedSemiSupervisedDataLoader(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=True,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            hard_ref_sampling_wrong_weight=self.hard_ref_sampling_wrong_weight,
            hard_ref_sampling_correct_weight=self.hard_ref_sampling_correct_weight,
            hard_ref_sampling_max_wrong_fraction=self.hard_ref_sampling_max_wrong_fraction,
            hard_ref_sampling_min_wrong_per_label=self.hard_ref_sampling_min_wrong_per_label,
            hard_ref_sampling_seed=self.hard_ref_sampling_seed,
            **self.data_loader_kwargs,
        )


class HardRefSamplingCallback(Callback):
    """Update hard-reference sampling weights from reference-train prediction errors."""

    def __init__(
        self,
        *,
        model_dir: Path,
        wrong_weight: float = 10.0,
        correct_weight: float = 1.0,
        source: str = "reference_train_pred_errors",
        batch_size: int = 512,
    ):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.wrong_weight = float(wrong_weight)
        self.correct_weight = float(correct_weight)
        self.source = str(source)
        self.batch_size = int(batch_size)
        self.epoch_rows: list[dict[str, Any]] = []
        self.class_rows: list[dict[str, Any]] = []

    @staticmethod
    def _find_train_loader(trainer: Any) -> HardWeightedSemiSupervisedDataLoader | None:
        dl = getattr(trainer, "train_dataloader", None)
        if isinstance(dl, HardWeightedSemiSupervisedDataLoader):
            return dl
        return None

    @staticmethod
    def _labels_for_indices(adata_manager: AnnDataManager, indices: np.ndarray) -> np.ndarray:
        labels_state_registry = adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        labels = np.asarray(
            adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
            if hasattr(adata_manager, "get_from_registry")
            else adata_manager.adata.obs[labels_state_registry.original_key].to_numpy()
        ).reshape(-1)
        unlabeled = labels_state_registry.unlabeled_category
        return labels[indices], unlabeled

    def _predict_errors(
        self,
        pl_module: Any,
        train_loader: HardWeightedSemiSupervisedDataLoader,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        labelled_indices = np.concatenate([np.asarray(x, dtype=np.int64) for x in train_loader.labeled_locs])
        return self._predict_errors_for_indices(pl_module, train_loader, labelled_indices)

    def _predict_errors_for_indices(
        self,
        pl_module: Any,
        train_loader: HardWeightedSemiSupervisedDataLoader,
        labelled_indices: np.ndarray,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        module = pl_module.module
        adata_manager = train_loader.adata_manager
        labelled_indices = np.asarray(labelled_indices, dtype=np.int64)
        if labelled_indices.size == 0:
            return pd.DataFrame(), np.zeros(0, dtype=np.int64)

        loader = AnnDataLoader(
            adata_manager,
            indices=labelled_indices,
            shuffle=False,
            batch_size=self.batch_size,
            data_and_attributes=train_loader.data_and_attributes,
            drop_last=False,
            **train_loader.data_loader_kwargs,
        )
        was_training = bool(module.training)
        module.eval()
        rows = []
        error_indices = []
        cursor = 0
        labels_state = adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        try:
            categorical_mapping = list(labels_state.categorical_mapping)
            unlabeled_code = categorical_mapping.index(labels_state.unlabeled_category)
        except Exception:
            unlabeled_code = None
        with torch.no_grad():
            for batch in loader:
                batch = _move_tensor_tree_to_device(batch, pl_module.device)
                logits = module.classify_tensors(batch)
                if logits.ndim > 2:
                    logits = logits.reshape(-1, logits.shape[-1])
                pred = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(int)
                true = batch[REGISTRY_KEYS.LABELS_KEY].detach().cpu().numpy().reshape(-1).astype(int)
                n = int(true.shape[0])
                batch_indices = labelled_indices[cursor : cursor + n]
                cursor += n
                if unlabeled_code is not None:
                    labelled_mask = true != int(unlabeled_code)
                    pred = pred[labelled_mask]
                    true = true[labelled_mask]
                    batch_indices = batch_indices[labelled_mask]
                    if true.size == 0:
                        continue

                correct = pred == true
                supervision_codes = None
                if hasattr(module, "supervision_code_to_desc_indices") and "partial_supervision_code" in batch:
                    supervision_codes = (
                        batch["partial_supervision_code"].detach().cpu().numpy().reshape(-1).astype(int)
                    )
                    if unlabeled_code is not None:
                        supervision_codes = supervision_codes[labelled_mask]
                    allowed = getattr(module, "supervision_code_to_desc_indices", {})
                    correct = np.array(
                        [
                            int(p) in set(int(x) for x in allowed.get(int(c), [int(t)]))
                            if int(c) >= 0
                            else bool(p == t)
                            for p, t, c in zip(pred, true, supervision_codes, strict=False)
                        ],
                        dtype=bool,
                    )
                err_idx = batch_indices[~correct]
                error_indices.append(err_idx)
                for idx, t, p, ok in zip(batch_indices, true, pred, correct, strict=False):
                    rows.append(
                        {
                            "adata_index": int(idx),
                            "true_label_code": int(t),
                            "pred_label_code": int(p),
                            "is_error": bool(not ok),
                        }
                    )
        if was_training:
            module.train()
        errors = np.concatenate(error_indices) if error_indices else np.zeros(0, dtype=np.int64)
        return pd.DataFrame(rows), errors.astype(np.int64, copy=False)

    def on_train_epoch_end(self, trainer, pl_module):
        train_loader = self._find_train_loader(trainer)
        if train_loader is None:
            return
        pred_df, error_indices = self._predict_errors(pl_module, train_loader)
        n_obs = int(train_loader.adata_manager.adata.n_obs)
        weights = np.full(n_obs, self.correct_weight, dtype=np.float64)
        if error_indices.size:
            weights[error_indices] = self.wrong_weight
        train_loader.hard_ref_sampling_weights = weights

        labels, _ = self._labels_for_indices(train_loader.adata_manager, pred_df["adata_index"].to_numpy(dtype=np.int64))
        pred_df["true_label_registry"] = labels
        epoch = int(getattr(trainer, "current_epoch", -1))
        total = int(len(pred_df))
        n_error = int(pred_df["is_error"].sum()) if not pred_df.empty else 0
        sampled = np.asarray(getattr(train_loader, "hard_ref_sampling_last_sampled", []), dtype=np.int64)
        self.epoch_rows.append(
            {
                "epoch": epoch,
                "source": self.source,
                "n_reference_train_labelled": total,
                "n_error": n_error,
                "error_rate": n_error / max(total, 1),
                "wrong_weight": self.wrong_weight,
                "correct_weight": self.correct_weight,
                "n_sampled_last_epoch": int(sampled.size),
                "n_error_sampled_last_epoch": int(np.isin(sampled, error_indices).sum()) if sampled.size else 0,
            }
        )
        if not pred_df.empty:
            for label_code, grp in pred_df.groupby("true_label_code", sort=True):
                class_idx = grp["adata_index"].to_numpy(dtype=np.int64)
                class_error = grp.loc[grp["is_error"], "adata_index"].to_numpy(dtype=np.int64)
                class_sampled = sampled[np.isin(sampled, class_idx)] if sampled.size else np.zeros(0, dtype=np.int64)
                self.class_rows.append(
                    {
                        "epoch": epoch,
                        "true_label_code": int(label_code),
                        "n_reference_train_labelled": int(len(grp)),
                        "n_error": int(len(class_error)),
                        "error_rate": len(class_error) / max(len(grp), 1),
                        "n_sampled_last_epoch": int(class_sampled.size),
                        "n_error_sampled_last_epoch": int(np.isin(class_sampled, class_error).sum())
                        if class_sampled.size
                        else 0,
                    }
                )
        self.model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.epoch_rows).to_csv(self.model_dir / "hard_ref_sampling_epoch_summary.csv", index=False)
        class_df = pd.DataFrame(self.class_rows)
        class_df.to_csv(self.model_dir / "hard_ref_sampling_by_class_epoch.csv", index=False)
        if not class_df.empty:
            latest = class_df.loc[class_df["epoch"].eq(epoch)].copy()
            latest.to_csv(self.model_dir / "reference_train_error_by_class.csv", index=False)
        try:
            datamodule = getattr(trainer, "datamodule", None)
            val_idx = getattr(datamodule, "val_idx", None)
            if val_idx is not None:
                val_idx = np.asarray(val_idx, dtype=np.int64)
                if val_idx.size:
                    val_df, _ = self._predict_errors_for_indices(pl_module, train_loader, val_idx)
                    if not val_df.empty:
                        val_summary = (
                            val_df.groupby("true_label_code", sort=True)["is_error"]
                            .agg(n_reference_val_labelled="count", n_error="sum")
                            .reset_index()
                        )
                        val_summary["error_rate"] = val_summary["n_error"] / val_summary["n_reference_val_labelled"].clip(lower=1)
                        val_summary["epoch"] = epoch
                        val_summary.to_csv(self.model_dir / "reference_val_error_by_class.csv", index=False)
        except Exception as exc:
            (self.model_dir / "reference_val_error_by_class_unavailable.txt").write_text(
                f"{type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
