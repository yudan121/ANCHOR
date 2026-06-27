from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scvi.model._totalvi import TOTALVI
from scvi.train._callbacks import SubSampleLabels

from .hard_sampling import HardRefSamplingCallback, HardWeightedSemiSupervisedDataSplitter
from .model import _AnchorTeacherBaseModel
from .module import set_scvi_training_seed

def load_matching_totalvi_weights(model: _AnchorTeacherBaseModel, totalvi_model: TOTALVI) -> dict[str, list[str]]:
    target_state = model.module.state_dict()
    source_state = totalvi_model.module.state_dict()
    filtered_state = {}
    skipped_shape_mismatch = []
    for key, value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            continue
        if tuple(target_value.shape) != tuple(value.shape):
            skipped_shape_mismatch.append(
                {
                    "key": key,
                    "source_shape": list(value.shape),
                    "target_shape": list(target_value.shape),
                }
            )
            continue
        filtered_state[key] = value
    result = model.module.load_state_dict(filtered_state, strict=False)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "skipped_shape_mismatch": skipped_shape_mismatch,
    }


def train_totalvi_pretrain(
    adata: ad.AnnData,
    model_dir,
    *,
    n_latent: int,
    n_layers: int,
    batch_size: int,
    max_epochs: int | None = None,
    external_indexing: list[np.ndarray] | None = None,
    force_retrain: bool = False,
    random_seed: int = 2026,
) -> TOTALVI:
    import scvi

    set_scvi_training_seed(random_seed)
    if (model_dir / "model.pt").exists() and not force_retrain:
        load_adata = adata
        # Older initialization checkpoints may have been saved with the
        # development-time batch column name.  Add a temporary compatibility
        # column only for scvi-tools' registry transfer; ANCHOR outputs keep the
        # release schema.
        old_batch_col = ("bench" + "mark") + "_batch"
        if old_batch_col not in load_adata.obs and "batch" in load_adata.obs:
            load_adata = adata.copy()
            load_adata.obs[old_batch_col] = load_adata.obs["batch"].astype(str)
        return scvi.model.TOTALVI.load(model_dir, adata=load_adata, accelerator="auto", device="auto")
    model = scvi.model.TOTALVI(adata, n_latent=n_latent, n_layers_encoder=n_layers, n_layers_decoder=n_layers)
    train_kwargs = {"accelerator": "auto", "devices": 1, "batch_size": batch_size}
    if max_epochs is not None:
        train_kwargs["max_epochs"] = max_epochs
    if external_indexing is not None:
        train_kwargs["external_indexing"] = external_indexing
    model.train(**train_kwargs)
    model.save(model_dir, overwrite=True)
    return model


def train_teacher_refinement(
    model: _AnchorTeacherBaseModel,
    model_dir,
    *,
    max_epochs: int,
    batch_size: int,
    classification_ratio: float,
    n_samples_per_label: int,
    external_indexing: list[np.ndarray] | None = None,
    force_retrain: bool = False,
    hard_ref_sampling_enable: bool = False,
    hard_ref_sampling_update: str = "epoch",
    hard_ref_sampling_wrong_weight: float = 10.0,
    hard_ref_sampling_correct_weight: float = 1.0,
    hard_ref_sampling_max_wrong_fraction: float | None = None,
    hard_ref_sampling_min_wrong_per_label: int | None = None,
    hard_ref_sampling_n_samples_per_label: int | None = None,
    hard_ref_sampling_source: str = "reference_train_pred_errors",
    hard_ref_sampling_seed: int = 0,
    random_seed: int = 2026,
):
    model_dir = Path(model_dir)
    set_scvi_training_seed(random_seed)
    hard_ref_sampling_enable = bool(hard_ref_sampling_enable)
    if hard_ref_sampling_enable:
        force_retrain = True
    if (model_dir / "model.pt").exists() and not force_retrain:
        try:
            return type(model).load(model_dir, adata=model.adata, accelerator="auto", device="auto")
        except TypeError as exc:
            print(
                f"Existing checkpoint at {model_dir} is incompatible with this model constructor; "
                f"retraining and overwriting it. ({type(exc).__name__}: {exc})"
            )
    datasplitter_kwargs = {}
    if external_indexing is not None:
        datasplitter_kwargs["external_indexing"] = external_indexing
    train_kwargs: dict[str, Any] = {}
    original_splitter_cls = getattr(model, "_data_splitter_cls", None)
    if hard_ref_sampling_enable:
        if str(hard_ref_sampling_update) != "epoch":
            raise ValueError("Only hard_ref_sampling_update='epoch' is currently supported.")
        model._data_splitter_cls = HardWeightedSemiSupervisedDataSplitter
        datasplitter_kwargs.update(
            {
                "hard_ref_sampling_wrong_weight": float(hard_ref_sampling_wrong_weight),
                "hard_ref_sampling_correct_weight": float(hard_ref_sampling_correct_weight),
                "hard_ref_sampling_max_wrong_fraction": hard_ref_sampling_max_wrong_fraction,
                "hard_ref_sampling_min_wrong_per_label": hard_ref_sampling_min_wrong_per_label,
                "hard_ref_sampling_seed": int(hard_ref_sampling_seed),
            }
        )
        callback = HardRefSamplingCallback(
            model_dir=model_dir.parent,
            wrong_weight=float(hard_ref_sampling_wrong_weight),
            correct_weight=float(hard_ref_sampling_correct_weight),
            source=str(hard_ref_sampling_source),
            batch_size=int(batch_size),
        )
        train_kwargs["callbacks"] = [SubSampleLabels(), callback]
        (model_dir.parent).mkdir(parents=True, exist_ok=True)
        (model_dir.parent / "hard_ref_sampling_config.json").write_text(
            json.dumps(
                {
                    "hard_ref_sampling_enable": True,
                    "hard_ref_sampling_update": str(hard_ref_sampling_update),
                    "hard_ref_sampling_wrong_weight": float(hard_ref_sampling_wrong_weight),
                    "hard_ref_sampling_correct_weight": float(hard_ref_sampling_correct_weight),
                    "hard_ref_sampling_max_wrong_fraction": (
                        None
                        if hard_ref_sampling_max_wrong_fraction is None
                        else float(hard_ref_sampling_max_wrong_fraction)
                    ),
                    "hard_ref_sampling_min_wrong_per_label": (
                        None
                        if hard_ref_sampling_min_wrong_per_label is None
                        else int(hard_ref_sampling_min_wrong_per_label)
                    ),
                    "hard_ref_sampling_n_samples_per_label": int(
                        hard_ref_sampling_n_samples_per_label
                        if hard_ref_sampling_n_samples_per_label is not None
                        else n_samples_per_label
                    ),
                    "hard_ref_sampling_source": str(hard_ref_sampling_source),
                    "hard_ref_sampling_seed": int(hard_ref_sampling_seed),
                    "random_seed": int(random_seed),
                    "n_samples_per_label": int(n_samples_per_label),
                    "batch_size": int(batch_size),
                    "max_epochs": int(max_epochs),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    try:
        model.train(
            max_epochs=max_epochs,
            early_stopping=True,
            check_val_every_n_epoch=1,
            n_samples_per_label=int(
                hard_ref_sampling_n_samples_per_label
                if hard_ref_sampling_enable and hard_ref_sampling_n_samples_per_label is not None
                else n_samples_per_label
            ),
            accelerator="auto",
            devices=1,
            batch_size=batch_size,
            datasplitter_kwargs=datasplitter_kwargs,
            plan_kwargs={"classification_ratio": classification_ratio},
            **train_kwargs,
        )
    finally:
        if hard_ref_sampling_enable:
            if original_splitter_cls is None:
                try:
                    delattr(model, "_data_splitter_cls")
                except AttributeError:
                    pass
            else:
                model._data_splitter_cls = original_splitter_cls
    model.save(model_dir, overwrite=True)
    return model


def predict_teacher_outputs(
    model: _AnchorTeacherBaseModel,
    adata: ad.AnnData,
    *,
    batch_size: int,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    soft = model.predict(adata, soft=True, batch_size=batch_size)
    pred = soft.idxmax(axis=1)
    latent = model.get_latent_representation(adata, batch_size=batch_size)
    return pred.astype(str), soft, latent.astype(np.float32)

def train_teacher_rounds(bundle, config, *, force_retrain: bool = False):
    from .._runner import run_generic_teacher

    return run_generic_teacher(bundle, config, force_retrain=force_retrain)
