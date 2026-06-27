from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .builder import TeacherDataBundle
from .config import ExperimentConfig
from .output import (
    history_to_dataframe,
    plot_confusion_heatmap,
    query_confusion_counts,
    write_json,
    write_teacher_stage_outputs,
)
from .teacher import (
    AnchorTeacherModel,
    load_matching_totalvi_weights,
    predict_teacher_outputs,
    set_scvi_training_seed,
    train_teacher_refinement,
    train_totalvi_pretrain,
)
from .teacher.pseudolabels import (
    UnifiedTeacherPseudoConfig,
    bottomup_as_flat_selection,
    bottomup_as_partial_selection,
    build_teacher_repeat_table,
    select_unified_bottomup_teacher_pseudolabels,
    write_teacher_bottomup_selection_outputs,
)
from .partial import (
    DEFAULT_HIDDEN_PARENT_ANCHOR_METHOD,
    DEFAULT_HIDDEN_PARENT_ANCHOR_STRATEGY,
    apply_hidden_parent_anchor_obs,
    apply_partial_flat_leaf_pseudolabel_obs,
    select_hidden_parent_anchor_cells,
)
from .partial import (
    PARTIAL_TRAIN_LABEL_COL,
    compute_collapsed_predictions_from_soft,
    compute_partial_collapsed_overall_metrics,
    compute_partial_fine_overall_metrics,
    compute_partial_hidden_pair_fine_accuracy,
    normalize_partial_branch_specs,
)
from .partial import (
    AnchorPartialTeacherModel,
    HIDDEN_BALANCE_MODE_KL_PBAR_UNIFORM,
    SMALLCLASS_CE_MODE_OFF as PARTIAL_SMALLCLASS_CE_MODE_OFF,
    SMALLCLASS_CE_MODE_OVERSAMPLE as PARTIAL_SMALLCLASS_CE_MODE_OVERSAMPLE,
)
from .teacher import (
    ANCHOR_ADATA_INDEX_KEY,
    PAIR_QUERY_PSEUDO_SELECTED_KEY,
    PAIR_QUERY_PSEUDO_TARGET_KEY,
    apply_pair_query_pseudolabels,
)
from .teacher import (
    AnchorPseudoTeacherModel,
    SMALLCLASS_CE_MODE_OFF,
    SMALLCLASS_CE_MODE_OVERSAMPLE,
    build_smallclass_augmentation_config,
)


def _setup_totalvi(adata, config: ExperimentConfig) -> None:
    import scvi

    scvi.model.TOTALVI.setup_anndata(
        adata,
        layer=config.counts_layer,
        batch_key="batch",
        protein_expression_obsm_key=config.heldout_protein_obsm_key,
    )


def _ensure_anchor_adata_index(adata) -> None:
    if ANCHOR_ADATA_INDEX_KEY not in adata.obs:
        adata.obs[ANCHOR_ADATA_INDEX_KEY] = np.arange(adata.n_obs, dtype=np.int64)


def _setup_teacher(model_cls: type, adata, config: ExperimentConfig, *, include_pseudo: bool = False) -> None:
    _ensure_anchor_adata_index(adata)
    kwargs: dict[str, Any] = {}
    if include_pseudo:
        kwargs.update(
            {
                "pseudo_selected_key": PAIR_QUERY_PSEUDO_SELECTED_KEY,
                "pseudo_target_key": PAIR_QUERY_PSEUDO_TARGET_KEY,
            }
        )
    model_cls.setup_anndata(
        adata,
        layer=config.counts_layer,
        batch_key="batch",
        protein_expression_obsm_key=config.heldout_protein_obsm_key,
        labels_key=config.label_key,
        unlabeled_category=config.teacher.unlabeled_category,
        **kwargs,
    )


def _setup_partial_teacher(adata, config: ExperimentConfig) -> None:
    _ensure_anchor_adata_index(adata)
    AnchorPartialTeacherModel.setup_anndata(
        adata,
        layer=config.counts_layer,
        batch_key="batch",
        protein_expression_obsm_key=config.heldout_protein_obsm_key,
        labels_key=PARTIAL_TRAIN_LABEL_COL,
        unlabeled_category=config.teacher.unlabeled_category,
    )


def _train_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    if config.teacher.hard_sampling_start_epoch is not None:
        raise NotImplementedError(
            "TeacherConfig.hard_sampling_start_epoch is declared but not wired into "
            "HardRefSamplingCallback yet. Leave it as None to use the default: hard weighted "
            "reference sampling for the full teacher stage."
        )
    return dict(config.teacher.hard_sampling_kwargs())


def _teacher_pseudo_config(config: ExperimentConfig) -> UnifiedTeacherPseudoConfig:
    """Build the pseudo-label selector config used between teacher rounds."""

    kwargs: dict[str, Any] = dict(
        pseudo_selection_mode=str(config.teacher.teacher_pseudo_selection_mode),
        query_pseudolabel_ratio=float(config.teacher.teacher_pseudo_query_pseudolabel_ratio),
        robust_marker_tail_fraction=float(config.teacher.teacher_pseudo_robust_marker_tail_fraction),
        robust_no_marker_tail_fraction=float(config.teacher.teacher_pseudo_robust_no_marker_tail_fraction),
        robust_hidden_tail_fraction=float(config.teacher.teacher_pseudo_robust_hidden_tail_fraction),
        robust_elbow_floor_fraction=float(config.teacher.teacher_pseudo_robust_elbow_floor_fraction),
    )
    kwargs.update(dict(config.teacher.teacher_pseudo_overrides))
    allowed = set(UnifiedTeacherPseudoConfig.__dataclass_fields__)
    return UnifiedTeacherPseudoConfig(**{key: value for key, value in kwargs.items() if key in allowed})


def _hidden_balance_mode(config: ExperimentConfig) -> str:
    mode = str(config.teacher.hidden_balance_mode)
    if mode != HIDDEN_BALANCE_MODE_KL_PBAR_UNIFORM:
        raise ValueError(
            "ANCHOR currently supports hidden_balance_mode="
            f"{HIDDEN_BALANCE_MODE_KL_PBAR_UNIFORM!r}; got {mode!r}"
        )
    return mode


def train_totalvi_initialization(
    bundle: TeacherDataBundle,
    config: ExperimentConfig,
    *,
    force_retrain: bool = False,
):
    _setup_totalvi(bundle.adata_model, config)
    return train_totalvi_pretrain(
        bundle.adata_model,
        config.stage_dir("totalvi_init_model"),
        n_latent=config.teacher.n_latent,
        n_layers=config.teacher.n_layers,
        batch_size=config.teacher.batch_size,
        max_epochs=None,
        external_indexing=bundle.external_indexing,
        force_retrain=force_retrain,
        random_seed=config.teacher.random_seed,
    )


def train_teacher_round0(bundle: TeacherDataBundle, config: ExperimentConfig, *, force_retrain: bool = False):
    """Train the initial supervised teacher before query pseudo-labels are used."""
    stage_dir = config.stage_dir("round0")
    model_dir = stage_dir / "model"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _setup_teacher(AnchorTeacherModel, bundle.adata_model, config)
    set_scvi_training_seed(config.teacher.random_seed)
    model = AnchorTeacherModel(
        bundle.adata_model,
        n_labels=len(bundle.label_categories),
        prior_spec=bundle.prior_spec,
        protein_names=bundle.protein_names,
        protein_teacher_stats=bundle.protein_teacher_stats,
        standard_normal_prior_enable=config.teacher.standard_normal_prior_enable,
        standard_normal_prior_loss_weight=config.teacher.standard_normal_prior_loss_weight,
        standard_normal_prior_warmup_steps=config.teacher.standard_normal_prior_warmup_steps,
        standard_normal_prior_ramp_steps=config.teacher.standard_normal_prior_ramp_steps,
        standard_normal_prior_safe_mode=config.teacher.standard_normal_prior_safe_mode,
        standard_normal_prior_detach_outliers=config.teacher.standard_normal_prior_detach_outliers,
        standard_normal_prior_skip_extreme_z1=config.teacher.standard_normal_prior_skip_extreme_z1,
        standard_normal_prior_extreme_z1_threshold=config.teacher.standard_normal_prior_extreme_z1_threshold,
        standard_normal_prior_min_scale=config.teacher.standard_normal_prior_min_scale,
        standard_normal_prior_max_scale=config.teacher.standard_normal_prior_max_scale,
        standard_normal_prior_max_abs_loc=config.teacher.standard_normal_prior_max_abs_loc,
        standard_normal_prior_max_reconstruction=config.teacher.standard_normal_prior_max_reconstruction,
        standard_normal_prior_max_kl=config.teacher.standard_normal_prior_max_kl,
        standard_normal_prior_detach_scale_multiplier=config.teacher.standard_normal_prior_detach_scale_multiplier,
        standard_normal_prior_detach_loss_multiplier=config.teacher.standard_normal_prior_detach_loss_multiplier,
        n_latent=config.teacher.n_latent,
        n_layers_encoder=config.teacher.n_layers,
        n_layers_decoder=config.teacher.n_layers,
    )
    if config.teacher.totalvi_init:
        totalvi = train_totalvi_initialization(bundle, config, force_retrain=force_retrain)
        _setup_teacher(AnchorTeacherModel, bundle.adata_model, config)
        report = load_matching_totalvi_weights(model, totalvi)
        write_json(stage_dir / "totalvi_weight_load_report.json", report)
    model = train_teacher_refinement(
        model,
        model_dir,
        max_epochs=config.teacher.round0_epochs,
        batch_size=config.teacher.batch_size,
        classification_ratio=config.teacher.classification_ratio,
        n_samples_per_label=config.teacher.n_samples_per_label,
        external_indexing=bundle.external_indexing,
        force_retrain=force_retrain,
        random_seed=config.teacher.random_seed,
        **_train_kwargs(config),
    )
    hist = history_to_dataframe(getattr(model, "history", getattr(model, "history_", None)))
    if not hist.empty:
        hist.to_csv(stage_dir / "history.csv", index=False)
    return model


def _init_partial_model(
    bundle: TeacherDataBundle,
    config: ExperimentConfig,
    *,
    query_pseudolabel_fine_ratio: float,
    query_pseudolabel_coarse_ratio: float,
    hidden_balance_enable: bool = False,
    hidden_parent_anchor_ce_enable: bool = False,
    smallclass_ce_mode: str = PARTIAL_SMALLCLASS_CE_MODE_OFF,
    smallclass_repeat_by_label: list[int] | None = None,
):
    if not bundle.partial_label_spec:
        raise ValueError("partial hidden model requested but partial_label_spec is empty")
    if not bundle.partial_supervision_categories or not bundle.supervision_label_to_desc_indices:
        raise ValueError("partial hidden model requested before partial supervision columns were built")
    set_scvi_training_seed(config.teacher.random_seed)
    return AnchorPartialTeacherModel(
        bundle.adata_model,
        fine_output_labels=bundle.label_categories,
        supervision_categories=bundle.partial_supervision_categories,
        supervision_label_to_desc_indices=bundle.supervision_label_to_desc_indices,
        prior_spec=bundle.prior_spec,
        protein_names=bundle.protein_names,
        protein_teacher_stats=bundle.protein_teacher_stats,
        standard_normal_prior_enable=config.teacher.standard_normal_prior_enable,
        standard_normal_prior_loss_weight=config.teacher.standard_normal_prior_loss_weight,
        standard_normal_prior_warmup_steps=config.teacher.standard_normal_prior_warmup_steps,
        standard_normal_prior_ramp_steps=config.teacher.standard_normal_prior_ramp_steps,
        standard_normal_prior_safe_mode=config.teacher.standard_normal_prior_safe_mode,
        standard_normal_prior_detach_outliers=config.teacher.standard_normal_prior_detach_outliers,
        standard_normal_prior_skip_extreme_z1=config.teacher.standard_normal_prior_skip_extreme_z1,
        standard_normal_prior_extreme_z1_threshold=config.teacher.standard_normal_prior_extreme_z1_threshold,
        standard_normal_prior_min_scale=config.teacher.standard_normal_prior_min_scale,
        standard_normal_prior_max_scale=config.teacher.standard_normal_prior_max_scale,
        standard_normal_prior_max_abs_loc=config.teacher.standard_normal_prior_max_abs_loc,
        standard_normal_prior_max_reconstruction=config.teacher.standard_normal_prior_max_reconstruction,
        standard_normal_prior_max_kl=config.teacher.standard_normal_prior_max_kl,
        standard_normal_prior_detach_scale_multiplier=config.teacher.standard_normal_prior_detach_scale_multiplier,
        standard_normal_prior_detach_loss_multiplier=config.teacher.standard_normal_prior_detach_loss_multiplier,
        query_pseudolabel_fine_ratio=float(query_pseudolabel_fine_ratio),
        query_pseudolabel_coarse_ratio=float(query_pseudolabel_coarse_ratio),
        hidden_balance_enable=bool(hidden_balance_enable),
        hidden_balance_lambda=float(config.teacher.hidden_balance_lambda) if hidden_balance_enable else 0.0,
        hidden_balance_branches=bundle.partial_hidden_branches,
        hidden_balance_mode=_hidden_balance_mode(config),
        hidden_balance_min_parent_mass=float(config.teacher.hidden_balance_min_parent_mass),
        hidden_parent_anchor_ce_enable=bool(hidden_parent_anchor_ce_enable),
        hidden_parent_anchor_ce_lambda=(
            float(config.teacher.hidden_parent_anchor_ce_lambda) if hidden_parent_anchor_ce_enable else 0.0
        ),
        hidden_parent_anchor_branches=bundle.partial_hidden_branches,
        smallclass_ce_mode=smallclass_ce_mode,
        smallclass_repeat_by_label=smallclass_repeat_by_label,
        n_latent=config.teacher.n_latent,
        n_layers_encoder=config.teacher.n_layers,
        n_layers_decoder=config.teacher.n_layers,
    )


def _clone_partial_model(
    source_model,
    bundle: TeacherDataBundle,
    config: ExperimentConfig,
    *,
    query_pseudolabel_fine_ratio: float,
    query_pseudolabel_coarse_ratio: float,
    hidden_balance_enable: bool = False,
    hidden_parent_anchor_ce_enable: bool = False,
    smallclass_ce_mode: str = PARTIAL_SMALLCLASS_CE_MODE_OFF,
    smallclass_repeat_by_label: list[int] | None = None,
):
    model = _init_partial_model(
        bundle,
        config,
        query_pseudolabel_fine_ratio=query_pseudolabel_fine_ratio,
        query_pseudolabel_coarse_ratio=query_pseudolabel_coarse_ratio,
        hidden_balance_enable=hidden_balance_enable,
        hidden_parent_anchor_ce_enable=hidden_parent_anchor_ce_enable,
        smallclass_ce_mode=smallclass_ce_mode,
        smallclass_repeat_by_label=smallclass_repeat_by_label,
    )
    load_result = model.module.load_state_dict(source_model.module.state_dict(), strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "State-dict clone for hidden partial teacher had mismatches: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    return model


def _write_stage_prediction(model, bundle: TeacherDataBundle, config: ExperimentConfig, *, stage: str) -> dict[str, Path]:
    pred, soft, latent = predict_teacher_outputs(
        model,
        bundle.adata_model,
        batch_size=config.teacher.batch_size,
    )
    return write_teacher_stage_outputs(
        adata=bundle.adata_model,
        out_dir=config.stage_dir(stage),
        stage=stage,
        pred=pred,
        soft=soft,
        latent=latent,
        prior_spec=bundle.prior_spec,
        reference_name=config.reference_name,
        query_name=config.query_name,
    )


def _summarize_hidden_branch_probabilities(
    bundle: TeacherDataBundle,
    soft: pd.DataFrame,
    *,
    query_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    query_mask = bundle.adata_model.obs["ref_query_col"].astype(str).eq(query_name)
    for spec in normalize_partial_branch_specs(bundle.partial_label_spec):
        true_mask = query_mask & bundle.adata_model.obs["true_label"].astype(str).isin(list(spec.children))
        n_query = int(true_mask.sum())
        row: dict[str, Any] = {
            "pair_key": str(spec.key),
            "parent_axis_label": str(spec.parent_label),
            "children": "|".join(str(x) for x in spec.children),
            "n_query": n_query,
        }
        child_cols = [str(child) for child in spec.children if str(child) in soft.columns]
        if n_query == 0 or not child_cols:
            row.update({"mean_top_child_conf": np.nan, "mean_parent_mass": np.nan})
            rows.append(row)
            continue
        pair_soft = soft.loc[bundle.adata_model.obs_names[true_mask], child_cols].astype(float)
        branch_total = pair_soft.sum(axis=1).clip(lower=1e-8)
        branch_local = pair_soft.div(branch_total, axis=0)
        row.update(
            {
                "mean_parent_mass": float(branch_total.mean()),
                "mean_top_child_conf": float(branch_local.max(axis=1).mean()),
            }
        )
        for child in child_cols:
            row[f"mean_raw_mass__{child}"] = float(pair_soft[child].mean())
            row[f"argmax_share__{child}"] = float(branch_local.idxmax(axis=1).astype(str).eq(child).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_hidden_stage_extras(
    *,
    bundle: TeacherDataBundle,
    config: ExperimentConfig,
    stage: str,
    paths: dict[str, Path],
    soft: pd.DataFrame,
) -> None:
    out_dir = config.stage_dir(stage)
    pred_col = f"pred_{stage}"
    collapsed_soft, collapsed_pred, collapsed_conf = compute_collapsed_predictions_from_soft(
        soft,
        partial_label_spec=bundle.partial_label_spec,
        fine_output_labels=bundle.label_categories,
    )
    collapsed_col = f"pred_collapsed_{stage}"
    bundle.adata_model.obs[collapsed_col] = collapsed_pred.reindex(bundle.adata_model.obs_names.astype(str)).astype(str).to_numpy()
    bundle.adata_model.obs[f"confidence_collapsed_{stage}"] = (
        collapsed_conf.reindex(bundle.adata_model.obs_names.astype(str)).astype(float).to_numpy()
    )
    compute_partial_fine_overall_metrics(
        bundle.adata_model.obs,
        fine_pred_col=pred_col,
        query_name=config.query_name,
    ).to_frame().T.to_csv(out_dir / "summary_metrics_fine.csv", index=False)
    compute_partial_collapsed_overall_metrics(
        bundle.adata_model.obs,
        collapsed_pred_col=collapsed_col,
        partial_label_spec=bundle.partial_label_spec,
        query_name=config.query_name,
    ).to_frame().T.to_csv(out_dir / "summary_metrics_collapsed.csv", index=False)
    by_pair, overall = compute_partial_hidden_pair_fine_accuracy(
        bundle.adata_model.obs,
        partial_label_spec=bundle.partial_label_spec,
        fine_pred_col=pred_col,
        collapsed_pred_col=collapsed_col,
        query_name=config.query_name,
    )
    by_pair.to_csv(out_dir / "hidden_pair_metrics_by_pair.csv", index=False)
    overall.to_csv(out_dir / "hidden_pair_metrics_overall.csv", index=False)
    _summarize_hidden_branch_probabilities(bundle, soft, query_name=config.query_name).to_csv(
        out_dir / f"{stage}_hidden_branch_probability_summary.csv",
        index=False,
    )
    collapsed_counts = query_confusion_counts(
        bundle.adata_model.obs,
        collapsed_col,
        label_col="partial_collapsed_true_label",
        query_name=config.query_name,
    )
    collapsed_counts.to_csv(out_dir / "confusion_counts_collapsed.csv")
    plot_confusion_heatmap(
        collapsed_counts,
        title=f"{stage} collapsed query confusion",
        out_path=out_dir / "confusion_heatmap_collapsed.png",
    )
    bundle.adata_model.write_h5ad(paths["results_h5ad"])


def train_teacher_round(
    previous_model,
    *,
    previous_soft: pd.DataFrame,
    previous_latent_query: np.ndarray,
    bundle: TeacherDataBundle,
    config: ExperimentConfig,
    round_id: int,
    force_retrain: bool = False,
):
    """Refine the teacher with query pseudo-labels selected from the previous round."""
    pseudo_cfg = _teacher_pseudo_config(config)
    stage = f"round{round_id}"
    stage_dir = config.stage_dir(stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    pseudo_df, by_class, vetoed, evidence, score_availability = select_unified_bottomup_teacher_pseudolabels(
        query_obs=bundle.query_obs,
        soft=previous_soft.reindex(bundle.query_index).loc[:, bundle.label_categories],
        protein_arcsinh=bundle.protein_arcsinh,
        prior_spec=bundle.prior_spec,
        label_categories=bundle.label_categories,
        partial_label_spec=None,
        leaf_marker_specs=bundle.leaf_marker_specs,
        label_col="true_label",
        teacher_latent=previous_latent_query,
        enable_hidden_rescue=False,
        config=pseudo_cfg,
    )
    write_teacher_bottomup_selection_outputs(
        results_dir=stage_dir,
        pseudo_df=pseudo_df,
        by_class=by_class,
        vetoed=vetoed,
        evidence=evidence,
        score_availability=score_availability,
        prefix=pseudo_cfg.output_prefix(),
    )
    selection = bottomup_as_flat_selection(
        pseudo_df=pseudo_df,
        by_class=by_class,
        score_availability=score_availability,
        label_categories=bundle.label_categories,
        strategy=f"teacher_round{round_id}",
    )
    apply_pair_query_pseudolabels(bundle.adata_model, selection.pair_bundle)
    if config.teacher.pseudo_smallclass_oversample:
        repeat_table = build_teacher_repeat_table(
            selection.by_class,
            label_categories=bundle.label_categories,
            config=pseudo_cfg,
        )
        repeat_table.to_csv(stage_dir / "teacher_pseudolabel_smallclass_repeat_by_class.csv", index=False)
        aug_config = build_smallclass_augmentation_config(
            repeat_table,
            label_categories=bundle.label_categories,
            leaf_marker_specs=bundle.leaf_marker_specs,
            protein_names=bundle.protein_names,
            gene_names=list(bundle.adata_model.var_names.astype(str)),
        )
        smallclass_ce_mode = SMALLCLASS_CE_MODE_OVERSAMPLE
        repeat_by_label = aug_config.repeat_by_label
        positive_protein = aug_config.positive_protein_indices_by_label
        negative_protein = aug_config.negative_protein_indices_by_label
        positive_gene = aug_config.positive_gene_indices_by_label
        negative_gene = aug_config.negative_gene_indices_by_label
    else:
        smallclass_ce_mode = SMALLCLASS_CE_MODE_OFF
        repeat_by_label = [0] * len(bundle.label_categories)
        positive_protein = {}
        negative_protein = {}
        positive_gene = {}
        negative_gene = {}
    _setup_teacher(AnchorPseudoTeacherModel, bundle.adata_model, config, include_pseudo=True)
    set_scvi_training_seed(config.teacher.random_seed)
    model = AnchorPseudoTeacherModel(
        bundle.adata_model,
        n_labels=len(bundle.label_categories),
        prior_spec=bundle.prior_spec,
        protein_names=bundle.protein_names,
        protein_teacher_stats=bundle.protein_teacher_stats,
        standard_normal_prior_enable=config.teacher.standard_normal_prior_enable,
        standard_normal_prior_loss_weight=config.teacher.standard_normal_prior_loss_weight,
        standard_normal_prior_warmup_steps=config.teacher.standard_normal_prior_warmup_steps,
        standard_normal_prior_ramp_steps=config.teacher.standard_normal_prior_ramp_steps,
        standard_normal_prior_safe_mode=config.teacher.standard_normal_prior_safe_mode,
        standard_normal_prior_detach_outliers=config.teacher.standard_normal_prior_detach_outliers,
        standard_normal_prior_skip_extreme_z1=config.teacher.standard_normal_prior_skip_extreme_z1,
        standard_normal_prior_extreme_z1_threshold=config.teacher.standard_normal_prior_extreme_z1_threshold,
        standard_normal_prior_min_scale=config.teacher.standard_normal_prior_min_scale,
        standard_normal_prior_max_scale=config.teacher.standard_normal_prior_max_scale,
        standard_normal_prior_max_abs_loc=config.teacher.standard_normal_prior_max_abs_loc,
        standard_normal_prior_max_reconstruction=config.teacher.standard_normal_prior_max_reconstruction,
        standard_normal_prior_max_kl=config.teacher.standard_normal_prior_max_kl,
        standard_normal_prior_detach_scale_multiplier=config.teacher.standard_normal_prior_detach_scale_multiplier,
        standard_normal_prior_detach_loss_multiplier=config.teacher.standard_normal_prior_detach_loss_multiplier,
        query_pseudolabel_classification_ratio=pseudo_cfg.query_pseudolabel_ratio,
        n_latent=config.teacher.n_latent,
        n_layers_encoder=config.teacher.n_layers,
        n_layers_decoder=config.teacher.n_layers,
        smallclass_ce_mode=smallclass_ce_mode,
        smallclass_repeat_by_label=repeat_by_label,
        smallclass_positive_protein_indices_by_label=positive_protein,
        smallclass_negative_protein_indices_by_label=negative_protein,
        smallclass_positive_gene_indices_by_label=positive_gene,
        smallclass_negative_gene_indices_by_label=negative_gene,
    )
    load_result = model.module.load_state_dict(previous_model.module.state_dict(), strict=False)
    if load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected state dict keys in teacher round{round_id}: {load_result.unexpected_keys}")
    model = train_teacher_refinement(
        model,
        stage_dir / "model",
        max_epochs=config.teacher.round1_epochs if round_id == 1 else config.teacher.round2_epochs,
        batch_size=config.teacher.batch_size,
        classification_ratio=config.teacher.classification_ratio,
        n_samples_per_label=config.teacher.n_samples_per_label,
        external_indexing=bundle.external_indexing,
        force_retrain=force_retrain,
        random_seed=config.teacher.random_seed,
        **_train_kwargs(config),
    )
    hist = history_to_dataframe(getattr(model, "history", getattr(model, "history_", None)))
    if not hist.empty:
        hist.to_csv(stage_dir / "history.csv", index=False)
    return model


def train_hidden_teacher_round0(bundle: TeacherDataBundle, config: ExperimentConfig, *, force_retrain: bool = False):
    """Train the partial-label teacher warmup and hidden-parent anchor stage."""
    stage_dir = config.stage_dir("round0")
    warmup_dir = stage_dir / "warmup_model"
    balance_dir = stage_dir / "model"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _setup_partial_teacher(bundle.adata_model, config)
    warmup = _init_partial_model(
        bundle,
        config,
        query_pseudolabel_fine_ratio=0.0,
        query_pseudolabel_coarse_ratio=0.0,
        hidden_balance_enable=False,
        hidden_parent_anchor_ce_enable=False,
    )
    if config.teacher.totalvi_init:
        totalvi = train_totalvi_initialization(bundle, config, force_retrain=force_retrain)
        _setup_partial_teacher(bundle.adata_model, config)
        report = load_matching_totalvi_weights(warmup, totalvi)
        write_json(stage_dir / "warmup_totalvi_weight_load_report.json", report)
    warmup = train_teacher_refinement(
        warmup,
        warmup_dir,
        max_epochs=config.teacher.round0_warmup_epochs,
        batch_size=config.teacher.batch_size,
        classification_ratio=config.teacher.classification_ratio,
        n_samples_per_label=config.teacher.n_samples_per_label,
        external_indexing=bundle.external_indexing,
        force_retrain=force_retrain,
        random_seed=config.teacher.random_seed,
        **_train_kwargs(config),
    )
    warmup_hist = history_to_dataframe(getattr(warmup, "history", getattr(warmup, "history_", None)))
    if not warmup_hist.empty:
        warmup_hist.to_csv(stage_dir / "warmup_history.csv", index=False)
    _warmup_pred, warmup_soft, _warmup_latent = predict_teacher_outputs(
        warmup,
        bundle.adata_model,
        batch_size=config.teacher.batch_size,
    )
    anchor_selection = select_hidden_parent_anchor_cells(
        query_obs=bundle.query_obs,
        soft=warmup_soft.loc[bundle.query_index, bundle.label_categories].copy(),
        protein_arcsinh=bundle.protein_arcsinh,
        fine_output_labels=bundle.label_categories,
        partial_label_spec=bundle.partial_label_spec,
        label_col="true_label",
        top_k_per_child=config.teacher.hidden_parent_anchor_top_k_per_child,
        parent_posterior_thresholds=config.teacher.hidden_parent_anchor_parent_posterior_thresholds,
        method=DEFAULT_HIDDEN_PARENT_ANCHOR_METHOD,
        strategy=DEFAULT_HIDDEN_PARENT_ANCHOR_STRATEGY,
        leaf_marker_specs=bundle.leaf_marker_specs,
    )
    anchor_selection.write_outputs(stage_dir, prefix="round0_hidden_parent_anchor_ce_balancekl")
    if int(anchor_selection.cell_level.shape[0]) == 0:
        write_json(
            stage_dir / "hidden_anchor_failure.json",
            {
                "reason": "no_hidden_parent_anchor_cells_selected",
                "partial_label_spec": {k: list(v) for k, v in bundle.partial_label_spec.items()},
                "top_k_per_child": int(config.teacher.hidden_parent_anchor_top_k_per_child),
                "parent_posterior_thresholds": list(config.teacher.hidden_parent_anchor_parent_posterior_thresholds),
            },
        )
        raise RuntimeError("No hidden parent anchor cells were selected for hidden round0")
    apply_hidden_parent_anchor_obs(
        bundle.adata_model,
        anchor_selection,
        branch_order=bundle.partial_hidden_branches,
    )
    model = _clone_partial_model(
        warmup,
        bundle,
        config,
        query_pseudolabel_fine_ratio=0.0,
        query_pseudolabel_coarse_ratio=0.0,
        hidden_balance_enable=True,
        hidden_parent_anchor_ce_enable=True,
    )
    model = train_teacher_refinement(
        model,
        balance_dir,
        max_epochs=config.teacher.round0_balance_kl_epochs,
        batch_size=config.teacher.batch_size,
        classification_ratio=config.teacher.classification_ratio,
        n_samples_per_label=config.teacher.n_samples_per_label,
        external_indexing=bundle.external_indexing,
        force_retrain=force_retrain,
        random_seed=config.teacher.random_seed,
        **_train_kwargs(config),
    )
    hist = history_to_dataframe(getattr(model, "history", getattr(model, "history_", None)))
    if not hist.empty:
        hist.to_csv(stage_dir / "history.csv", index=False)
    write_json(
        stage_dir / "training_config.json",
        {
            "strategy": "hidden_parent_anchor_ce_balancekl",
            "warmup_epochs": int(config.teacher.round0_warmup_epochs),
            "anchor_balance_epochs": int(config.teacher.round0_balance_kl_epochs),
            "hidden_balance_enable": True,
            "hidden_balance_lambda": float(config.teacher.hidden_balance_lambda),
            "hidden_balance_mode": str(config.teacher.hidden_balance_mode),
            "hidden_balance_min_parent_mass": float(config.teacher.hidden_balance_min_parent_mass),
            "hidden_parent_anchor_ce_enable": True,
            "hidden_parent_anchor_ce_lambda": float(config.teacher.hidden_parent_anchor_ce_lambda),
            "n_hidden_parent_anchor_cells": int(anchor_selection.cell_level.shape[0]),
            "teacher_train_kwargs": _train_kwargs(config),
        },
    )
    return model


def train_hidden_teacher_round(
    previous_model,
    *,
    previous_soft: pd.DataFrame,
    previous_latent_query: np.ndarray,
    bundle: TeacherDataBundle,
    config: ExperimentConfig,
    round_id: int,
    force_retrain: bool = False,
):
    """Refine a partial-label teacher with leaf pseudo-labels after hidden warmup."""
    pseudo_cfg = _teacher_pseudo_config(config)
    stage = f"round{round_id}"
    stage_dir = config.stage_dir(stage)
    stage_dir.mkdir(parents=True, exist_ok=True)
    pseudo_df, by_class, vetoed, evidence, score_availability = select_unified_bottomup_teacher_pseudolabels(
        query_obs=bundle.query_obs,
        soft=previous_soft.reindex(bundle.query_index).loc[:, bundle.label_categories],
        protein_arcsinh=bundle.protein_arcsinh,
        prior_spec=bundle.prior_spec,
        label_categories=bundle.label_categories,
        partial_label_spec=bundle.partial_label_spec,
        leaf_marker_specs=bundle.leaf_marker_specs,
        label_col="true_label",
        teacher_latent=previous_latent_query,
        enable_hidden_rescue=True,
        config=pseudo_cfg,
    )
    write_teacher_bottomup_selection_outputs(
        results_dir=stage_dir,
        pseudo_df=pseudo_df,
        by_class=by_class,
        vetoed=vetoed,
        evidence=evidence,
        score_availability=score_availability,
        prefix=pseudo_cfg.output_prefix(),
    )
    selection = bottomup_as_partial_selection(
        pseudo_df=pseudo_df,
        by_class=by_class,
        score_availability=score_availability,
        evidence=evidence,
        strategy=f"hidden_teacher_round{round_id}",
    )
    apply_partial_flat_leaf_pseudolabel_obs(
        bundle.adata_model,
        selection,
        fine_output_labels=bundle.label_categories,
        round_idx=round_id,
        source_name=f"hidden_teacher_round{round_id}",
    )
    if config.teacher.pseudo_smallclass_oversample:
        repeat_table = build_teacher_repeat_table(
            selection.by_class,
            label_categories=bundle.label_categories,
            config=pseudo_cfg,
        )
        repeat_table.to_csv(stage_dir / "teacher_pseudolabel_smallclass_repeat_by_class.csv", index=False)
        repeat_by_label = repeat_table.sort_values("label_index")["aug_repeats_per_cell"].astype(int).tolist()
        smallclass_ce_mode = PARTIAL_SMALLCLASS_CE_MODE_OVERSAMPLE
    else:
        repeat_by_label = [0] * len(bundle.label_categories)
        smallclass_ce_mode = PARTIAL_SMALLCLASS_CE_MODE_OFF
    model = _clone_partial_model(
        previous_model,
        bundle,
        config,
        query_pseudolabel_fine_ratio=pseudo_cfg.query_pseudolabel_ratio,
        query_pseudolabel_coarse_ratio=0.0,
        smallclass_ce_mode=smallclass_ce_mode,
        smallclass_repeat_by_label=repeat_by_label,
    )
    model = train_teacher_refinement(
        model,
        stage_dir / "model",
        max_epochs=config.teacher.round1_epochs if round_id == 1 else config.teacher.round2_epochs,
        batch_size=config.teacher.batch_size,
        classification_ratio=config.teacher.classification_ratio,
        n_samples_per_label=config.teacher.n_samples_per_label,
        external_indexing=bundle.external_indexing,
        force_retrain=force_retrain,
        random_seed=config.teacher.random_seed,
        **_train_kwargs(config),
    )
    hist = history_to_dataframe(getattr(model, "history", getattr(model, "history_", None)))
    if not hist.empty:
        hist.to_csv(stage_dir / "history.csv", index=False)
    return model


def run_generic_teacher(bundle: TeacherDataBundle, config: ExperimentConfig, *, force_retrain: bool = False):
    """Run the three-stage teacher pipeline.

    Standard runs use supervised round0 followed by two pseudo-label
    refinement rounds.  Partial-label runs replace round0 with a hidden-branch
    warmup and anchor-balancing stage before the same refinement pattern.
    """
    if bundle.hidden_branch_detected and config.teacher.hidden_branch_mode != "off":
        model0 = train_hidden_teacher_round0(bundle, config, force_retrain=force_retrain)
        paths0 = _write_stage_prediction(model0, bundle, config, stage="round0")
        soft0_all = pd.read_csv(paths0["soft_probs_csv"], index_col=0)
        _write_hidden_stage_extras(bundle=bundle, config=config, stage="round0", paths=paths0, soft=soft0_all)
        soft0 = soft0_all.reindex(bundle.query_index).loc[:, bundle.label_categories]
        latent0 = bundle.adata_model.obsm["X_round0"][bundle.query_mask.to_numpy()]

        model1 = train_hidden_teacher_round(
            model0,
            previous_soft=soft0,
            previous_latent_query=latent0,
            bundle=bundle,
            config=config,
            round_id=1,
            force_retrain=force_retrain,
        )
        paths1 = _write_stage_prediction(model1, bundle, config, stage="round1")
        soft1_all = pd.read_csv(paths1["soft_probs_csv"], index_col=0)
        _write_hidden_stage_extras(bundle=bundle, config=config, stage="round1", paths=paths1, soft=soft1_all)
        soft1 = soft1_all.reindex(bundle.query_index).loc[:, bundle.label_categories]
        latent1 = bundle.adata_model.obsm["X_round1"][bundle.query_mask.to_numpy()]

        model2 = train_hidden_teacher_round(
            model1,
            previous_soft=soft1,
            previous_latent_query=latent1,
            bundle=bundle,
            config=config,
            round_id=2,
            force_retrain=force_retrain,
        )
        paths2 = _write_stage_prediction(model2, bundle, config, stage="round2")
        soft2_all = pd.read_csv(paths2["soft_probs_csv"], index_col=0)
        _write_hidden_stage_extras(bundle=bundle, config=config, stage="round2", paths=paths2, soft=soft2_all)
        comparison = []
        for stage, paths in [("round0", paths0), ("round1", paths1), ("round2", paths2)]:
            row = pd.read_csv(paths["summary_csv"]).iloc[0].to_dict()
            row["stage"] = stage
            row["summary_csv"] = str(paths["summary_csv"])
            row["results_h5ad"] = str(paths["results_h5ad"])
            comparison.append(row)
        pd.DataFrame(comparison).to_csv(config.root_dir / "teacher_round_comparison.csv", index=False)
        return paths0, paths1, paths2
    model0 = train_teacher_round0(bundle, config, force_retrain=force_retrain)
    paths0 = _write_stage_prediction(model0, bundle, config, stage="round0")
    soft0 = pd.read_csv(paths0["soft_probs_csv"], index_col=0).reindex(bundle.query_index).loc[:, bundle.label_categories]
    latent0 = bundle.adata_model.obsm["X_round0"][bundle.query_mask.to_numpy()]

    model1 = train_teacher_round(
        model0,
        previous_soft=soft0,
        previous_latent_query=latent0,
        bundle=bundle,
        config=config,
        round_id=1,
        force_retrain=force_retrain,
    )
    paths1 = _write_stage_prediction(model1, bundle, config, stage="round1")
    soft1 = pd.read_csv(paths1["soft_probs_csv"], index_col=0).reindex(bundle.query_index).loc[:, bundle.label_categories]
    latent1 = bundle.adata_model.obsm["X_round1"][bundle.query_mask.to_numpy()]

    model2 = train_teacher_round(
        model1,
        previous_soft=soft1,
        previous_latent_query=latent1,
        bundle=bundle,
        config=config,
        round_id=2,
        force_retrain=force_retrain,
    )
    paths2 = _write_stage_prediction(model2, bundle, config, stage="round2")
    comparison = []
    for stage, paths in [("round0", paths0), ("round1", paths1), ("round2", paths2)]:
        row = pd.read_csv(paths["summary_csv"]).iloc[0].to_dict()
        row["stage"] = stage
        row["summary_csv"] = str(paths["summary_csv"])
        row["results_h5ad"] = str(paths["results_h5ad"])
        comparison.append(row)
    pd.DataFrame(comparison).to_csv(config.root_dir / "teacher_round_comparison.csv", index=False)
    return paths0, paths1, paths2
