from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .markers import MarkerTree


@dataclass(frozen=True)
class ColumnMap:
    """Input column names required by ANCHOR."""

    batch_key: str
    celltype_key: str
    query_label_key: str | None = None
    split_key: str | None = None
    hidden_branch_key: str | None = None
    sample_key: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ColumnMap":
        return cls(**{k: v for k, v in dict(data).items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class AnchorSelectionConfig:
    """Parameters for adaptive-tail marker-guided anchor selection."""

    posterior_threshold: float = 0.95
    parent_pool_threshold: float = 0.20
    child_conditional_threshold: float = 0.05
    hard_contradiction_quantile: float = 0.90
    soft_contradiction_quantile: float = 0.75
    soft_contradiction_penalty: float = 0.25
    wide_candidate_multiplier: int = 8
    marker_tail_fraction: float = 0.25
    no_marker_tail_fraction: float = 0.10
    hidden_tail_fraction: float = 0.10
    marker_max_cap: int = 50
    no_marker_max_cap: int = 10
    hidden_max_cap_per_child: int = 10
    marker_min_select_if_any: int = 3
    no_marker_min_select_if_any: int = 2
    hidden_min_select_if_any: int = 0
    elbow_drop_ratio: float = 5.0
    elbow_absolute_min_drop: float = 0.03
    elbow_floor_fraction: float = 0.40
    marker_min_elbow_count: int = 5
    no_marker_min_elbow_count: int = 2
    hidden_min_elbow_count: int = 1
    marker_pseudo_weight: float = 1.0
    hidden_pseudo_weight: float = 0.5
    no_marker_pseudo_weight: float = 0.25

    def as_student_overrides(self) -> dict[str, Any]:
        return {
            "pseudo_selection_mode": "adaptive_tail_robust_elbow",
            "posterior_threshold": self.posterior_threshold,
            "parent_pool_threshold": self.parent_pool_threshold,
            "child_conditional_threshold": self.child_conditional_threshold,
            "hard_contradiction_quantile": self.hard_contradiction_quantile,
            "soft_contradiction_quantile": self.soft_contradiction_quantile,
            "soft_contradiction_penalty": self.soft_contradiction_penalty,
            "wide_candidate_multiplier": self.wide_candidate_multiplier,
            "adaptive_marker_tail_fraction": self.marker_tail_fraction,
            "adaptive_no_marker_tail_fraction": self.no_marker_tail_fraction,
            "adaptive_hidden_tail_fraction": self.hidden_tail_fraction,
            "adaptive_marker_max_cap": self.marker_max_cap,
            "adaptive_no_marker_max_cap": self.no_marker_max_cap,
            "adaptive_hidden_max_cap_per_child": self.hidden_max_cap_per_child,
            "adaptive_marker_min_select_if_any": self.marker_min_select_if_any,
            "adaptive_no_marker_min_select_if_any": self.no_marker_min_select_if_any,
            "adaptive_hidden_min_select_if_any": self.hidden_min_select_if_any,
            "adaptive_elbow_drop_ratio": self.elbow_drop_ratio,
            "adaptive_elbow_absolute_min_drop": self.elbow_absolute_min_drop,
            "adaptive_elbow_floor_fraction": self.elbow_floor_fraction,
            "adaptive_marker_min_elbow_count": self.marker_min_elbow_count,
            "adaptive_no_marker_min_elbow_count": self.no_marker_min_elbow_count,
            "adaptive_hidden_min_elbow_count": self.hidden_min_elbow_count,
            "adaptive_marker_pseudo_weight": self.marker_pseudo_weight,
            "adaptive_hidden_pseudo_weight": self.hidden_pseudo_weight,
            "adaptive_no_marker_pseudo_weight": self.no_marker_pseudo_weight,
        }

    def as_teacher_pseudo_kwargs(self) -> dict[str, Any]:
        return {
            "pseudo_selection_mode": "adaptive_tail_robust_elbow",
            "posterior_threshold": self.posterior_threshold,
            "parent_pool_threshold": self.parent_pool_threshold,
            "child_conditional_threshold": self.child_conditional_threshold,
            "hard_contradiction_quantile": self.hard_contradiction_quantile,
            "soft_contradiction_quantile": self.soft_contradiction_quantile,
            "soft_contradiction_penalty": self.soft_contradiction_penalty,
            "wide_candidate_multiplier": self.wide_candidate_multiplier,
            "robust_marker_tail_fraction": self.marker_tail_fraction,
            "robust_no_marker_tail_fraction": self.no_marker_tail_fraction,
            "robust_hidden_tail_fraction": self.hidden_tail_fraction,
            "adaptive_marker_max_cap": self.marker_max_cap,
            "adaptive_no_marker_max_cap": self.no_marker_max_cap,
            "adaptive_hidden_max_cap_per_child": self.hidden_max_cap_per_child,
            "robust_marker_min_select_if_any": self.marker_min_select_if_any,
            "robust_no_marker_min_select_if_any": self.no_marker_min_select_if_any,
            "robust_hidden_min_select_if_any": self.hidden_min_select_if_any,
            "robust_elbow_drop_ratio": self.elbow_drop_ratio,
            "robust_elbow_absolute_min_drop": self.elbow_absolute_min_drop,
            "robust_elbow_floor_fraction": self.elbow_floor_fraction,
            "robust_marker_min_elbow_count": self.marker_min_elbow_count,
            "robust_no_marker_min_elbow_count": self.no_marker_min_elbow_count,
            "robust_hidden_min_elbow_count": self.hidden_min_elbow_count,
            "adaptive_marker_pseudo_weight": self.marker_pseudo_weight,
            "adaptive_hidden_pseudo_weight": self.hidden_pseudo_weight,
            "adaptive_no_marker_pseudo_weight": self.no_marker_pseudo_weight,
        }


@dataclass(frozen=True)
class RhoPolicyConfig:
    """Node-wise teacher-weight policy for the conditional KL loss."""

    parent_pool_threshold: float = 0.20
    release_strength: float = 1.0
    min_parent_pool_for_policy: int = 50
    partial_protein_power_threshold: float = 0.50
    strong_protein_power_threshold: float = 0.75
    partial_challenge_threshold: float = 0.10
    strong_challenge_threshold: float = 0.20
    partial_rna_protection_max: float = 0.70
    strong_rna_protection_max: float = 0.45
    partial_release_rho: float = 0.50
    strong_release_rho: float = 0.10

    def as_policy_params(self) -> dict[str, Any]:
        return {
            "min_parent_pool_for_policy": self.min_parent_pool_for_policy,
            "partial_protein_power_threshold": self.partial_protein_power_threshold,
            "strong_protein_power_threshold": self.strong_protein_power_threshold,
            "partial_challenge_threshold": self.partial_challenge_threshold,
            "strong_challenge_threshold": self.strong_challenge_threshold,
            "partial_rna_protection_max": self.partial_rna_protection_max,
            "strong_rna_protection_max": self.strong_rna_protection_max,
            "partial_release_rho": self.partial_release_rho,
            "strong_release_rho": self.strong_release_rho,
        }


@dataclass(frozen=True)
class TeacherConfig:
    batch_size: int = 512
    totalvi_init: bool = True
    random_seed: int = 2026
    n_latent: int = 30
    n_layers: int = 2
    round0_epochs: int = 20
    round1_epochs: int = 10
    round2_epochs: int = 10
    n_samples_per_label: int = 100
    classification_ratio: int = 50
    hard_ref_sampling_enable: bool = True
    hard_ref_sampling_update: str = "epoch"
    hard_ref_sampling_wrong_weight: float = 10.0
    hard_ref_sampling_correct_weight: float = 1.0
    hard_ref_sampling_max_wrong_fraction: float = 0.3
    hard_ref_sampling_min_wrong_per_label: int | None = None
    hard_ref_sampling_source: str = "reference_train_pred_errors"
    hard_ref_sampling_seed: int = 2026
    hard_sampling_start_epoch: int | None = None
    hidden_branch_mode: str = "auto"
    round0_warmup_epochs: int = 5
    round0_balance_kl_epochs: int = 15
    hidden_balance_lambda: float = 1.0
    hidden_balance_mode: str = "kl_pbar_uniform"
    hidden_balance_min_parent_mass: float = 1.0
    hidden_parent_anchor_ce_lambda: float = 10.0
    hidden_parent_anchor_top_k_per_child: int = 10
    hidden_parent_anchor_parent_posterior_thresholds: tuple[float, ...] = (0.5, 0.2)
    fixed_split_seed: int = 0
    fixed_split_train_fraction: float = 0.90
    unlabeled_category: str = "Unknown"
    standard_normal_prior_enable: bool = True
    standard_normal_prior_loss_weight: float = 1.0
    standard_normal_prior_warmup_steps: int = 0
    standard_normal_prior_ramp_steps: int = 0
    standard_normal_prior_safe_mode: bool = False
    standard_normal_prior_detach_outliers: bool = False
    standard_normal_prior_skip_extreme_z1: bool = False
    standard_normal_prior_extreme_z1_threshold: float = 30.0
    standard_normal_prior_extreme_z1_initial_threshold: float | None = None
    standard_normal_prior_extreme_z1_ramp_steps: int = 0
    standard_normal_prior_min_scale: float = 1e-4
    standard_normal_prior_max_scale: float = 1e3
    standard_normal_prior_max_abs_loc: float = 50.0
    standard_normal_prior_max_reconstruction: float = 1e3
    standard_normal_prior_max_kl: float = 1e4
    standard_normal_prior_detach_scale_multiplier: float = 1.0
    standard_normal_prior_detach_loss_multiplier: float = 1.0
    teacher_pseudo_selection_mode: str = "adaptive_tail_robust_elbow"
    teacher_pseudo_query_pseudolabel_ratio: float = 5.0
    teacher_pseudo_robust_marker_tail_fraction: float = 0.25
    teacher_pseudo_robust_no_marker_tail_fraction: float = 0.10
    teacher_pseudo_robust_hidden_tail_fraction: float = 0.10
    teacher_pseudo_robust_elbow_floor_fraction: float = 0.40
    teacher_pseudo_overrides: Mapping[str, Any] = field(default_factory=dict)
    pseudo_smallclass_oversample: bool = True
    protein_likelihood: str = "nb_mixture"

    def with_overrides(self, **kwargs: Any) -> "TeacherConfig":
        return replace(self, **kwargs)

    def to_model_kwargs(self) -> dict[str, Any]:
        if self.protein_likelihood != "nb_mixture":
            raise ValueError("ANCHOR currently supports protein_likelihood='nb_mixture' only.")
        return {
            "batch_size": self.batch_size,
            "totalvi_init": self.totalvi_init,
            "random_seed": self.random_seed,
            "n_latent": self.n_latent,
            "n_layers": self.n_layers,
            "round0_epochs": self.round0_epochs,
            "round1_epochs": self.round1_epochs,
            "round2_epochs": self.round2_epochs,
            "n_samples_per_label": self.n_samples_per_label,
            "classification_ratio": self.classification_ratio,
            "hard_ref_sampling_enable": self.hard_ref_sampling_enable,
            "hard_ref_sampling_wrong_weight": self.hard_ref_sampling_wrong_weight,
            "hard_ref_sampling_correct_weight": self.hard_ref_sampling_correct_weight,
            "hard_ref_sampling_max_wrong_fraction": self.hard_ref_sampling_max_wrong_fraction,
            "round0_warmup_epochs": self.round0_warmup_epochs,
            "round0_balance_kl_epochs": self.round0_balance_kl_epochs,
            "hidden_balance_lambda": self.hidden_balance_lambda,
            "hidden_parent_anchor_ce_lambda": self.hidden_parent_anchor_ce_lambda,
            "hidden_parent_anchor_top_k_per_child": self.hidden_parent_anchor_top_k_per_child,
            "hidden_parent_anchor_parent_posterior_thresholds": self.hidden_parent_anchor_parent_posterior_thresholds,
            "fixed_split_seed": self.fixed_split_seed,
            "fixed_split_train_fraction": self.fixed_split_train_fraction,
            "standard_normal_prior_enable": True,
        }

    def hard_sampling_kwargs(self) -> dict[str, Any]:
        return {
            "hard_ref_sampling_enable": self.hard_ref_sampling_enable,
            "hard_ref_sampling_update": self.hard_ref_sampling_update,
            "hard_ref_sampling_wrong_weight": self.hard_ref_sampling_wrong_weight,
            "hard_ref_sampling_correct_weight": self.hard_ref_sampling_correct_weight,
            "hard_ref_sampling_max_wrong_fraction": self.hard_ref_sampling_max_wrong_fraction,
            "hard_ref_sampling_min_wrong_per_label": self.hard_ref_sampling_min_wrong_per_label,
            "hard_ref_sampling_n_samples_per_label": self.n_samples_per_label,
            "hard_ref_sampling_source": self.hard_ref_sampling_source,
            "hard_ref_sampling_seed": self.hard_ref_sampling_seed,
        }


@dataclass(frozen=True)
class StudentConfig:
    batch_size: int = 512
    max_epochs: int = 100
    random_seed: int = 2026
    protein_panel: str = "allprotein"
    teacher_soft_kl_schedule: Mapping[str, Any] = field(
        default_factory=lambda: {
            "mode": "linear",
            "start": 2.0,
            "end": 0.5,
            "hold_epochs": 10,
            "decay_end_epoch": 40,
        }
    )
    rank_loss_weight: float = 1.0
    rank_loss_use_global_child_scores: bool = False
    prototype_logit_weight: float = 1.0
    prototype_ce_lambda: float = 0.5
    graph_consistency_lambda: float = 0.5
    student_loss_weight_overrides: Mapping[str, float] = field(default_factory=dict)
    enable_hidden_rescue: bool = False
    safety_guard_fallback_to_teacher_round2: bool = True
    rho_parent_pool_threshold: float = 0.20
    rho_release_strength: float = 1.0
    rho_policy_params: Mapping[str, Any] = field(default_factory=dict)
    rho_policy_seed: int = 2026
    rho_fail_on_missing_audit: bool = True

    def with_overrides(self, **kwargs: Any) -> "StudentConfig":
        return replace(self, **kwargs)

    def config_overrides(self) -> dict[str, Any]:
        return {
            "teacher_soft_kl_schedule": dict(self.teacher_soft_kl_schedule),
            "pseudo_selection_mode": "adaptive_tail_robust_elbow",
            "rank_loss_weight": self.rank_loss_weight,
            "rank_loss_use_global_child_scores": self.rank_loss_use_global_child_scores,
            "prototype_logit_weight": self.prototype_logit_weight,
            "prototype_ce_lambda": self.prototype_ce_lambda,
            "graph_consistency_lambda": self.graph_consistency_lambda,
            "student_loss_weight_overrides": dict(self.student_loss_weight_overrides),
            "rho_policy_params": dict(self.rho_policy_params),
            "enable_hidden_rescue": self.enable_hidden_rescue,
            "auto_tree_rank_specs": True,
        }


@dataclass(frozen=True)
class AnchorConfig:
    columns: ColumnMap
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    selection: AnchorSelectionConfig = field(default_factory=AnchorSelectionConfig)
    rho: RhoPolicyConfig = field(default_factory=RhoPolicyConfig)
    counts_layer: str = "counts"
    protein_obsm_key: str = "protein_expression"
    heldout_protein_obsm_key: str = "protein_expression_heldout"
    reference_name: str = "reference"
    query_name: str = "query"
    label_key: str = "teacher_label"

    def with_overrides(self, **kwargs: Any) -> "AnchorConfig":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved internal run config used by the self-contained pipeline."""

    run_name: str
    results_dir: Path
    columns: ColumnMap
    marker_tree: MarkerTree
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    counts_layer: str = "counts"
    protein_obsm_key: str = "protein_expression"
    heldout_protein_obsm_key: str = "protein_expression_heldout"
    reference_name: str = "reference"
    query_name: str = "query"
    label_key: str = "teacher_label"

    @property
    def root_dir(self) -> Path:
        return Path(self.results_dir) / self.run_name

    def stage_dir(self, stage: str) -> Path:
        return self.root_dir / stage


def apply_anchor_overrides(config: AnchorConfig, overrides: Mapping[str, Any] | None) -> AnchorConfig:
    """Apply nested config overrides such as {"selection": {"marker_max_cap": 30}}."""

    if not overrides:
        return config
    updates: dict[str, Any] = {}
    for key, value in dict(overrides).items():
        if key in {"teacher", "student", "selection", "rho", "columns"}:
            current = getattr(config, key)
            if not isinstance(value, Mapping):
                raise TypeError(f"Override for {key!r} must be a mapping.")
            updates[key] = replace(current, **dict(value))
        else:
            updates[key] = value
    return replace(config, **updates)
