from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Tuple

from .builder import build_student_data_bundle, build_teacher_data
from .config import (
    AnchorConfig,
    ColumnMap,
    ExperimentConfig,
    StudentConfig,
    TeacherConfig,
    apply_anchor_overrides,
)
from .data import CanonicalInputs, load_inputs, validate_inputs
from .markers import MarkerTree, load_marker_tree
from .output import write_json
from .safety import SafetyGuardDecision, choose_final_stage, write_final_decision


@dataclass(frozen=True)
class StageResult:
    stage: str
    path: Path
    summary_csv: Path
    confusion_counts_csv: Path
    confusion_heatmap_png: Path
    results_h5ad: Path | None = None
    soft_probs_csv: Path | None = None


@dataclass(frozen=True)
class ExperimentResult:
    root_dir: Path
    round0: StageResult | None
    round1: StageResult | None
    round2: StageResult | None
    student: StageResult | None
    final_source: str
    final_dir: Path
    safety: SafetyGuardDecision


@dataclass(frozen=True)
class AnchorRunResult:
    root_dir: Path
    round0_dir: Path
    round1_dir: Path
    round2_dir: Path
    student_dir: Path
    final_dir: Path
    final_source: str
    safety_triggered: bool
    safety_reason: str


TeacherRunner = Callable[..., Tuple[StageResult, StageResult, StageResult]]
StudentRunner = Callable[[CanonicalInputs, ExperimentConfig, StageResult], StageResult]


def _runner_accepts_force_retrain(runner: TeacherRunner) -> bool:
    signature = inspect.signature(runner)
    return "force_retrain" in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )


def _stage_result_from_paths(stage: str, path: str | Path, paths: dict[str, Path]) -> StageResult:
    path = Path(path)
    return StageResult(
        stage=stage,
        path=path,
        summary_csv=paths.get("summary_csv", path / "summary_metrics.csv"),
        confusion_counts_csv=paths.get("confusion_counts_csv", path / "confusion_counts.csv"),
        confusion_heatmap_png=paths.get("confusion_heatmap_png", path / "confusion_heatmap.png"),
        results_h5ad=paths.get("results_h5ad", path / "results.h5ad"),
        soft_probs_csv=paths.get("soft_probs_csv", path / "soft_probs_all_cells.csv"),
    )


def _dataclass_payload(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _dataclass_payload(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _dataclass_payload(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_dataclass_payload(v) for v in value]
    return value


def _write_config(config: ExperimentConfig) -> None:
    config.root_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": config.run_name,
        "results_dir": str(config.results_dir),
        "columns": config.columns.__dict__,
        "marker_tree": config.marker_tree.to_dict(),
        "teacher": config.teacher.__dict__,
        "student": {
            **config.student.__dict__,
            "teacher_soft_kl_schedule": dict(config.student.teacher_soft_kl_schedule),
        },
        "counts_layer": config.counts_layer,
        "protein_obsm_key": config.protein_obsm_key,
        "heldout_protein_obsm_key": config.heldout_protein_obsm_key,
    }
    (config.root_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _canonicalize_student_outputs(student_dir: Path) -> StageResult:
    _copy_if_exists(student_dir / "student_summary_metrics.csv", student_dir / "summary_metrics.csv")
    _copy_if_exists(student_dir / "confusion" / "student_confusion_counts.csv", student_dir / "confusion_counts.csv")
    _copy_if_exists(
        student_dir / "confusion" / "student_confusion_heatmap.png",
        student_dir / "confusion_heatmap.png",
    )
    return StageResult(
        stage="student",
        path=student_dir,
        summary_csv=student_dir / "summary_metrics.csv",
        confusion_counts_csv=student_dir / "confusion_counts.csv",
        confusion_heatmap_png=student_dir / "confusion_heatmap.png",
        results_h5ad=None,
        soft_probs_csv=student_dir / "student_soft_probs.csv",
    )


def _copy_totalvi_initialization(
    *,
    source_dir: Path | None,
    target_dir: Path,
    allow_existing: bool,
) -> None:
    if source_dir is None:
        return
    source_model = source_dir / "model.pt"
    if not source_model.exists():
        raise FileNotFoundError(f"Missing source totalVI initialization: {source_model}")
    if target_dir.exists():
        if not (target_dir / "model.pt").exists():
            raise FileExistsError(f"Target totalVI directory exists but has no model.pt: {target_dir}")
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir)


def _assert_no_existing_training_outputs(root_dir: Path) -> None:
    blockers = [
        root_dir / "round0" / "model" / "model.pt",
        root_dir / "round1" / "model" / "model.pt",
        root_dir / "round2" / "model" / "model.pt",
        root_dir / "student" / "model.pt",
    ]
    existing = [path for path in blockers if path.exists()]
    if existing:
        raise RuntimeError(
            "Teacher/student checkpoints already exist. Use a fresh run_name/results_dir "
            "or set allow_resume_from_existing=True.\n"
            + "\n".join(str(path) for path in existing)
        )


def _resolved_experiment_config(
    *,
    config: AnchorConfig,
    marker_tree: MarkerTree,
    results_dir: Path,
    run_name: str,
) -> ExperimentConfig:
    if config.teacher.protein_likelihood != "nb_mixture":
        raise ValueError("ANCHOR currently supports protein_likelihood='nb_mixture' only.")
    teacher = config.teacher.with_overrides(
        teacher_pseudo_selection_mode="adaptive_tail_robust_elbow",
        teacher_pseudo_query_pseudolabel_ratio=5.0,
        teacher_pseudo_robust_marker_tail_fraction=config.selection.marker_tail_fraction,
        teacher_pseudo_robust_no_marker_tail_fraction=config.selection.no_marker_tail_fraction,
        teacher_pseudo_robust_hidden_tail_fraction=config.selection.hidden_tail_fraction,
        teacher_pseudo_robust_elbow_floor_fraction=config.selection.elbow_floor_fraction,
        teacher_pseudo_overrides=config.selection.as_teacher_pseudo_kwargs(),
    )
    student = config.student.with_overrides(
        rho_parent_pool_threshold=config.rho.parent_pool_threshold,
        rho_release_strength=config.rho.release_strength,
        rho_policy_params=config.rho.as_policy_params(),
    )
    return ExperimentConfig(
        run_name=str(run_name),
        results_dir=Path(results_dir),
        columns=ColumnMap(**_dataclass_payload(config.columns)),
        marker_tree=marker_tree,
        teacher=teacher,
        student=student,
        counts_layer=config.counts_layer,
        protein_obsm_key=config.protein_obsm_key,
        heldout_protein_obsm_key=config.heldout_protein_obsm_key,
        reference_name=config.reference_name,
        query_name=config.query_name,
        label_key=config.label_key,
    )


def run_teacher(
    inputs: CanonicalInputs,
    config: ExperimentConfig,
    *,
    runner: TeacherRunner | None = None,
    force_retrain: bool = False,
) -> Tuple[StageResult, StageResult, StageResult]:
    validate_inputs(
        inputs,
        config.columns,
        counts_layer=config.counts_layer,
        protein_obsm_key=config.protein_obsm_key,
        heldout_protein_obsm_key=config.heldout_protein_obsm_key,
    )
    _write_config(config)
    if runner is None:
        from ._runner import run_generic_teacher

        bundle = build_teacher_data(inputs, config)
        paths0, paths1, paths2 = run_generic_teacher(bundle, config, force_retrain=force_retrain)
        return (
            _stage_result_from_paths("round0", config.stage_dir("round0"), paths0),
            _stage_result_from_paths("round1", config.stage_dir("round1"), paths1),
            _stage_result_from_paths("round2", config.stage_dir("round2"), paths2),
        )
    if _runner_accepts_force_retrain(runner):
        return runner(inputs, config, force_retrain=force_retrain)
    return runner(inputs, config)


def _run_student_stage(
    inputs: CanonicalInputs,
    config: ExperimentConfig,
    round2: StageResult,
    *,
    runner: StudentRunner | None = None,
) -> StageResult:
    validate_inputs(
        inputs,
        config.columns,
        counts_layer=config.counts_layer,
        protein_obsm_key=config.protein_obsm_key,
        heldout_protein_obsm_key=config.heldout_protein_obsm_key,
    )
    if runner is not None:
        return runner(inputs, config, round2)
    if round2.results_h5ad is None or round2.soft_probs_csv is None:
        raise ValueError("round2 StageResult must include results_h5ad and soft_probs_csv for student training")

    student_dir = config.stage_dir("student")
    teacher_bundle = build_teacher_data(inputs, config)
    bundle = build_student_data_bundle(
        teacher_results_h5ad=round2.results_h5ad,
        teacher_soft_csv=round2.soft_probs_csv,
        results_dir=student_dir,
        prior_spec=teacher_bundle.prior_spec,
        leaf_marker_specs=teacher_bundle.leaf_marker_specs,
        protein_obsm_key=config.protein_obsm_key,
        batch_key=config.columns.batch_key,
        query_name=config.query_name,
        protein_panel=config.student.protein_panel,
        partial_label_spec=teacher_bundle.partial_label_spec,
    )

    from .rho import build_rho_policy_kl_specs_for_bundle, compute_rho_policy_audit

    rho_audit = compute_rho_policy_audit(
        bundle=bundle,
        teacher_bundle=teacher_bundle,
        teacher_results_h5ad=round2.results_h5ad,
        output_dir=student_dir / "tree_reliability_rho_policy_audit",
        dataset=config.run_name,
        setting="anchor",
        teacher_source="round2",
        reference_name=config.reference_name,
        seed=config.student.rho_policy_seed,
        parent_pool_threshold=config.student.rho_parent_pool_threshold,
        release_strength=config.student.rho_release_strength,
        policy_params=config.student.rho_policy_params,
        fail_on_missing_audit=config.student.rho_fail_on_missing_audit,
    )
    rho_table = rho_audit["frames"]["node"]
    rho_table.to_csv(student_dir / "rho_node_audit_source_rows.csv", index=False)
    rho_specs, rho_table = build_rho_policy_kl_specs_for_bundle(
        bundle,
        rho_table,
        fail_on_missing_audit=config.student.rho_fail_on_missing_audit,
    )

    from .student import run_bottomup_treeguard_student_from_bundle

    config_overrides = config.student.config_overrides()
    config_overrides["rho_audit_dir"] = "tree_reliability_rho_policy_audit"
    config_overrides["rho_node_audit_csv"] = "tree_reliability_rho_policy_audit/rho_node_audit.csv"
    config_overrides["rho_node_audit_source_rows_csv"] = "rho_node_audit_source_rows.csv"
    if teacher_bundle.partial_label_spec:
        config_overrides["enable_hidden_rescue"] = True
    run_bottomup_treeguard_student_from_bundle(
        bundle=bundle,
        results_dir=student_dir,
        random_seed=config.student.random_seed,
        max_epochs=config.student.max_epochs,
        batch_size=config.student.batch_size,
        config_overrides=config_overrides,
        conditional_kl_specs_override=rho_specs,
        conditional_kl_table_override=rho_table,
    )
    return _canonicalize_student_outputs(student_dir)


def run_experiment(
    inputs: CanonicalInputs,
    config: ExperimentConfig,
    *,
    teacher_runner: TeacherRunner | None = None,
    student_runner: StudentRunner | None = None,
    force_retrain: bool = False,
) -> ExperimentResult:
    round0, round1, round2 = run_teacher(inputs, config, runner=teacher_runner, force_retrain=force_retrain)
    student = _run_student_stage(inputs, config, round2, runner=student_runner)
    final_stage_dir, decision = choose_final_stage(
        student_dir=student.path,
        teacher_round2_dir=round2.path,
        fallback_to_teacher_round2=config.student.safety_guard_fallback_to_teacher_round2,
    )
    final_dir = config.root_dir / "final"
    write_final_decision(final_dir, decision)
    return ExperimentResult(
        root_dir=config.root_dir,
        round0=round0,
        round1=round1,
        round2=round2,
        student=student,
        final_source=decision.final_source,
        final_dir=final_stage_dir,
        safety=decision,
    )


def _existing_round2_stage(config: ExperimentConfig) -> StageResult:
    round2_dir = config.stage_dir("round2")
    round2 = StageResult(
        stage="round2",
        path=round2_dir,
        summary_csv=round2_dir / "summary_metrics.csv",
        confusion_counts_csv=round2_dir / "confusion_counts.csv",
        confusion_heatmap_png=round2_dir / "confusion_heatmap.png",
        results_h5ad=round2_dir / "results.h5ad",
        soft_probs_csv=round2_dir / "soft_probs_all_cells.csv",
    )
    missing = [
        path
        for path in (round2.results_h5ad, round2.soft_probs_csv)
        if path is None or not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot run student-only mode because required round2 teacher outputs are missing:\n"
            + "\n".join(str(path) for path in missing)
        )
    return round2


def run_student_from_existing_teacher(
    inputs: CanonicalInputs,
    config: ExperimentConfig,
    *,
    student_runner: StudentRunner | None = None,
) -> ExperimentResult:
    """Run only the student stage from existing round2 teacher outputs."""

    validate_inputs(
        inputs,
        config.columns,
        counts_layer=config.counts_layer,
        protein_obsm_key=config.protein_obsm_key,
        heldout_protein_obsm_key=config.heldout_protein_obsm_key,
    )
    _write_config(config)
    round2 = _existing_round2_stage(config)
    student = _run_student_stage(inputs, config, round2, runner=student_runner)
    final_stage_dir, decision = choose_final_stage(
        student_dir=student.path,
        teacher_round2_dir=round2.path,
        fallback_to_teacher_round2=config.student.safety_guard_fallback_to_teacher_round2,
    )
    final_dir = config.root_dir / "final"
    write_final_decision(final_dir, decision)
    return ExperimentResult(
        root_dir=config.root_dir,
        round0=None,
        round1=None,
        round2=round2,
        student=student,
        final_source=decision.final_source,
        final_dir=final_stage_dir,
        safety=decision,
    )


def _make_anchor_config(
    *,
    batch_key: str,
    celltype_key: str,
    query_label_key: str | None = None,
    split_key: str | None = None,
    hidden_branch_key: str | None = None,
    sample_key: str | None = None,
    counts_layer: str = "counts",
    protein_obsm_key: str = "protein_expression",
    heldout_protein_obsm_key: str = "protein_expression_heldout",
    reference_name: str = "reference",
    query_name: str = "query",
    label_key: str = "teacher_label",
    batch_size: int | None = None,
    student_max_epochs: int | None = None,
    random_seed: int | None = None,
    teacher_overrides: Mapping[str, Any] | None = None,
    student_overrides: Mapping[str, Any] | None = None,
    selection_overrides: Mapping[str, Any] | None = None,
    rho_overrides: Mapping[str, Any] | None = None,
) -> AnchorConfig:
    """Build the internal config from user-facing run arguments."""

    config = AnchorConfig(
        columns=ColumnMap(
            batch_key=batch_key,
            celltype_key=celltype_key,
            query_label_key=query_label_key,
            split_key=split_key,
            hidden_branch_key=hidden_branch_key,
            sample_key=sample_key,
        ),
        counts_layer=counts_layer,
        protein_obsm_key=protein_obsm_key,
        heldout_protein_obsm_key=heldout_protein_obsm_key,
        reference_name=reference_name,
        query_name=query_name,
        label_key=label_key,
    )
    teacher_updates = dict(teacher_overrides or {})
    student_updates = dict(student_overrides or {})
    selection_updates = dict(selection_overrides or {})
    rho_updates = dict(rho_overrides or {})
    if batch_size is not None:
        teacher_updates["batch_size"] = int(batch_size)
        student_updates["batch_size"] = int(batch_size)
    if student_max_epochs is not None:
        student_updates["max_epochs"] = int(student_max_epochs)
    if random_seed is not None:
        seed = int(random_seed)
        teacher_updates["random_seed"] = seed
        teacher_updates["hard_ref_sampling_seed"] = seed
        student_updates["random_seed"] = seed
        student_updates["rho_policy_seed"] = seed
    return apply_anchor_overrides(
        config,
        {
            "teacher": teacher_updates,
            "student": student_updates,
            "selection": selection_updates,
            "rho": rho_updates,
        },
    )


def _anchor_result_from_experiment(result: ExperimentResult) -> AnchorRunResult:
    return AnchorRunResult(
        root_dir=Path(result.root_dir),
        round0_dir=Path(result.round0.path) if result.round0 is not None else Path(result.root_dir) / "round0",
        round1_dir=Path(result.round1.path) if result.round1 is not None else Path(result.root_dir) / "round1",
        round2_dir=Path(result.round2.path) if result.round2 is not None else Path(result.root_dir) / "round2",
        student_dir=Path(result.student.path) if result.student is not None else Path(result.root_dir) / "student",
        final_dir=Path(result.final_dir),
        final_source=str(result.final_source),
        safety_triggered=bool(result.safety.triggered),
        safety_reason=str(result.safety.reason),
    )


def run_anchor(
    *,
    reference: str | Path | Any,
    query: str | Path | Any,
    marker_tree: str | Path | Mapping[str, Any] | MarkerTree,
    results_dir: str | Path,
    run_name: str,
    batch_key: str,
    celltype_key: str,
    query_label_key: str | None = None,
    split_key: str | None = None,
    hidden_branch_key: str | None = None,
    sample_key: str | None = None,
    counts_layer: str = "counts",
    protein_obsm_key: str = "protein_expression",
    heldout_protein_obsm_key: str = "protein_expression_heldout",
    reference_name: str = "reference",
    query_name: str = "query",
    label_key: str = "teacher_label",
    batch_size: int | None = None,
    student_max_epochs: int | None = None,
    random_seed: int | None = None,
    teacher_overrides: Mapping[str, Any] | None = None,
    student_overrides: Mapping[str, Any] | None = None,
    selection_overrides: Mapping[str, Any] | None = None,
    rho_overrides: Mapping[str, Any] | None = None,
    source_totalvi_init_dir: str | Path | None = None,
    force_retrain: bool = False,
    allow_resume_from_existing: bool = False,
) -> AnchorRunResult:
    """Run the full ANCHOR teacher-student pipeline.

    ``reference`` and ``query`` may be paths to ``.h5ad`` files or loaded
    AnnData objects. ``marker_tree`` may be a JSON path, nested dict, or
    ``MarkerTree``. Most users should keep the default training parameters and
    only pass overrides for dataset-specific smoke tests or ablations.
    """

    config = _make_anchor_config(
        batch_key=batch_key,
        celltype_key=celltype_key,
        query_label_key=query_label_key,
        split_key=split_key,
        hidden_branch_key=hidden_branch_key,
        sample_key=sample_key,
        counts_layer=counts_layer,
        protein_obsm_key=protein_obsm_key,
        heldout_protein_obsm_key=heldout_protein_obsm_key,
        reference_name=reference_name,
        query_name=query_name,
        label_key=label_key,
        batch_size=batch_size,
        student_max_epochs=student_max_epochs,
        random_seed=random_seed,
        teacher_overrides=teacher_overrides,
        student_overrides=student_overrides,
        selection_overrides=selection_overrides,
        rho_overrides=rho_overrides,
    )
    inputs = load_inputs(reference=reference, query=query, marker_tree=marker_tree)
    resolved = _resolved_experiment_config(
        config=config,
        marker_tree=inputs.marker_tree,
        results_dir=Path(results_dir),
        run_name=str(run_name),
    )
    resolved.root_dir.mkdir(parents=True, exist_ok=True)
    write_json(resolved.root_dir / "anchor_config.json", _dataclass_payload(config))
    if not allow_resume_from_existing:
        _assert_no_existing_training_outputs(resolved.root_dir)
    _copy_totalvi_initialization(
        source_dir=Path(source_totalvi_init_dir) if source_totalvi_init_dir else None,
        target_dir=resolved.stage_dir("totalvi_init_model"),
        allow_existing=allow_resume_from_existing,
    )
    return _anchor_result_from_experiment(run_experiment(inputs, resolved, force_retrain=force_retrain))


def run_student(
    *,
    reference: str | Path | Any,
    query: str | Path | Any,
    marker_tree: str | Path | Mapping[str, Any] | MarkerTree,
    results_dir: str | Path,
    run_name: str,
    batch_key: str,
    celltype_key: str,
    query_label_key: str | None = None,
    split_key: str | None = None,
    hidden_branch_key: str | None = None,
    sample_key: str | None = None,
    counts_layer: str = "counts",
    protein_obsm_key: str = "protein_expression",
    heldout_protein_obsm_key: str = "protein_expression_heldout",
    reference_name: str = "reference",
    query_name: str = "query",
    label_key: str = "teacher_label",
    batch_size: int | None = None,
    student_max_epochs: int | None = None,
    random_seed: int | None = None,
    teacher_overrides: Mapping[str, Any] | None = None,
    student_overrides: Mapping[str, Any] | None = None,
    selection_overrides: Mapping[str, Any] | None = None,
    rho_overrides: Mapping[str, Any] | None = None,
) -> AnchorRunResult:
    """Train only the student from existing round-2 teacher outputs."""

    config = _make_anchor_config(
        batch_key=batch_key,
        celltype_key=celltype_key,
        query_label_key=query_label_key,
        split_key=split_key,
        hidden_branch_key=hidden_branch_key,
        sample_key=sample_key,
        counts_layer=counts_layer,
        protein_obsm_key=protein_obsm_key,
        heldout_protein_obsm_key=heldout_protein_obsm_key,
        reference_name=reference_name,
        query_name=query_name,
        label_key=label_key,
        batch_size=batch_size,
        student_max_epochs=student_max_epochs,
        random_seed=random_seed,
        teacher_overrides=teacher_overrides,
        student_overrides=student_overrides,
        selection_overrides=selection_overrides,
        rho_overrides=rho_overrides,
    )
    inputs = load_inputs(reference=reference, query=query, marker_tree=marker_tree)
    resolved = _resolved_experiment_config(
        config=config,
        marker_tree=inputs.marker_tree,
        results_dir=Path(results_dir),
        run_name=str(run_name),
    )
    resolved.root_dir.mkdir(parents=True, exist_ok=True)
    write_json(resolved.root_dir / "anchor_config.json", _dataclass_payload(config))
    return _anchor_result_from_experiment(run_student_from_existing_teacher(inputs, resolved))
