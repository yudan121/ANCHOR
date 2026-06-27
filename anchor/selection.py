from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AnchorSelectionConfig
from . import student as anchor_student


@dataclass(frozen=True)
class AnchorSelectionResult:
    cell_level: pd.DataFrame
    by_class: pd.DataFrame
    vetoed_candidates: pd.DataFrame
    evidence: pd.DataFrame


def select_anchor_pseudolabels(
    bundle: Any,
    *,
    config: AnchorSelectionConfig | None = None,
    enable_hidden_rescue: bool | None = None,
) -> AnchorSelectionResult:
    """Select final query anchors for student training.

    This public facade is used after the final teacher round.  Teacher
    refinement uses the related pseudo-label selector in
    ``anchor.teacher.pseudolabels``.
    """

    cfg = config or AnchorSelectionConfig()
    hidden_rescue = bool(bundle.partial_label_spec) if enable_hidden_rescue is None else bool(enable_hidden_rescue)
    overrides = cfg.as_student_overrides()
    pseudo_df, by_class, vetoed, evidence = anchor_student.select_bottomup_treeguard_pseudolabels(
        bundle,
        posterior_threshold=float(overrides["posterior_threshold"]),
        max_marker_pseudo_per_class=20,
        max_no_marker_pseudo_per_class=int(cfg.no_marker_max_cap),
        max_hidden_rescue_per_child=int(cfg.hidden_max_cap_per_child),
        wide_candidate_multiplier=int(cfg.wide_candidate_multiplier),
        parent_pool_threshold=float(cfg.parent_pool_threshold),
        child_conditional_threshold=float(cfg.child_conditional_threshold),
        enable_hidden_rescue=hidden_rescue,
        hard_contradiction_quantile=float(cfg.hard_contradiction_quantile),
        soft_contradiction_quantile=float(cfg.soft_contradiction_quantile),
        soft_contradiction_penalty=float(cfg.soft_contradiction_penalty),
        pseudo_selection_mode="adaptive_tail_robust_elbow",
        adaptive_config=overrides,
    )
    return AnchorSelectionResult(
        cell_level=pseudo_df,
        by_class=by_class,
        vetoed_candidates=vetoed,
        evidence=evidence,
    )


def write_anchor_selection(result: AnchorSelectionResult, out_dir: str | Path, *, prefix: str = "anchor") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.cell_level.to_csv(out / f"{prefix}_pseudolabel_cell_level.csv", index=False)
    result.by_class.to_csv(out / f"{prefix}_pseudolabel_by_class.csv", index=False)
    result.vetoed_candidates.to_csv(out / f"{prefix}_vetoed_candidates.csv", index=False)
    result.evidence.to_csv(out / f"{prefix}_evidence_summary.csv", index=False)


__all__ = ["AnchorSelectionResult", "select_anchor_pseudolabels", "write_anchor_selection"]
