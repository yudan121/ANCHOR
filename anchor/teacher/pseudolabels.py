"""Teacher-round pseudo-label selection.

This module selects query cells that are confident enough to refine the teacher
between rounds.  It shares the same marker-guided evidence sources as the final
student anchor selector, but its outputs are written back to the teacher
AnnData object rather than used directly by the student.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from ..partial.labels import (
    GENERIC_FLAT_SCORE_MODE,
    compute_flat_leaf_target_score,
)
from ..partial.pseudolabels import PartialFlatLeafPseudoLabelSelection
from ..student.anchors import _ancestor_contradiction, _direct_sibling_margin
from ..student.bundle import (
    _child_conditional,
    _finalize_pseudolabel_rows,
    _knn_purity,
    _node_posterior,
)
from .model import PairQueryPseudoLabelBundle

@dataclass
class FlatLeafPseudoLabelSelection:
    cell_level: pd.DataFrame
    by_class: pd.DataFrame
    overall: pd.DataFrame
    score_availability: pd.DataFrame
    pair_bundle: PairQueryPseudoLabelBundle


@dataclass(frozen=True)
class UnifiedTeacherPseudoConfig:
    pseudo_selection_mode: str = "adaptive_tail_robust_elbow"
    posterior_threshold: float = 0.95
    max_marker_pseudo_per_class: int = 20
    max_no_marker_pseudo_per_class: int = 10
    wide_candidate_multiplier: int = 8
    hard_contradiction_quantile: float = 0.90
    soft_contradiction_quantile: float = 0.75
    soft_contradiction_penalty: float = 0.25
    parent_pool_threshold: float = 0.20
    child_conditional_threshold: float = 0.05
    max_hidden_rescue_per_child: int = 10
    query_pseudolabel_ratio: float = 5.0
    smallclass_min_effective_selected_per_class: int = 50
    smallclass_max_repeats_per_cell: int = 9
    adaptive_marker_tail_fraction: float = 0.15
    adaptive_no_marker_tail_fraction: float = 0.05
    adaptive_hidden_tail_fraction: float = 0.10
    robust_marker_tail_fraction: float = 0.25
    robust_no_marker_tail_fraction: float = 0.10
    robust_hidden_tail_fraction: float = 0.10
    adaptive_marker_max_cap: int = 50
    adaptive_no_marker_max_cap: int = 10
    adaptive_hidden_max_cap_per_child: int = 10
    adaptive_marker_min_select_if_any: int = 1
    adaptive_no_marker_min_select_if_any: int = 1
    adaptive_hidden_min_select_if_any: int = 0
    robust_marker_min_select_if_any: int = 3
    robust_no_marker_min_select_if_any: int = 2
    robust_hidden_min_select_if_any: int = 0
    robust_elbow_drop_ratio: float = 5.0
    robust_elbow_absolute_min_drop: float = 0.03
    robust_elbow_floor_fraction: float = 0.40
    robust_marker_min_elbow_count: int = 5
    robust_no_marker_min_elbow_count: int = 2
    robust_hidden_min_elbow_count: int = 1
    adaptive_strong_reliability_threshold: float = 0.70
    adaptive_medium_reliability_threshold: float = 0.55
    adaptive_strong_effective_target: int = 50
    adaptive_medium_effective_target: int = 30
    adaptive_weak_effective_target: int = 20
    adaptive_no_marker_effective_target: int = 10
    adaptive_marker_max_repeats_per_cell: int = 9
    adaptive_hidden_max_repeats_per_cell: int = 4
    adaptive_no_marker_max_repeats_per_cell: int = 2
    adaptive_marker_pseudo_weight: float = 1.0
    adaptive_hidden_pseudo_weight: float = 0.5
    adaptive_no_marker_pseudo_weight: float = 0.25

    def selector_overrides(self, *, enable_hidden_rescue: bool) -> dict[str, Any]:
        return {
            "posterior_threshold": self.posterior_threshold,
            "max_marker_pseudo_per_class": self.max_marker_pseudo_per_class,
            "max_no_marker_pseudo_per_class": self.max_no_marker_pseudo_per_class,
            "wide_candidate_multiplier": self.wide_candidate_multiplier,
            "hard_contradiction_quantile": self.hard_contradiction_quantile,
            "soft_contradiction_quantile": self.soft_contradiction_quantile,
            "soft_contradiction_penalty": self.soft_contradiction_penalty,
            "parent_pool_threshold": self.parent_pool_threshold,
            "child_conditional_threshold": self.child_conditional_threshold,
            "max_hidden_rescue_per_child": self.max_hidden_rescue_per_child,
            "enable_hidden_rescue": bool(enable_hidden_rescue),
        }

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "pseudo_selection_mode": self.pseudo_selection_mode,
            "posterior_threshold": self.posterior_threshold,
            "max_marker_pseudo_per_class": self.max_marker_pseudo_per_class,
            "max_no_marker_pseudo_per_class": self.max_no_marker_pseudo_per_class,
            "wide_candidate_multiplier": self.wide_candidate_multiplier,
            "hard_contradiction_quantile": self.hard_contradiction_quantile,
            "soft_contradiction_quantile": self.soft_contradiction_quantile,
            "soft_contradiction_penalty": self.soft_contradiction_penalty,
            "parent_pool_threshold": self.parent_pool_threshold,
            "child_conditional_threshold": self.child_conditional_threshold,
            "max_hidden_rescue_per_child": self.max_hidden_rescue_per_child,
            "query_pseudolabel_ratio": self.query_pseudolabel_ratio,
            "smallclass_min_effective_selected_per_class": self.smallclass_min_effective_selected_per_class,
            "smallclass_max_repeats_per_cell": self.smallclass_max_repeats_per_cell,
            "adaptive_marker_tail_fraction": self.adaptive_marker_tail_fraction,
            "adaptive_no_marker_tail_fraction": self.adaptive_no_marker_tail_fraction,
            "adaptive_hidden_tail_fraction": self.adaptive_hidden_tail_fraction,
            "robust_marker_tail_fraction": self.robust_marker_tail_fraction,
            "robust_no_marker_tail_fraction": self.robust_no_marker_tail_fraction,
            "robust_hidden_tail_fraction": self.robust_hidden_tail_fraction,
            "adaptive_marker_max_cap": self.adaptive_marker_max_cap,
            "adaptive_no_marker_max_cap": self.adaptive_no_marker_max_cap,
            "adaptive_hidden_max_cap_per_child": self.adaptive_hidden_max_cap_per_child,
            "adaptive_marker_min_select_if_any": self.adaptive_marker_min_select_if_any,
            "adaptive_no_marker_min_select_if_any": self.adaptive_no_marker_min_select_if_any,
            "adaptive_hidden_min_select_if_any": self.adaptive_hidden_min_select_if_any,
            "robust_marker_min_select_if_any": self.robust_marker_min_select_if_any,
            "robust_no_marker_min_select_if_any": self.robust_no_marker_min_select_if_any,
            "robust_hidden_min_select_if_any": self.robust_hidden_min_select_if_any,
            "robust_elbow_drop_ratio": self.robust_elbow_drop_ratio,
            "robust_elbow_absolute_min_drop": self.robust_elbow_absolute_min_drop,
            "robust_elbow_floor_fraction": self.robust_elbow_floor_fraction,
            "robust_marker_min_elbow_count": self.robust_marker_min_elbow_count,
            "robust_no_marker_min_elbow_count": self.robust_no_marker_min_elbow_count,
            "robust_hidden_min_elbow_count": self.robust_hidden_min_elbow_count,
            "adaptive_strong_reliability_threshold": self.adaptive_strong_reliability_threshold,
            "adaptive_medium_reliability_threshold": self.adaptive_medium_reliability_threshold,
            "adaptive_strong_effective_target": self.adaptive_strong_effective_target,
            "adaptive_medium_effective_target": self.adaptive_medium_effective_target,
            "adaptive_weak_effective_target": self.adaptive_weak_effective_target,
            "adaptive_no_marker_effective_target": self.adaptive_no_marker_effective_target,
            "adaptive_marker_max_repeats_per_cell": self.adaptive_marker_max_repeats_per_cell,
            "adaptive_hidden_max_repeats_per_cell": self.adaptive_hidden_max_repeats_per_cell,
            "adaptive_no_marker_max_repeats_per_cell": self.adaptive_no_marker_max_repeats_per_cell,
            "adaptive_marker_pseudo_weight": self.adaptive_marker_pseudo_weight,
            "adaptive_hidden_pseudo_weight": self.adaptive_hidden_pseudo_weight,
            "adaptive_no_marker_pseudo_weight": self.adaptive_no_marker_pseudo_weight,
        }

    @property
    def is_adaptive_tail(self) -> bool:
        return str(self.pseudo_selection_mode).lower() == "adaptive_tail_robust_elbow"

    @property
    def is_adaptive_tail_robust_elbow(self) -> bool:
        return str(self.pseudo_selection_mode).lower() == "adaptive_tail_robust_elbow"

    def output_prefix(self) -> str:
        return "teacher_adaptive_tail_robust_elbow"


def _children_map(prior_spec: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        str(parent): [str(child) for child in children]
        for parent, children in prior_spec.get("tree_spec", {}).get("children", {}).items()
    }


def _parent_map(prior_spec: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for parent, children in _children_map(prior_spec).items():
        for child in children:
            out[str(child)] = str(parent)
    return out


def _descendant_leaves(prior_spec: Mapping[str, Any], node: str, labels: Sequence[str]) -> list[str]:
    label_set = {str(x) for x in labels}
    desc = prior_spec.get("tree_spec", {}).get("descendants", {}).get(str(node), [str(node)])
    return [str(x) for x in desc if str(x) in label_set]


def _score_signed_markers(
    protein: pd.DataFrame,
    signed_spec: Mapping[str, Any],
    *,
    index: pd.Index,
) -> tuple[pd.Series | None, dict[str, Any]]:
    pos_values = signed_spec.get("positive", {}) if isinstance(signed_spec, Mapping) else {}
    neg_values = signed_spec.get("negative", {}) if isinstance(signed_spec, Mapping) else {}
    pos = [str(x) for x in (pos_values.keys() if isinstance(pos_values, Mapping) else pos_values)]
    neg = [str(x) for x in (neg_values.keys() if isinstance(neg_values, Mapping) else neg_values)]
    available = set(protein.columns.astype(str))
    pos_avail = [m for m in pos if m in available]
    neg_avail = [m for m in neg if m in available]
    if not pos_avail and not neg_avail:
        return None, {
            "score_available": False,
            "score_type": "no_marker_score_available",
            "positive_markers": "|".join(pos),
            "negative_markers": "|".join(neg),
            "missing_markers": "|".join(sorted(set(pos + neg) - available)),
        }
    score = pd.Series(0.0, index=index, dtype=float)
    if pos_avail:
        score = score + protein.loc[index, pos_avail].astype(float).mean(axis=1)
    if neg_avail:
        score = score - protein.loc[index, neg_avail].astype(float).mean(axis=1)
    return score.astype(float), {
        "score_available": True,
        "score_type": "mean_positive_minus_negative",
        "positive_markers": "|".join(pos_avail),
        "negative_markers": "|".join(neg_avail),
        "missing_markers": "|".join(sorted(set(pos + neg) - available)),
    }


def _build_teacher_target_scores(
    *,
    protein_arcsinh: pd.DataFrame,
    labels: Sequence[str],
    prior_spec: Mapping[str, Any],
    leaf_marker_specs: Mapping[str, Mapping[str, Any]] | None,
    score_mode: str = GENERIC_FLAT_SCORE_MODE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    index = pd.Index(protein_arcsinh.index.astype(str))
    leaf_marker_specs = leaf_marker_specs or {}
    scores: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    parents = _parent_map(prior_spec)
    for label in [str(x) for x in labels]:
        score: pd.Series | None = None
        meta: dict[str, Any] = {}
        if label in leaf_marker_specs:
            score, meta = compute_flat_leaf_target_score(
                label,
                protein_arcsinh=protein_arcsinh,
                leaf_marker_specs=leaf_marker_specs,
                score_mode=score_mode,
            )
        if score is None:
            parent = parents.get(label, "")
            class_spec = (
                prior_spec.get("branch_teacher_specs", {})
                .get(str(parent), {})
                .get("classes", {})
                .get(label, {})
            )
            score, meta = _score_signed_markers(protein_arcsinh, class_spec, index=index)
        if score is None:
            score = pd.Series(np.nan, index=index, dtype=float)
        scores[label] = score.reindex(index).astype(float)
        rows.append({"target_label": label, **meta})
    score_df = pd.DataFrame(scores, index=index)
    available = score_df.replace([np.inf, -np.inf], np.nan).notna().any(axis=0)
    return score_df, available.astype(bool), pd.DataFrame(rows)


def _collapsed_pred_for_partial_nodes(
    soft: pd.DataFrame,
    *,
    prior_spec: Mapping[str, Any],
    labels: Sequence[str],
    partial_label_spec: Mapping[str, Sequence[str]] | None,
) -> pd.Series:
    if not partial_label_spec:
        return soft.idxmax(axis=1).astype(str)
    nodes = [str(x) for x in partial_label_spec]
    if not nodes:
        return soft.idxmax(axis=1).astype(str)
    masses: dict[str, pd.Series] = {}
    for node in nodes:
        leaves = [leaf for leaf in _descendant_leaves(prior_spec, node, labels) if leaf in soft.columns]
        if leaves:
            masses[node] = soft.loc[:, leaves].sum(axis=1).astype(float)
    if not masses:
        return soft.idxmax(axis=1).astype(str)
    return pd.DataFrame(masses, index=soft.index).idxmax(axis=1).astype(str)


def _knn_purity_from_latent(latent: np.ndarray | None, pred: pd.Series, *, k: int = 15) -> pd.Series:
    if latent is None:
        return pd.Series(1.0, index=pred.index, dtype=float)
    try:
        return _knn_purity(np.asarray(latent, dtype=np.float32), pred.astype(str), k=k)
    except Exception:
        return pd.Series(1.0, index=pred.index, dtype=float)



def _rank_high(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not values.notna().any():
        return pd.Series(0.5, index=series.index, dtype=float)
    return values.rank(method="average", pct=True).fillna(0.5).astype(float)


def _rank_low(series: pd.Series) -> pd.Series:
    return (1.0 - _rank_high(series)).clip(0.0, 1.0)


def _add_adaptive_reliability(df: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    out = df.copy()
    parts: list[pd.Series] = []
    if "target_posterior" in out:
        parts.append(_rank_high(out["target_posterior"]))
    if "teacher_confidence" in out:
        parts.append(_rank_high(out["teacher_confidence"]))
    if "knn_purity" in out:
        parts.append(_rank_high(out["knn_purity"]))
    if "ancestor_max_contradiction_quantile" in out:
        parts.append(_rank_low(out["ancestor_max_contradiction_quantile"]))
    if mode in {"marker", "hidden"}:
        if "target_score" in out:
            parts.append(_rank_high(out["target_score"]))
        if "direct_sibling_margin" in out and out["direct_sibling_margin"].replace([np.inf, -np.inf], np.nan).notna().any():
            parts.append(_rank_high(out["direct_sibling_margin"]))
    if mode == "hidden":
        if "parent_posterior" in out:
            parts.append(_rank_high(out["parent_posterior"]))
        if "child_conditional_posterior" in out:
            parts.append(_rank_high(out["child_conditional_posterior"]))
    out["adaptive_reliability_score"] = pd.concat(parts, axis=1).mean(axis=1).clip(0.0, 1.0) if parts else 0.5
    out["adaptive_reliability_rank"] = out["adaptive_reliability_score"].rank(method="first", ascending=False).astype(int)
    return out


def _adaptive_tail_count_details(
    scores: pd.Series,
    *,
    fraction: float,
    min_if_any: int,
    max_cap: int,
    robust: bool = False,
    drop_ratio: float = 3.0,
    absolute_min_drop: float = 0.0,
    floor_fraction: float = 0.0,
    min_elbow_count: int = 1,
) -> dict[str, Any]:
    clean = pd.to_numeric(scores, errors="coerce").dropna().sort_values(ascending=False)
    n = int(clean.shape[0])
    if n <= 0:
        return {
            "n_select": 0,
            "selection_reason": "no_eligible_candidates",
            "n_eligible_for_tail": 0,
            "tail_fraction": float(fraction),
            "base_count": 0,
            "floor_count": 0,
            "max_cap": int(max_cap),
            "min_if_any": int(min_if_any),
            "elbow_rank": 0,
            "largest_drop": np.nan,
            "median_positive_drop": np.nan,
            "drop_ratio_threshold": float(drop_ratio),
            "absolute_min_drop": float(absolute_min_drop),
            "min_elbow_count": int(min_elbow_count),
        }
    base = int(np.ceil(n * float(fraction)))
    base = int(np.clip(base, int(min_if_any), int(max_cap)))
    base = min(base, n)
    floor = int(np.ceil(base * float(floor_fraction))) if robust else int(min_if_any)
    floor = int(np.clip(max(int(min_if_any), floor), 0, min(int(max_cap), n)))
    details = {
        "n_eligible_for_tail": n,
        "tail_fraction": float(fraction),
        "base_count": int(base),
        "floor_count": int(floor),
        "max_cap": int(max_cap),
        "min_if_any": int(min_if_any),
        "elbow_rank": 0,
        "largest_drop": np.nan,
        "median_positive_drop": np.nan,
        "drop_ratio_threshold": float(drop_ratio),
        "absolute_min_drop": float(absolute_min_drop),
        "min_elbow_count": int(min_elbow_count),
    }
    if n < max(4, int(min_if_any) + 2):
        return {**details, "n_select": int(base), "selection_reason": "fraction_tail_small_pool"}
    head = clean.iloc[: min(n, int(max_cap) + 1)].to_numpy(dtype=float)
    drops = head[:-1] - head[1:]
    pos = drops[drops > 0]
    if pos.size:
        median = float(np.median(pos))
        elbow_idx = int(np.argmax(drops))
        largest = float(drops[elbow_idx])
        elbow_count = int(elbow_idx + 1)
        details.update({"elbow_rank": elbow_count, "largest_drop": largest, "median_positive_drop": median})
        if not robust:
            if median > 0 and largest >= 3.0 * median and elbow_count >= int(min_if_any):
                n_select = int(np.clip(elbow_count, int(min_if_any), min(int(max_cap), n)))
                return {**details, "n_select": n_select, "selection_reason": "adaptive_elbow"}
        else:
            if median <= 0 or largest < float(drop_ratio) * median:
                return {**details, "n_select": int(base), "selection_reason": "fraction_tail"}
            if largest < float(absolute_min_drop):
                return {**details, "n_select": int(base), "selection_reason": "robust_elbow_ignored_small_drop"}
            if elbow_count < int(min_elbow_count):
                return {**details, "n_select": int(base), "selection_reason": "robust_elbow_ignored_too_early"}
            if elbow_count < floor:
                return {**details, "n_select": int(floor), "selection_reason": "robust_elbow_floor_limited"}
            n_select = int(np.clip(elbow_count, floor, min(int(max_cap), n)))
            return {**details, "n_select": n_select, "selection_reason": "robust_elbow"}
    return {**details, "n_select": int(base), "selection_reason": "fraction_tail"}


def _adaptive_tail_count(scores: pd.Series, *, fraction: float, min_if_any: int, max_cap: int) -> tuple[int, str]:
    details = _adaptive_tail_count_details(scores, fraction=fraction, min_if_any=min_if_any, max_cap=max_cap)
    return int(details["n_select"]), str(details["selection_reason"])



def _adaptive_tail_selection_params(config: UnifiedTeacherPseudoConfig, *, kind: str) -> dict[str, Any]:
    if kind == "marker":
        tail_fraction = config.robust_marker_tail_fraction if config.is_adaptive_tail_robust_elbow else config.adaptive_marker_tail_fraction
        min_if_any = config.robust_marker_min_select_if_any if config.is_adaptive_tail_robust_elbow else config.adaptive_marker_min_select_if_any
        max_cap = config.adaptive_marker_max_cap
        min_elbow_count = config.robust_marker_min_elbow_count
    elif kind == "no_marker":
        tail_fraction = config.robust_no_marker_tail_fraction if config.is_adaptive_tail_robust_elbow else config.adaptive_no_marker_tail_fraction
        min_if_any = config.robust_no_marker_min_select_if_any if config.is_adaptive_tail_robust_elbow else config.adaptive_no_marker_min_select_if_any
        max_cap = config.adaptive_no_marker_max_cap
        min_elbow_count = config.robust_no_marker_min_elbow_count
    elif kind == "hidden":
        tail_fraction = config.robust_hidden_tail_fraction if config.is_adaptive_tail_robust_elbow else config.adaptive_hidden_tail_fraction
        min_if_any = config.robust_hidden_min_select_if_any if config.is_adaptive_tail_robust_elbow else config.adaptive_hidden_min_select_if_any
        max_cap = config.adaptive_hidden_max_cap_per_child
        min_elbow_count = config.robust_hidden_min_elbow_count
    else:
        raise ValueError(f"Unknown adaptive tail kind: {kind}")
    return {
        "fraction": float(tail_fraction),
        "min_if_any": int(min_if_any),
        "max_cap": int(max_cap),
        "robust": bool(config.is_adaptive_tail_robust_elbow),
        "drop_ratio": float(config.robust_elbow_drop_ratio),
        "absolute_min_drop": float(config.robust_elbow_absolute_min_drop),
        "floor_fraction": float(config.robust_elbow_floor_fraction),
        "min_elbow_count": int(min_elbow_count),
    }


def _attach_adaptive_tail_details(df: pd.DataFrame, details: Mapping[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    prefix_map = {
        "base_count": "adaptive_tail_base_count",
        "floor_count": "adaptive_tail_floor_count",
        "elbow_rank": "adaptive_tail_elbow_rank",
        "largest_drop": "adaptive_tail_largest_drop",
        "median_positive_drop": "adaptive_tail_median_positive_drop",
        "absolute_min_drop": "adaptive_tail_absolute_min_drop",
        "drop_ratio_threshold": "adaptive_tail_drop_ratio_threshold",
        "tail_fraction": "adaptive_tail_fraction",
    }
    for src, dst in prefix_map.items():
        out[dst] = details.get(src, np.nan)
    return out

def _append_teacher_pseudolabel_rows(
    *,
    bundle: SimpleNamespace,
    rows: list[dict[str, Any]],
    selected: pd.DataFrame,
    target: str,
    tier: str,
    mode: str,
    weight: pd.Series | float,
    selection_reason: str,
) -> None:
    if selected.empty:
        return
    soft = bundle.teacher_soft
    pred = bundle.teacher_pred.astype(str)
    conf = bundle.teacher_confidence.astype(float)
    knn = bundle.knn_purity.astype(float)
    true = bundle.query.obs["true_label"].astype(str) if "true_label" in bundle.query.obs else pd.Series("", index=bundle.query_index)
    if isinstance(weight, pd.Series):
        weights = weight.reindex(selected.index).astype(float)
    else:
        weights = pd.Series(float(weight), index=selected.index, dtype=float)
    for rank, (cell_id, row) in enumerate(selected.iterrows(), start=1):
        cell_id = str(cell_id)
        true_label = str(true.loc[cell_id]) if cell_id in true.index else ""
        rows.append(
            {
                "cell_id": cell_id,
                "target_label": str(target),
                "selection_tier": str(tier),
                "candidate_mode": str(mode),
                "teacher_pred_label": str(pred.loc[cell_id]) if cell_id in pred.index else "",
                "teacher_confidence": float(conf.loc[cell_id]) if cell_id in conf.index else np.nan,
                "target_posterior": float(soft.loc[cell_id, target]) if cell_id in soft.index and target in soft.columns else np.nan,
                "target_score": float(row.get("target_score", np.nan)),
                "leaf_local_score": float(row.get("leaf_local_score", np.nan)),
                "direct_sibling_margin": float(row.get("direct_sibling_margin", np.nan)),
                "ancestor_max_contradiction": float(row.get("ancestor_max_contradiction", 0.0)),
                "ancestor_max_contradiction_quantile": float(row.get("ancestor_max_contradiction_quantile", 0.0)),
                "ancestor_max_contradiction_node": str(row.get("ancestor_max_contradiction_node", "")),
                "ancestor_soft_penalty": float(row.get("ancestor_soft_penalty", 0.0)),
                "ancestor_vetoed_before_final": bool(row.get("ancestor_vetoed_before_final", False)),
                "knn_purity": float(knn.loc[cell_id]) if cell_id in knn.index else np.nan,
                "pseudo_weight": float(weights.loc[cell_id]) if cell_id in weights.index else 1.0,
                "selection_rank": int(rank),
                "true_label": true_label,
                "is_correct_pseudolabel": bool(true_label == str(target)) if true_label else np.nan,
                "adaptive_reliability_score": float(row.get("adaptive_reliability_score", np.nan)),
                "adaptive_reliability_rank": int(row.get("adaptive_reliability_rank", rank)) if pd.notna(row.get("adaptive_reliability_rank", rank)) else int(rank),
                "adaptive_tail_selection_reason": str(selection_reason),
                "adaptive_candidate_pool_size": int(row.get("adaptive_candidate_pool_size", selected.shape[0])),
                "adaptive_tail_base_count": int(row.get("adaptive_tail_base_count", 0)) if pd.notna(row.get("adaptive_tail_base_count", 0)) else 0,
                "adaptive_tail_floor_count": int(row.get("adaptive_tail_floor_count", 0)) if pd.notna(row.get("adaptive_tail_floor_count", 0)) else 0,
                "adaptive_tail_elbow_rank": int(row.get("adaptive_tail_elbow_rank", 0)) if pd.notna(row.get("adaptive_tail_elbow_rank", 0)) else 0,
                "adaptive_tail_largest_drop": float(row.get("adaptive_tail_largest_drop", np.nan)),
                "adaptive_tail_median_positive_drop": float(row.get("adaptive_tail_median_positive_drop", np.nan)),
                "adaptive_tail_absolute_min_drop": float(row.get("adaptive_tail_absolute_min_drop", np.nan)),
            }
        )


def _adaptive_candidate_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["leaf_sort"] = out["direct_sibling_margin"].where(out["direct_sibling_margin"].notna(), out["leaf_local_score"]).fillna(-1e9)
    return out.sort_values(
        ["adaptive_reliability_score", "leaf_sort", "target_posterior", "teacher_confidence", "knn_purity"],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    )


def _select_adaptive_tail_teacher_pseudolabels(bundle: SimpleNamespace, *, config: UnifiedTeacherPseudoConfig, enable_hidden_rescue: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = bundle.query_index
    soft = bundle.teacher_soft.reindex(index)
    pred = bundle.teacher_pred.reindex(index).astype(str)
    collapsed_pred = bundle.teacher_collapsed_pred.reindex(index).astype(str)
    conf = bundle.teacher_confidence.reindex(index).astype(float)
    knn = bundle.knn_purity.reindex(index).astype(float)
    parent_map = _parent_map(bundle.prior_spec)
    hidden_children = {str(child) for values in bundle.partial_label_spec.values() for child in values} if enable_hidden_rescue else set()
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    veto_frames: list[pd.DataFrame] = []
    evidence_rows: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []

    for target in bundle.label_names:
        if target not in soft.columns:
            summary_rows.append({"target_label": target, "status": "missing_teacher_probability", "n_used_for_training": 0})
            evidence_rows.append({"target_label": target, "status": "missing_teacher_probability", "n_eligible_candidates": 0})
            continue
        parent = parent_map.get(str(target), "")
        target_score, direct_margin, score_available = _direct_sibling_margin(bundle, target)
        finite_score = target_score.replace([np.inf, -np.inf], np.nan).notna()
        contradiction = _ancestor_contradiction(bundle, target)
        mode_kind = "marker" if score_available else "no_marker"
        if score_available:
            base_mask = pred.eq(target) & finite_score
            if int(base_mask.sum()) == 0:
                base_mask = soft[target].astype(float).gt(0) & finite_score
            cand = pd.DataFrame(
                {
                    "target_score": target_score.loc[base_mask].astype(float),
                    "leaf_local_score": target_score.loc[base_mask].astype(float),
                    "direct_sibling_margin": direct_margin.loc[base_mask].astype(float),
                    "target_posterior": soft.loc[base_mask, target].astype(float),
                    "teacher_confidence": conf.loc[base_mask].astype(float),
                    "knn_purity": knn.loc[base_mask].astype(float),
                }
            )
            mode = "pred_same_leaf_local_marker"
            tail_params = _adaptive_tail_selection_params(config, kind="marker")
            max_cap = int(tail_params["max_cap"])
            pseudo_weight = float(config.adaptive_marker_pseudo_weight)
        else:
            base_mask = pred.eq(target)
            if int(base_mask.sum()) == 0:
                base_mask = soft[target].astype(float).gt(0)
            cand = pd.DataFrame(
                {
                    "target_score": np.nan,
                    "leaf_local_score": np.nan,
                    "direct_sibling_margin": np.nan,
                    "target_posterior": soft.loc[base_mask, target].astype(float),
                    "teacher_confidence": conf.loc[base_mask].astype(float),
                    "knn_purity": knn.loc[base_mask].astype(float),
                },
                index=index[base_mask],
            )
            mode = "pred_same_confidence_knn_no_marker"
            tail_params = _adaptive_tail_selection_params(config, kind="no_marker")
            max_cap = int(tail_params["max_cap"])
            pseudo_weight = float(config.adaptive_no_marker_pseudo_weight)
        cand = cand.join(contradiction, how="left").fillna(
            {"ancestor_max_contradiction": 0.0, "ancestor_max_contradiction_quantile": 0.0, "ancestor_max_contradiction_node": ""}
        )
        cand["ancestor_vetoed_before_final"] = cand["ancestor_max_contradiction_quantile"].ge(float(config.hard_contradiction_quantile)) & cand[
            "ancestor_max_contradiction"
        ].gt(0)
        cand["ancestor_soft_penalty"] = np.where(
            cand["ancestor_max_contradiction_quantile"].ge(float(config.soft_contradiction_quantile)) & cand["ancestor_max_contradiction"].gt(0),
            float(config.soft_contradiction_penalty),
            0.0,
        )
        cand = _add_adaptive_reliability(cand, mode=mode_kind)
        wide_n = max(max_cap * int(config.wide_candidate_multiplier), max_cap)
        wide = _adaptive_candidate_sort(cand).head(wide_n).copy()
        if not wide.empty:
            veto = wide.loc[wide["ancestor_vetoed_before_final"].astype(bool)].copy()
            if not veto.empty:
                veto["target_label"] = str(target)
                veto["candidate_mode"] = mode
                veto["would_rank_without_veto"] = np.arange(1, veto.shape[0] + 1)
                veto_frames.append(veto.reset_index(names="cell_id"))
        eligible = _adaptive_candidate_sort(wide.loc[~wide["ancestor_vetoed_before_final"].astype(bool)].copy())
        tail_details = _adaptive_tail_count_details(eligible["adaptive_reliability_score"], **tail_params)
        n_select = int(tail_details["n_select"])
        reason = str(tail_details["selection_reason"])
        selected = _attach_adaptive_tail_details(eligible.head(n_select).copy(), tail_details)
        if not selected.empty:
            selected["adaptive_candidate_pool_size"] = int(eligible.shape[0])
        _append_teacher_pseudolabel_rows(
            bundle=bundle,
            rows=rows,
            selected=selected,
            target=target,
            tier="tier_adaptive_tail_leaf_treeguard",
            mode=mode,
            weight=pseudo_weight,
            selection_reason=reason,
        )
        if not eligible.empty:
            audit = _attach_adaptive_tail_details(eligible.copy(), tail_details).reset_index(names="cell_id")
            audit["target_label"] = str(target)
            audit["candidate_mode"] = mode
            audit["adaptive_selected"] = audit["cell_id"].astype(str).isin(selected.index.astype(str))
            audit["adaptive_tail_final_count"] = int(n_select)
            audit["adaptive_tail_selection_reason"] = reason
            audit_frames.append(audit)

        rescue = pd.DataFrame()
        rescue_eligible = pd.DataFrame()
        rescue_reason = "hidden_rescue_disabled"
        rescue_tail_details = _adaptive_tail_count_details(
            pd.Series(dtype=float),
            **_adaptive_tail_selection_params(config, kind="hidden"),
        )
        if enable_hidden_rescue and score_available and target in hidden_children:
            parent_mass = _node_posterior(soft, parent, prior_spec=bundle.prior_spec, label_names=bundle.label_names) if parent else pd.Series(0.0, index=index)
            child_cond = _child_conditional(soft, target, parent, prior_spec=bundle.prior_spec, label_names=bundle.label_names) if parent else pd.Series(0.0, index=index)
            pool_mask = (
                (parent_mass.ge(float(config.parent_pool_threshold)) | collapsed_pred.eq(parent))
                & child_cond.ge(float(config.child_conditional_threshold))
                & finite_score
                & ~pd.Series(index.isin(selected.index), index=index)
            )
            rescue = pd.DataFrame(
                {
                    "target_score": target_score.loc[pool_mask].astype(float),
                    "leaf_local_score": target_score.loc[pool_mask].astype(float),
                    "direct_sibling_margin": direct_margin.loc[pool_mask].astype(float),
                    "target_posterior": soft.loc[pool_mask, target].astype(float),
                    "teacher_confidence": conf.loc[pool_mask].astype(float),
                    "parent_posterior": parent_mass.loc[pool_mask].astype(float),
                    "child_conditional_posterior": child_cond.loc[pool_mask].astype(float),
                    "knn_purity": knn.loc[pool_mask].astype(float),
                }
            )
            rescue = rescue.join(contradiction, how="left").fillna(
                {"ancestor_max_contradiction": 0.0, "ancestor_max_contradiction_quantile": 0.0, "ancestor_max_contradiction_node": ""}
            )
            rescue["ancestor_vetoed_before_final"] = rescue["ancestor_max_contradiction_quantile"].ge(float(config.hard_contradiction_quantile)) & rescue[
                "ancestor_max_contradiction"
            ].gt(0)
            rescue["ancestor_soft_penalty"] = np.where(
                rescue["ancestor_max_contradiction_quantile"].ge(float(config.soft_contradiction_quantile)) & rescue["ancestor_max_contradiction"].gt(0),
                float(config.soft_contradiction_penalty),
                0.0,
            )
            rescue = _add_adaptive_reliability(rescue, mode="hidden")
            rescue_tail_params = _adaptive_tail_selection_params(config, kind="hidden")
            rescue_wide_n = max(int(rescue_tail_params["max_cap"]) * int(config.wide_candidate_multiplier), int(rescue_tail_params["max_cap"]))
            rescue_wide = _adaptive_candidate_sort(rescue).head(rescue_wide_n).copy()
            rescue_eligible = _adaptive_candidate_sort(rescue_wide.loc[~rescue_wide["ancestor_vetoed_before_final"].astype(bool)].copy())
            rescue_tail_details = _adaptive_tail_count_details(rescue_eligible["adaptive_reliability_score"], **rescue_tail_params)
            n_rescue = int(rescue_tail_details["n_select"])
            rescue_reason = str(rescue_tail_details["selection_reason"])
            rescue = _attach_adaptive_tail_details(rescue_eligible.head(n_rescue).copy(), rescue_tail_details)
            if not rescue.empty:
                rescue["adaptive_candidate_pool_size"] = int(rescue_eligible.shape[0])
            _append_teacher_pseudolabel_rows(
                bundle=bundle,
                rows=rows,
                selected=rescue,
                target=target,
                tier="tier_adaptive_tail_hidden_parent_rescue",
                mode="parent_pool_leaf_marker_treeguard",
                weight=float(config.adaptive_hidden_pseudo_weight),
                selection_reason=rescue_reason,
            )
            if not rescue_eligible.empty:
                audit = _attach_adaptive_tail_details(rescue_eligible.copy(), rescue_tail_details).reset_index(names="cell_id")
                audit["target_label"] = str(target)
                audit["candidate_mode"] = "parent_pool_leaf_marker_treeguard"
                audit["adaptive_selected"] = audit["cell_id"].astype(str).isin(rescue.index.astype(str))
                audit["adaptive_tail_final_count"] = int(n_rescue)
                audit["adaptive_tail_selection_reason"] = rescue_reason
                audit_frames.append(audit)

        selected_rel = pd.concat([selected.get("adaptive_reliability_score", pd.Series(dtype=float)), rescue.get("adaptive_reliability_score", pd.Series(dtype=float))], axis=0)
        mean_rel = float(selected_rel.mean()) if not selected_rel.empty else np.nan
        if not score_available:
            class_strength = "no_marker"
            effective_target = int(config.adaptive_no_marker_effective_target)
            max_repeats = int(config.adaptive_no_marker_max_repeats_per_cell)
        elif int(rescue.shape[0]) > 0 and int(rescue.shape[0]) >= int(selected.shape[0]):
            class_strength = "hidden_rescue"
            effective_target = int(config.adaptive_weak_effective_target)
            max_repeats = int(config.adaptive_hidden_max_repeats_per_cell)
        elif pd.notna(mean_rel) and mean_rel >= float(config.adaptive_strong_reliability_threshold):
            class_strength = "strong_marker"
            effective_target = int(config.adaptive_strong_effective_target)
            max_repeats = int(config.adaptive_marker_max_repeats_per_cell)
        elif pd.notna(mean_rel) and mean_rel >= float(config.adaptive_medium_reliability_threshold):
            class_strength = "medium_marker"
            effective_target = int(config.adaptive_medium_effective_target)
            max_repeats = int(config.adaptive_marker_max_repeats_per_cell)
        else:
            class_strength = "weak_marker"
            effective_target = int(config.adaptive_weak_effective_target)
            max_repeats = int(config.adaptive_hidden_max_repeats_per_cell)
        evidence_rows.append(
            {
                "target_label": str(target),
                "parent_node": str(parent),
                "score_available": bool(score_available),
                "n_base_candidates": int(cand.shape[0]),
                "n_eligible_candidates": int(eligible.shape[0]),
                "n_selected_before_conflict": int(selected.shape[0] + rescue.shape[0]),
                "n_hidden_rescue_candidates": int(rescue_eligible.shape[0]),
                "n_hidden_rescue_selected": int(rescue.shape[0]),
                "hidden_rescue_enabled": bool(enable_hidden_rescue),
                "adaptive_class_strength": class_strength,
                "adaptive_mean_reliability": mean_rel,
                "adaptive_effective_target": int(effective_target),
                "adaptive_max_repeats_per_cell": int(max_repeats),
                "adaptive_tail_selection_reason": reason,
                "adaptive_hidden_selection_reason": rescue_reason,
                "adaptive_tail_base_count": int(tail_details.get("base_count", 0)),
                "adaptive_tail_floor_count": int(tail_details.get("floor_count", 0)),
                "adaptive_tail_elbow_rank": int(tail_details.get("elbow_rank", 0)),
                "adaptive_tail_largest_drop": float(tail_details.get("largest_drop", np.nan)),
                "adaptive_tail_median_positive_drop": float(tail_details.get("median_positive_drop", np.nan)),
                "adaptive_tail_absolute_min_drop": float(tail_details.get("absolute_min_drop", np.nan)),
                "adaptive_tail_fraction": float(tail_details.get("tail_fraction", np.nan)),
                "adaptive_hidden_base_count": int(rescue_tail_details.get("base_count", 0)),
                "adaptive_hidden_floor_count": int(rescue_tail_details.get("floor_count", 0)),
                "adaptive_hidden_elbow_rank": int(rescue_tail_details.get("elbow_rank", 0)),
                "adaptive_hidden_largest_drop": float(rescue_tail_details.get("largest_drop", np.nan)),
                "adaptive_hidden_median_positive_drop": float(rescue_tail_details.get("median_positive_drop", np.nan)),
                "status": "ok" if int(selected.shape[0] + rescue.shape[0]) else "zero_selected_after_adaptive_tail",
            }
        )
        summary_rows.append(
            {
                "target_label": str(target),
                "score_available": bool(score_available),
                "n_pred_same": int(pred.eq(target).sum()),
                "n_base_candidates": int(cand.shape[0]),
                "n_eligible_candidates": int(eligible.shape[0]),
                "n_tier_adaptive_tail_leaf_treeguard": int(selected.shape[0]),
                "n_tier_adaptive_tail_hidden_parent_rescue": int(rescue.shape[0]),
                "hidden_rescue_enabled": bool(enable_hidden_rescue),
                "adaptive_class_strength": class_strength,
                "adaptive_mean_reliability": mean_rel,
                "adaptive_effective_target": int(effective_target),
                "adaptive_max_repeats_per_cell": int(max_repeats),
                "adaptive_tail_selection_reason": reason,
                "adaptive_tail_base_count": int(tail_details.get("base_count", 0)),
                "adaptive_tail_floor_count": int(tail_details.get("floor_count", 0)),
                "adaptive_tail_elbow_rank": int(tail_details.get("elbow_rank", 0)),
                "adaptive_tail_largest_drop": float(tail_details.get("largest_drop", np.nan)),
                "adaptive_tail_median_positive_drop": float(tail_details.get("median_positive_drop", np.nan)),
                "adaptive_tail_absolute_min_drop": float(tail_details.get("absolute_min_drop", np.nan)),
                "status": "ok" if int(selected.shape[0] + rescue.shape[0]) else "zero_selected_after_adaptive_tail",
            }
        )

    pseudo_df, by_class = _finalize_pseudolabel_rows(rows, summary_rows)
    if not by_class.empty:
        for col in ["adaptive_effective_target", "adaptive_max_repeats_per_cell"]:
            if col in by_class.columns:
                by_class[col] = pd.to_numeric(by_class[col], errors="coerce")
    vetoed = pd.concat(veto_frames, ignore_index=True, sort=False) if veto_frames else pd.DataFrame()
    evidence = pd.DataFrame(evidence_rows).sort_values("target_label", kind="mergesort").reset_index(drop=True)
    candidate_audit = pd.concat(audit_frames, ignore_index=True, sort=False) if audit_frames else pd.DataFrame()
    missing = evidence.loc[evidence.get("status", pd.Series(dtype=str)).astype(str).ne("ok")].copy() if not evidence.empty else pd.DataFrame()
    reliability = by_class.copy()
    pseudo_df.attrs["adaptive_candidate_audit"] = candidate_audit
    pseudo_df.attrs["adaptive_missing_or_low_coverage"] = missing
    pseudo_df.attrs["adaptive_reliability_by_class"] = reliability
    return pseudo_df, by_class, vetoed, evidence

def select_unified_bottomup_teacher_pseudolabels(
    *,
    query_obs: pd.DataFrame,
    soft: pd.DataFrame,
    protein_arcsinh: pd.DataFrame,
    prior_spec: Mapping[str, Any],
    label_categories: Sequence[str],
    partial_label_spec: Mapping[str, Sequence[str]] | None = None,
    leaf_marker_specs: Mapping[str, Mapping[str, Any]] | None = None,
    label_col: str = "true_label",
    teacher_latent: np.ndarray | None = None,
    enable_hidden_rescue: bool = False,
    config: UnifiedTeacherPseudoConfig | None = None,
    score_mode: str = GENERIC_FLAT_SCORE_MODE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select query pseudo-labels used between teacher refinement rounds.

    These pseudo-labels update the teacher in round1/round2.  Final student
    anchors are selected later from the final teacher state.
    """
    cfg = config or UnifiedTeacherPseudoConfig()
    query_index = pd.Index(query_obs.index.astype(str))
    soft = soft.reindex(query_index).loc[:, [str(x) for x in label_categories]].astype(float)
    protein_arcsinh = protein_arcsinh.reindex(query_index)
    target_scores, score_available, score_availability = _build_teacher_target_scores(
        protein_arcsinh=protein_arcsinh,
        labels=label_categories,
        prior_spec=prior_spec,
        leaf_marker_specs=leaf_marker_specs,
        score_mode=score_mode,
    )
    pred = soft.idxmax(axis=1).astype(str)
    collapsed_pred = _collapsed_pred_for_partial_nodes(
        soft,
        prior_spec=prior_spec,
        labels=label_categories,
        partial_label_spec=partial_label_spec if enable_hidden_rescue else None,
    )
    query = ad.AnnData(
        X=np.zeros((len(query_index), 0), dtype=np.float32),
        obs=query_obs.reindex(query_index).copy(),
    )
    if label_col in query.obs and "true_label" not in query.obs:
        query.obs["true_label"] = query.obs[label_col].astype(str)
    bundle = SimpleNamespace(
        query=query,
        query_index=query_index,
        label_names=[str(x) for x in label_categories],
        label_to_idx={str(label): i for i, label in enumerate(label_categories)},
        teacher_soft=soft,
        teacher_pred=pred,
        teacher_collapsed_pred=collapsed_pred,
        teacher_confidence=soft.max(axis=1).astype(float),
        prior_spec=dict(prior_spec),
        partial_label_spec={str(k): tuple(str(x) for x in v) for k, v in (partial_label_spec or {}).items()},
        leaf_marker_specs=dict(leaf_marker_specs or {}),
        target_scores=target_scores,
        target_score_available=score_available,
        knn_purity=_knn_purity_from_latent(teacher_latent, pred, k=15),
        protein_arcsinh=protein_arcsinh,
    )
    if not cfg.is_adaptive_tail:
        raise ValueError("ANCHOR currently supports teacher pseudo-selection mode 'adaptive_tail_robust_elbow' only.")
    pseudo_df, by_class, vetoed, evidence = _select_adaptive_tail_teacher_pseudolabels(
        bundle,
        config=cfg,
        enable_hidden_rescue=enable_hidden_rescue,
    )
    by_class = by_class.copy()
    if "n_selected" not in by_class.columns:
        by_class["n_selected"] = by_class.get("n_used_for_training", 0)
    if "n_used_for_training" not in by_class.columns:
        by_class["n_used_for_training"] = by_class["n_selected"]
    by_class["n_selected"] = by_class["n_used_for_training"].astype(int)
    pseudo_df.attrs["adaptive_reliability_by_class"] = by_class.copy()
    if not evidence.empty:
        missing = evidence.loc[evidence.get("status", pd.Series(dtype=str)).astype(str).ne("ok")].copy()
        pseudo_df.attrs["adaptive_missing_or_low_coverage"] = missing
    return pseudo_df, by_class, vetoed, evidence, score_availability


def _teacher_overall(pseudo_df: pd.DataFrame, *, method: str, strategy: str) -> pd.DataFrame:
    if pseudo_df.empty:
        return pd.DataFrame(
            [{"method": method, "strategy": strategy, "n_selected": 0, "n_used_for_training": 0, "pseudo_precision": np.nan}]
        )
    used = pseudo_df.loc[pseudo_df["used_for_training"].astype(bool)].copy()
    return pd.DataFrame(
        [
            {
                "method": method,
                "strategy": strategy,
                "n_selected": int(pseudo_df.shape[0]),
                "n_used_for_training": int(used.shape[0]),
                "n_labels_with_selected": int(used["target_label"].nunique()) if not used.empty else 0,
                "pseudo_precision": float(used["is_correct_pseudolabel"].mean()) if not used.empty else np.nan,
            }
        ]
    )


def _pair_bundle_from_bottomup(pseudo_df: pd.DataFrame, label_categories: Sequence[str]) -> PairQueryPseudoLabelBundle:
    if pseudo_df.empty:
        empty = pd.DataFrame()
        return PairQueryPseudoLabelBundle(cell_level=empty, summary=empty, conflicts=empty, counts_by_label=empty)
    label_to_idx = {str(label): idx for idx, label in enumerate(label_categories)}
    cell_level = pseudo_df.copy()
    cell_level["obs_name"] = cell_level["cell_id"].astype(str)
    cell_level["pseudo_label"] = cell_level["target_label"].astype(str)
    cell_level["pseudo_target_index"] = cell_level["pseudo_label"].map(label_to_idx).fillna(-1).astype(int)
    cell_level["pair_key"] = cell_level.get("candidate_mode", "bottomup_treeguard").astype(str)
    cell_level["strategy"] = cell_level.get("selection_tier", "bottomup_treeguard").astype(str)
    cell_level["method"] = "unified_bottomup_treeguard_teacher"
    cell_level["prediction_confidence"] = cell_level.get("teacher_confidence", cell_level.get("target_posterior", np.nan)).astype(float)
    if "true_label" not in cell_level and "true_label" in cell_level:
        cell_level["true_label"] = cell_level["true_label"].astype(str)
    grouped = cell_level.groupby(["pair_key", "pseudo_label", "strategy"], dropna=False)
    summary = (
        grouped.agg(
            n_selected=("obs_name", "nunique"),
            n_used_for_training=("used_for_training", "sum"),
            n_conflicts=("is_conflict", "sum"),
            n_correct_pseudolabel=("is_correct_pseudolabel", "sum"),
        )
        .reset_index()
    )
    summary["pseudo_label_precision_all_selected"] = np.where(
        summary["n_selected"].gt(0),
        summary["n_correct_pseudolabel"] / summary["n_selected"],
        np.nan,
    )
    counts = (
        cell_level.loc[cell_level["used_for_training"].astype(bool)]
        .groupby("pseudo_label")["obs_name"]
        .nunique()
        .rename("n_used_for_training")
        .reset_index()
    )
    conflicts = cell_level.loc[cell_level["is_conflict"].astype(bool)].copy()
    return PairQueryPseudoLabelBundle(
        cell_level=cell_level.reset_index(drop=True),
        summary=summary.reset_index(drop=True),
        conflicts=conflicts.reset_index(drop=True),
        counts_by_label=counts.reset_index(drop=True),
    )


def bottomup_as_flat_selection(
    *,
    pseudo_df: pd.DataFrame,
    by_class: pd.DataFrame,
    score_availability: pd.DataFrame,
    label_categories: Sequence[str],
    method: str = "unified_bottomup_treeguard_teacher",
    strategy: str = "bottomup_treeguard_full_mode",
) -> FlatLeafPseudoLabelSelection:
    cell_level = pseudo_df.copy()
    if "true_label" not in cell_level and "true_label" in cell_level:
        cell_level["true_label"] = cell_level["true_label"].astype(str)
    if "pred_label" not in cell_level and "teacher_pred_label" in cell_level:
        cell_level["pred_label"] = cell_level["teacher_pred_label"].astype(str)
    cell_level["method"] = method
    cell_level["strategy"] = strategy
    overall = _teacher_overall(cell_level, method=method, strategy=strategy)
    return FlatLeafPseudoLabelSelection(
        cell_level=cell_level.reset_index(drop=True),
        by_class=by_class.reset_index(drop=True),
        overall=overall,
        score_availability=score_availability.reset_index(drop=True),
        pair_bundle=_pair_bundle_from_bottomup(cell_level, label_categories),
    )


def bottomup_as_partial_selection(
    *,
    pseudo_df: pd.DataFrame,
    by_class: pd.DataFrame,
    score_availability: pd.DataFrame,
    evidence: pd.DataFrame,
    method: str = "unified_bottomup_treeguard_teacher",
    strategy: str = "bottomup_treeguard_partial_label_mode",
) -> PartialFlatLeafPseudoLabelSelection:
    cell_level = pseudo_df.copy()
    if "true_label" not in cell_level and "true_label" in cell_level:
        cell_level["true_label"] = cell_level["true_label"].astype(str)
    if "pred_label" not in cell_level and "teacher_pred_label" in cell_level:
        cell_level["pred_label"] = cell_level["teacher_pred_label"].astype(str)
    cell_level["method"] = method
    cell_level["strategy"] = strategy
    overall = _teacher_overall(cell_level, method=method, strategy=strategy)
    hidden_summary = evidence.copy()
    if "n_hidden_rescue_selected" in hidden_summary.columns:
        hidden_summary = hidden_summary.loc[hidden_summary["n_hidden_rescue_selected"].astype(int).gt(0)].copy()
    return PartialFlatLeafPseudoLabelSelection(
        cell_level=cell_level.reset_index(drop=True),
        by_class=by_class.reset_index(drop=True),
        overall=overall,
        score_availability=score_availability.reset_index(drop=True),
        hidden_parent_summary=hidden_summary.reset_index(drop=True),
    )


def write_teacher_bottomup_selection_outputs(
    *,
    results_dir: Path,
    pseudo_df: pd.DataFrame,
    by_class: pd.DataFrame,
    vetoed: pd.DataFrame,
    evidence: pd.DataFrame,
    score_availability: pd.DataFrame,
    prefix: str = "teacher_bottomup_treeguard",
) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    pseudo_df.to_csv(results_dir / f"{prefix}_pseudolabel_cell_level.csv", index=False)
    by_class.to_csv(results_dir / f"{prefix}_pseudolabel_by_class.csv", index=False)
    vetoed.to_csv(results_dir / f"{prefix}_vetoed_candidates.csv", index=False)
    evidence.to_csv(results_dir / f"{prefix}_anchor_evidence_summary.csv", index=False)
    score_availability.to_csv(results_dir / f"{prefix}_score_availability.csv", index=False)
    candidate_audit = pseudo_df.attrs.get("adaptive_candidate_audit") if hasattr(pseudo_df, "attrs") else None
    if isinstance(candidate_audit, pd.DataFrame):
        candidate_audit.to_csv(results_dir / f"{prefix}_candidate_audit.csv", index=False)
    missing = pseudo_df.attrs.get("adaptive_missing_or_low_coverage") if hasattr(pseudo_df, "attrs") else None
    if isinstance(missing, pd.DataFrame):
        missing.to_csv(results_dir / f"{prefix}_missing_or_low_coverage.csv", index=False)
    reliability = pseudo_df.attrs.get("adaptive_reliability_by_class") if hasattr(pseudo_df, "attrs") else None
    if isinstance(reliability, pd.DataFrame):
        reliability.to_csv(results_dir / f"{prefix}_reliability_by_class.csv", index=False)
    _teacher_overall(pseudo_df, method="unified_bottomup_treeguard_teacher", strategy=prefix).to_csv(
        results_dir / f"{prefix}_pseudolabel_overall.csv",
        index=False,
    )



def build_adaptive_tail_repeat_table(
    by_class: pd.DataFrame,
    *,
    label_categories: Sequence[str],
    config: UnifiedTeacherPseudoConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_label = by_class.set_index(by_class["target_label"].astype(str)) if not by_class.empty and "target_label" in by_class else pd.DataFrame()
    for label_index, label in enumerate([str(label) for label in label_categories]):
        n_selected = 0
        target_effective = int(config.adaptive_weak_effective_target)
        max_repeats = int(config.adaptive_hidden_max_repeats_per_cell)
        strength = "missing"
        mean_rel = np.nan
        if not by_label.empty and label in by_label.index:
            row = by_label.loc[label]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            value = row.get("n_selected", row.get("n_used_for_training", 0))
            n_selected = int(value) if pd.notna(value) else 0
            if pd.notna(row.get("adaptive_effective_target", np.nan)):
                target_effective = int(row.get("adaptive_effective_target"))
            if pd.notna(row.get("adaptive_max_repeats_per_cell", np.nan)):
                max_repeats = int(row.get("adaptive_max_repeats_per_cell"))
            strength = str(row.get("adaptive_class_strength", strength))
            mean_rel = row.get("adaptive_mean_reliability", np.nan)
        repeat = 0
        if 0 < n_selected < int(target_effective):
            repeat = int(np.ceil(float(target_effective) / float(n_selected))) - 1
            repeat = max(0, min(int(max_repeats), repeat))
        rows.append(
            {
                "label_index": int(label_index),
                "target_label": label,
                "n_selected": int(n_selected),
                "adaptive_class_strength": strength,
                "adaptive_mean_reliability": mean_rel,
                "adaptive_effective_target": int(target_effective),
                "adaptive_max_repeats_per_cell": int(max_repeats),
                "aug_repeats_per_cell": int(repeat),
                "n_augmented_views": int(n_selected * repeat),
                "effective_training_count": int(n_selected * (1 + repeat)),
            }
        )
    return pd.DataFrame(rows)


def build_teacher_repeat_table(
    by_class: pd.DataFrame,
    *,
    label_categories: Sequence[str],
    config: UnifiedTeacherPseudoConfig,
) -> pd.DataFrame:
    if not config.is_adaptive_tail:
        raise ValueError("ANCHOR currently supports teacher pseudo-selection mode 'adaptive_tail_robust_elbow' only.")
    return build_adaptive_tail_repeat_table(by_class, label_categories=label_categories, config=config)

__all__ = [
    "UnifiedTeacherPseudoConfig",
    "bottomup_as_flat_selection",
    "bottomup_as_partial_selection",
    "build_adaptive_tail_repeat_table",
    "build_teacher_repeat_table",
    "select_unified_bottomup_teacher_pseudolabels",
    "write_teacher_bottomup_selection_outputs",
]
