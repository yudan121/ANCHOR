"""Final student anchor selection.

The selector ranks query cells using agreement between teacher predictions,
protein marker evidence and local neighborhood purity.  Selected anchors carry
reliability weights and supervise the final student model through weighted
pseudo-label cross-entropy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .bundle import (
    StudentDataBundle,
    _build_target_scores,
    _child_conditional,
    _descendant_leaves,
    _detect_teacher_collapsed_pred,
    _direct_parent_map,
    _finalize_pseudolabel_rows,
    _knn_purity,
    _node_posterior,
)
from .losses import _build_branch_rank_specs

SAFETY_GUARD_POLICY_VERSION = "rare_anchor_loss_guard_v1"
SAFETY_GUARD_MAX_RARE_ANCHOR_LOSS_THRESHOLD = 0.30
SAFETY_GUARD_TOP5_MEAN_RARE_ANCHOR_LOSS_THRESHOLD = 0.15


DEFAULT_BOTTOMUP_CONFIG: dict[str, Any] = {
    "posterior_threshold": 0.95,
    "max_marker_pseudo_per_class": 20,
    "max_no_marker_pseudo_per_class": 10,
    "max_hidden_rescue_per_child": 10,
    "enable_hidden_rescue": True,
    "wide_candidate_multiplier": 8,
    "parent_pool_threshold": 0.20,
    "child_conditional_threshold": 0.05,
    "hard_contradiction_quantile": 0.90,
    "soft_contradiction_quantile": 0.75,
    "soft_contradiction_penalty": 0.25,
    "rank_loss_weight": 0.20,
    "rank_loss_use_global_child_scores": False,
    "auto_tree_rank_specs": True,
    "rank_min_locality_weight": 0.05,
    "prototype_logit_weight": 1.0,
    "prototype_ce_lambda": 0.5,
    "graph_consistency_lambda": 0.5,
    "student_loss_weight_overrides": {},
    "graph_refresh_every": 5,
    "graph_k": 15,
    "pseudo_selection_mode": "adaptive_tail_robust_elbow",
    "adaptive_marker_tail_fraction": 0.25,
    "adaptive_no_marker_tail_fraction": 0.10,
    "adaptive_hidden_tail_fraction": 0.10,
    "adaptive_marker_max_cap": 50,
    "adaptive_no_marker_max_cap": 10,
    "adaptive_hidden_max_cap_per_child": 10,
    "adaptive_marker_min_select_if_any": 3,
    "adaptive_no_marker_min_select_if_any": 0,
    "adaptive_hidden_min_select_if_any": 0,
    "adaptive_elbow_drop_ratio": 5.0,
    "adaptive_elbow_absolute_min_drop": 0.03,
    "adaptive_elbow_floor_fraction": 0.40,
    "adaptive_marker_min_elbow_count": 5,
    "adaptive_no_marker_min_elbow_count": 2,
    "adaptive_hidden_min_elbow_count": 1,
    "adaptive_marker_pseudo_weight": 1.0,
    "adaptive_hidden_pseudo_weight": 0.5,
    "adaptive_no_marker_pseudo_weight": 0.25,
    "adaptive_no_marker_min_reliability": 0.75,
    "adaptive_no_marker_soft_floor_posterior": 0.95,
    "adaptive_no_marker_soft_floor_knn": 0.50,
    "adaptive_no_marker_small_pool_threshold": 10,
    "adaptive_no_marker_small_pool_low_knn": 0.50,
    "adaptive_no_marker_small_pool_penalty": 0.50,
    "adaptive_anchor_floor_min_per_class": 0,
}


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


def _path_to_leaf(prior_spec: Mapping[str, Any], leaf: str) -> list[str]:
    parents = _parent_map(prior_spec)
    path = [str(leaf)]
    current = str(leaf)
    seen = {current}
    while current in parents:
        current = str(parents[current])
        if current in seen:
            break
        path.append(current)
        seen.add(current)
    return list(reversed(path))


def _child_on_path(prior_spec: Mapping[str, Any], node: str, target_leaf: str, label_names: list[str]) -> str | None:
    for child in _children_map(prior_spec).get(str(node), []):
        if str(child) == str(target_leaf):
            return str(child)
        if str(target_leaf) in set(_descendant_leaves(child, prior_spec, label_names)):
            return str(child)
    return None


def _marker_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [str(x) for x in value.keys()]
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def _score_from_branch_spec(protein: pd.DataFrame, class_spec: Mapping[str, Any]) -> pd.Series | None:
    pos = [m for m in _marker_list(class_spec.get("positive", [])) if m in protein.columns]
    neg = [m for m in _marker_list(class_spec.get("negative", [])) if m in protein.columns]
    if not pos and not neg:
        return None
    score = pd.Series(0.0, index=protein.index, dtype=float)
    if pos:
        score = score + protein.loc[:, pos].astype(float).mean(axis=1)
    if neg:
        score = score - protein.loc[:, neg].astype(float).mean(axis=1)
    return score.astype(float)


def _score_for_child(bundle: StudentDataBundle, node: str, child: str) -> pd.Series | None:
    if str(child) in bundle.target_scores:
        score = bundle.target_scores[str(child)].reindex(bundle.query_index).astype(float)
        if score.replace([np.inf, -np.inf], np.nan).notna().any():
            return score
    class_spec = (
        bundle.prior_spec.get("branch_teacher_specs", {})
        .get(str(node), {})
        .get("classes", {})
        .get(str(child), {})
    )
    score = _score_from_branch_spec(bundle.protein_arcsinh.reindex(bundle.query_index), class_spec)
    return score.reindex(bundle.query_index).astype(float) if score is not None else None


def _child_score_any(bundle: StudentDataBundle, child: str) -> pd.Series | None:
    parent = _parent_map(bundle.prior_spec).get(str(child), "")
    if parent:
        return _score_for_child(bundle, parent, str(child))
    if str(child) in bundle.target_scores:
        return bundle.target_scores[str(child)].reindex(bundle.query_index).astype(float)
    return None


def _direct_sibling_margin(bundle: StudentDataBundle, target: str) -> tuple[pd.Series, pd.Series, bool]:
    index = bundle.query_index
    parents = _parent_map(bundle.prior_spec)
    parent = parents.get(str(target), "")
    target_score = _child_score_any(bundle, str(target))
    if target_score is None:
        target_score = pd.Series(np.nan, index=index, dtype=float)
    sibling_scores: list[pd.Series] = []
    if parent:
        for sibling in _children_map(bundle.prior_spec).get(parent, []):
            if str(sibling) == str(target):
                continue
            score = _score_for_child(bundle, parent, str(sibling))
            if score is not None:
                sibling_scores.append(score.reindex(index).astype(float))
    if sibling_scores:
        best_sibling = pd.concat(sibling_scores, axis=1).max(axis=1)
        margin = target_score - best_sibling
    else:
        margin = pd.Series(np.nan, index=index, dtype=float)
    available = bool(target_score.replace([np.inf, -np.inf], np.nan).notna().any())
    return target_score.astype(float), margin.astype(float), available


def _build_generic_branch_rank_specs(
    bundle: StudentDataBundle,
    *,
    branches: list[str] | tuple[str, ...] | None = None,
    use_teacher_parent_pool: bool = False,
    min_locality_weight: float = 0.05,
) -> list[dict[str, Any]]:
    branch_filter = {str(x) for x in branches} if branches is not None else None
    specs: list[dict[str, Any]] = []
    for branch, branch_spec in bundle.prior_spec.get("branch_teacher_specs", {}).items():
        branch = str(branch)
        if branch_filter is not None and branch not in branch_filter:
            continue
        parent_desc = [
            bundle.label_to_idx[x]
            for x in _descendant_leaves(branch, bundle.prior_spec, bundle.label_names)
            if x in bundle.label_to_idx
        ]
        if not parent_desc:
            continue
        locality_weight = float(np.clip(2.0 / max(float(len(parent_desc)), 1.0), float(min_locality_weight), 1.0))
        children: list[dict[str, Any]] = []
        for child in branch_spec.get("children", []):
            child = str(child)
            desc = [
                bundle.label_to_idx[x]
                for x in _descendant_leaves(child, bundle.prior_spec, bundle.label_names)
                if x in bundle.label_to_idx
            ]
            if not desc:
                continue
            score = _score_for_child(bundle, branch, child)
            if score is None:
                continue
            score = score.reindex(bundle.query_index).replace([np.inf, -np.inf], np.nan).astype(float)
            if not score.notna().any():
                continue
            label_idx = int(bundle.label_to_idx[child]) if child in bundle.label_to_idx else int(desc[0])
            children.append(
                {
                    "child": child,
                    "desc_indices": desc,
                    "label_idx": label_idx,
                    "score_values": score.fillna(-1e6).to_numpy(dtype=np.float32),
                }
            )
        if len(children) >= 2:
            specs.append(
                {
                    "branch": branch,
                    "parent_desc_indices": parent_desc,
                    "children": children,
                    "use_teacher_parent_pool": bool(use_teacher_parent_pool),
                    "rank_weight": locality_weight,
                    "n_descendant_leaves": int(len(parent_desc)),
                }
            )
    return specs


def _rank_specs_summary(rank_specs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in rank_specs:
        rows.append(
            {
                "branch": str(spec.get("branch", "")),
                "n_descendant_leaves": int(spec.get("n_descendant_leaves", len(spec.get("parent_desc_indices", [])))),
                "n_rank_children": int(len(spec.get("children", []))),
                "rank_weight": float(spec.get("rank_weight", 1.0)),
                "use_teacher_parent_pool": bool(spec.get("use_teacher_parent_pool", False)),
                "children": ";".join(str(child.get("child", "")) for child in spec.get("children", [])),
            }
        )
    columns = ["branch", "n_descendant_leaves", "n_rank_children", "rank_weight", "use_teacher_parent_pool", "children"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(["rank_weight", "branch"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _class_balanced_sampling_summary(
    pseudo_df: pd.DataFrame,
    pseudo_by_class: pd.DataFrame,
    *,
    samples_per_label: int,
    sample_by_weight: bool,
) -> pd.DataFrame:
    sampling_summary = pseudo_by_class.copy()
    if "n_used_for_training" not in sampling_summary.columns:
        sampling_summary["n_used_for_training"] = 0
    sampling_summary["class_balanced_samples_per_step"] = np.where(
        sampling_summary["n_used_for_training"].astype(int).gt(0),
        int(samples_per_label),
        0,
    )
    sampling_summary["class_balanced_sampling_with_replacement"] = sampling_summary["n_used_for_training"].astype(int).between(1, int(samples_per_label) - 1)
    sampling_summary["class_balanced_sample_by_weight"] = bool(sample_by_weight)
    if not pseudo_df.empty and "target_label" in pseudo_df.columns and "pseudo_weight" in pseudo_df.columns:
        weights = pseudo_df.loc[pseudo_df["used_for_training"].astype(bool)].groupby("target_label")["pseudo_weight"].mean()
        sampling_summary["mean_pseudo_weight"] = sampling_summary["target_label"].map(weights)
    else:
        sampling_summary["mean_pseudo_weight"] = np.nan
    return sampling_summary


def _write_student_safety_guard_report(
    *,
    bundle: StudentDataBundle,
    pred: pd.Series,
    pseudo_by_class: pd.DataFrame,
    results_dir: Path,
) -> dict[str, Any]:
    teacher_counts = bundle.teacher_pred.reindex(bundle.query_index).astype(str).value_counts()
    student_counts = pred.astype(str).value_counts()
    if not pseudo_by_class.empty and "target_label" in pseudo_by_class.columns and "n_used_for_training" in pseudo_by_class.columns:
        anchor_counts = pseudo_by_class.set_index("target_label")["n_used_for_training"].astype(float)
    else:
        anchor_counts = pd.Series(dtype=float)
    rows: list[dict[str, Any]] = []
    for label in bundle.label_names:
        teacher_count = int(teacher_counts.get(label, 0))
        student_count = int(student_counts.get(label, 0))
        anchor_count = float(anchor_counts.get(label, 0.0))
        if teacher_count > 0:
            support_drop_frac = max(0.0, float(teacher_count - student_count) / float(teacher_count))
            rare_anchor_loss = support_drop_frac / float(np.sqrt((teacher_count + 1.0) * (anchor_count + 1.0)))
        else:
            support_drop_frac = 0.0
            rare_anchor_loss = 0.0
        rows.append(
            {
                "target_label": str(label),
                "teacher_pred_count": int(teacher_count),
                "student_pred_count": int(student_count),
                "anchor_count": float(anchor_count),
                "support_drop_frac": float(support_drop_frac),
                "rare_anchor_loss": float(rare_anchor_loss),
                "fragile_zeroed": bool(anchor_count <= 2 and teacher_count > 0 and student_count == 0),
            }
        )
    by_class = pd.DataFrame(rows).sort_values("rare_anchor_loss", ascending=False, kind="mergesort").reset_index(drop=True)
    top5_mean = float(by_class["rare_anchor_loss"].head(5).mean()) if not by_class.empty else 0.0
    max_loss = float(by_class["rare_anchor_loss"].max()) if not by_class.empty else 0.0
    guard_trigger = bool(
        max_loss >= SAFETY_GUARD_MAX_RARE_ANCHOR_LOSS_THRESHOLD
        and top5_mean >= SAFETY_GUARD_TOP5_MEAN_RARE_ANCHOR_LOSS_THRESHOLD
    )
    summary = pd.DataFrame(
        [
            {
                "safety_guard_policy_version": SAFETY_GUARD_POLICY_VERSION,
                "max_rare_anchor_loss": max_loss,
                "top5_mean_rare_anchor_loss": top5_mean,
                "guard_trigger": guard_trigger,
                "max_loss_threshold": SAFETY_GUARD_MAX_RARE_ANCHOR_LOSS_THRESHOLD,
                "top5_mean_threshold": SAFETY_GUARD_TOP5_MEAN_RARE_ANCHOR_LOSS_THRESHOLD,
                "n_fragile_zeroed": int(by_class["fragile_zeroed"].sum()) if not by_class.empty else 0,
            }
        ]
    )
    by_class.to_csv(results_dir / "student_safety_guard_by_class.csv", index=False)
    summary.to_csv(results_dir / "student_safety_guard_report.csv", index=False)
    return {"by_class": by_class, "summary": summary, "guard_trigger": guard_trigger}


def _ancestor_contradiction(bundle: StudentDataBundle, target: str) -> pd.DataFrame:
    index = bundle.query_index
    rows = pd.DataFrame(index=index)
    path = _path_to_leaf(bundle.prior_spec, str(target))
    if len(path) <= 2:
        rows["ancestor_max_contradiction"] = 0.0
        rows["ancestor_max_contradiction_quantile"] = 0.0
        rows["ancestor_max_contradiction_node"] = ""
        return rows
    # Exclude the target itself and the direct parent; direct-parent marker is the main leaf-local evidence.
    ancestors = path[:-2]
    contradiction_values: list[pd.Series] = []
    contradiction_q: list[pd.Series] = []
    contradiction_nodes: list[str] = []
    for node in ancestors:
        target_child = _child_on_path(bundle.prior_spec, node, str(target), bundle.label_names)
        if target_child is None:
            continue
        target_score = _score_for_child(bundle, node, target_child)
        if target_score is None:
            continue
        sibling_scores = []
        for sibling in _children_map(bundle.prior_spec).get(node, []):
            if str(sibling) == str(target_child):
                continue
            score = _score_for_child(bundle, node, str(sibling))
            if score is not None:
                sibling_scores.append(score.reindex(index).astype(float))
        if not sibling_scores:
            continue
        best_sibling = pd.concat(sibling_scores, axis=1).max(axis=1)
        raw = (best_sibling - target_score.reindex(index).astype(float)).replace([np.inf, -np.inf], np.nan)
        raw = raw.where(raw.gt(0), 0.0).fillna(0.0)
        q = raw.rank(pct=True, method="average").fillna(0.0)
        contradiction_values.append(raw.astype(float))
        contradiction_q.append(q.astype(float))
        contradiction_nodes.append(str(node))
    if not contradiction_values:
        rows["ancestor_max_contradiction"] = 0.0
        rows["ancestor_max_contradiction_quantile"] = 0.0
        rows["ancestor_max_contradiction_node"] = ""
        return rows
    val_df = pd.concat(contradiction_values, axis=1)
    q_df = pd.concat(contradiction_q, axis=1)
    val_df.columns = contradiction_nodes
    q_df.columns = contradiction_nodes
    max_node = q_df.idxmax(axis=1).fillna("")
    rows["ancestor_max_contradiction"] = val_df.max(axis=1).astype(float)
    rows["ancestor_max_contradiction_quantile"] = q_df.max(axis=1).astype(float)
    rows["ancestor_max_contradiction_node"] = max_node.astype(str)
    return rows


def _append_candidate_rows(
    *,
    bundle: StudentDataBundle,
    rows: list[dict[str, Any]],
    selected: pd.DataFrame,
    target: str,
    tier: str,
    mode: str,
    weight: pd.Series,
    selection_reason: str | None = None,
) -> None:
    pred = bundle.teacher_pred.reindex(bundle.query_index).astype(str)
    conf = bundle.teacher_confidence.reindex(bundle.query_index).astype(float)
    soft = bundle.teacher_soft.reindex(bundle.query_index)
    knn = bundle.knn_purity.reindex(bundle.query_index).astype(float)
    for rank, (cell_id, row) in enumerate(selected.iterrows(), start=1):
        true_label = str(bundle.query.obs.loc[cell_id, "true_label"]) if "true_label" in bundle.query.obs else ""
        rows.append(
            {
                "cell_id": str(cell_id),
                "target_label": str(target),
                "selection_tier": str(tier),
                "candidate_mode": str(mode),
                "teacher_pred_label": str(pred.loc[cell_id]),
                "teacher_confidence": float(conf.loc[cell_id]),
                "target_posterior": float(soft.loc[cell_id, target]) if target in soft.columns else np.nan,
                "target_score": float(row.get("target_score", np.nan)),
                "leaf_local_score": float(row.get("leaf_local_score", np.nan)),
                "direct_sibling_margin": float(row.get("direct_sibling_margin", np.nan)),
                "ancestor_max_contradiction": float(row.get("ancestor_max_contradiction", 0.0)),
                "ancestor_max_contradiction_quantile": float(row.get("ancestor_max_contradiction_quantile", 0.0)),
                "ancestor_max_contradiction_node": str(row.get("ancestor_max_contradiction_node", "")),
                "ancestor_soft_penalty": float(row.get("ancestor_soft_penalty", 0.0)),
                "ancestor_vetoed_before_final": bool(row.get("ancestor_vetoed_before_final", False)),
                "knn_purity": float(knn.loc[cell_id]),
                "pseudo_weight": float(weight.loc[cell_id]) if cell_id in weight.index else 1.0,
                "selection_rank": int(rank),
                "adaptive_reliability_score": float(row.get("adaptive_reliability_score", np.nan)),
                "adaptive_reliability_rank": int(row.get("adaptive_reliability_rank", rank)) if pd.notna(row.get("adaptive_reliability_rank", rank)) else int(rank),
                "adaptive_tail_selection_reason": str(selection_reason or row.get("adaptive_tail_selection_reason", "")),
                "adaptive_candidate_pool_size": int(row.get("adaptive_candidate_pool_size", selected.shape[0])) if pd.notna(row.get("adaptive_candidate_pool_size", selected.shape[0])) else int(selected.shape[0]),
                "adaptive_tail_base_count": int(row.get("adaptive_tail_base_count", 0)) if pd.notna(row.get("adaptive_tail_base_count", 0)) else 0,
                "adaptive_tail_floor_count": int(row.get("adaptive_tail_floor_count", 0)) if pd.notna(row.get("adaptive_tail_floor_count", 0)) else 0,
                "adaptive_tail_elbow_rank": int(row.get("adaptive_tail_elbow_rank", 0)) if pd.notna(row.get("adaptive_tail_elbow_rank", 0)) else 0,
                "adaptive_tail_largest_drop": float(row.get("adaptive_tail_largest_drop", np.nan)),
                "adaptive_tail_median_positive_drop": float(row.get("adaptive_tail_median_positive_drop", np.nan)),
                "adaptive_tail_absolute_min_drop": float(row.get("adaptive_tail_absolute_min_drop", np.nan)),
                "true_label": true_label,
                "is_correct_pseudolabel": bool(true_label == str(target)) if true_label else np.nan,
            }
        )


def _wide_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["leaf_sort"] = out["direct_sibling_margin"].where(out["direct_sibling_margin"].notna(), out["leaf_local_score"]).fillna(-1e9)
    return out.sort_values(
        ["leaf_sort", "target_posterior", "teacher_confidence", "knn_purity"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )


def _final_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["leaf_sort"] = out["direct_sibling_margin"].where(out["direct_sibling_margin"].notna(), out["leaf_local_score"]).fillna(-1e9)
    out["final_sort"] = out["leaf_sort"] - out["ancestor_soft_penalty"].astype(float)
    return out.sort_values(
        ["final_sort", "leaf_sort", "target_posterior", "teacher_confidence", "knn_purity"],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    )


def _rank_high(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not clean.notna().any():
        return pd.Series(0.5, index=values.index, dtype=float)
    return clean.rank(pct=True, method="average").fillna(0.5).astype(float)


def _rank_low(values: pd.Series) -> pd.Series:
    return 1.0 - _rank_high(values)


def _adaptive_tail_count_details(
    scores: pd.Series,
    *,
    fraction: float,
    min_if_any: int,
    max_cap: int,
    drop_ratio: float,
    absolute_min_drop: float,
    floor_fraction: float,
    min_elbow_count: int,
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
            "absolute_min_drop": float(absolute_min_drop),
        }
    base = int(np.ceil(n * float(fraction)))
    base = int(np.clip(base, int(min_if_any), int(max_cap)))
    base = min(base, n)
    floor = int(np.ceil(base * float(floor_fraction)))
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
        "absolute_min_drop": float(absolute_min_drop),
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


def _attach_adaptive_tail_details(df: pd.DataFrame, details: Mapping[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    mapping = {
        "base_count": "adaptive_tail_base_count",
        "floor_count": "adaptive_tail_floor_count",
        "elbow_rank": "adaptive_tail_elbow_rank",
        "largest_drop": "adaptive_tail_largest_drop",
        "median_positive_drop": "adaptive_tail_median_positive_drop",
        "absolute_min_drop": "adaptive_tail_absolute_min_drop",
        "tail_fraction": "adaptive_tail_fraction",
    }
    for src, dst in mapping.items():
        out[dst] = details.get(src, np.nan)
    return out


def _add_adaptive_reliability(cand: pd.DataFrame, *, mode: str, config: Mapping[str, Any]) -> pd.DataFrame:
    if cand.empty:
        out = cand.copy()
        out["adaptive_reliability_score"] = pd.Series(dtype=float)
        out["adaptive_reliability_rank"] = pd.Series(dtype=int)
        return out
    out = cand.copy()
    parts = [
        _rank_high(out.get("target_posterior", pd.Series(np.nan, index=out.index))),
        _rank_high(out.get("teacher_confidence", pd.Series(np.nan, index=out.index))),
        _rank_high(out.get("knn_purity", pd.Series(np.nan, index=out.index))),
        _rank_low(out.get("ancestor_max_contradiction_quantile", pd.Series(0.0, index=out.index))),
    ]
    if mode in {"marker", "hidden"}:
        parts.append(_rank_high(out.get("target_score", pd.Series(np.nan, index=out.index))))
        if out.get("direct_sibling_margin", pd.Series(np.nan, index=out.index)).replace([np.inf, -np.inf], np.nan).notna().any():
            parts.append(_rank_high(out["direct_sibling_margin"]))
    if mode == "hidden":
        parts.append(_rank_high(out.get("parent_posterior", pd.Series(np.nan, index=out.index))))
        parts.append(_rank_high(out.get("child_conditional_posterior", pd.Series(np.nan, index=out.index))))
    score = pd.concat(parts, axis=1).mean(axis=1).clip(0.0, 1.0)
    if mode == "no_marker":
        small_pool = int(out.shape[0]) < int(config.get("adaptive_no_marker_small_pool_threshold", 10))
        low_knn = pd.to_numeric(out.get("knn_purity", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0).lt(
            float(config.get("adaptive_no_marker_small_pool_low_knn", 0.50))
        )
        if small_pool:
            score = score.where(~low_knn, score * float(config.get("adaptive_no_marker_small_pool_penalty", 0.50)))
    out["adaptive_reliability_score"] = score.astype(float)
    out["adaptive_reliability_rank"] = out["adaptive_reliability_score"].rank(method="first", ascending=False).astype(int)
    return out


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


def _adaptive_tail_params(config: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    if kind == "marker":
        return {
            "fraction": float(config.get("adaptive_marker_tail_fraction", 0.25)),
            "min_if_any": int(config.get("adaptive_marker_min_select_if_any", 3)),
            "max_cap": int(config.get("adaptive_marker_max_cap", 50)),
            "min_elbow_count": int(config.get("adaptive_marker_min_elbow_count", 5)),
        }
    if kind == "hidden":
        return {
            "fraction": float(config.get("adaptive_hidden_tail_fraction", 0.10)),
            "min_if_any": int(config.get("adaptive_hidden_min_select_if_any", 0)),
            "max_cap": int(config.get("adaptive_hidden_max_cap_per_child", 10)),
            "min_elbow_count": int(config.get("adaptive_hidden_min_elbow_count", 1)),
        }
    return {
        "fraction": float(config.get("adaptive_no_marker_tail_fraction", 0.10)),
        "min_if_any": int(config.get("adaptive_no_marker_min_select_if_any", 0)),
        "max_cap": int(config.get("adaptive_no_marker_max_cap", 10)),
        "min_elbow_count": int(config.get("adaptive_no_marker_min_elbow_count", 2)),
    }


def _select_adaptive_tail_softfloor_pseudolabels(
    bundle: StudentDataBundle,
    *,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = bundle.query_index
    soft = bundle.teacher_soft.reindex(index)
    pred = bundle.teacher_pred.reindex(index).astype(str)
    collapsed_pred = bundle.teacher_collapsed_pred.reindex(index).astype(str)
    conf = bundle.teacher_confidence.reindex(index).astype(float)
    knn = bundle.knn_purity.reindex(index).astype(float)
    parent_map = _parent_map(bundle.prior_spec)
    enable_hidden_rescue = bool(config.get("enable_hidden_rescue", False))
    hidden_children = {str(child) for values in bundle.partial_label_spec.values() for child in values} if enable_hidden_rescue else set()
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    veto_frames: list[pd.DataFrame] = []
    evidence_rows: list[dict[str, Any]] = []
    audit_frames: list[pd.DataFrame] = []
    missing_rows: list[dict[str, Any]] = []
    floor_rows: list[dict[str, Any]] = []

    for target in bundle.label_names:
        if target not in soft.columns:
            summary_rows.append({"target_label": target, "status": "missing_teacher_probability", "n_used_for_training": 0})
            missing_rows.append({"target_label": target, "status": "missing_teacher_probability"})
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
            pseudo_weight = float(config.get("adaptive_marker_pseudo_weight", 1.0))
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
            pseudo_weight = float(config.get("adaptive_no_marker_pseudo_weight", 0.25))
        cand = cand.join(contradiction, how="left").fillna(
            {"ancestor_max_contradiction": 0.0, "ancestor_max_contradiction_quantile": 0.0, "ancestor_max_contradiction_node": ""}
        )
        cand["ancestor_vetoed_before_final"] = cand["ancestor_max_contradiction_quantile"].ge(float(config.get("hard_contradiction_quantile", 0.90))) & cand[
            "ancestor_max_contradiction"
        ].gt(0)
        cand["ancestor_soft_penalty"] = np.where(
            cand["ancestor_max_contradiction_quantile"].ge(float(config.get("soft_contradiction_quantile", 0.75))) & cand["ancestor_max_contradiction"].gt(0),
            float(config.get("soft_contradiction_penalty", 0.25)),
            0.0,
        )
        cand = _add_adaptive_reliability(cand, mode=mode_kind, config=config)
        params = _adaptive_tail_params(config, kind=mode_kind)
        sorted_cand = _adaptive_candidate_sort(cand)
        wide_n = max(int(params["max_cap"]) * int(config.get("wide_candidate_multiplier", 8)), int(params["max_cap"]))
        wide = sorted_cand.head(wide_n).copy()
        if not wide.empty:
            veto = wide.loc[wide["ancestor_vetoed_before_final"].astype(bool)].copy()
            if not veto.empty:
                veto["target_label"] = str(target)
                veto["candidate_mode"] = mode
                veto["would_rank_without_veto"] = np.arange(1, veto.shape[0] + 1)
                veto_frames.append(veto.reset_index(names="cell_id"))
        eligible = _adaptive_candidate_sort(wide.loc[~wide["ancestor_vetoed_before_final"].astype(bool)].copy())
        if not score_available:
            min_rel = float(config.get("adaptive_no_marker_min_reliability", 0.75))
            primary_eligible = eligible.loc[eligible["adaptive_reliability_score"].ge(min_rel)].copy()
        else:
            primary_eligible = eligible
        tail_details = _adaptive_tail_count_details(
            primary_eligible["adaptive_reliability_score"],
            **params,
            drop_ratio=float(config.get("adaptive_elbow_drop_ratio", 5.0)),
            absolute_min_drop=float(config.get("adaptive_elbow_absolute_min_drop", 0.03)),
            floor_fraction=float(config.get("adaptive_elbow_floor_fraction", 0.40)),
        )
        n_select = int(tail_details["n_select"])
        reason = str(tail_details["selection_reason"])
        selected = _attach_adaptive_tail_details(primary_eligible.head(n_select).copy(), tail_details)
        original_adaptive_tail_count = int(selected.shape[0])
        if not score_available and selected.empty and not eligible.empty:
            floor_mask = eligible["target_posterior"].ge(float(config.get("adaptive_no_marker_soft_floor_posterior", 0.95))) & eligible["knn_purity"].ge(
                float(config.get("adaptive_no_marker_soft_floor_knn", 0.50))
            )
            floor_pool = _adaptive_candidate_sort(eligible.loc[floor_mask].copy())
            if not floor_pool.empty:
                reason = "no_marker_soft_floor_top1"
                tail_details = {**tail_details, "n_select": 1, "selection_reason": reason}
                selected = _attach_adaptive_tail_details(floor_pool.head(1).copy(), tail_details)
            else:
                reason = "no_marker_failed_soft_floor"
        if not selected.empty:
            selected["adaptive_candidate_pool_size"] = int(eligible.shape[0])
            selected["adaptive_tail_selection_reason"] = reason
        selected_weight = pd.Series(float(pseudo_weight), index=selected.index, dtype=float)
        _append_candidate_rows(
            bundle=bundle,
            rows=rows,
            selected=selected,
            target=target,
            tier="tier_adaptive_tail_leaf_treeguard",
            mode=mode,
            weight=selected_weight,
            selection_reason=reason,
        )
        if not eligible.empty:
            audit = _attach_adaptive_tail_details(eligible.copy(), tail_details).reset_index(names="cell_id")
            audit["target_label"] = str(target)
            audit["candidate_mode"] = mode
            audit["adaptive_selected"] = audit["cell_id"].astype(str).isin(selected.index.astype(str))
            audit["adaptive_tail_final_count"] = int(selected.shape[0])
            audit["adaptive_tail_selection_reason"] = reason
            audit_frames.append(audit)
        rescue = pd.DataFrame()
        if enable_hidden_rescue and score_available and target in hidden_children:
            # Keep hidden-rescue behavior conservative for future setting-B runs.
            parent_mass = _node_posterior(soft, parent, prior_spec=bundle.prior_spec, label_names=bundle.label_names) if parent else pd.Series(0.0, index=index)
            child_cond = _child_conditional(soft, target, parent, prior_spec=bundle.prior_spec, label_names=bundle.label_names) if parent else pd.Series(0.0, index=index)
            pool_mask = (
                (parent_mass.ge(float(config.get("parent_pool_threshold", 0.20))) | collapsed_pred.eq(parent))
                & child_cond.ge(float(config.get("child_conditional_threshold", 0.05)))
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
            rescue["ancestor_vetoed_before_final"] = rescue["ancestor_max_contradiction_quantile"].ge(float(config.get("hard_contradiction_quantile", 0.90))) & rescue[
                "ancestor_max_contradiction"
            ].gt(0)
            rescue["ancestor_soft_penalty"] = np.where(
                rescue["ancestor_max_contradiction_quantile"].ge(float(config.get("soft_contradiction_quantile", 0.75))) & rescue["ancestor_max_contradiction"].gt(0),
                float(config.get("soft_contradiction_penalty", 0.25)),
                0.0,
            )
            rescue = _add_adaptive_reliability(rescue, mode="hidden", config=config)
            rescue_params = _adaptive_tail_params(config, kind="hidden")
            rescue_wide = _adaptive_candidate_sort(rescue).head(max(int(rescue_params["max_cap"]) * int(config.get("wide_candidate_multiplier", 8)), int(rescue_params["max_cap"]))).copy()
            rescue_eligible = _adaptive_candidate_sort(rescue_wide.loc[~rescue_wide["ancestor_vetoed_before_final"].astype(bool)].copy())
            rescue_details = _adaptive_tail_count_details(
                rescue_eligible["adaptive_reliability_score"],
                **rescue_params,
                drop_ratio=float(config.get("adaptive_elbow_drop_ratio", 5.0)),
                absolute_min_drop=float(config.get("adaptive_elbow_absolute_min_drop", 0.03)),
                floor_fraction=float(config.get("adaptive_elbow_floor_fraction", 0.40)),
            )
            rescue = _attach_adaptive_tail_details(rescue_eligible.head(int(rescue_details["n_select"])).copy(), rescue_details)
            if not rescue.empty:
                rescue["adaptive_candidate_pool_size"] = int(rescue_eligible.shape[0])
                rescue_reason = str(rescue_details["selection_reason"])
                _append_candidate_rows(
                    bundle=bundle,
                    rows=rows,
                    selected=rescue,
                    target=target,
                    tier="tier_adaptive_tail_hidden_parent_rescue",
                    mode="parent_pool_leaf_marker_treeguard",
                    weight=pd.Series(float(config.get("adaptive_hidden_pseudo_weight", 0.5)), index=rescue.index, dtype=float),
                    selection_reason=rescue_reason,
                )
        selected_ids = set(selected.index.astype(str))
        rescue_ids = set(rescue.index.astype(str)) if not rescue.empty else set()
        total_selected = int(selected.shape[0] + rescue.shape[0])
        floor_min = int(config.get("adaptive_anchor_floor_min_per_class", 0) or 0)
        floor_topup = pd.DataFrame()
        if floor_min > 0 and total_selected < floor_min and not eligible.empty:
            topup_pool = eligible.loc[
                ~eligible.index.astype(str).isin(selected_ids | rescue_ids)
            ].copy()
            n_topup = max(0, min(int(floor_min - total_selected), int(topup_pool.shape[0])))
            if n_topup > 0:
                floor_topup = _attach_adaptive_tail_details(topup_pool.head(n_topup).copy(), tail_details)
                floor_topup["adaptive_candidate_pool_size"] = int(eligible.shape[0])
                floor_topup["adaptive_tail_selection_reason"] = "adaptive_anchor_floor_topup"
                _append_candidate_rows(
                    bundle=bundle,
                    rows=rows,
                    selected=floor_topup,
                    target=target,
                    tier="tier_adaptive_anchor_floor_topup",
                    mode=mode,
                    weight=pd.Series(float(pseudo_weight), index=floor_topup.index, dtype=float),
                    selection_reason="adaptive_anchor_floor_topup",
                )
                total_selected += int(floor_topup.shape[0])
                floor_rows.append(
                    {
                        "target_label": str(target),
                        "score_available": bool(score_available),
                        "n_selected_before_floor": int(total_selected - floor_topup.shape[0]),
                        "n_floor_topup": int(floor_topup.shape[0]),
                        "floor_min_per_class": int(floor_min),
                        "n_eligible_candidates": int(eligible.shape[0]),
                        "max_topup_reliability": float(floor_topup["adaptive_reliability_score"].max()),
                        "min_topup_reliability": float(floor_topup["adaptive_reliability_score"].min()),
                        "max_topup_target_posterior": float(floor_topup["target_posterior"].max()),
                        "min_topup_target_posterior": float(floor_topup["target_posterior"].min()),
                        "max_topup_knn_purity": float(floor_topup["knn_purity"].max()),
                        "min_topup_knn_purity": float(floor_topup["knn_purity"].min()),
                    }
                )
        if total_selected == 0:
            missing_rows.append(
                {
                    "target_label": str(target),
                    "status": reason if reason else "zero_selected_after_adaptive_tail",
                    "score_available": bool(score_available),
                    "n_base_candidates": int(cand.shape[0]),
                    "n_eligible_candidates": int(eligible.shape[0]),
                    "max_reliability": float(eligible["adaptive_reliability_score"].max()) if not eligible.empty else np.nan,
                    "max_target_posterior": float(eligible["target_posterior"].max()) if not eligible.empty else np.nan,
                    "max_knn_purity": float(eligible["knn_purity"].max()) if not eligible.empty else np.nan,
                }
            )
        evidence_rows.append(
            {
                "target_label": str(target),
                "parent_node": str(parent),
                "score_available": bool(score_available),
                "n_base_candidates": int(cand.shape[0]),
                "n_eligible_candidates": int(eligible.shape[0]),
                "n_selected_before_conflict": total_selected,
                "n_hidden_rescue_selected": int(rescue.shape[0]),
                "n_anchor_floor_topup_selected": int(floor_topup.shape[0]) if not floor_topup.empty else 0,
                "hidden_rescue_enabled": bool(enable_hidden_rescue),
                "adaptive_tail_selection_reason": reason,
                "adaptive_tail_base_count": int(tail_details.get("base_count", 0)),
                "adaptive_tail_floor_count": int(tail_details.get("floor_count", 0)),
                "adaptive_tail_elbow_rank": int(tail_details.get("elbow_rank", 0)),
                "adaptive_tail_largest_drop": float(tail_details.get("largest_drop", np.nan)),
                "adaptive_tail_median_positive_drop": float(tail_details.get("median_positive_drop", np.nan)),
                "adaptive_mean_reliability": float(selected["adaptive_reliability_score"].mean()) if not selected.empty else np.nan,
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
                "n_tier_adaptive_anchor_floor_topup": int(floor_topup.shape[0]) if not floor_topup.empty else 0,
                "original_adaptive_tail_count": int(original_adaptive_tail_count),
                "eligible_count": int(eligible.shape[0]),
                "hidden_rescue_enabled": bool(enable_hidden_rescue),
                "adaptive_tail_selection_reason": reason,
                "status": "ok" if total_selected else reason,
            }
        )

    pseudo_df, by_class = _finalize_pseudolabel_rows(rows, summary_rows)
    vetoed = pd.concat(veto_frames, ignore_index=True, sort=False) if veto_frames else pd.DataFrame()
    evidence = pd.DataFrame(evidence_rows).sort_values("target_label", kind="mergesort").reset_index(drop=True)
    candidate_audit = pd.concat(audit_frames, ignore_index=True, sort=False) if audit_frames else pd.DataFrame()
    missing_columns = [
        "target_label",
        "status",
        "score_available",
        "n_base_candidates",
        "n_eligible_candidates",
        "max_reliability",
        "max_target_posterior",
        "max_knn_purity",
    ]
    missing = pd.DataFrame(missing_rows, columns=missing_columns) if missing_rows else pd.DataFrame(columns=missing_columns)
    pseudo_df.attrs["adaptive_candidate_audit"] = candidate_audit
    pseudo_df.attrs["adaptive_missing_or_low_coverage"] = missing
    pseudo_df.attrs["adaptive_reliability_by_class"] = by_class.copy()
    floor_audit_columns = [
        "target_label",
        "score_available",
        "n_selected_before_floor",
        "n_floor_topup",
        "floor_min_per_class",
        "n_eligible_candidates",
        "max_topup_reliability",
        "min_topup_reliability",
        "max_topup_target_posterior",
        "min_topup_target_posterior",
        "max_topup_knn_purity",
        "min_topup_knn_purity",
    ]
    pseudo_df.attrs["adaptive_anchor_floor_topup"] = (
        pd.DataFrame(floor_rows, columns=floor_audit_columns)
        if floor_rows
        else pd.DataFrame(columns=floor_audit_columns)
    )
    return pseudo_df, by_class, vetoed, evidence


def select_bottomup_treeguard_pseudolabels(
    bundle: StudentDataBundle,
    *,
    max_marker_pseudo_per_class: int = int(DEFAULT_BOTTOMUP_CONFIG["max_marker_pseudo_per_class"]),
    max_no_marker_pseudo_per_class: int = int(DEFAULT_BOTTOMUP_CONFIG["max_no_marker_pseudo_per_class"]),
    max_hidden_rescue_per_child: int = int(DEFAULT_BOTTOMUP_CONFIG["max_hidden_rescue_per_child"]),
    posterior_threshold: float = DEFAULT_BOTTOMUP_CONFIG["posterior_threshold"],
    wide_candidate_multiplier: int = int(DEFAULT_BOTTOMUP_CONFIG["wide_candidate_multiplier"]),
    parent_pool_threshold: float = DEFAULT_BOTTOMUP_CONFIG["parent_pool_threshold"],
    child_conditional_threshold: float = DEFAULT_BOTTOMUP_CONFIG["child_conditional_threshold"],
    enable_hidden_rescue: bool = bool(DEFAULT_BOTTOMUP_CONFIG["enable_hidden_rescue"]),
    hard_contradiction_quantile: float = DEFAULT_BOTTOMUP_CONFIG["hard_contradiction_quantile"],
    soft_contradiction_quantile: float = DEFAULT_BOTTOMUP_CONFIG["soft_contradiction_quantile"],
    soft_contradiction_penalty: float = DEFAULT_BOTTOMUP_CONFIG["soft_contradiction_penalty"],
    pseudo_selection_mode: str = "adaptive_tail_robust_elbow",
    adaptive_config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select weighted query anchors for the final student objective.

    Selected rows enter the student's weighted pseudo-label cross-entropy.
    Marker-supported, no-marker and hidden-rescue anchors are assigned different
    reliability weights by the adaptive-tail selector.
    """
    pseudo_mode = str(pseudo_selection_mode).lower()
    if pseudo_mode != "adaptive_tail_robust_elbow":
        raise ValueError("ANCHOR currently supports pseudo_selection_mode='adaptive_tail_robust_elbow' only.")
    config = dict(DEFAULT_BOTTOMUP_CONFIG)
    if adaptive_config:
        config.update({str(k): v for k, v in adaptive_config.items()})
    config.update(
        {
            "posterior_threshold": posterior_threshold,
            "max_marker_pseudo_per_class": max_marker_pseudo_per_class,
            "max_no_marker_pseudo_per_class": max_no_marker_pseudo_per_class,
            "max_hidden_rescue_per_child": max_hidden_rescue_per_child,
            "wide_candidate_multiplier": wide_candidate_multiplier,
            "parent_pool_threshold": parent_pool_threshold,
            "child_conditional_threshold": child_conditional_threshold,
            "enable_hidden_rescue": enable_hidden_rescue,
            "hard_contradiction_quantile": hard_contradiction_quantile,
            "soft_contradiction_quantile": soft_contradiction_quantile,
            "soft_contradiction_penalty": soft_contradiction_penalty,
            "pseudo_selection_mode": "adaptive_tail_robust_elbow",
        }
    )
    return _select_adaptive_tail_softfloor_pseudolabels(bundle, config=config)
