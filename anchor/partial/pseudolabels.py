"""Pseudo-label and anchor helpers for partial-label branches.

Partial-label runs use these helpers when the reference contains a coarse
parent label but the marker tree defines finer children.  The selected cells
provide hidden-parent anchor supervision and leaf-level pseudo-labels during
teacher refinement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from .labels import (
    GENERIC_FLAT_SCORE_MODE,
    HIDDEN_PARENT_ANCHOR_BRANCH_KEY,
    HIDDEN_PARENT_ANCHOR_CHILD_KEY,
    HIDDEN_PARENT_ANCHOR_WEIGHT_KEY,
    PARTIAL_QUERY_PSEUDO_COARSE_TARGET_KEY,
    PARTIAL_QUERY_PSEUDO_COARSE_WEIGHT_KEY,
    PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY,
    PARTIAL_QUERY_PSEUDO_FINE_WEIGHT_KEY,
    PARTIAL_QUERY_PSEUDO_MODE_KEY,
    PARTIAL_QUERY_PSEUDO_ROUND_KEY,
    PARTIAL_QUERY_PSEUDO_SELECTED_KEY,
    PARTIAL_QUERY_PSEUDO_SOURCE_KEY,
    compute_collapsed_predictions_from_soft,
    compute_flat_leaf_target_score,
    default_alternative_leaf_marker_specs,
)

DEFAULT_PARTIAL_FLAT_SELECTION_METHOD = "partial_label_flat_leaf"
DEFAULT_PARTIAL_FLAT_SELECTION_STRATEGY = "generic_score_top20_pred_same_posterior95_max10"
DEFAULT_PARTIAL_HIDDEN_SELECTION_MODE = "pred_top10"
PARENT_POOL_PARTIAL_HIDDEN_SELECTION_MODE = "parent_pool_top10"
PREDSAME_TREE_PARENT_RESCUE_SELECTION_MODE = "pred_same_lowconf_then_tree_parent_rescue_top10"
PREDSAME_TREE_PARENT_RESCUE_TOP30_CAP50_SELECTION_MODE = (
    "pred_same_lowconf_then_tree_parent_rescue_top30_cap50"
)
PREDSAME_TREE_PARENT_RESCUE_ADAPTIVE_SELECTION_MODE = (
    "pred_same_lowconf_then_tree_parent_rescue_adaptive"
)


@dataclass
class PartialHierarchicalPseudoLabelBundle:
    cell_level: pd.DataFrame
    by_class: pd.DataFrame
    by_mode: pd.DataFrame
    overall: pd.DataFrame


def build_partial_hierarchical_pseudolabel_bundle(
    selection_cell_level: pd.DataFrame,
    *,
    fine_output_labels: Sequence[str],
    supervision_categories: Sequence[str],
) -> PartialHierarchicalPseudoLabelBundle:
    fine_output_labels = [str(x) for x in fine_output_labels]
    supervision_categories = [str(x) for x in supervision_categories]
    fine_to_index = {label: idx for idx, label in enumerate(fine_output_labels)}
    coarse_to_index = {label: idx for idx, label in enumerate(supervision_categories)}

    cell_level = selection_cell_level.copy()
    if cell_level.empty:
        empty = pd.DataFrame()
        return PartialHierarchicalPseudoLabelBundle(empty, empty, empty, empty)

    selected = cell_level.loc[cell_level["selected_for_training"].astype(bool)].copy()
    if selected.empty:
        overall = pd.DataFrame(
            [
                {
                    "n_selected_total": 0,
                    "n_selected_fine": 0,
                    "n_selected_coarse_only": 0,
                    "overall_pseudo_precision": np.nan,
                    "fine_pseudo_precision": np.nan,
                    "coarse_only_pseudo_precision": np.nan,
                    "mean_weight": np.nan,
                }
            ]
        )
        by_mode = pd.DataFrame(
            columns=["pseudo_mode", "n_selected", "n_correct", "pseudo_precision", "mean_weight", "median_weight"]
        )
        by_class = pd.DataFrame(
            columns=["pseudo_mode", "pseudo_label", "n_selected", "n_correct", "pseudo_precision", "mean_weight"]
        )
        return PartialHierarchicalPseudoLabelBundle(selected, by_class, by_mode, overall)

    selected["obs_name"] = selected["cell_id"].astype(str)
    selected["pseudo_mode"] = selected["pseudo_mode"].astype(str)
    selected["pseudo_fine_target_index"] = -1
    selected["pseudo_coarse_target_index"] = -1

    fine_mask = selected["pseudo_mode"].astype(str).eq("fine")
    coarse_mask = selected["pseudo_mode"].astype(str).eq("coarse_only")
    if bool(fine_mask.any()):
        selected.loc[fine_mask, "pseudo_fine_target_index"] = (
            selected.loc[fine_mask, "pseudo_fine_label"].astype(str).map(fine_to_index).fillna(-1).astype(int)
        )
    if bool(coarse_mask.any()):
        selected.loc[coarse_mask, "pseudo_coarse_target_index"] = (
            selected.loc[coarse_mask, "pseudo_coarse_label"].astype(str).map(coarse_to_index).fillna(-1).astype(int)
        )

    selected["pseudo_label"] = np.where(
        selected["pseudo_mode"].astype(str).eq("fine"),
        selected["pseudo_fine_label"].astype(str),
        selected["pseudo_coarse_label"].astype(str),
    )
    selected["is_correct"] = selected["true_label"].astype(str).eq(selected["pseudo_label"].astype(str))

    by_mode = (
        selected.groupby("pseudo_mode", dropna=False)
        .agg(
            n_selected=("obs_name", "nunique"),
            n_correct=("is_correct", "sum"),
            pseudo_precision=("is_correct", "mean"),
            mean_weight=("pseudo_weight", "mean"),
            median_weight=("pseudo_weight", "median"),
        )
        .reset_index()
        .sort_values("pseudo_mode", kind="mergesort")
        .reset_index(drop=True)
    )
    by_class = (
        selected.groupby(["pseudo_mode", "pseudo_label"], dropna=False)
        .agg(
            n_selected=("obs_name", "nunique"),
            n_correct=("is_correct", "sum"),
            pseudo_precision=("is_correct", "mean"),
            mean_weight=("pseudo_weight", "mean"),
            median_weight=("pseudo_weight", "median"),
        )
        .reset_index()
        .sort_values(["pseudo_mode", "pseudo_label"], kind="mergesort")
        .reset_index(drop=True)
    )
    overall = pd.DataFrame(
        [
            {
                "n_selected_total": int(selected.shape[0]),
                "n_selected_fine": int(fine_mask.sum()),
                "n_selected_coarse_only": int(coarse_mask.sum()),
                "overall_pseudo_precision": float(selected["is_correct"].mean()),
                "fine_pseudo_precision": float(selected.loc[fine_mask, "is_correct"].mean()) if bool(fine_mask.any()) else np.nan,
                "coarse_only_pseudo_precision": float(selected.loc[coarse_mask, "is_correct"].mean()) if bool(coarse_mask.any()) else np.nan,
                "mean_weight": float(selected["pseudo_weight"].mean()),
            }
        ]
    )
    return PartialHierarchicalPseudoLabelBundle(
        cell_level=selected.reset_index(drop=True),
        by_class=by_class,
        by_mode=by_mode,
        overall=overall,
    )


def apply_partial_hierarchical_pseudolabel_obs(
    adata: ad.AnnData,
    bundle: PartialHierarchicalPseudoLabelBundle,
    *,
    round_idx: int,
    source_name: str = "partial_hierarchical_tree",
) -> None:
    adata.obs[PARTIAL_QUERY_PSEUDO_SELECTED_KEY] = 0.0
    adata.obs[PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY] = -1.0
    adata.obs[PARTIAL_QUERY_PSEUDO_FINE_WEIGHT_KEY] = 0.0
    adata.obs[PARTIAL_QUERY_PSEUDO_COARSE_TARGET_KEY] = -1.0
    adata.obs[PARTIAL_QUERY_PSEUDO_COARSE_WEIGHT_KEY] = 0.0
    adata.obs[PARTIAL_QUERY_PSEUDO_MODE_KEY] = ""
    adata.obs[PARTIAL_QUERY_PSEUDO_ROUND_KEY] = float(round_idx)
    adata.obs[PARTIAL_QUERY_PSEUDO_SOURCE_KEY] = str(source_name)
    if bundle.cell_level.empty:
        return

    selected = bundle.cell_level.copy()
    idx = pd.Index(selected["obs_name"].astype(str))
    adata.obs.loc[idx, PARTIAL_QUERY_PSEUDO_SELECTED_KEY] = 1.0
    adata.obs.loc[idx, PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY] = (
        selected["pseudo_fine_target_index"].astype(float).to_numpy()
    )
    adata.obs.loc[idx, PARTIAL_QUERY_PSEUDO_FINE_WEIGHT_KEY] = np.where(
        selected["pseudo_mode"].astype(str).eq("fine"),
        selected["pseudo_weight"].astype(float).to_numpy(),
        0.0,
    )
    adata.obs.loc[idx, PARTIAL_QUERY_PSEUDO_COARSE_TARGET_KEY] = (
        selected["pseudo_coarse_target_index"].astype(float).to_numpy()
    )
    adata.obs.loc[idx, PARTIAL_QUERY_PSEUDO_COARSE_WEIGHT_KEY] = np.where(
        selected["pseudo_mode"].astype(str).eq("coarse_only"),
        selected["pseudo_weight"].astype(float).to_numpy(),
        0.0,
    )
    adata.obs.loc[idx, PARTIAL_QUERY_PSEUDO_MODE_KEY] = selected["pseudo_mode"].astype(str).to_numpy()
    adata.obs.loc[idx, PARTIAL_QUERY_PSEUDO_ROUND_KEY] = float(round_idx)
    adata.obs.loc[idx, PARTIAL_QUERY_PSEUDO_SOURCE_KEY] = str(source_name)


DEFAULT_HIDDEN_PARENT_ANCHOR_METHOD = "hidden_parent_anchor_ce"
DEFAULT_HIDDEN_PARENT_ANCHOR_STRATEGY = "parent_pred_or_posterior_marker_top10"


@dataclass
class PartialFlatLeafPseudoLabelSelection:
    cell_level: pd.DataFrame
    by_class: pd.DataFrame
    overall: pd.DataFrame
    score_availability: pd.DataFrame
    hidden_parent_summary: pd.DataFrame

    def write_outputs(self, results_dir: Path, *, prefix: str) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        self.cell_level.to_csv(results_dir / f"{prefix}_cell_level.csv", index=False)
        self.by_class.to_csv(results_dir / f"{prefix}_by_class.csv", index=False)
        self.overall.to_csv(results_dir / f"{prefix}_overall.csv", index=False)
        self.score_availability.to_csv(results_dir / f"{prefix}_score_availability.csv", index=False)
        self.hidden_parent_summary.to_csv(results_dir / f"{prefix}_hidden_parent_summary.csv", index=False)


@dataclass
class HiddenParentAnchorSelection:
    cell_level: pd.DataFrame
    summary_by_branch: pd.DataFrame
    conflicts: pd.DataFrame

    def write_outputs(self, results_dir: Path, *, prefix: str) -> None:
        results_dir.mkdir(parents=True, exist_ok=True)
        self.cell_level.to_csv(results_dir / f"{prefix}_anchor_cell_level.csv", index=False)
        self.summary_by_branch.to_csv(results_dir / f"{prefix}_anchor_summary_by_branch.csv", index=False)
        self.conflicts.to_csv(results_dir / f"{prefix}_anchor_conflicts.csv", index=False)


def _align_index(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.astype(str)
    return out.reindex(index)


def _sigmoid_series(values: pd.Series) -> pd.Series:
    values = values.astype(float).clip(lower=-50.0, upper=50.0)
    return 1.0 / (1.0 + np.exp(-values))


def _clip_series(values: pd.Series, lower: float, upper: float) -> pd.Series:
    return values.astype(float).clip(lower=float(lower), upper=float(upper))


def _direct_parent_for_label(
    label: str,
    *,
    leaf_marker_specs: Mapping[str, Mapping[str, Sequence[str]]],
    prior_spec: Mapping[str, Any] | None,
) -> str:
    label = str(label)
    tree_spec = (prior_spec or {}).get("tree_spec", {}) if isinstance(prior_spec, Mapping) else {}
    parent_map = {str(k): str(v) for k, v in tree_spec.get("parent", {}).items()}
    if label in parent_map:
        return parent_map[label]
    spec = leaf_marker_specs.get(label, {})
    parent = spec.get("parent", "") if isinstance(spec, Mapping) else ""
    return str(parent) if parent is not None else ""


def _descendant_leaf_names(
    node: str,
    *,
    prior_spec: Mapping[str, Any] | None,
    fine_output_labels: Sequence[str],
) -> list[str]:
    node = str(node)
    label_set = {str(label) for label in fine_output_labels}
    if node in label_set:
        return [node]
    tree_spec = (prior_spec or {}).get("tree_spec", {}) if isinstance(prior_spec, Mapping) else {}
    leaves = [str(x) for x in tree_spec.get("descendants", {}).get(node, [])]
    return [leaf for leaf in leaves if leaf in label_set]


def _node_posterior(
    node: str,
    *,
    soft: pd.DataFrame,
    prior_spec: Mapping[str, Any] | None,
    fine_output_labels: Sequence[str],
) -> pd.Series:
    leaves = _descendant_leaf_names(node, prior_spec=prior_spec, fine_output_labels=fine_output_labels)
    leaves = [leaf for leaf in leaves if leaf in soft.columns]
    if not leaves:
        return pd.Series(np.nan, index=soft.index, dtype=float)
    return soft.loc[:, leaves].sum(axis=1).astype(float)


def _collapsed_pred_mask_for_node(
    node: str,
    *,
    soft: pd.DataFrame,
    prior_spec: Mapping[str, Any] | None,
    fine_output_labels: Sequence[str],
) -> pd.Series:
    node = str(node)
    tree_spec = (prior_spec or {}).get("tree_spec", {}) if isinstance(prior_spec, Mapping) else {}
    parent_map = {str(k): str(v) for k, v in tree_spec.get("parent", {}).items()}
    children_map = {str(k): [str(x) for x in v] for k, v in tree_spec.get("children", {}).items()}
    grandparent = parent_map.get(node, "")
    siblings = children_map.get(grandparent, []) if grandparent else []
    if node not in siblings:
        node_mass = _node_posterior(node, soft=soft, prior_spec=prior_spec, fine_output_labels=fine_output_labels)
        return node_mass.gt(0)
    sibling_masses = {
        sibling: _node_posterior(sibling, soft=soft, prior_spec=prior_spec, fine_output_labels=fine_output_labels)
        for sibling in siblings
    }
    mass_df = pd.DataFrame(sibling_masses, index=soft.index)
    return mass_df.idxmax(axis=1).astype(str).eq(node)


def _score_from_signed_markers(
    *,
    signed_spec: Mapping[str, Any],
    protein_arcsinh: pd.DataFrame,
    index: pd.Index,
) -> tuple[pd.Series | None, dict[str, Any]]:
    pos_values = signed_spec.get("positive", {}) if isinstance(signed_spec, Mapping) else {}
    neg_values = signed_spec.get("negative", {}) if isinstance(signed_spec, Mapping) else {}
    pos = [str(marker) for marker in (pos_values.keys() if isinstance(pos_values, Mapping) else pos_values)]
    neg = [str(marker) for marker in (neg_values.keys() if isinstance(neg_values, Mapping) else neg_values)]
    available = set(protein_arcsinh.columns.astype(str))
    pos_avail = [marker for marker in pos if marker in available]
    neg_avail = [marker for marker in neg if marker in available]
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
        score = score + protein_arcsinh.loc[index, pos_avail].astype(float).mean(axis=1)
    if neg_avail:
        score = score - protein_arcsinh.loc[index, neg_avail].astype(float).mean(axis=1)
    return score.astype(float), {
        "score_available": True,
        "score_type": "generic_feature_mean_pos_minus_neg",
        "positive_markers": "|".join(pos_avail),
        "negative_markers": "|".join(neg_avail),
        "missing_markers": "|".join(sorted(set(pos + neg) - available)),
    }


def _score_tree_child_under_parent(
    child: str,
    parent: str,
    *,
    protein_arcsinh: pd.DataFrame,
    leaf_marker_specs: Mapping[str, Mapping[str, Sequence[str]]],
    prior_spec: Mapping[str, Any] | None,
    score_mode: str,
) -> pd.Series | None:
    child = str(child)
    parent = str(parent)
    if child in leaf_marker_specs:
        score, _ = compute_flat_leaf_target_score(
            child,
            protein_arcsinh=protein_arcsinh,
            leaf_marker_specs=leaf_marker_specs,
            score_mode=score_mode,
        )
        return score
    branch_specs = (prior_spec or {}).get("branch_teacher_specs", {}) if isinstance(prior_spec, Mapping) else {}
    signed_spec = branch_specs.get(parent, {}).get("classes", {}).get(child, {})
    score, _ = _score_from_signed_markers(
        signed_spec=signed_spec,
        protein_arcsinh=protein_arcsinh,
        index=pd.Index(protein_arcsinh.index.astype(str)),
    )
    return score


def _resolve_conflicts(cell_level: pd.DataFrame) -> pd.DataFrame:
    if cell_level.empty:
        return cell_level.copy()
    dedup = (
        cell_level.sort_values(
            ["cell_id", "target_label", "target_score", "target_posterior", "selection_rank"],
            ascending=[True, True, False, False, True],
            kind="mergesort",
        )
        .groupby(["cell_id", "target_label"], dropna=False, as_index=False)
        .first()
        .copy()
    )
    target_counts = dedup.groupby("cell_id", dropna=False)["target_label"].nunique().rename("n_target_labels_per_cell")
    dedup = dedup.merge(target_counts.reset_index(), on="cell_id", how="left")
    dedup["is_conflict"] = dedup["n_target_labels_per_cell"].astype(int).gt(1)
    dedup["used_for_training"] = ~dedup["is_conflict"].astype(bool)
    if "pseudo_weight" not in dedup.columns:
        dedup["pseudo_weight"] = 1.0
    dedup["pseudo_weight"] = np.where(
        dedup["used_for_training"].astype(bool),
        dedup["pseudo_weight"].astype(float),
        0.0,
    )
    return dedup.reset_index(drop=True)


def _resolve_anchor_conflicts(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        return raw.copy(), pd.DataFrame()
    raw = raw.reset_index(drop=True).copy()
    raw["_raw_anchor_row"] = np.arange(raw.shape[0], dtype=int)
    same_branch_sorted = raw.sort_values(
        ["cell_id", "branch", "score_margin", "target_score", "parent_posterior", "child"],
        ascending=[True, True, False, False, False, True],
        kind="mergesort",
    )
    same_branch_keep = same_branch_sorted.groupby(["cell_id", "branch"], dropna=False).head(1)["_raw_anchor_row"]
    same_branch_dropped = raw.loc[~raw["_raw_anchor_row"].isin(set(same_branch_keep))].copy()
    same_branch_dropped["conflict_type"] = "same_branch_child"

    stage1 = raw.loc[raw["_raw_anchor_row"].isin(set(same_branch_keep))].copy()
    cross_branch_sorted = stage1.sort_values(
        ["cell_id", "parent_posterior", "score_margin", "target_score", "branch"],
        ascending=[True, False, False, False, True],
        kind="mergesort",
    )
    cross_branch_keep = cross_branch_sorted.groupby("cell_id", dropna=False).head(1)["_raw_anchor_row"]
    cross_branch_dropped = stage1.loc[~stage1["_raw_anchor_row"].isin(set(cross_branch_keep))].copy()
    cross_branch_dropped["conflict_type"] = "cross_branch"

    kept = stage1.loc[stage1["_raw_anchor_row"].isin(set(cross_branch_keep))].copy()
    kept["used_for_anchor_ce"] = True
    kept["anchor_weight"] = kept["anchor_weight"].astype(float)
    conflicts = pd.concat([same_branch_dropped, cross_branch_dropped], axis=0, ignore_index=True, sort=False)
    if conflicts.empty:
        conflicts = pd.DataFrame(columns=list(raw.columns) + ["conflict_type"])
    kept = kept.drop(columns=["_raw_anchor_row"], errors="ignore").reset_index(drop=True)
    conflicts = conflicts.drop(columns=["_raw_anchor_row"], errors="ignore").reset_index(drop=True)
    return kept, conflicts


def select_hidden_parent_anchor_cells(
    *,
    query_obs: pd.DataFrame,
    soft: pd.DataFrame,
    protein_arcsinh: pd.DataFrame,
    fine_output_labels: Sequence[str],
    partial_label_spec: Mapping[str, Sequence[str]],
    label_col: str = "true_label",
    top_k_per_child: int = 10,
    parent_posterior_thresholds: Sequence[float] = (0.5, 0.2),
    method: str = DEFAULT_HIDDEN_PARENT_ANCHOR_METHOD,
    strategy: str = DEFAULT_HIDDEN_PARENT_ANCHOR_STRATEGY,
    score_mode: str = GENERIC_FLAT_SCORE_MODE,
    leaf_marker_specs: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> HiddenParentAnchorSelection:
    """Select child anchors inside coarse reference branches.

    These anchors are used only by partial-label teacher training, where a
    parent label is observed in the reference but its marker-defined children
    are hidden from the reference annotation.
    """
    query_index = pd.Index(query_obs.index.astype(str))
    fine_output_labels = [str(x) for x in fine_output_labels]
    soft = _align_index(soft, query_index)
    protein_arcsinh = _align_index(protein_arcsinh, query_index)
    truth = query_obs.reindex(query_index)[label_col].astype(str)
    collapsed_soft, collapsed_pred, _ = compute_collapsed_predictions_from_soft(
        soft,
        partial_label_spec=partial_label_spec,
        fine_output_labels=fine_output_labels,
    )
    collapsed_soft = _align_index(collapsed_soft, query_index)
    collapsed_pred = collapsed_pred.reindex(query_index).astype(str)
    leaf_marker_specs = leaf_marker_specs or default_alternative_leaf_marker_specs()

    raw_rows: list[dict[str, Any]] = []
    top_k_per_child = max(0, int(top_k_per_child))
    for branch, children in partial_label_spec.items():
        branch = str(branch)
        children = [str(child) for child in children if str(child) in fine_output_labels]
        if not children or branch not in collapsed_soft.columns:
            continue
        parent_posterior = collapsed_soft[branch].astype(float)
        child_scores: dict[str, pd.Series] = {}
        child_meta: dict[str, dict[str, Any]] = {}
        for child in children:
            score, meta = compute_flat_leaf_target_score(
                child,
                protein_arcsinh=protein_arcsinh,
                leaf_marker_specs=leaf_marker_specs,
                score_mode=score_mode,
            )
            if score is None:
                continue
            child_scores[child] = score.reindex(query_index).astype(float)
            child_meta[child] = dict(meta)
        if not child_scores:
            continue

        for child_index, child in enumerate(children):
            score = child_scores.get(child)
            if score is None:
                continue
            candidate_mask = collapsed_pred.eq(branch)
            pool_source = "collapsed_pred"
            finite_in_pool = int(np.isfinite(score.loc[candidate_mask]).sum())
            for threshold in parent_posterior_thresholds:
                if finite_in_pool >= top_k_per_child:
                    break
                candidate_mask = parent_posterior.ge(float(threshold))
                pool_source = f"parent_posterior_ge_{float(threshold):g}"
                finite_in_pool = int(np.isfinite(score.loc[candidate_mask]).sum())

            candidates = query_index[np.asarray(candidate_mask, dtype=bool)]
            candidate_df = pd.DataFrame(index=candidates)
            candidate_df["target_score"] = score.reindex(candidates).astype(float)
            candidate_df["parent_posterior"] = parent_posterior.reindex(candidates).astype(float)
            candidate_df["collapsed_pred_label"] = collapsed_pred.reindex(candidates).astype(str)
            candidate_df["cell_id"] = candidate_df.index.astype(str)
            for other_child, other_score in child_scores.items():
                if other_child == child:
                    continue
                candidate_df[f"score__{other_child}"] = other_score.reindex(candidates).astype(float)
            other_score_cols = [col for col in candidate_df.columns if col.startswith("score__")]
            if other_score_cols:
                candidate_df["best_other_child_score"] = candidate_df[other_score_cols].max(axis=1)
            else:
                candidate_df["best_other_child_score"] = np.nan
            candidate_df["score_margin"] = candidate_df["target_score"] - candidate_df["best_other_child_score"].fillna(0.0)
            candidate_df = (
                candidate_df.replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["target_score", "parent_posterior"])
                .sort_values(
                    ["target_score", "parent_posterior", "cell_id"],
                    ascending=[False, False, True],
                    kind="mergesort",
                )
                .head(top_k_per_child)
            )
            score_type = str(child_meta.get(child, {}).get("score_type", ""))
            for rank, (cell_id, row) in enumerate(candidate_df.iterrows(), start=1):
                raw_rows.append(
                    {
                        "method": str(method),
                        "strategy": str(strategy),
                        "cell_id": str(cell_id),
                        "branch": branch,
                        "child": child,
                        "child_index": int(child_index),
                        "true_label": str(truth.loc[cell_id]),
                        "collapsed_pred_label": str(row["collapsed_pred_label"]),
                        "parent_posterior": float(row["parent_posterior"]),
                        "pool_source": str(pool_source),
                        "target_score": float(row["target_score"]),
                        "best_other_child_score": (
                            float(row["best_other_child_score"])
                            if pd.notna(row["best_other_child_score"])
                            else np.nan
                        ),
                        "score_margin": float(row["score_margin"]),
                        "score_rank": int(rank),
                        "anchor_weight": 1.0,
                        "score_type": score_type,
                        "score_mode": str(score_mode),
                        "is_correct_anchor": bool(str(truth.loc[cell_id]) == child),
                    }
                )

    raw = pd.DataFrame(raw_rows)
    cell_level, conflicts = _resolve_anchor_conflicts(raw)
    if cell_level.empty:
        summary = pd.DataFrame(
            columns=[
                "branch",
                "child",
                "n_selected",
                "n_correct_anchor",
                "anchor_precision",
                "mean_parent_posterior",
                "mean_target_score",
                "mean_score_margin",
            ]
        )
    else:
        summary = (
            cell_level.groupby(["branch", "child"], dropna=False)
            .agg(
                n_selected=("cell_id", "nunique"),
                n_correct_anchor=("is_correct_anchor", "sum"),
                anchor_precision=("is_correct_anchor", "mean"),
                mean_parent_posterior=("parent_posterior", "mean"),
                mean_target_score=("target_score", "mean"),
                mean_score_margin=("score_margin", "mean"),
                pool_sources=("pool_source", lambda s: "|".join(sorted(pd.unique(s.astype(str))))),
                pool_source_counts=(
                    "pool_source",
                    lambda s: "|".join(
                        f"{key}:{int(value)}"
                        for key, value in s.astype(str).value_counts().sort_index().items()
                    ),
                ),
            )
            .reset_index()
            .sort_values(["branch", "child"], kind="mergesort")
            .reset_index(drop=True)
        )
    return HiddenParentAnchorSelection(
        cell_level=cell_level,
        summary_by_branch=summary,
        conflicts=conflicts,
    )


def _select_partial_flat_leaf_pseudolabels_predsame_tree_parent_rescue(
    *,
    query_obs: pd.DataFrame,
    soft: pd.DataFrame,
    protein_arcsinh: pd.DataFrame,
    fine_output_labels: Sequence[str],
    prior_spec: Mapping[str, Any] | None,
    label_col: str,
    posterior_threshold: float,
    top_fraction: float,
    max_selected_per_class: int | None,
    method: str,
    strategy: str,
    score_mode: str,
    hidden_selection_mode: str,
    leaf_marker_specs: Mapping[str, Mapping[str, Sequence[str]]],
    rescue_parent_posterior_thresholds: Sequence[float],
    rescue_child_conditional_min: float,
    lowconf_weight_bounds: tuple[float, float],
    rescue_weight_bounds: tuple[float, float],
) -> PartialFlatLeafPseudoLabelSelection:
    query_index = pd.Index(query_obs.index.astype(str))
    fine_output_labels = [str(x) for x in fine_output_labels]
    soft = _align_index(soft, query_index)
    protein_arcsinh = _align_index(protein_arcsinh, query_index)
    truth = query_obs.reindex(query_index)[label_col].astype(str)
    pred = soft.idxmax(axis=1).astype(str)
    confidence = soft.max(axis=1).astype(float)
    tree_spec = (prior_spec or {}).get("tree_spec", {}) if isinstance(prior_spec, Mapping) else {}
    children_map = {str(k): [str(x) for x in v] for k, v in tree_spec.get("children", {}).items()}

    selected_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    target_cap = int(max_selected_per_class) if max_selected_per_class is not None else None
    fractional_cap_mode = hidden_selection_mode == PREDSAME_TREE_PARENT_RESCUE_TOP30_CAP50_SELECTION_MODE
    adaptive_cap_mode = hidden_selection_mode == PREDSAME_TREE_PARENT_RESCUE_ADAPTIVE_SELECTION_MODE

    def _rank_keep(df: pd.DataFrame, n_keep: int, sort_cols: list[str], ascending: list[bool]) -> pd.DataFrame:
        if df.empty or n_keep <= 0:
            return df.head(0).copy()
        return df.sort_values(sort_cols, ascending=ascending, kind="mergesort").head(int(n_keep)).copy()

    def _fractional_keep_count(n_candidates: int) -> int:
        if n_candidates <= 0:
            return 0
        n_keep = max(1, int(math.ceil(float(top_fraction) * int(n_candidates))))
        if target_cap is not None:
            n_keep = min(n_keep, target_cap)
        return int(n_keep)

    def _adaptive_keep_decision(pred_same_df: pd.DataFrame) -> dict[str, Any]:
        if pred_same_df.empty:
            return {
                "adaptive_quality_tier": "none",
                "adaptive_top_fraction": 0.0,
                "adaptive_cap": 0,
                "adaptive_desired_total": 0,
                "adaptive_median_target_posterior": np.nan,
                "adaptive_median_child_conditional_posterior": np.nan,
                "adaptive_median_score_margin": np.nan,
            }
        med_post = float(pred_same_df["target_posterior"].median())
        med_child = (
            float(pred_same_df["child_conditional_posterior"].median())
            if "child_conditional_posterior" in pred_same_df
            else np.nan
        )
        med_margin = (
            float(pred_same_df["score_margin"].median())
            if "score_margin" in pred_same_df and bool(pred_same_df["score_margin"].notna().any())
            else np.nan
        )
        margin_ok_high = np.isnan(med_margin) or med_margin >= 0.25
        margin_ok_mid = np.isnan(med_margin) or med_margin >= 0.0
        child_high = np.isnan(med_child) or med_child >= 0.75
        child_mid = np.isnan(med_child) or med_child >= 0.50
        if med_post >= 0.90 and child_high and margin_ok_high:
            tier, frac, cap = "high", min(float(top_fraction), 0.30), min(target_cap or 50, 50)
        elif med_post >= 0.75 and child_mid and margin_ok_mid:
            tier, frac, cap = "medium", min(float(top_fraction), 0.20), min(target_cap or 30, 30)
        else:
            tier, frac, cap = "conservative", 0.0, min(target_cap or 10, 10)
        if tier == "conservative":
            desired_total = min(int(cap), int(pred_same_df.shape[0]))
        else:
            desired_total = min(int(cap), max(1, int(math.ceil(float(frac) * int(pred_same_df.shape[0])))))
        return {
            "adaptive_quality_tier": tier,
            "adaptive_top_fraction": float(frac),
            "adaptive_cap": int(cap),
            "adaptive_desired_total": int(desired_total),
            "adaptive_median_target_posterior": med_post,
            "adaptive_median_child_conditional_posterior": med_child,
            "adaptive_median_score_margin": med_margin,
        }

    def _append_rows(
        *,
        kept: pd.DataFrame,
        target_label: str,
        parent_label: str,
        candidate_mode: str,
        selection_tier: str,
        score_type: str,
    ) -> None:
        for rank, (cell_id, row) in enumerate(kept.iterrows(), start=1):
            selected_rows.append(
                {
                    "method": str(method),
                    "strategy": str(strategy),
                    "hidden_selection_mode": hidden_selection_mode,
                    "cell_id": str(cell_id),
                    "target_label": str(target_label),
                    "true_label": str(truth.loc[cell_id]),
                    "pred_label": str(row.get("pred_label", "")),
                    "prediction_confidence": float(row.get("prediction_confidence", np.nan)),
                    "collapsed_pred_label": str(row.get("collapsed_pred_label", "")),
                    "parent_label": str(parent_label),
                    "candidate_mode": str(candidate_mode),
                    "selection_tier": str(selection_tier),
                    "target_posterior": float(row.get("target_posterior", np.nan)),
                    "collapsed_parent_posterior": float(row.get("parent_posterior", np.nan))
                    if pd.notna(row.get("parent_posterior", np.nan))
                    else np.nan,
                    "parent_posterior": float(row.get("parent_posterior", np.nan))
                    if pd.notna(row.get("parent_posterior", np.nan))
                    else np.nan,
                    "child_conditional_posterior": float(row.get("child_conditional_posterior", np.nan))
                    if pd.notna(row.get("child_conditional_posterior", np.nan))
                    else np.nan,
                    "target_score": float(row["target_score"]),
                    "best_sibling_label": str(row.get("best_sibling_label", "")),
                    "best_sibling_score": float(row.get("best_sibling_score", np.nan))
                    if pd.notna(row.get("best_sibling_score", np.nan))
                    else np.nan,
                    "score_margin": float(row.get("score_margin", np.nan))
                    if pd.notna(row.get("score_margin", np.nan))
                    else np.nan,
                    "pseudo_weight": float(row.get("pseudo_weight", 1.0)),
                    "selection_rank": int(rank),
                    "is_hidden_label": bool(parent_label),
                    "is_correct_pseudolabel": bool(str(truth.loc[cell_id]) == str(target_label)),
                    "score_type": str(score_type),
                    "score_mode": str(score_mode),
                }
            )

    for target_label in fine_output_labels:
        target_label = str(target_label)
        meta_base = {
            "target_label": target_label,
            "method": str(method),
            "strategy": str(strategy),
            "score_mode": str(score_mode),
            "hidden_selection_mode": hidden_selection_mode,
            "max_selected_per_class": max_selected_per_class,
        }
        if target_label not in soft.columns:
            availability_rows.append(
                {
                    **meta_base,
                    "score_available": False,
                    "score_type": "missing_probability_column",
                    "candidate_mode": "missing_probability_column",
                    "parent_label": "",
                    "n_candidate": 0,
                    "n_selected": 0,
                }
            )
            continue

        score, score_meta = compute_flat_leaf_target_score(
            target_label,
            protein_arcsinh=protein_arcsinh,
            leaf_marker_specs=leaf_marker_specs,
            score_mode=score_mode,
        )
        meta = {**meta_base, **score_meta}
        parent_label = _direct_parent_for_label(
            target_label,
            leaf_marker_specs=leaf_marker_specs,
            prior_spec=prior_spec,
        )
        meta["parent_label"] = parent_label
        meta["candidate_mode"] = hidden_selection_mode
        target_posterior = soft[target_label].astype(float)
        if score is None:
            meta.update(
                {
                    "n_candidate": 0,
                    "n_selected": 0,
                    "n_used_for_training": 0,
                    "n_conflicts": 0,
                    "n_correct_pseudolabel": 0,
                    "pseudo_precision_all_selected": np.nan,
                    "pseudo_precision_used_for_training": np.nan,
                    "mean_target_score": np.nan,
                    "median_target_score": np.nan,
                }
            )
            availability_rows.append(meta.copy())
            continue

        base_df = pd.DataFrame(index=query_index)
        base_df["target_score"] = score.reindex(query_index).astype(float)
        base_df["target_posterior"] = target_posterior.reindex(query_index).astype(float)
        base_df["cell_id"] = base_df.index.astype(str)
        base_df["pred_label"] = pred.reindex(query_index).astype(str)
        base_df["prediction_confidence"] = confidence.reindex(query_index).astype(float)
        if parent_label:
            parent_posterior = _node_posterior(
                parent_label,
                soft=soft,
                prior_spec=prior_spec,
                fine_output_labels=fine_output_labels,
            ).reindex(query_index).astype(float)
        else:
            parent_posterior = pd.Series(np.nan, index=query_index, dtype=float)
        base_df["parent_posterior"] = parent_posterior
        base_df["child_conditional_posterior"] = (
            base_df["target_posterior"] / base_df["parent_posterior"].replace(0, np.nan)
        )
        base_df["collapsed_pred_label"] = ""
        base_df = base_df.replace([np.inf, -np.inf], np.nan)

        selected_ids: set[str] = set()
        score_type = str(meta.get("score_type", ""))
        pred_same_finite = base_df.loc[
            base_df["pred_label"].eq(target_label) & base_df["target_score"].notna()
        ].copy()
        high_conf = pred_same_finite.loc[pred_same_finite["target_posterior"].gt(float(posterior_threshold))].copy()
        adaptive_decision = (
            _adaptive_keep_decision(pred_same_finite)
            if adaptive_cap_mode
            else {
                "adaptive_quality_tier": "",
                "adaptive_top_fraction": np.nan,
                "adaptive_cap": np.nan,
                "adaptive_desired_total": np.nan,
                "adaptive_median_target_posterior": np.nan,
                "adaptive_median_child_conditional_posterior": np.nan,
                "adaptive_median_score_margin": np.nan,
            }
        )
        meta.update(adaptive_decision)

        if adaptive_cap_mode:
            desired_total = int(adaptive_decision["adaptive_desired_total"])
            n_tier1_keep = min(int(high_conf.shape[0]), desired_total)
        elif fractional_cap_mode:
            desired_total = _fractional_keep_count(pred_same_finite.shape[0])
            n_tier1_keep = min(int(high_conf.shape[0]), desired_total)
        else:
            desired_total = target_cap if target_cap is not None else 0
            n_tier1_keep = max(1, int(math.ceil(float(top_fraction) * high_conf.shape[0]))) if not high_conf.empty else 0
            if target_cap is not None and n_tier1_keep:
                n_tier1_keep = min(n_tier1_keep, target_cap)
        tier1 = _rank_keep(
            high_conf,
            n_tier1_keep,
            ["target_score", "target_posterior", "cell_id"],
            [False, False, True],
        )
        if not tier1.empty:
            tier1["pseudo_weight"] = 1.0
            _append_rows(
                kept=tier1,
                target_label=target_label,
                parent_label=parent_label,
                candidate_mode="pred_same_posterior95",
                selection_tier="tier1_pred_same_posterior95",
                score_type=score_type,
            )
            selected_ids.update(tier1["cell_id"].astype(str))

        current_selected = len(selected_ids)
        if (not fractional_cap_mode) and (not adaptive_cap_mode):
            desired_total = target_cap if target_cap is not None else current_selected
        if (
            (not fractional_cap_mode)
            and (not adaptive_cap_mode)
            and target_cap is None
            and current_selected == 0
            and not pred_same_finite.empty
        ):
            desired_total = max(1, int(math.ceil(float(top_fraction) * pred_same_finite.shape[0])))
        tier2 = pred_same_finite.head(0).copy()
        if current_selected < desired_total:
            need = int(desired_total - current_selected)
            lowconf = pred_same_finite.loc[~pred_same_finite["cell_id"].astype(str).isin(selected_ids)].copy()
            tier2 = _rank_keep(
                lowconf,
                need,
                ["target_score", "target_posterior", "cell_id"],
                [False, False, True],
            )
            if not tier2.empty:
                tier2["pseudo_weight"] = _clip_series(
                    tier2["target_posterior"],
                    float(lowconf_weight_bounds[0]),
                    float(lowconf_weight_bounds[1]),
                )
                _append_rows(
                    kept=tier2,
                    target_label=target_label,
                    parent_label=parent_label,
                    candidate_mode="pred_same_lowconf",
                    selection_tier="tier2_pred_same_lowconf",
                    score_type=score_type,
                )
                selected_ids.update(tier2["cell_id"].astype(str))

        tier3 = base_df.head(0).copy()
        rescue_pool_source = ""
        rescue_allowed = pred_same_finite.empty and parent_label and (
            len(selected_ids) < desired_total or fractional_cap_mode or adaptive_cap_mode
        )
        if rescue_allowed:
            sibling_scores: dict[str, pd.Series] = {}
            for sibling in children_map.get(parent_label, []):
                sibling_score = _score_tree_child_under_parent(
                    sibling,
                    parent_label,
                    protein_arcsinh=protein_arcsinh,
                                leaf_marker_specs=leaf_marker_specs,
                    prior_spec=prior_spec,
                    score_mode=score_mode,
                )
                if sibling_score is not None:
                    sibling_scores[str(sibling)] = sibling_score.reindex(query_index).astype(float)
            other_scores = {k: v for k, v in sibling_scores.items() if str(k) != target_label}
            if other_scores:
                sibling_df = pd.DataFrame(other_scores, index=query_index)
                base_df["best_sibling_score"] = sibling_df.max(axis=1)
                base_df["best_sibling_label"] = sibling_df.idxmax(axis=1).astype(str)
                base_df["score_margin"] = base_df["target_score"] - base_df["best_sibling_score"]
                candidate_mask = _collapsed_pred_mask_for_node(
                    parent_label,
                    soft=soft,
                    prior_spec=prior_spec,
                    fine_output_labels=fine_output_labels,
                ).reindex(query_index).fillna(False).astype(bool)
                rescue_pool_source = "tree_parent_collapsed_pred"
                provisional_need = int(max(1, desired_total - len(selected_ids)))
                for threshold in rescue_parent_posterior_thresholds:
                    finite_count = int(
                        base_df.loc[
                            candidate_mask
                            & base_df["target_score"].notna()
                            & base_df["score_margin"].notna()
                            & ~base_df["cell_id"].astype(str).isin(selected_ids)
                        ].shape[0]
                    )
                    if finite_count >= provisional_need:
                        break
                    candidate_mask = base_df["parent_posterior"].ge(float(threshold))
                    rescue_pool_source = f"tree_parent_posterior_ge_{float(threshold):g}"
                rescue_df = base_df.loc[
                    candidate_mask
                    & base_df["target_score"].notna()
                    & base_df["score_margin"].ge(0.0)
                    & base_df["child_conditional_posterior"].ge(float(rescue_child_conditional_min))
                    & ~base_df["cell_id"].astype(str).isin(selected_ids)
                ].copy()
                if adaptive_cap_mode:
                    rescue_decision = _adaptive_keep_decision(rescue_df)
                    desired_total = int(rescue_decision["adaptive_desired_total"])
                    for key, value in rescue_decision.items():
                        meta[f"rescue_{key}"] = value
                elif fractional_cap_mode:
                    desired_total = _fractional_keep_count(rescue_df.shape[0])
                need = int(desired_total - len(selected_ids))
                tier3 = _rank_keep(
                    rescue_df,
                    need,
                    ["score_margin", "target_score", "parent_posterior", "child_conditional_posterior", "cell_id"],
                    [False, False, False, False, True],
                )
                if not tier3.empty:
                    tier3["pseudo_weight"] = _clip_series(
                        tier3["parent_posterior"] * _sigmoid_series(tier3["score_margin"]),
                        float(rescue_weight_bounds[0]),
                        float(rescue_weight_bounds[1]),
                    )
                    _append_rows(
                        kept=tier3,
                        target_label=target_label,
                        parent_label=parent_label,
                        candidate_mode=rescue_pool_source,
                        selection_tier="tier3_tree_parent_rescue",
                        score_type=score_type,
                    )
                    selected_ids.update(tier3["cell_id"].astype(str))

        selected_truth = truth.reindex(list(selected_ids)).astype(str) if selected_ids else pd.Series(dtype=str)
        n_selected = int(len(selected_ids))
        n_correct = int(selected_truth.eq(target_label).sum()) if n_selected else 0
        meta.update(
            {
                "n_candidate": int(pred_same_finite.shape[0]),
                "n_candidate_tier1_pred_same_posterior95": int(high_conf.shape[0]),
                "n_candidate_tier2_pred_same_finite": int(pred_same_finite.shape[0]),
                "n_candidate_tier3_tree_parent_rescue": int(tier3.shape[0]),
                "n_selected": n_selected,
                "n_selected_tier1": int(tier1.shape[0]),
                "n_selected_tier2": int(tier2.shape[0]),
                "n_selected_tier3": int(tier3.shape[0]),
                "n_used_for_training": n_selected,
                "n_conflicts": 0,
                "n_correct_pseudolabel": n_correct,
                "pseudo_precision_all_selected": float(n_correct) / float(n_selected) if n_selected else np.nan,
                "pseudo_precision_used_for_training": float(n_correct) / float(n_selected) if n_selected else np.nan,
                "mean_target_score": float(base_df.loc[list(selected_ids), "target_score"].mean()) if n_selected else np.nan,
                "median_target_score": float(base_df.loc[list(selected_ids), "target_score"].median()) if n_selected else np.nan,
                "rescue_pool_source": rescue_pool_source,
            }
        )
        availability_rows.append(meta.copy())

    cell_level = _resolve_conflicts(pd.DataFrame(selected_rows))
    score_availability = pd.DataFrame(availability_rows)

    if cell_level.empty:
        overall = pd.DataFrame(
            [
                {
                    "method": str(method),
                    "strategy": str(strategy),
                    "hidden_selection_mode": hidden_selection_mode,
                    "n_selected": 0,
                    "n_used_for_training": 0,
                    "n_conflicts": 0,
                    "n_correct_pseudolabel": 0,
                    "pseudo_precision_all_selected": np.nan,
                    "pseudo_precision_used_for_training": np.nan,
                    "n_labels_with_selected": 0,
                }
            ]
        )
        return PartialFlatLeafPseudoLabelSelection(
            cell_level=cell_level,
            by_class=pd.DataFrame(),
            overall=overall,
            score_availability=score_availability.reset_index(drop=True),
            hidden_parent_summary=pd.DataFrame(),
        )

    by_class = (
        cell_level.groupby(["target_label", "parent_label", "candidate_mode", "selection_tier"], dropna=False)
        .agg(
            n_selected=("cell_id", "nunique"),
            n_used_for_training=("used_for_training", "sum"),
            n_conflicts=("is_conflict", "sum"),
            n_correct_pseudolabel=("is_correct_pseudolabel", "sum"),
            pseudo_precision_all_selected=("is_correct_pseudolabel", "mean"),
            pseudo_precision_used_for_training=(
                "is_correct_pseudolabel",
                lambda s: float(s[cell_level.loc[s.index, "used_for_training"].astype(bool)].mean())
                if bool(cell_level.loc[s.index, "used_for_training"].astype(bool).any())
                else np.nan,
            ),
            mean_target_score=("target_score", "mean"),
            mean_target_posterior=("target_posterior", "mean"),
            mean_parent_posterior=("parent_posterior", "mean"),
            mean_child_conditional_posterior=("child_conditional_posterior", "mean"),
            mean_score_margin=("score_margin", "mean"),
            mean_pseudo_weight=("pseudo_weight", "mean"),
        )
        .reset_index()
        .sort_values(["target_label", "selection_tier", "candidate_mode"], kind="mergesort")
        .reset_index(drop=True)
    )
    n_selected = int(cell_level.shape[0])
    n_used = int(cell_level["used_for_training"].astype(bool).sum())
    n_correct = int(cell_level["is_correct_pseudolabel"].astype(bool).sum())
    used_mask = cell_level["used_for_training"].astype(bool)
    used_precision = float(cell_level.loc[used_mask, "is_correct_pseudolabel"].mean()) if bool(used_mask.any()) else np.nan
    overall = pd.DataFrame(
        [
            {
                "method": str(method),
                "strategy": str(strategy),
                "hidden_selection_mode": hidden_selection_mode,
                "n_selected": n_selected,
                "n_used_for_training": n_used,
                "n_conflicts": int(cell_level["is_conflict"].astype(bool).sum()),
                "n_correct_pseudolabel": n_correct,
                "pseudo_precision_all_selected": float(n_correct) / float(n_selected) if n_selected else np.nan,
                "pseudo_precision_used_for_training": used_precision,
                "n_labels_with_selected": int(cell_level["target_label"].nunique()),
            }
        ]
    )
    hidden_parent_summary = (
        cell_level.loc[cell_level["parent_label"].astype(str).ne("")]
        .groupby(["parent_label", "target_label", "candidate_mode", "selection_tier"], dropna=False)
        .agg(
            n_selected=("cell_id", "nunique"),
            n_used_for_training=("used_for_training", "sum"),
            n_conflicts=("is_conflict", "sum"),
            mean_target_score=("target_score", "mean"),
            mean_target_posterior=("target_posterior", "mean"),
            mean_parent_posterior=("parent_posterior", "mean"),
            mean_score_margin=("score_margin", "mean"),
            mean_pseudo_weight=("pseudo_weight", "mean"),
        )
        .reset_index()
        .sort_values(["parent_label", "target_label", "selection_tier"], kind="mergesort")
        .reset_index(drop=True)
    )
    return PartialFlatLeafPseudoLabelSelection(
        cell_level=cell_level.reset_index(drop=True),
        by_class=by_class,
        overall=overall.reset_index(drop=True),
        score_availability=score_availability.reset_index(drop=True),
        hidden_parent_summary=hidden_parent_summary,
    )


def select_partial_flat_leaf_pseudolabels(
    *,
    query_obs: pd.DataFrame,
    soft: pd.DataFrame,
    protein_arcsinh: pd.DataFrame,
    fine_output_labels: Sequence[str],
    partial_label_spec: Mapping[str, Sequence[str]],
    label_col: str = "true_label",
    posterior_threshold: float = 0.95,
    top_fraction: float = 0.20,
    max_selected_per_class: int | None = 10,
    method: str = DEFAULT_PARTIAL_FLAT_SELECTION_METHOD,
    strategy: str = DEFAULT_PARTIAL_FLAT_SELECTION_STRATEGY,
    score_mode: str = GENERIC_FLAT_SCORE_MODE,
    hidden_selection_mode: str = DEFAULT_PARTIAL_HIDDEN_SELECTION_MODE,
    leaf_marker_specs: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    prior_spec: Mapping[str, Any] | None = None,
    rescue_parent_posterior_thresholds: Sequence[float] = (0.5, 0.2),
    rescue_child_conditional_min: float = 0.05,
    lowconf_weight_bounds: tuple[float, float] = (0.3, 0.7),
    rescue_weight_bounds: tuple[float, float] = (0.2, 0.5),
) -> PartialFlatLeafPseudoLabelSelection:
    """Select leaf pseudo-labels for partial-label teacher refinement."""
    hidden_selection_mode = str(hidden_selection_mode)
    if hidden_selection_mode not in {
        DEFAULT_PARTIAL_HIDDEN_SELECTION_MODE,
        PARENT_POOL_PARTIAL_HIDDEN_SELECTION_MODE,
        PREDSAME_TREE_PARENT_RESCUE_SELECTION_MODE,
        PREDSAME_TREE_PARENT_RESCUE_TOP30_CAP50_SELECTION_MODE,
        PREDSAME_TREE_PARENT_RESCUE_ADAPTIVE_SELECTION_MODE,
    }:
        raise ValueError(f"Unsupported hidden_selection_mode={hidden_selection_mode!r}")

    query_index = pd.Index(query_obs.index.astype(str))
    protein_arcsinh = _align_index(protein_arcsinh, query_index)
    leaf_marker_specs = leaf_marker_specs or default_alternative_leaf_marker_specs()
    if hidden_selection_mode in {
        PREDSAME_TREE_PARENT_RESCUE_SELECTION_MODE,
        PREDSAME_TREE_PARENT_RESCUE_TOP30_CAP50_SELECTION_MODE,
        PREDSAME_TREE_PARENT_RESCUE_ADAPTIVE_SELECTION_MODE,
    }:
        return _select_partial_flat_leaf_pseudolabels_predsame_tree_parent_rescue(
            query_obs=query_obs,
            soft=soft,
            protein_arcsinh=protein_arcsinh,
            fine_output_labels=fine_output_labels,
            prior_spec=prior_spec,
            label_col=label_col,
            posterior_threshold=posterior_threshold,
            top_fraction=top_fraction,
            max_selected_per_class=max_selected_per_class,
            method=method,
            strategy=strategy,
            score_mode=score_mode,
            hidden_selection_mode=hidden_selection_mode,
            leaf_marker_specs=leaf_marker_specs,
            rescue_parent_posterior_thresholds=rescue_parent_posterior_thresholds,
            rescue_child_conditional_min=rescue_child_conditional_min,
            lowconf_weight_bounds=lowconf_weight_bounds,
            rescue_weight_bounds=rescue_weight_bounds,
        )

    fine_output_labels = [str(x) for x in fine_output_labels]
    soft = _align_index(soft, query_index)
    truth = query_obs.reindex(query_index)[label_col].astype(str)
    pred = soft.idxmax(axis=1).astype(str)
    confidence = soft.max(axis=1).astype(float)
    collapsed_soft, collapsed_pred, collapsed_conf = compute_collapsed_predictions_from_soft(
        soft,
        partial_label_spec=partial_label_spec,
        fine_output_labels=fine_output_labels,
    )
    collapsed_soft = _align_index(collapsed_soft, query_index)
    collapsed_pred = collapsed_pred.reindex(query_index).astype(str)
    collapsed_conf = collapsed_conf.reindex(query_index).astype(float)
    child_to_parent = {
        str(child): str(parent)
        for parent, children in partial_label_spec.items()
        for child in children
    }

    selected_rows: list[dict[str, Any]] = []
    by_class_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []

    for target_label in fine_output_labels:
        meta_base = {
            "target_label": str(target_label),
            "method": str(method),
            "strategy": str(strategy),
            "score_mode": str(score_mode),
            "hidden_selection_mode": hidden_selection_mode,
            "max_selected_per_class": max_selected_per_class,
        }
        if target_label not in soft.columns:
            row = {
                **meta_base,
                "score_available": False,
                "score_type": "missing_probability_column",
                "candidate_mode": "missing_probability_column",
                "parent_label": str(child_to_parent.get(str(target_label), "")),
                "n_candidate": 0,
                "n_selected": 0,
            }
            availability_rows.append(row)
            by_class_rows.append(row)
            continue

        score, meta = compute_flat_leaf_target_score(
            target_label,
            protein_arcsinh=protein_arcsinh,
            leaf_marker_specs=leaf_marker_specs,
            score_mode=score_mode,
        )
        meta = {**meta_base, **meta}
        parent_label = str(child_to_parent.get(str(target_label), ""))
        is_hidden_label = bool(parent_label)
        target_posterior = soft[target_label].astype(float)
        if is_hidden_label and hidden_selection_mode == PARENT_POOL_PARTIAL_HIDDEN_SELECTION_MODE:
            candidate_mask = collapsed_pred.eq(parent_label)
            candidate_mode = "hidden_parent_pool"
        else:
            candidate_mask = pred.eq(target_label) & target_posterior.gt(float(posterior_threshold))
            candidate_mode = "pred_same_posterior95"
        candidates = query_index[np.asarray(candidate_mask, dtype=bool)]
        meta["parent_label"] = parent_label
        meta["candidate_mode"] = candidate_mode
        meta["n_candidate"] = int(candidates.size)

        if score is None:
            meta["n_selected"] = 0
            meta["n_used_for_training"] = 0
            meta["n_conflicts"] = 0
            meta["n_correct_pseudolabel"] = 0
            meta["pseudo_precision_all_selected"] = np.nan
            meta["pseudo_precision_used_for_training"] = np.nan
            meta["mean_target_score"] = np.nan
            meta["median_target_score"] = np.nan
            availability_rows.append(meta.copy())
            by_class_rows.append(meta.copy())
            continue

        candidate_df = pd.DataFrame(index=candidates)
        candidate_df["target_score"] = score.reindex(candidates).astype(float)
        candidate_df["target_posterior"] = target_posterior.reindex(candidates).astype(float)
        candidate_df["cell_id"] = candidate_df.index.astype(str)
        candidate_df["pred_label"] = pred.reindex(candidates).astype(str)
        candidate_df["prediction_confidence"] = confidence.reindex(candidates).astype(float)
        candidate_df["collapsed_pred_label"] = collapsed_pred.reindex(candidates).astype(str)
        candidate_df["collapsed_parent_posterior"] = (
            collapsed_soft[parent_label].reindex(candidates).astype(float)
            if parent_label and parent_label in collapsed_soft.columns
            else np.nan
        )
        candidate_df = candidate_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_score", "target_posterior"])
        n_keep = max(1, int(math.ceil(float(top_fraction) * candidate_df.shape[0]))) if not candidate_df.empty else 0
        if max_selected_per_class is not None and n_keep:
            n_keep = min(n_keep, int(max_selected_per_class))
        kept = candidate_df.sort_values(
            ["target_score", "target_posterior", "cell_id"],
            ascending=[False, False, True],
            kind="mergesort",
        ).head(n_keep)

        selected_truth = truth.reindex(kept.index).astype(str)
        n_selected = int(kept.shape[0])
        n_correct = int(selected_truth.eq(target_label).sum()) if n_selected else 0
        precision = float(n_correct) / float(n_selected) if n_selected else np.nan
        meta["n_candidate_with_finite_score"] = int(candidate_df.shape[0])
        meta["n_selected"] = n_selected
        meta["n_correct_pseudolabel"] = n_correct
        meta["pseudo_precision_all_selected"] = precision
        meta["pseudo_precision_used_for_training"] = precision
        meta["n_used_for_training"] = n_selected
        meta["n_conflicts"] = 0
        meta["mean_target_score"] = float(kept["target_score"].mean()) if n_selected else np.nan
        meta["median_target_score"] = float(kept["target_score"].median()) if n_selected else np.nan
        availability_rows.append(meta.copy())
        by_class_rows.append(meta.copy())

        for rank, (cell_id, row) in enumerate(kept.iterrows(), start=1):
            selected_rows.append(
                {
                    "method": str(method),
                    "strategy": str(strategy),
                    "hidden_selection_mode": hidden_selection_mode,
                    "cell_id": str(cell_id),
                    "target_label": str(target_label),
                    "true_label": str(truth.loc[cell_id]),
                    "pred_label": str(row["pred_label"]),
                    "prediction_confidence": float(row["prediction_confidence"]),
                    "collapsed_pred_label": str(row["collapsed_pred_label"]),
                    "parent_label": parent_label,
                    "candidate_mode": candidate_mode,
                    "target_posterior": float(row["target_posterior"]),
                    "collapsed_parent_posterior": (
                        float(row["collapsed_parent_posterior"])
                        if pd.notna(row["collapsed_parent_posterior"])
                        else np.nan
                    ),
                    "target_score": float(row["target_score"]),
                    "selection_rank": int(rank),
                    "is_hidden_label": bool(is_hidden_label),
                    "is_correct_pseudolabel": bool(str(truth.loc[cell_id]) == str(target_label)),
                    "score_type": str(meta.get("score_type", "")),
                    "score_mode": str(score_mode),
                }
            )

    cell_level = _resolve_conflicts(pd.DataFrame(selected_rows))
    by_class = pd.DataFrame(by_class_rows)
    score_availability = pd.DataFrame(availability_rows)

    if cell_level.empty:
        overall = pd.DataFrame(
            [
                {
                    "method": str(method),
                    "strategy": str(strategy),
                    "hidden_selection_mode": hidden_selection_mode,
                    "n_selected": 0,
                    "n_used_for_training": 0,
                    "n_conflicts": 0,
                    "n_correct_pseudolabel": 0,
                    "pseudo_precision_all_selected": np.nan,
                    "pseudo_precision_used_for_training": np.nan,
                    "n_labels_with_selected": 0,
                }
            ]
        )
        hidden_parent_summary = pd.DataFrame()
        return PartialFlatLeafPseudoLabelSelection(
            cell_level=cell_level,
            by_class=by_class.reset_index(drop=True),
            overall=overall,
            score_availability=score_availability.reset_index(drop=True),
            hidden_parent_summary=hidden_parent_summary,
        )

    by_class = (
        cell_level.groupby(["target_label", "parent_label", "candidate_mode"], dropna=False)
        .agg(
            n_selected=("cell_id", "nunique"),
            n_used_for_training=("used_for_training", "sum"),
            n_conflicts=("is_conflict", "sum"),
            n_correct_pseudolabel=("is_correct_pseudolabel", "sum"),
            pseudo_precision_all_selected=("is_correct_pseudolabel", "mean"),
            pseudo_precision_used_for_training=(
                "is_correct_pseudolabel",
                lambda s: float(s[cell_level.loc[s.index, "used_for_training"].astype(bool)].mean())
                if bool(cell_level.loc[s.index, "used_for_training"].astype(bool).any())
                else np.nan,
            ),
            mean_target_score=("target_score", "mean"),
            mean_target_posterior=("target_posterior", "mean"),
        )
        .reset_index()
        .sort_values(["target_label", "candidate_mode"], kind="mergesort")
        .reset_index(drop=True)
    )
    n_selected = int(cell_level.shape[0])
    n_used = int(cell_level["used_for_training"].astype(bool).sum())
    n_correct = int(cell_level["is_correct_pseudolabel"].astype(bool).sum())
    used_mask = cell_level["used_for_training"].astype(bool)
    used_precision = float(cell_level.loc[used_mask, "is_correct_pseudolabel"].mean()) if bool(used_mask.any()) else np.nan
    overall = pd.DataFrame(
        [
            {
                "method": str(method),
                "strategy": str(strategy),
                "hidden_selection_mode": hidden_selection_mode,
                "n_selected": n_selected,
                "n_used_for_training": n_used,
                "n_conflicts": int(cell_level["is_conflict"].astype(bool).sum()),
                "n_correct_pseudolabel": n_correct,
                "pseudo_precision_all_selected": float(n_correct) / float(n_selected) if n_selected else np.nan,
                "pseudo_precision_used_for_training": used_precision,
                "n_labels_with_selected": int(cell_level["target_label"].nunique()),
            }
        ]
    )
    hidden_parent_summary = (
        cell_level.loc[cell_level["is_hidden_label"].astype(bool)]
        .groupby(["parent_label", "target_label", "candidate_mode"], dropna=False)
        .agg(
            n_selected=("cell_id", "nunique"),
            n_used_for_training=("used_for_training", "sum"),
            n_conflicts=("is_conflict", "sum"),
            mean_target_score=("target_score", "mean"),
            mean_target_posterior=("target_posterior", "mean"),
            mean_parent_posterior=("collapsed_parent_posterior", "mean"),
        )
        .reset_index()
        .sort_values(["parent_label", "target_label"], kind="mergesort")
        .reset_index(drop=True)
    )

    return PartialFlatLeafPseudoLabelSelection(
        cell_level=cell_level.reset_index(drop=True),
        by_class=by_class,
        overall=overall.reset_index(drop=True),
        score_availability=score_availability.reset_index(drop=True),
        hidden_parent_summary=hidden_parent_summary,
    )


def apply_partial_flat_leaf_pseudolabel_obs(
    adata,
    selection: PartialFlatLeafPseudoLabelSelection,
    *,
    fine_output_labels: Sequence[str],
    round_idx: int,
    source_name: str = "partial_flat_leaf",
) -> None:
    adata.obs[PARTIAL_QUERY_PSEUDO_SELECTED_KEY] = 0.0
    adata.obs[PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY] = -1.0
    adata.obs[PARTIAL_QUERY_PSEUDO_FINE_WEIGHT_KEY] = 0.0
    adata.obs[PARTIAL_QUERY_PSEUDO_COARSE_TARGET_KEY] = -1.0
    adata.obs[PARTIAL_QUERY_PSEUDO_COARSE_WEIGHT_KEY] = 0.0
    adata.obs[PARTIAL_QUERY_PSEUDO_MODE_KEY] = ""
    adata.obs[PARTIAL_QUERY_PSEUDO_ROUND_KEY] = float(round_idx)
    adata.obs[PARTIAL_QUERY_PSEUDO_SOURCE_KEY] = str(source_name)

    if selection.cell_level.empty:
        return

    fine_to_index = {str(label): idx for idx, label in enumerate([str(x) for x in fine_output_labels])}
    selected = selection.cell_level.loc[selection.cell_level["used_for_training"].astype(bool)].copy()
    if selected.empty:
        return
    obs_index = pd.Index(selected["cell_id"].astype(str))
    adata.obs.loc[obs_index, PARTIAL_QUERY_PSEUDO_SELECTED_KEY] = 1.0
    adata.obs.loc[obs_index, PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY] = (
        selected["target_label"].astype(str).map(fine_to_index).fillna(-1).astype(float).to_numpy()
    )
    adata.obs.loc[obs_index, PARTIAL_QUERY_PSEUDO_FINE_WEIGHT_KEY] = (
        selected["pseudo_weight"].astype(float).to_numpy()
    )
    adata.obs.loc[obs_index, PARTIAL_QUERY_PSEUDO_MODE_KEY] = "flat_fine"
    adata.obs.loc[obs_index, PARTIAL_QUERY_PSEUDO_ROUND_KEY] = float(round_idx)
    adata.obs.loc[obs_index, PARTIAL_QUERY_PSEUDO_SOURCE_KEY] = str(source_name)


def apply_hidden_parent_anchor_obs(
    adata,
    selection: HiddenParentAnchorSelection,
    *,
    branch_order: Sequence[str],
) -> None:
    adata.obs[HIDDEN_PARENT_ANCHOR_BRANCH_KEY] = -1.0
    adata.obs[HIDDEN_PARENT_ANCHOR_CHILD_KEY] = -1.0
    adata.obs[HIDDEN_PARENT_ANCHOR_WEIGHT_KEY] = 0.0
    if selection.cell_level.empty:
        return
    branch_to_code = {str(branch): int(code) for code, branch in enumerate([str(x) for x in branch_order])}
    selected = selection.cell_level.loc[selection.cell_level["used_for_anchor_ce"].astype(bool)].copy()
    if selected.empty:
        return
    selected["branch_code"] = selected["branch"].astype(str).map(branch_to_code).fillna(-1).astype(float)
    selected = selected.loc[selected["branch_code"].ge(0)].copy()
    if selected.empty:
        return
    obs_index = pd.Index(selected["cell_id"].astype(str))
    adata.obs.loc[obs_index, HIDDEN_PARENT_ANCHOR_BRANCH_KEY] = selected["branch_code"].to_numpy(dtype=np.float32)
    adata.obs.loc[obs_index, HIDDEN_PARENT_ANCHOR_CHILD_KEY] = (
        selected["child_index"].astype(float).to_numpy(dtype=np.float32)
    )
    adata.obs.loc[obs_index, HIDDEN_PARENT_ANCHOR_WEIGHT_KEY] = (
        selected["anchor_weight"].astype(float).to_numpy(dtype=np.float32)
    )
