from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .labels import (
    collapse_partial_values,
    compute_collapsed_predictions_from_soft,
    normalize_partial_branch_specs,
)

def compute_partial_fine_overall_metrics(
    obs: pd.DataFrame,
    *,
    fine_pred_col: str,
    label_col: str = "true_label",
    split_col: str = "ref_query_col",
    query_name: str = "query",
) -> pd.Series:
    mask = obs[split_col].astype(str).eq(str(query_name))
    y_true = obs.loc[mask, label_col].astype(str)
    y_pred = obs.loc[mask, fine_pred_col].astype(str)
    return pd.Series(
        {
            "query_accuracy": accuracy_score(y_true, y_pred),
            "query_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "query_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
    )


def compute_partial_collapsed_overall_metrics(
    obs: pd.DataFrame,
    *,
    collapsed_pred_col: str,
    partial_label_spec: dict[str, Sequence[str]],
    label_col: str = "true_label",
    split_col: str = "ref_query_col",
    query_name: str = "query",
) -> pd.Series:
    mask = obs[split_col].astype(str).eq(str(query_name))
    y_true = collapse_partial_values(obs.loc[mask, label_col].astype(str), partial_label_spec=partial_label_spec)
    y_pred = obs.loc[mask, collapsed_pred_col].astype(str)
    return pd.Series(
        {
            "query_accuracy": accuracy_score(y_true, y_pred),
            "query_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "query_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }
    )


def compute_partial_hidden_pair_fine_accuracy(
    obs: pd.DataFrame,
    *,
    partial_label_spec: dict[str, Sequence[str]],
    fine_pred_col: str,
    collapsed_pred_col: str,
    label_col: str = "true_label",
    split_col: str = "ref_query_col",
    query_name: str = "query",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    branch_specs = [spec for spec in normalize_partial_branch_specs(partial_label_spec) if len(spec.children) == 2]
    query_mask = obs[split_col].astype(str).eq(str(query_name))
    rows: list[dict[str, Any]] = []
    selected_union_masks: list[pd.Series] = []

    for spec in branch_specs:
        true_mask = query_mask & obs[label_col].astype(str).isin(list(spec.children))
        selected_union_masks.append(true_mask)
        n_query = int(true_mask.sum())
        if n_query == 0:
            rows.append(
                {
                    "pair_key": str(spec.key),
                    "parent_axis_label": str(spec.parent_label),
                    "label_a": str(spec.label_a),
                    "label_b": str(spec.label_b),
                    "n_query": 0,
                    "n_label_a": 0,
                    "n_label_b": 0,
                    "n_correct": 0,
                    "pair_accuracy": np.nan,
                    "recall_label_a": np.nan,
                    "recall_label_b": np.nan,
                    "balanced_accuracy": np.nan,
                    "majority_class_accuracy": np.nan,
                    "out_of_pair_prediction_rate": np.nan,
                    "parent_axis_prediction_rate": np.nan,
                }
            )
            continue
        y_true = obs.loc[true_mask, label_col].astype(str)
        y_pred = obs.loc[true_mask, fine_pred_col].astype(str)
        collapsed_pred = obs.loc[true_mask, collapsed_pred_col].astype(str)
        correct = y_true.eq(y_pred)
        out_of_pair = ~y_pred.isin(list(spec.children))
        parent_axis = collapsed_pred.eq(str(spec.parent_label))
        mask_a = y_true.eq(str(spec.label_a))
        mask_b = y_true.eq(str(spec.label_b))
        n_a = int(mask_a.sum())
        n_b = int(mask_b.sum())
        recall_a = float(y_pred.loc[mask_a].eq(str(spec.label_a)).mean()) if n_a > 0 else np.nan
        recall_b = float(y_pred.loc[mask_b].eq(str(spec.label_b)).mean()) if n_b > 0 else np.nan
        rows.append(
            {
                "pair_key": str(spec.key),
                "parent_axis_label": str(spec.parent_label),
                "label_a": str(spec.label_a),
                "label_b": str(spec.label_b),
                "n_query": n_query,
                "n_label_a": n_a,
                "n_label_b": n_b,
                "n_correct": int(correct.sum()),
                "pair_accuracy": float(correct.mean()),
                "recall_label_a": recall_a,
                "recall_label_b": recall_b,
                "balanced_accuracy": float(np.nanmean([recall_a, recall_b])),
                "majority_class_accuracy": float(max(n_a, n_b) / max(n_query, 1)),
                "out_of_pair_prediction_rate": float(out_of_pair.mean()),
                "parent_axis_prediction_rate": float(parent_axis.mean()),
            }
        )

    by_pair = pd.DataFrame(rows)
    if not selected_union_masks:
        return by_pair, pd.DataFrame()

    union_mask = selected_union_masks[0].copy()
    for mask in selected_union_masks[1:]:
        union_mask = union_mask | mask
    n_query = int(union_mask.sum())
    if n_query == 0:
        return by_pair, pd.DataFrame(
            [
                {
                    "n_query": 0,
                    "n_correct": 0,
                    "pair_accuracy": np.nan,
                    "macro_pair_accuracy": np.nan,
                    "macro_pair_balanced_accuracy": np.nan,
                    "macro_label_recall": np.nan,
                    "out_of_pair_prediction_rate": np.nan,
                    "parent_axis_prediction_rate": np.nan,
                }
            ]
        )

    y_true = obs.loc[union_mask, label_col].astype(str)
    y_pred = obs.loc[union_mask, fine_pred_col].astype(str)
    collapsed_pred = obs.loc[union_mask, collapsed_pred_col].astype(str)
    true_to_allowed = {str(child): set(spec.children) for spec in branch_specs for child in spec.children}
    true_to_parent = {str(child): str(spec.parent_label) for spec in branch_specs for child in spec.children}
    correct_mask = pd.Series(False, index=y_true.index)
    out_of_pair_mask = pd.Series(False, index=y_true.index)
    parent_axis_mask = pd.Series(False, index=y_true.index)
    for cell_id in y_true.index:
        true_label = str(y_true.loc[cell_id])
        pred_label = str(y_pred.loc[cell_id])
        correct_mask.loc[cell_id] = pred_label == true_label
        out_of_pair_mask.loc[cell_id] = pred_label not in true_to_allowed[true_label]
        parent_axis_mask.loc[cell_id] = str(collapsed_pred.loc[cell_id]) == true_to_parent[true_label]
    overall = pd.DataFrame(
        [
            {
                "n_query": n_query,
                "n_correct": int(correct_mask.sum()),
                "pair_accuracy": float(correct_mask.mean()),
                "macro_pair_accuracy": float(by_pair["pair_accuracy"].mean()) if not by_pair.empty else np.nan,
                "macro_pair_balanced_accuracy": float(by_pair["balanced_accuracy"].mean()) if not by_pair.empty else np.nan,
                "macro_label_recall": float(
                    np.nanmean(
                        np.r_[by_pair["recall_label_a"].to_numpy(dtype=float), by_pair["recall_label_b"].to_numpy(dtype=float)]
                    )
                ) if not by_pair.empty else np.nan,
                "out_of_pair_prediction_rate": float(out_of_pair_mask.mean()),
                "parent_axis_prediction_rate": float(parent_axis_mask.mean()),
            }
        ]
    )
    return by_pair, overall


def build_analysis_coarse_fallback_predictions(
    soft: pd.DataFrame,
    *,
    partial_label_spec: dict[str, Sequence[str]],
    branch_top_child_conf_threshold: float = 0.6,
) -> pd.DataFrame:
    soft = soft.copy()
    fine_pred = soft.idxmax(axis=1).astype(str)
    child_to_parent = {
        str(child): str(parent)
        for parent, children in partial_label_spec.items()
        for child in children
    }
    rows: list[dict[str, Any]] = []
    for cell_id in soft.index.astype(str):
        pred_label = str(fine_pred.loc[cell_id])
        fallback_pred = pred_label
        branch_parent = child_to_parent.get(pred_label)
        branch_conf = np.nan
        used_fallback = False
        if branch_parent is not None:
            child_probs = soft.loc[cell_id, list(partial_label_spec[str(branch_parent)])].astype(float)
            denom = float(child_probs.sum())
            if denom > 0:
                branch_conf = float(child_probs.max() / denom)
            else:
                branch_conf = 0.0
            if branch_conf < float(branch_top_child_conf_threshold):
                fallback_pred = str(branch_parent)
                used_fallback = True
        rows.append(
            {
                "cell_id": str(cell_id),
                "fine_pred_label": pred_label,
                "analysis_pred_label": str(fallback_pred),
                "branch_parent_label": str(branch_parent) if branch_parent is not None else pd.NA,
                "branch_top_child_conf": float(branch_conf) if pd.notna(branch_conf) else np.nan,
                "used_coarse_fallback": bool(used_fallback),
            }
        )
    return pd.DataFrame(rows)
