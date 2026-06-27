from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import NearestNeighbors

from .bundle import StudentDataBundle, _node_posterior, _softmax_entropy
from .losses import _build_branch_rank_specs
from ..partial import compute_partial_hidden_pair_fine_accuracy

SAFETY_GUARD_POLICY_VERSION = "rare_anchor_loss_guard_v1"
SAFETY_GUARD_MAX_RARE_ANCHOR_LOSS_THRESHOLD = 0.30
SAFETY_GUARD_TOP5_MEAN_RARE_ANCHOR_LOSS_THRESHOLD = 0.15

def _write_classification_report(
    obs: pd.DataFrame,
    results_dir: Path,
    *,
    pred_col: str,
    label_col: str,
) -> pd.DataFrame:
    report = pd.DataFrame(
        classification_report(
            obs[label_col].astype(str),
            obs[pred_col].astype(str),
            output_dict=True,
            zero_division=0,
        )
    ).T
    report.to_csv(results_dir / "classification_report_query.csv")
    return report


def _branch_subset_metrics(
    obs: pd.DataFrame,
    prior_spec: Mapping[str, Any],
    *,
    pred_col: str,
    label_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = ["branch", "n_query", "accuracy", "macro_f1"]
    for branch, desc_idx in prior_spec.get("branch_to_desc_indices", {}).items():
        labels = [prior_spec["fine_classes"][int(i)] for i in desc_idx]
        mask = obs[label_col].astype(str).isin(labels)
        if int(mask.sum()) == 0:
            continue
        rows.append(
            {
                "branch": str(branch),
                "n_query": int(mask.sum()),
                "accuracy": float(
                    accuracy_score(obs.loc[mask, label_col].astype(str), obs.loc[mask, pred_col].astype(str))
                ),
                "macro_f1": float(
                    f1_score(
                        obs.loc[mask, label_col].astype(str),
                        obs.loc[mask, pred_col].astype(str),
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("branch", kind="mergesort").reset_index(drop=True)

def evaluate_and_write_student(
    bundle: StudentDataBundle,
    *,
    soft: pd.DataFrame,
    u_student: np.ndarray,
    z_recon: np.ndarray,
    protein_recon: np.ndarray,
    pseudo_df: pd.DataFrame,
    results_dir: Path,
) -> dict[str, Any]:
    pred = soft.idxmax(axis=1).astype(str)
    conf = soft.max(axis=1).astype(float)
    entropy = _softmax_entropy(soft)
    soft.to_csv(results_dir / "student_soft_probs.csv")
    obs = bundle.query.obs.copy()
    obs["ref_query_col"] = "query"
    obs["teacher_pred_label"] = bundle.teacher_pred.reindex(bundle.query_index).astype(str).to_numpy()
    obs["teacher_confidence"] = bundle.teacher_confidence.reindex(bundle.query_index).astype(float).to_numpy()
    obs["student_pred_label"] = pred.reindex(bundle.query_index).astype(str).to_numpy()
    obs["student_confidence"] = conf.reindex(bundle.query_index).astype(float).to_numpy()
    obs["student_entropy"] = entropy.reindex(bundle.query_index).astype(float).to_numpy()
    obs["student_correct"] = np.where(
        obs["student_pred_label"].astype(str).eq(obs["true_label"].astype(str)),
        "correct",
        "incorrect",
    )
    pseudo_label = pd.Series("Not selected", index=bundle.query_index, dtype=object)
    if not pseudo_df.empty:
        used = pseudo_df.loc[pseudo_df["used_for_training"].astype(bool)].drop_duplicates("cell_id", keep="first")
        pseudo_label.loc[pseudo_label.index.intersection(used["cell_id"].astype(str))] = (
            used.set_index(used["cell_id"].astype(str))["target_label"].astype(str).reindex(pseudo_label.index).dropna()
        )
    obs["student_pseudo_label"] = pseudo_label.reindex(bundle.query_index).astype(str).to_numpy()
    summary = pd.Series(
        {
            "query_accuracy": accuracy_score(obs["true_label"].astype(str), obs["student_pred_label"].astype(str)),
            "query_macro_f1": f1_score(
                obs["true_label"].astype(str),
                obs["student_pred_label"].astype(str),
                average="macro",
                zero_division=0,
            ),
            "query_weighted_f1": f1_score(
                obs["true_label"].astype(str),
                obs["student_pred_label"].astype(str),
                average="weighted",
                zero_division=0,
            ),
            "mean_confidence_query": float(obs["student_confidence"].mean()),
            "mean_entropy_query": float(obs["student_entropy"].mean()),
        }
    )
    summary.to_frame().T.to_csv(results_dir / "student_summary_metrics.csv", index=False)
    _write_classification_report(obs, results_dir, pred_col="student_pred_label", label_col="true_label")
    branch_df = _branch_subset_metrics(obs, bundle.prior_spec, pred_col="student_pred_label", label_col="true_label")
    branch_df.to_csv(results_dir / "student_branch_subset_metrics.csv", index=False)
    pair_by, pair_overall = compute_partial_hidden_pair_fine_accuracy(
        obs,
        partial_label_spec=bundle.partial_label_spec,
        fine_pred_col="student_pred_label",
        collapsed_pred_col="student_pred_label",
        label_col="true_label",
        split_col="ref_query_col",
        query_name="query",
    )
    pair_by.to_csv(results_dir / "student_hidden_pair_accuracy.csv", index=False)
    pair_overall.to_csv(results_dir / "student_hidden_pair_overall.csv", index=False)
    report = pd.DataFrame(
        classification_report(
            obs["true_label"].astype(str),
            obs["student_pred_label"].astype(str),
            output_dict=True,
            zero_division=0,
        )
    ).T
    report.to_csv(results_dir / "student_classification_report_query.csv")

    result = ad.AnnData(X=np.zeros((len(bundle.query_index), 0), dtype=np.float32), obs=obs)
    result.obsm["u_student"] = u_student.astype(np.float32)
    result.obsm["z_teacher"] = bundle.z_teacher_raw.astype(np.float32)
    result.obsm["z_recon_scaled"] = z_recon.astype(np.float32)
    result.obsm["protein_recon_scaled"] = protein_recon.astype(np.float32)
    result.write_h5ad(results_dir / "student_results.h5ad")

    recon_rows = []
    z_corr = np.corrcoef(bundle.z_teacher.reshape(-1), z_recon.reshape(-1))[0, 1]
    recon_rows.append({"target": "z_teacher", "pearson_flat": float(z_corr)})
    for j, marker in enumerate(bundle.protein_names):
        obs_vals = bundle.protein_features[:, j]
        rec_vals = protein_recon[:, j]
        if np.nanstd(obs_vals) < 1e-8 or np.nanstd(rec_vals) < 1e-8:
            corr = np.nan
        else:
            corr = float(np.corrcoef(obs_vals, rec_vals)[0, 1])
        recon_rows.append({"target": f"protein::{marker}", "pearson_flat": corr})
    recon_df = pd.DataFrame(recon_rows)
    recon_df.to_csv(results_dir / "student_reconstruction_summary.csv", index=False)
    rank_summary = _student_marker_rank_summary(bundle, soft)
    rank_summary.to_csv(results_dir / "student_marker_rank_summary.csv", index=False)
    return {
        "obs": obs,
        "summary": summary,
        "branch_subset_metrics": branch_df,
        "hidden_pair_accuracy": pair_by,
        "hidden_pair_overall": pair_overall,
        "classification_report": report,
        "reconstruction_summary": recon_df,
        "marker_rank_summary": rank_summary,
    }


def _student_marker_rank_summary(bundle: StudentDataBundle, soft: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rank_spec in _build_branch_rank_specs(bundle):
        branch = str(rank_spec["branch"])
        parent_mass = _node_posterior(soft, branch, prior_spec=bundle.prior_spec, label_names=bundle.label_names)
        for child_spec in rank_spec.get("children", []):
            child = str(child_spec.get("child"))
            label_idx = int(child_spec.get("label_idx", -1))
            if label_idx < 0 or label_idx >= len(bundle.label_names):
                continue
            score_label = str(bundle.label_names[label_idx])
            child_mass = _node_posterior(soft, child, prior_spec=bundle.prior_spec, label_names=bundle.label_names)
            child_cond = child_mass / parent_mass.clip(lower=1e-8)
            score = bundle.target_scores[score_label].reindex(bundle.query_index).replace([np.inf, -np.inf], np.nan)
            valid = score.notna() & child_cond.notna() & parent_mass.gt(0.05)
            if int(valid.sum()) < 3:
                continue
            score_v = score.loc[valid].astype(float)
            prob_v = child_cond.loc[valid].astype(float)
            rows.append(
                {
                    "branch": branch,
                    "child": child,
                    "score_label": score_label,
                    "n_cells": int(valid.sum()),
                    "pearson_score_childprob": float(score_v.corr(prob_v, method="pearson")),
                    "spearman_score_childprob": float(score_v.corr(prob_v, method="spearman")),
                    "score_median": float(score_v.median()),
                    "child_prob_median": float(prob_v.median()),
                    "child_prob_q25": float(prob_v.quantile(0.25)),
                    "child_prob_q75": float(prob_v.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)



def _student_knn_purity_summary(
    bundle: StudentDataBundle,
    *,
    u_student: np.ndarray,
    pred: pd.Series,
    results_dir: Path,
    label_col: str = "true_label",
    k: int = 15,
) -> pd.DataFrame:
    """Summarize local-neighborhood purity in student latent space.

    This is a diagnostic only: it should never affect training or prediction.
    Rows include the whole query set, each true label, and tree internal nodes
    with at least two descendant labels when available.
    """
    out_path = Path(results_dir) / "student_knn_purity_summary.csv"
    columns = [
        "axis",
        "n_query",
        "true_knn_purity",
        "pred_knn_purity",
        "pred_neighbor_flip_rate",
    ]
    if u_student.shape[0] <= 1:
        out = pd.DataFrame(columns=columns)
        out.to_csv(out_path, index=False)
        return out

    n_neighbors = min(int(k) + 1, int(u_student.shape[0]))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(u_student)
    neighbors = nn.kneighbors(u_student, return_distance=False)[:, 1:]
    if neighbors.shape[1] == 0:
        out = pd.DataFrame(columns=columns)
        out.to_csv(out_path, index=False)
        return out

    obs = bundle.query.obs.copy()
    obs.index = obs.index.astype(str)
    true_labels = obs[label_col].astype(str).reindex(bundle.query_index).to_numpy()
    pred_labels = pred.reindex(bundle.query_index).astype(str).to_numpy()
    label_set = set(str(x) for x in bundle.label_names)

    def _row(axis: str, mask: np.ndarray) -> dict[str, Any] | None:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return None
        return {
            "axis": axis,
            "n_query": int(idx.size),
            "true_knn_purity": float(np.mean([np.mean(true_labels[neighbors[i]] == true_labels[i]) for i in idx])),
            "pred_knn_purity": float(np.mean([np.mean(pred_labels[neighbors[i]] == pred_labels[i]) for i in idx])),
            "pred_neighbor_flip_rate": float(np.mean([np.mean(pred_labels[neighbors[i]] != pred_labels[i]) for i in idx])),
        }

    rows: list[dict[str, Any]] = []
    all_row = _row("all", np.ones(len(true_labels), dtype=bool))
    if all_row is not None:
        rows.append(all_row)

    for label in sorted(label_set):
        row = _row(f"label::{label}", true_labels == label)
        if row is not None:
            rows.append(row)

    descendants = bundle.prior_spec.get("tree_spec", {}).get("descendants", {})
    for node, leaves in sorted(descendants.items(), key=lambda item: str(item[0])):
        leaf_labels = [str(x) for x in leaves if str(x) in label_set]
        if len(leaf_labels) < 2 or len(leaf_labels) == len(label_set):
            continue
        row = _row(f"node::{node}", np.isin(true_labels, leaf_labels))
        if row is not None:
            rows.append(row)

    out = pd.DataFrame(rows, columns=columns)
    out.to_csv(out_path, index=False)
    return out
