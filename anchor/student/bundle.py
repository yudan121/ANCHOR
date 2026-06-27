from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from ..partial.labels import (
    GENERIC_FLAT_SCORE_MODE,
    build_fine_output_labels,
    compute_flat_leaf_target_score,
)
from ..protein import build_protein_feature_tables, protein_obsm_to_frame


@dataclass
class StudentDataBundle:
    query: ad.AnnData
    query_index: pd.Index
    label_names: list[str]
    label_to_idx: dict[str, int]
    true_label_idx: np.ndarray
    z_teacher_raw: np.ndarray
    z_teacher: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray
    protein_raw: pd.DataFrame
    protein_arcsinh: pd.DataFrame
    protein_features: np.ndarray
    protein_names: list[str]
    protein_panel: str
    protein_mean: np.ndarray
    protein_std: np.ndarray
    batch_idx: np.ndarray
    batch_names: list[str]
    teacher_soft: pd.DataFrame
    teacher_pred: pd.Series
    teacher_collapsed_pred: pd.Series
    teacher_confidence: pd.Series
    prior_spec: dict[str, Any]
    partial_label_spec: dict[str, tuple[str, ...]]
    leaf_marker_specs: dict[str, dict[str, list[str]]]
    target_scores: pd.DataFrame
    target_score_available: pd.Series
    knn_purity: pd.Series


class QueryTeacherStudentDataset(Dataset):
    def __init__(
        self,
        *,
        z: np.ndarray,
        protein: np.ndarray,
        batch_idx: np.ndarray,
        teacher_probs: np.ndarray,
        teacher_confidence: np.ndarray,
        pseudo_target: np.ndarray,
        pseudo_weight: np.ndarray,
        selected_mask: np.ndarray,
        label_score: np.ndarray,
        score_ok: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.z = torch.as_tensor(z, dtype=torch.float32)
        self.protein = torch.as_tensor(protein, dtype=torch.float32)
        self.batch_idx = torch.as_tensor(batch_idx, dtype=torch.long)
        self.teacher_probs = torch.as_tensor(teacher_probs, dtype=torch.float32)
        self.teacher_confidence = torch.as_tensor(teacher_confidence, dtype=torch.float32)
        self.pseudo_target = torch.as_tensor(pseudo_target, dtype=torch.long)
        self.pseudo_weight = torch.as_tensor(pseudo_weight, dtype=torch.float32)
        self.selected_mask = torch.as_tensor(selected_mask, dtype=torch.bool)
        self.label_score = torch.as_tensor(label_score, dtype=torch.float32)
        self.score_ok = torch.as_tensor(score_ok, dtype=torch.bool)
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "z": self.z[idx],
            "protein": self.protein[idx],
            "batch_idx": self.batch_idx[idx],
            "teacher_probs": self.teacher_probs[idx],
            "teacher_confidence": self.teacher_confidence[idx],
            "pseudo_target": self.pseudo_target[idx],
            "pseudo_weight": self.pseudo_weight[idx],
            "selected_mask": self.selected_mask[idx],
            "label_score": self.label_score[idx],
            "score_ok": self.score_ok[idx],
            "index": self.indices[idx],
        }



def _softmax_entropy(probs: pd.DataFrame) -> pd.Series:
    arr = probs.to_numpy(dtype=np.float64)
    return pd.Series(-(arr * np.log(np.clip(arr, 1e-12, None))).sum(axis=1), index=probs.index)


def _zscore(values: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    std = np.where(std < eps, 1.0, std).astype(np.float32)
    scaled = ((values - mean.reshape(1, -1)) / std.reshape(1, -1)).astype(np.float32)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled, mean, std


def _robust_feature_z(df: pd.DataFrame, eps: float = 1e-6) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    values = df.to_numpy(dtype=np.float32)
    median = np.nanmedian(values, axis=0).astype(np.float32)
    mad = np.nanmedian(np.abs(values - median.reshape(1, -1)), axis=0).astype(np.float32)
    scale = np.where(mad * 1.4826 < eps, 1.0, mad * 1.4826).astype(np.float32)
    scaled = ((values - median.reshape(1, -1)) / scale.reshape(1, -1)).astype(np.float32)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns), median, scale


def _detect_teacher_columns(teacher: ad.AnnData, *, preferred_prefix: str | None = None) -> tuple[str, str, str, str]:
    pred_cols = [
        c
        for c in teacher.obs.columns
        if str(c).startswith("pred_") and not str(c).startswith("pred_collapsed_")
    ]
    conf_cols = [c for c in teacher.obs.columns if str(c).startswith("confidence_")]
    entropy_cols = [c for c in teacher.obs.columns if str(c).startswith("entropy_")]
    if preferred_prefix:
        pred = f"pred_{preferred_prefix}"
        conf = f"confidence_{preferred_prefix}"
        entropy = f"entropy_{preferred_prefix}"
        latent = f"X_{preferred_prefix}"
        if pred in teacher.obs and latent in teacher.obsm:
            return pred, conf, entropy, latent
    if not pred_cols:
        raise KeyError("Teacher results h5ad has no pred_* obs column.")
    pred_col = pred_cols[-1]
    suffix = pred_col.removeprefix("pred_")
    conf_col = f"confidence_{suffix}" if f"confidence_{suffix}" in teacher.obs else (conf_cols[-1] if conf_cols else "")
    entropy_col = f"entropy_{suffix}" if f"entropy_{suffix}" in teacher.obs else (entropy_cols[-1] if entropy_cols else "")
    latent_key = f"X_{suffix}" if f"X_{suffix}" in teacher.obsm else list(teacher.obsm.keys())[0]
    return pred_col, conf_col, entropy_col, latent_key


def _infer_student_label_names(
    *,
    teacher: ad.AnnData,
    pred_col: str,
    teacher_soft_csv: Path | None,
) -> tuple[list[str], str]:
    if teacher_soft_csv is not None and Path(teacher_soft_csv).exists():
        columns = pd.read_csv(teacher_soft_csv, nrows=0, index_col=0).columns.astype(str).tolist()
        return build_fine_output_labels(columns), f"teacher_soft_csv_columns:{teacher_soft_csv}"

    pred = teacher.obs[pred_col]
    if hasattr(pred.dtype, "categories"):
        categories = [str(x) for x in pred.dtype.categories]
        return build_fine_output_labels(categories), f"teacher_pred_categories:{pred_col}"

    observed_pred = sorted(pred.astype(str).dropna().unique().tolist())
    if observed_pred:
        return build_fine_output_labels(observed_pred), f"teacher_pred_observed_values:{pred_col}"

    raise ValueError("Could not infer student label names from teacher soft probabilities or teacher predictions.")


def _detect_teacher_collapsed_pred(teacher: ad.AnnData, *, suffix: str, query_index: pd.Index) -> pd.Series:
    preferred = f"collapsed_pred_{suffix}"
    if preferred in teacher.obs:
        return teacher.obs[preferred].astype(str).reindex(query_index)
    preferred = f"pred_collapsed_{suffix}"
    if preferred in teacher.obs:
        return teacher.obs[preferred].astype(str).reindex(query_index)
    cols = [c for c in teacher.obs.columns if str(c).startswith("collapsed_pred_")]
    if cols:
        return teacher.obs[cols[-1]].astype(str).reindex(query_index)
    cols = [c for c in teacher.obs.columns if str(c).startswith("pred_collapsed_")]
    if cols:
        return teacher.obs[cols[-1]].astype(str).reindex(query_index)
    return pd.Series("", index=query_index, dtype=object)


def _load_or_build_teacher_soft(
    *,
    teacher: ad.AnnData,
    query_index: pd.Index,
    label_names: Sequence[str],
    pred_col: str,
    conf_col: str,
    teacher_soft_csv: Path | None,
    out_csv: Path | None,
) -> tuple[pd.DataFrame, str]:
    label_names = [str(x) for x in label_names]
    if teacher_soft_csv is not None and Path(teacher_soft_csv).exists():
        soft = pd.read_csv(teacher_soft_csv, index_col=0)
        soft.index = soft.index.astype(str)
        soft = soft.reindex(query_index).loc[:, label_names].astype(np.float32)
        if out_csv is not None:
            soft.to_csv(out_csv)
        return soft, f"csv:{teacher_soft_csv}"

    n_labels = len(label_names)
    if n_labels < 2:
        raise ValueError("Need at least two labels to build teacher soft targets.")
    pred = teacher.obs[pred_col].astype(str).reindex(query_index)
    conf = (
        teacher.obs[conf_col].astype(float).reindex(query_index)
        if conf_col and conf_col in teacher.obs
        else pd.Series(0.90, index=query_index, dtype=float)
    )
    conf = conf.clip(lower=1.0 / n_labels, upper=0.999)
    arr = np.zeros((len(query_index), n_labels), dtype=np.float32)
    remainder = ((1.0 - conf.to_numpy(dtype=np.float32)) / float(n_labels - 1)).reshape(-1, 1)
    arr[:, :] = remainder
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    for row_idx, label in enumerate(pred.astype(str).tolist()):
        if label in label_to_idx:
            arr[row_idx, label_to_idx[label]] = float(conf.iloc[row_idx])
    arr = arr / np.clip(arr.sum(axis=1, keepdims=True), 1e-8, None)
    soft = pd.DataFrame(arr, index=query_index, columns=label_names)
    if out_csv is not None:
        soft.to_csv(out_csv)
    return soft, "pred_confidence_uniform_remainder"


def _collect_tree_marker_panel(prior_spec: Mapping[str, Any], protein_names: Sequence[str]) -> list[str]:
    available = {str(x) for x in protein_names}
    markers: set[str] = set()
    for branch_spec in prior_spec.get("branch_teacher_specs", {}).values():
        for child_spec in branch_spec.get("classes", {}).values():
            for sign in ("positive", "negative"):
                values = child_spec.get(sign, {})
                if isinstance(values, Mapping):
                    markers.update(str(marker) for marker in values)
                else:
                    markers.update(str(marker) for marker in values)
    return sorted(marker for marker in markers if marker in available)


def _descendant_leaves(node: str, prior_spec: Mapping[str, Any], label_names: Sequence[str]) -> list[str]:
    label_set = {str(x) for x in label_names}
    node = str(node)
    if node in label_set:
        return [node]
    leaves = [str(x) for x in prior_spec.get("tree_spec", {}).get("descendants", {}).get(node, [])]
    return [leaf for leaf in leaves if leaf in label_set]


def _node_posterior(
    soft: pd.DataFrame,
    node: str,
    *,
    prior_spec: Mapping[str, Any],
    label_names: Sequence[str],
) -> pd.Series:
    leaves = [leaf for leaf in _descendant_leaves(node, prior_spec, label_names) if leaf in soft.columns]
    if not leaves:
        return pd.Series(np.nan, index=soft.index, dtype=float)
    return soft.loc[:, leaves].sum(axis=1).astype(float)


def _direct_parent_map(prior_spec: Mapping[str, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in prior_spec.get("tree_spec", {}).get("parent", {}).items()}


def _child_conditional(
    soft: pd.DataFrame,
    child: str,
    parent: str,
    *,
    prior_spec: Mapping[str, Any],
    label_names: Sequence[str],
) -> pd.Series:
    child_mass = _node_posterior(soft, child, prior_spec=prior_spec, label_names=label_names)
    parent_mass = _node_posterior(soft, parent, prior_spec=prior_spec, label_names=label_names)
    return child_mass / parent_mass.clip(lower=1e-8)


def _score_from_signed_spec(
    spec: Mapping[str, Any],
    protein: pd.DataFrame,
) -> pd.Series | None:
    pos_values = spec.get("positive", {}) if isinstance(spec, Mapping) else {}
    neg_values = spec.get("negative", {}) if isinstance(spec, Mapping) else {}
    pos = [str(x) for x in (pos_values.keys() if isinstance(pos_values, Mapping) else pos_values)]
    neg = [str(x) for x in (neg_values.keys() if isinstance(neg_values, Mapping) else neg_values)]
    pos = [marker for marker in pos if marker in protein.columns]
    neg = [marker for marker in neg if marker in protein.columns]
    if not pos and not neg:
        return None
    score = pd.Series(0.0, index=protein.index, dtype=float)
    if pos:
        score = score + protein.loc[:, pos].astype(float).mean(axis=1)
    if neg:
        score = score - protein.loc[:, neg].astype(float).mean(axis=1)
    return score.astype(float)


def _build_target_scores(
    *,
    protein_arcsinh: pd.DataFrame,
    leaf_marker_specs: Mapping[str, Mapping[str, Sequence[str]]],
    prior_spec: Mapping[str, Any],
    label_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    scores: dict[str, pd.Series] = {}
    rows: list[dict[str, Any]] = []
    for label in label_names:
        score, meta = compute_flat_leaf_target_score(
            str(label),
            protein_arcsinh=protein_arcsinh,
            leaf_marker_specs=leaf_marker_specs,
            score_mode=GENERIC_FLAT_SCORE_MODE,
        )
        if score is None:
            score = pd.Series(np.nan, index=protein_arcsinh.index, dtype=float)
        scores[str(label)] = score.reindex(protein_arcsinh.index).astype(float)
        rows.append({"label": str(label), **meta})
    score_df = pd.DataFrame(scores, index=protein_arcsinh.index)
    availability = score_df.notna().any(axis=0)
    return score_df, availability.astype(bool), pd.DataFrame(rows)


def _knn_purity(z_scaled: np.ndarray, labels: pd.Series, *, k: int = 15) -> pd.Series:
    if z_scaled.shape[0] <= 1:
        return pd.Series(1.0, index=labels.index, dtype=float)
    n_neighbors = min(int(k) + 1, z_scaled.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(z_scaled)
    neigh = nn.kneighbors(z_scaled, return_distance=False)
    label_values = labels.astype(str).to_numpy()
    purity = np.zeros(z_scaled.shape[0], dtype=np.float32)
    for i in range(z_scaled.shape[0]):
        nbs = neigh[i, 1:] if neigh.shape[1] > 1 else neigh[i]
        if nbs.size == 0:
            purity[i] = 1.0
        else:
            purity[i] = float(np.mean(label_values[nbs] == label_values[i]))
    return pd.Series(purity, index=labels.index, dtype=float)


_PSEUDO_COLUMNS = [
    "cell_id",
    "target_label",
    "selection_tier",
    "candidate_mode",
    "teacher_pred_label",
    "teacher_confidence",
    "target_posterior",
    "target_score",
    "knn_purity",
    "pseudo_weight",
    "selection_rank",
    "true_label",
    "is_correct_pseudolabel",
    "n_target_labels_per_cell",
    "is_conflict",
    "used_for_training",
]


def _finalize_pseudolabel_rows(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    if raw.empty:
        summary["n_used_for_training"] = 0
        summary["pseudo_precision"] = np.nan
        return pd.DataFrame(columns=_PSEUDO_COLUMNS), summary.sort_values("target_label", kind="mergesort").reset_index(drop=True)
    dedup = (
        raw.sort_values(
            ["cell_id", "pseudo_weight", "teacher_confidence", "target_score", "selection_rank"],
            ascending=[True, False, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["cell_id", "target_label"], keep="first")
        .copy()
    )
    counts = dedup.groupby("cell_id")["target_label"].nunique().rename("n_target_labels_per_cell")
    dedup = dedup.merge(counts.reset_index(), on="cell_id", how="left")
    dedup["is_conflict"] = dedup["n_target_labels_per_cell"].astype(int).gt(1)
    dedup["used_for_training"] = ~dedup["is_conflict"]
    dedup.loc[dedup["is_conflict"], "pseudo_weight"] = 0.0
    used = dedup.loc[dedup["used_for_training"].astype(bool)]
    used_counts = used.groupby("target_label").size().rename("n_used_for_training").reset_index()
    precision = used.groupby("target_label")["is_correct_pseudolabel"].mean().rename("pseudo_precision").reset_index()
    summary = summary.merge(used_counts, on="target_label", how="left").merge(precision, on="target_label", how="left")
    summary["n_used_for_training"] = summary["n_used_for_training"].fillna(0).astype(int)
    return dedup.reset_index(drop=True), summary.sort_values("target_label", kind="mergesort").reset_index(drop=True)


def _build_label_score_arrays(bundle: StudentDataBundle) -> tuple[np.ndarray, np.ndarray]:
    label_score_df = bundle.target_scores.reindex(index=bundle.query_index, columns=bundle.label_names)
    label_score = label_score_df.to_numpy(dtype=np.float32)
    label_score = np.nan_to_num(label_score, nan=-1e6, posinf=-1e6, neginf=-1e6)
    score_ok = np.zeros_like(label_score, dtype=bool)
    for label_idx, label in enumerate(bundle.label_names):
        values = bundle.target_scores[str(label)].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        if values.empty:
            score_ok[:, label_idx] = True
            continue
        threshold = float(values.quantile(0.20))
        score_ok[:, label_idx] = (
            bundle.target_scores[str(label)]
            .reindex(bundle.query_index)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(threshold)
            .to_numpy(dtype=float)
            >= threshold
        )
    return label_score.astype(np.float32), score_ok
