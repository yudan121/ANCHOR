"""Node-wise teacher-weight policy for student conditional KL.

Each internal tree node receives a teacher weight based on protein power,
teacher challenge and RNA protection.  The resulting table is also converted
into conditional-KL specifications consumed by the student training loop.

Hybrid rho v1 shares the same thresholds and three-level rho assignment as the
formal-compatible policy.  Its intentional change is limited to the protein
power / teacher-challenge signal: marker support is computed by direct child
score argmax and top1-vs-top2 margin instead of the formal-compatible binary
GMM split.  RNA protection remains the reference child validation trust times
query teacher-latent RNA structure trust.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from . import student as base_student
from .config import RhoPolicyConfig
from .student.bundle import _detect_teacher_columns

if TYPE_CHECKING:  # pragma: no cover
    from .builder import TeacherDataBundle


DEFAULT_POLICY_PARAMS: dict[str, float | int] = {
    "rna_knn_k": 15,
    "rna_structure_min_child_n": 20,
    "rna_silhouette_sample_max": 2000,
    "rna_knn_purity_low": 0.55,
    "rna_knn_purity_high": 0.80,
    "rna_silhouette_low": 0.00,
    "rna_silhouette_high": 0.20,
    "rna_margin_low": 0.00,
    "rna_margin_high": 0.25,
    "query_rna_knn_weight": 0.45,
    "query_rna_silhouette_weight": 0.35,
    "query_rna_margin_weight": 0.20,
    "min_parent_pool_for_policy": 50,
    "partial_protein_power_threshold": 0.50,
    "strong_protein_power_threshold": 0.75,
    "partial_challenge_threshold": 0.10,
    "strong_challenge_threshold": 0.20,
    "partial_rna_protection_max": 0.70,
    "strong_rna_protection_max": 0.45,
    "partial_release_rho": 0.50,
    "strong_release_rho": 0.10,
}


def _robust_z(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    values = df.to_numpy(dtype=np.float32)
    median = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - median.reshape(1, -1)), axis=0)
    scale = np.where(mad * 1.4826 < eps, 1.0, mad * 1.4826).astype(np.float32)
    z = ((values - median.reshape(1, -1)) / scale.reshape(1, -1)).astype(np.float32)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def _descendant_leaves(prior_spec: Mapping[str, Any], node: str, label_names: Sequence[str]) -> list[str]:
    labels = {str(x) for x in label_names}
    node = str(node)
    if node in labels:
        return [node]
    leaves = [str(x) for x in prior_spec.get("tree_spec", {}).get("descendants", {}).get(node, [])]
    return [leaf for leaf in leaves if leaf in labels]


def _child_for_label(
    label: str,
    *,
    children: Sequence[str],
    child_leaf_sets: Sequence[set[str]],
) -> str | None:
    label = str(label)
    for child, leaves in zip(children, child_leaf_sets, strict=False):
        if label == str(child) or label in leaves:
            return str(child)
    return None


def _score_child_markers(prior_spec: Mapping[str, Any], node: str, child: str, protein_z: pd.DataFrame) -> pd.Series | None:
    child_spec = (
        prior_spec.get("branch_teacher_specs", {})
        .get(str(node), {})
        .get("classes", {})
        .get(str(child), {})
    )
    if not child_spec:
        return None
    positive = child_spec.get("positive", {})
    negative = child_spec.get("negative", {})
    pos = [str(x) for x in (positive.keys() if isinstance(positive, Mapping) else positive)]
    neg = [str(x) for x in (negative.keys() if isinstance(negative, Mapping) else negative)]
    pos = [x for x in pos if x in protein_z.columns]
    neg = [x for x in neg if x in protein_z.columns]
    if not pos and not neg:
        return None
    score = pd.Series(0.0, index=protein_z.index, dtype=float)
    if pos:
        score = score + protein_z.loc[:, pos].mean(axis=1).astype(float)
    if neg:
        score = score - protein_z.loc[:, neg].mean(axis=1).astype(float)
    return score.astype(float)


def _binary_gmm(delta: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    finite = np.isfinite(delta)
    pred = np.full(delta.shape[0], -1, dtype=int)
    if int(finite.sum()) < 20:
        return pred, {
            "score_sep_z": np.nan,
            "cluster_balance": np.nan,
            "minority_cluster_n": np.nan,
            "gmm_low_mean": np.nan,
            "gmm_high_mean": np.nan,
        }
    x = delta[finite].reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=int(seed), covariance_type="full")
    raw = gmm.fit_predict(x)
    means = gmm.means_.reshape(-1)
    order = np.argsort(means)
    mapped = np.zeros_like(raw)
    mapped[raw == order[0]] = 0
    mapped[raw == order[1]] = 1
    pred[np.where(finite)[0]] = mapped
    counts = np.bincount(mapped, minlength=2).astype(float)
    variances = np.asarray(gmm.covariances_).reshape(-1)
    pooled_sd = math.sqrt(float(np.mean(np.maximum(variances, 1e-8))))
    return pred, {
        "score_sep_z": float(abs(means[order[1]] - means[order[0]]) / max(pooled_sd, 1e-8)),
        "cluster_balance": float(counts.min() / max(counts.sum(), 1.0)),
        "minority_cluster_n": float(counts.min()),
        "gmm_low_mean": float(means[order[0]]),
        "gmm_high_mean": float(means[order[1]]),
    }


def _normalized_entropy(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob.astype(np.float64), 1e-12, 1.0)
    denom = max(math.log(float(prob.shape[1])), 1e-8)
    return (-(prob * np.log(prob)).sum(axis=1) / denom).astype(float)


def _scale01(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip((float(value) - float(low)) / max(float(high) - float(low), 1e-8), 0.0, 1.0))


def _rna_trust_floor_from_accuracy(acc: float) -> float:
    if not np.isfinite(acc):
        return 0.0
    return float(np.clip((float(acc) - 0.70) / 0.20, 0.0, 1.0) * 0.85)


def _query_rna_structure_metrics(
    *,
    z_query: np.ndarray | None,
    pool_mask: np.ndarray,
    teacher_child_idx: np.ndarray,
    n_children: int,
    seed: int,
    policy_params: Mapping[str, float | int],
) -> dict[str, float | str]:
    if z_query is None or int(pool_mask.sum()) < int(policy_params["min_parent_pool_for_policy"]):
        return {
            "query_rna_knn_child_purity_v": np.nan,
            "query_rna_silhouette_v": np.nan,
            "query_rna_centroid_margin_v": np.nan,
            "query_rna_structure_trust_v": 0.0,
            "query_rna_structure_source": "unavailable_no_latent_or_small_pool",
        }

    z_pool = z_query[pool_mask]
    y_pool = np.asarray(teacher_child_idx[pool_mask], dtype=int)
    counts = np.bincount(y_pool, minlength=int(n_children))
    min_child_n = int(policy_params["rna_structure_min_child_n"])
    valid_children = np.where(counts >= min_child_n)[0]
    keep = np.isin(y_pool, valid_children)
    if len(valid_children) < 2 or int(keep.sum()) < int(policy_params["min_parent_pool_for_policy"]):
        return {
            "query_rna_knn_child_purity_v": np.nan,
            "query_rna_silhouette_v": np.nan,
            "query_rna_centroid_margin_v": np.nan,
            "query_rna_structure_trust_v": 0.0,
            "query_rna_structure_source": "unavailable_unbalanced_teacher_child_assignments",
        }

    z_use = z_pool[keep]
    y_use = y_pool[keep]
    n = int(z_use.shape[0])
    k = min(int(policy_params["rna_knn_k"]) + 1, n)
    knn_purity = np.nan
    if k >= 3:
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(z_use)
        neighbor_idx = nn.kneighbors(z_use, return_distance=False)[:, 1:]
        knn_purity = float((y_use[neighbor_idx] == y_use.reshape(-1, 1)).mean())

    silhouette = np.nan
    sample_max = int(policy_params["rna_silhouette_sample_max"])
    if len(np.unique(y_use)) >= 2 and n >= 20:
        sample_idx = np.arange(n)
        if n > sample_max:
            rng = np.random.default_rng(int(seed))
            sample_idx = rng.choice(sample_idx, size=sample_max, replace=False)
        sampled_labels = y_use[sample_idx]
        if len(np.unique(sampled_labels)) >= 2:
            try:
                silhouette = float(silhouette_score(z_use[sample_idx], sampled_labels, metric="euclidean"))
            except ValueError:
                silhouette = np.nan

    centroids = []
    child_ids = []
    for child_idx in sorted(np.unique(y_use)):
        mask = y_use == int(child_idx)
        if int(mask.sum()) >= min_child_n:
            centroids.append(np.nanmean(z_use[mask], axis=0))
            child_ids.append(int(child_idx))
    centroid_margin = np.nan
    if len(centroids) >= 2:
        centers = np.vstack(centroids).astype(np.float32)
        dist = np.linalg.norm(z_use[:, None, :] - centers[None, :, :], axis=2)
        assigned_center = np.asarray([child_ids.index(int(x)) for x in y_use], dtype=int)
        own = dist[np.arange(n), assigned_center]
        masked = dist.copy()
        masked[np.arange(n), assigned_center] = np.inf
        second = np.min(masked, axis=1)
        centroid_margin = float(np.nanmedian((second - own) / np.maximum(second, 1e-6)))

    knn_scaled = _scale01(knn_purity, float(policy_params["rna_knn_purity_low"]), float(policy_params["rna_knn_purity_high"]))
    sil_scaled = _scale01(silhouette, float(policy_params["rna_silhouette_low"]), float(policy_params["rna_silhouette_high"]))
    margin_scaled = _scale01(centroid_margin, float(policy_params["rna_margin_low"]), float(policy_params["rna_margin_high"]))
    trust = float(
        np.clip(
            float(policy_params["query_rna_knn_weight"]) * knn_scaled
            + float(policy_params["query_rna_silhouette_weight"]) * sil_scaled
            + float(policy_params["query_rna_margin_weight"]) * margin_scaled,
            0.0,
            1.0,
        )
    )
    return {
        "query_rna_knn_child_purity_v": knn_purity,
        "query_rna_silhouette_v": silhouette,
        "query_rna_centroid_margin_v": centroid_margin,
        "query_rna_structure_trust_v": trust,
        "query_rna_structure_source": "teacher_query_latent_structure",
    }


def _policy_locality_weight(n_descendant_leaves: int, direct_children_all_leaf: bool) -> float:
    if bool(direct_children_all_leaf):
        return 1.0
    return float(np.clip(1.25 / float(max(n_descendant_leaves, 1)), 0.05, 0.75))


def _assign_rho_policy(
    *,
    marker_complete: bool,
    n_parent_pool: int,
    protein_power: float,
    challenge: float,
    rna_protection: float,
    policy_locality_weight: float,
    partial_hidden_node: bool,
    policy_params: Mapping[str, float | int],
) -> tuple[str, float, str]:
    if not marker_complete:
        return "keep_teacher", 1.0, "no_complete_direct_child_marker_spec"
    if int(n_parent_pool) < int(policy_params["min_parent_pool_for_policy"]):
        return "keep_teacher", 1.0, "parent_pool_too_small"
    if protein_power < float(policy_params["partial_protein_power_threshold"]):
        return "keep_teacher", 1.0, "protein_power_below_partial_threshold"
    if challenge < float(policy_params["partial_challenge_threshold"]):
        return "keep_teacher", 1.0, "protein_not_challenging_teacher_enough"
    if rna_protection > float(policy_params["partial_rna_protection_max"]):
        return "keep_teacher", 1.0, "rna_reference_and_query_structure_protect_teacher"

    strong = (
        protein_power >= float(policy_params["strong_protein_power_threshold"])
        and challenge >= float(policy_params["strong_challenge_threshold"])
        and rna_protection <= float(policy_params["strong_rna_protection_max"])
        and (policy_locality_weight >= 0.75 or bool(partial_hidden_node))
    )
    if strong:
        return "strong_release", float(policy_params["strong_release_rho"]), "strong_protein_split_challenges_weak_rna_split"
    return "partial_release", float(policy_params["partial_release_rho"]), "moderate_protein_split_or_upper_node_conservative_release"


def _reference_validation_mask(
    *,
    teacher_bundle: "TeacherDataBundle",
    teacher: ad.AnnData,
    reference_name: str,
) -> tuple[pd.Series, str]:
    obs = teacher.obs
    ref_mask = obs["ref_query_col"].astype(str).eq(str(reference_name)) if "ref_query_col" in obs else pd.Series(False, index=obs.index)
    if teacher_bundle.external_indexing and len(teacher_bundle.external_indexing) >= 2:
        val_idx = np.asarray(teacher_bundle.external_indexing[1], dtype=np.int64)
        valid = pd.Series(False, index=obs.index)
        if teacher_bundle.adata_model.n_obs == teacher.n_obs:
            valid.iloc[val_idx[(val_idx >= 0) & (val_idx < teacher.n_obs)]] = True
        else:
            names = pd.Index(teacher_bundle.adata_model.obs_names.astype(str)[val_idx])
            valid.loc[valid.index.intersection(names)] = True
        return (valid & ref_mask).astype(bool), "reference_validation_external_index"
    return ref_mask.astype(bool), "reference_all_proxy_no_holdout"


def _reference_child_accuracy(
    *,
    teacher: ad.AnnData,
    pred_col: str,
    children: Sequence[str],
    child_leaf_sets: Sequence[set[str]],
    validation_mask: pd.Series,
) -> tuple[float, int, str]:
    if not validation_mask.any() or "true_label" not in teacher.obs:
        return np.nan, 0, "unavailable_no_reference_validation"
    labels = teacher.obs.loc[validation_mask, "true_label"].astype(str)
    true_child = labels.map(lambda x: _child_for_label(x, children=children, child_leaf_sets=child_leaf_sets))
    pred_labels = teacher.obs.loc[validation_mask, pred_col].astype(str)
    pred_child = pred_labels.map(lambda x: _child_for_label(x, children=children, child_leaf_sets=child_leaf_sets))
    keep = true_child.notna() & pred_child.notna()
    if int(keep.sum()) == 0:
        return np.nan, 0, "unavailable_no_child_labeled_validation_cells"
    return float((true_child.loc[keep].astype(str) == pred_child.loc[keep].astype(str)).mean()), int(keep.sum()), "reference_child_validation_accuracy"


def _summarize_rho(node_df: pd.DataFrame) -> pd.DataFrame:
    if node_df.empty:
        return pd.DataFrame(columns=["dataset", "setting", "teacher_source", "release_bin", "n_nodes", "summary_type"])
    continuous = (
        node_df.assign(
            release_bin=lambda x: pd.cut(
                x["rho_v"],
                bins=[-0.01, 0.25, 0.75, 1.01],
                labels=["strong_release", "partial_release", "keep_teacher"],
            )
        )
        .groupby(["dataset", "setting", "teacher_source", "release_bin"], dropna=False, observed=False)
        .size()
        .rename("n_nodes")
        .reset_index()
    )
    continuous["summary_type"] = "continuous_rho_v_bins"
    policy = (
        node_df.groupby(["dataset", "setting", "teacher_source", "rho_policy"], dropna=False, observed=False)
        .size()
        .rename("n_nodes")
        .reset_index()
        .rename(columns={"rho_policy": "release_bin"})
    )
    policy["summary_type"] = "discrete_policy"
    return pd.concat([continuous, policy], ignore_index=True, sort=False)


def compute_rho_policy_audit(
    *,
    bundle: base_student.StudentDataBundle,
    teacher_bundle: "TeacherDataBundle",
    teacher_results_h5ad: str | Path,
    output_dir: str | Path,
    dataset: str,
    setting: str,
    teacher_source: str = "round2",
    reference_name: str = "reference",
    seed: int = 2026,
    parent_pool_threshold: float = 0.20,
    release_strength: float = 1.0,
    policy_params: Mapping[str, float | int] | None = None,
    fail_on_missing_audit: bool = True,
) -> dict[str, Any]:
    """Compute node-wise teacher weights for the student conditional KL.

    The policy combines three node-level signals: protein power, teacher
    challenge and RNA protection.  Protein power and challenge determine how
    strongly protein evidence argues for release, while RNA protection keeps
    the student close to the teacher on branches that remain well supported by
    RNA-derived evidence.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params: dict[str, float | int] = dict(DEFAULT_POLICY_PARAMS)
    if policy_params:
        params.update({str(k): v for k, v in policy_params.items()})

    teacher = ad.read_h5ad(str(teacher_results_h5ad))
    teacher.obs_names_make_unique()
    pred_col, _conf_col, _entropy_col, _latent_key = _detect_teacher_columns(teacher, preferred_prefix=None)
    validation_mask, validation_source = _reference_validation_mask(
        teacher_bundle=teacher_bundle,
        teacher=teacher,
        reference_name=reference_name,
    )
    soft = bundle.teacher_soft.reindex(bundle.query_index).loc[:, bundle.label_names].astype(np.float32)
    protein_z = _robust_z(bundle.protein_arcsinh).reindex(bundle.query_index)
    z_query = np.asarray(bundle.z_teacher, dtype=np.float32) if bundle.z_teacher is not None else None
    partial_nodes = {str(x) for x in bundle.partial_label_spec}
    prior_spec = bundle.prior_spec
    label_names = [str(x) for x in bundle.label_names]
    query_index = bundle.query_index

    children_by_node = {
        str(parent): [str(child) for child in children]
        for parent, children in prior_spec.get("tree_spec", {}).get("children", {}).items()
    }
    node_rows: list[dict[str, Any]] = []
    label_diag_rows: list[dict[str, Any]] = []
    cell_rows: list[pd.DataFrame] = []

    for node, raw_children in sorted(children_by_node.items()):
        children: list[str] = []
        child_leaf_sets: list[set[str]] = []
        child_leaf_lists: list[list[str]] = []
        for child in raw_children:
            leaves = _descendant_leaves(prior_spec, child, label_names)
            if leaves:
                children.append(str(child))
                child_leaf_lists.append(leaves)
                child_leaf_sets.append(set(leaves))
        parent_leaves = _descendant_leaves(prior_spec, node, label_names)
        if len(children) < 2 or not parent_leaves:
            continue

        direct_children_all_leaf = bool(all(str(child) in set(label_names) for child in children))
        parent_mass = soft.loc[:, parent_leaves].sum(axis=1).astype(float)
        parent_pool = parent_mass.ge(float(parent_pool_threshold)).fillna(False)
        pool_mask = parent_pool.to_numpy(dtype=bool)
        n_parent_pool = int(parent_pool.sum())

        child_mass = np.stack([soft.loc[:, leaves].sum(axis=1).to_numpy(dtype=float) for leaves in child_leaf_lists], axis=1)
        child_cond = child_mass / np.clip(parent_mass.to_numpy(dtype=float).reshape(-1, 1), 1e-8, None)
        teacher_child_idx = np.argmax(child_cond, axis=1)
        teacher_child = np.asarray(children, dtype=object)[teacher_child_idx]
        teacher_child_conf = np.max(child_cond, axis=1)
        teacher_child_entropy = _normalized_entropy(child_cond)
        order = np.argsort(child_cond, axis=1)
        teacher_margin = child_cond[np.arange(child_cond.shape[0]), order[:, -1]] - child_cond[np.arange(child_cond.shape[0]), order[:, -2]]

        score_cols: list[pd.Series] = []
        marker_children_available = 0
        for child in children:
            score = _score_child_markers(prior_spec, node, child, protein_z)
            if score is None:
                score = pd.Series(np.nan, index=query_index, dtype=float)
            else:
                marker_children_available += 1
            score_cols.append(score.reindex(query_index).astype(float))
        marker_scores = pd.concat(score_cols, axis=1)
        marker_scores.columns = children
        marker_complete = marker_children_available == len(children)
        marker_coverage = marker_children_available / max(len(children), 1)

        marker_child = np.full(len(query_index), None, dtype=object)
        marker_valid_pool = 0
        median_marker_margin = np.nan
        child_support_min_fraction = np.nan
        protein_power = 0.0
        marker_pred_method = "unavailable"

        if marker_complete and n_parent_pool >= 20:
            pool_scores = marker_scores.loc[parent_pool].to_numpy(dtype=float)
            finite = np.isfinite(pool_scores).all(axis=1)
            marker_valid_pool = int(finite.sum())
            pool_positions = np.where(pool_mask)[0]
            valid_positions = pool_positions[finite]
            if marker_valid_pool >= 20:
                # Hybrid rho v1 intentionally uses the same direct child-score
                # assignment for binary and multi-child branches.  This is the
                # simplified protein-power/challenge signal relative to v1.
                score_order = np.argsort(pool_scores[finite], axis=1)
                argmax = score_order[:, -1]
                top = pool_scores[finite][np.arange(marker_valid_pool), argmax]
                second = pool_scores[finite][np.arange(marker_valid_pool), score_order[:, -2]]
                margins = top - second
                median_marker_margin = float(np.nanmedian(margins))
                marker_child[valid_positions] = np.asarray(children, dtype=object)[argmax]
                counts = np.bincount(argmax, minlength=len(children)).astype(float)
                fractions = counts / max(float(counts.sum()), 1.0)
                child_support_min_fraction = float(fractions.min())
                margin_score = _scale01(median_marker_margin, 0.25, 1.50)
                balance_score = float(np.clip(child_support_min_fraction / max(0.5 / len(children), 1e-8), 0.0, 1.0))
                protein_power = float(np.clip(0.50 * margin_score + 0.50 * balance_score, 0.0, 1.0))
                marker_pred_method = "simplified_score_argmax"

        marker_series = pd.Series(marker_child, index=query_index)
        valid_marker_mask = pool_mask & marker_series.notna().to_numpy()
        marker_teacher_disagreement = (
            float(np.mean(marker_series.loc[valid_marker_mask].astype(str).to_numpy() != teacher_child[valid_marker_mask]))
            if int(valid_marker_mask.sum())
            else 0.0
        )
        teacher_uncertainty = float(np.nanmean(teacher_child_entropy[pool_mask])) if n_parent_pool else 0.0
        teacher_margin_pool = float(np.nanmedian(teacher_margin[pool_mask])) if n_parent_pool else 0.0
        ref_child_acc = np.nan
        ref_child_n = 0
        rna_trust_source = "unavailable_hidden_or_mixed" if str(node) in partial_nodes else "unavailable"
        if str(node) not in partial_nodes:
            ref_child_acc, ref_child_n, rna_trust_source = _reference_child_accuracy(
                teacher=teacher,
                pred_col=pred_col,
                children=children,
                child_leaf_sets=child_leaf_sets,
                validation_mask=validation_mask,
            )
        rna_trust_floor = _rna_trust_floor_from_accuracy(ref_child_acc)
        rna_structure = _query_rna_structure_metrics(
            z_query=z_query,
            pool_mask=pool_mask,
            teacher_child_idx=teacher_child_idx,
            n_children=len(children),
            seed=seed,
            policy_params=params,
        )
        query_rna_structure_trust = float(rna_structure["query_rna_structure_trust_v"])
        # Keep the formal-compatible RNA protection term: trusted reference
        # child validation and clear query RNA structure protect teacher KL.
        rna_protection = float(np.clip(rna_trust_floor * query_rna_structure_trust, 0.0, 1.0))

        n_descendant_leaves = len(parent_leaves)
        locality_weight = float(np.clip(1.25 / float(max(n_descendant_leaves, 1)), 0.05, 1.0))
        release_driver = float(max(marker_teacher_disagreement, teacher_uncertainty))
        release_score = float(np.clip(protein_power * release_driver * locality_weight, 0.0, 1.0))
        rho = float(np.clip(max(rna_protection, 1.0 - float(release_strength) * release_score), 0.0, 1.0))
        if protein_power <= 1e-8:
            rho = 1.0
            release_score = 0.0

        policy_locality = _policy_locality_weight(n_descendant_leaves, direct_children_all_leaf)
        challenge = float(np.clip(protein_power * release_driver, 0.0, 1.0))
        policy, rho_discrete, policy_reason = _assign_rho_policy(
            marker_complete=bool(marker_complete),
            n_parent_pool=int(n_parent_pool),
            protein_power=float(protein_power),
            challenge=float(challenge * policy_locality),
            rna_protection=float(rna_protection),
            policy_locality_weight=float(policy_locality),
            partial_hidden_node=bool(str(node) in partial_nodes),
            policy_params=params,
        )

        row = {
            "dataset": str(dataset),
            "setting": str(setting),
            "teacher_source": str(teacher_source),
            "node": str(node),
            "children": "|".join(children),
            "n_children": int(len(children)),
            "n_descendant_leaves": int(n_descendant_leaves),
            "direct_children_all_leaf": bool(direct_children_all_leaf),
            "parent_pool_threshold": float(parent_pool_threshold),
            "n_parent_pool": int(n_parent_pool),
            "marker_children_available": int(marker_children_available),
            "marker_coverage": float(marker_coverage),
            "marker_complete": bool(marker_complete),
            "marker_pred_method": marker_pred_method,
            "marker_valid_pool": int(marker_valid_pool),
            "score_sep_z": np.nan,
            "cluster_balance": np.nan,
            "minority_cluster_n": np.nan,
            "median_marker_margin": median_marker_margin,
            "child_support_min_fraction": child_support_min_fraction,
            "mean_teacher_child_conf_pool": float(np.nanmean(teacher_child_conf[pool_mask])) if n_parent_pool else np.nan,
            "mean_teacher_child_entropy_pool": float(np.nanmean(teacher_child_entropy[pool_mask])) if n_parent_pool else np.nan,
            "median_teacher_child_margin_pool": teacher_margin_pool if n_parent_pool else np.nan,
            "marker_teacher_disagreement_v": float(marker_teacher_disagreement),
            "teacher_uncertainty_v": float(teacher_uncertainty),
            "protein_power_v": float(protein_power),
            "reference_child_accuracy": ref_child_acc,
            "reference_child_n": int(ref_child_n),
            "reference_validation_source": validation_source,
            "rna_trust_source": rna_trust_source,
            "rna_trust_floor_v": float(rna_trust_floor),
            "query_rna_knn_child_purity_v": float(rna_structure["query_rna_knn_child_purity_v"]),
            "query_rna_silhouette_v": float(rna_structure["query_rna_silhouette_v"]),
            "query_rna_centroid_margin_v": float(rna_structure["query_rna_centroid_margin_v"]),
            "query_rna_structure_trust_v": float(rna_structure["query_rna_structure_trust_v"]),
            "query_rna_structure_source": str(rna_structure["query_rna_structure_source"]),
            "rna_protection_v": float(rna_protection),
            "locality_weight_v": float(locality_weight),
            "policy_locality_weight_v": float(policy_locality),
            "challenge_v": float(challenge),
            "policy_challenge_v": float(challenge * policy_locality),
            "release_score_v": float(release_score),
            "rho_v": float(rho),
            "rho_policy": policy,
            "rho_discrete_recommended": float(rho_discrete),
            "rho_training_candidate_v": float(rho_discrete),
            "rho_policy_reason": policy_reason,
            "teacher_latent_key": "student_bundle_z_teacher",
            "release_strength": float(release_strength),
            "partial_hidden_node": bool(str(node) in partial_nodes),
            "rho_policy_variant": "hybrid_rho_v1",
        }
        node_rows.append(row)

        if n_parent_pool:
            cell_rows.append(
                pd.DataFrame(
                    {
                        "dataset": str(dataset),
                        "setting": str(setting),
                        "teacher_source": str(teacher_source),
                        "node": str(node),
                        "cell_id": query_index[pool_mask],
                        "teacher_parent_prob": parent_mass.loc[parent_pool].to_numpy(dtype=float),
                        "teacher_child": teacher_child[pool_mask],
                        "teacher_child_confidence": teacher_child_conf[pool_mask],
                        "teacher_child_entropy": teacher_child_entropy[pool_mask],
                        "marker_child": marker_series.loc[parent_pool].astype(object).to_numpy(),
                        "marker_teacher_disagreement": marker_series.loc[parent_pool].astype(str).to_numpy() != teacher_child[pool_mask],
                    }
                )
            )

        if "true_label" in bundle.query.obs:
            q_label = bundle.query.obs["true_label"].astype(str).reindex(query_index)
            true_child = q_label.map(lambda x: _child_for_label(x, children=children, child_leaf_sets=child_leaf_sets))
            diag_mask = parent_pool & true_child.notna()
            teacher_series = pd.Series(teacher_child, index=query_index)
            marker_keep = diag_mask & marker_series.notna()
            label_diag = dict(row)
            label_diag.update(
                {
                    "query_diag_true_in_node_n": int(diag_mask.sum()),
                    "query_teacher_child_accuracy_diag": float((teacher_series.loc[diag_mask].astype(str) == true_child.loc[diag_mask].astype(str)).mean())
                    if int(diag_mask.sum())
                    else np.nan,
                    "query_marker_child_accuracy_diag": float((marker_series.loc[marker_keep].astype(str) == true_child.loc[marker_keep].astype(str)).mean())
                    if int(marker_keep.sum())
                    else np.nan,
                    "query_marker_minus_teacher_accuracy_diag": np.nan,
                }
            )
            if np.isfinite(label_diag["query_teacher_child_accuracy_diag"]) and np.isfinite(label_diag["query_marker_child_accuracy_diag"]):
                label_diag["query_marker_minus_teacher_accuracy_diag"] = (
                    float(label_diag["query_marker_child_accuracy_diag"]) - float(label_diag["query_teacher_child_accuracy_diag"])
                )
            label_diag_rows.append(label_diag)

    node_df = pd.DataFrame(node_rows)
    if not node_df.empty:
        node_df = node_df.sort_values(["setting", "rho_v", "node"], kind="mergesort").reset_index(drop=True)
    diag_df = pd.DataFrame(label_diag_rows)
    if not diag_df.empty:
        diag_df = diag_df.sort_values(["setting", "rho_v", "node"], kind="mergesort").reset_index(drop=True)
    cell_df = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    if fail_on_missing_audit and node_df.empty:
        raise ValueError("rho_policy_conditional requested but simplified rho audit produced no node rows")

    paths = {
        "node": output_dir / "rho_node_audit.csv",
        "diag": output_dir / "rho_node_audit_with_label_diag.csv",
        "cell": output_dir / "rho_cell_parent_pool_audit.csv",
        "summary": output_dir / "rho_summary_by_setting.csv",
        "config": output_dir / "rho_audit_config.json",
    }
    summary = _summarize_rho(node_df)
    node_df.to_csv(paths["node"], index=False)
    diag_df.to_csv(paths["diag"], index=False)
    cell_df.to_csv(paths["cell"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paths["config"].write_text(
        json.dumps(
            {
                "policy_variant": "hybrid_rho_v1",
                "seed": int(seed),
                "parent_pool_threshold": float(parent_pool_threshold),
                "release_strength": float(release_strength),
                "formula": "protein_power=0.5*scaled_top1_top2_marker_margin+0.5*scaled_marker_child_balance; challenge=protein_power*max(marker_teacher_disagreement,teacher_entropy)*locality; rna_protection=reference_child_validation_trust*query_teacher_latent_structure_trust",
                "policy_formula": "three-level rho assignment using protein_power, challenge, and rna_protection",
                "policy_params": params,
                "dataset": str(dataset),
                "setting": str(setting),
                "teacher_source": str(teacher_source),
                "teacher_results_h5ad": str(teacher_results_h5ad),
                "partial_coarse_nodes": sorted(partial_nodes),
                "leakage_note": "rho_v uses teacher posterior and query protein marker scores only. Query true labels appear only in rho_node_audit_with_label_diag.csv.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "frames": {"node": node_df, "diag": diag_df, "cell": cell_df, "summary": summary},
        "paths": paths,
        "output_dir": output_dir,
    }


def build_rho_policy_kl_specs_for_bundle(
    bundle: base_student.StudentDataBundle,
    rho_table: pd.DataFrame,
    *,
    fail_on_missing_audit: bool = True,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if fail_on_missing_audit and (rho_table is None or rho_table.empty):
        raise ValueError("rho_policy_conditional requires a non-empty rho audit table")
    specs, table = base_student.build_rho_policy_kl_specs_from_table(
        prior_spec=bundle.prior_spec,
        label_names=bundle.label_names,
        rho_table=rho_table,
        default_rho=1.0,
        rho_col="rho_discrete_recommended",
    )
    if not specs:
        raise ValueError("rho_policy_conditional KL specs are empty")
    if fail_on_missing_audit and table["rho_source_col"].astype(str).eq("default_rho_missing_audit").all():
        raise ValueError("rho audit did not match any KL nodes; refusing all-default keep-teacher rho policy")
    return specs, table


def _rho_table_for_source(
    *,
    node_df: pd.DataFrame,
    dataset: str,
    setting: str,
    teacher_source: str,
) -> pd.DataFrame:
    sub = node_df[
        node_df["dataset"].astype(str).eq(str(dataset))
        & node_df["setting"].astype(str).eq(str(setting))
        & node_df["teacher_source"].astype(str).eq(str(teacher_source))
    ].copy()
    if sub.empty:
        raise ValueError(f"No rho audit rows for dataset={dataset}, setting={setting}, teacher_source={teacher_source}")
    return sub.reset_index(drop=True)


def _rho_specs_for_bundle(bundle: base_student.StudentDataBundle, rho_table: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    return build_rho_policy_kl_specs_for_bundle(bundle, rho_table, fail_on_missing_audit=True)


@dataclass(frozen=True)
class RhoPolicyResult:
    node_table: pd.DataFrame
    kl_specs: list[dict[str, Any]]
    kl_table: pd.DataFrame
    output_dir: Path


def compute_node_rho(
    *,
    bundle: Any,
    teacher_bundle: Any,
    teacher_results_h5ad: str | Path,
    output_dir: str | Path,
    dataset: str,
    config: RhoPolicyConfig | None = None,
    setting: str = "anchor_reorg",
    teacher_source: str = "round2",
    reference_name: str = "reference",
    seed: int = 2026,
) -> RhoPolicyResult:
    cfg = config or RhoPolicyConfig()
    audit = compute_rho_policy_audit(
        bundle=bundle,
        teacher_bundle=teacher_bundle,
        teacher_results_h5ad=teacher_results_h5ad,
        output_dir=output_dir,
        dataset=dataset,
        setting=setting,
        teacher_source=teacher_source,
        reference_name=reference_name,
        seed=seed,
        parent_pool_threshold=cfg.parent_pool_threshold,
        release_strength=cfg.release_strength,
        policy_params=cfg.as_policy_params(),
        fail_on_missing_audit=True,
    )
    node_table = audit["frames"]["node"]
    kl_specs, kl_table = build_rho_policy_kl_specs_for_bundle(
        bundle,
        node_table,
        fail_on_missing_audit=True,
    )
    return RhoPolicyResult(
        node_table=node_table,
        kl_specs=kl_specs,
        kl_table=kl_table,
        output_dir=Path(output_dir),
    )


def build_conditional_kl_specs(bundle: Any, rho_table: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    return build_rho_policy_kl_specs_for_bundle(
        bundle,
        rho_table,
        fail_on_missing_audit=True,
    )


__all__ = [
    "DEFAULT_POLICY_PARAMS",
    "RhoPolicyResult",
    "build_conditional_kl_specs",
    "compute_rho_policy_audit",
    "compute_node_rho",
    "build_rho_policy_kl_specs_for_bundle",
    "_rho_specs_for_bundle",
    "_rho_table_for_source",
]
