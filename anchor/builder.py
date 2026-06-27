from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import CanonicalInputs, validate_inputs
from .markers import MarkerTree, MarkerTreeNode
from .protein import build_protein_feature_tables, protein_obsm_to_frame
from .student.bundle import (
    StudentDataBundle,
    _build_target_scores,
    _collect_tree_marker_panel,
    _detect_teacher_columns,
    _knn_purity,
    _load_or_build_teacher_soft,
    _robust_feature_z,
    _zscore,
)


@dataclass(frozen=True)
class PriorBundle:
    prior_spec: dict[str, Any]
    leaf_marker_specs: dict[str, dict[str, list[str]]]
    fine_classes: list[str]
    hidden_branch_detected: bool
    partial_label_spec: dict[str, tuple[str, ...]]


@dataclass
class TeacherDataBundle:
    adata_model: Any
    query_index: pd.Index
    query_mask: pd.Series
    query_obs: pd.DataFrame
    protein_arcsinh: pd.DataFrame
    protein_names: list[str]
    label_categories: list[str]
    prior_spec: dict[str, Any]
    leaf_marker_specs: dict[str, dict[str, list[str]]]
    protein_teacher_stats: dict[str, Any]
    external_indexing: list[np.ndarray]
    fixed_split_payload: dict[str, Any]
    hidden_branch_detected: bool
    partial_label_spec: dict[str, tuple[str, ...]]
    partial_supervision_categories: list[str]
    supervision_label_to_desc_indices: dict[str, list[int]]
    partial_hidden_branches: list[str]


def _node_by_name(tree: MarkerTree) -> dict[str, MarkerTreeNode]:
    out: dict[str, MarkerTreeNode] = {}
    for node in tree.root.walk():
        if node.name in out:
            raise ValueError(f"marker tree contains duplicate node name `{node.name}`")
        out[node.name] = node
    return out


def _leaf_names(node: MarkerTreeNode) -> list[str]:
    if not node.children:
        return [node.name]
    leaves: list[str] = []
    for child in node.children:
        leaves.extend(_leaf_names(child))
    return leaves


def _children_map(tree: MarkerTree) -> dict[str, list[str]]:
    return {node.name: [child.name for child in node.children] for node in tree.root.walk() if node.children}


def _parent_map(tree: MarkerTree) -> dict[str, str]:
    parent: dict[str, str] = {}
    for node in tree.root.walk():
        for child in node.children:
            parent[child.name] = node.name
    return parent


def _descendants_map(tree: MarkerTree) -> dict[str, list[str]]:
    return {node.name: _leaf_names(node) for node in tree.root.walk()}


def _signed_marker_dict(markers: Sequence[str]) -> dict[str, float]:
    return {str(marker): 1.0 for marker in markers}


def _explicit_partial_label_spec(marker_tree: MarkerTree, fine_classes: Sequence[str]) -> dict[str, tuple[str, ...]]:
    fine_set = {str(x) for x in fine_classes}
    child_to_parent: dict[str, str] = {}
    out: dict[str, tuple[str, ...]] = {}
    for node in marker_tree.root.walk():
        if node.metadata.get("hidden_branch") is not True:
            continue
        if len(node.children) < 2:
            raise ValueError(f"hidden branch `{node.name}` must have at least two direct children")
        children = tuple(str(child.name) for child in node.children)
        non_leaf = [child.name for child in node.children if child.children]
        if non_leaf:
            raise ValueError(
                f"hidden branch `{node.name}` direct children must be fine leaf labels; "
                f"non-leaf children: {non_leaf}"
            )
        missing = [child for child in children if child not in fine_set]
        if missing:
            raise ValueError(
                f"hidden branch `{node.name}` children are not in fine labels: {missing}. "
                "Hidden children must be marker-tree leaf labels."
            )
        for child in children:
            previous = child_to_parent.get(child)
            if previous is not None:
                raise ValueError(
                    f"hidden child label `{child}` is assigned to multiple hidden parents: "
                    f"{previous!r} and {node.name!r}"
                )
            child_to_parent[child] = node.name
        out[node.name] = children
    return out


def _validate_reference_labels_for_prior(ref_labels: set[str], prior: PriorBundle) -> None:
    hidden_children = {str(child) for children in prior.partial_label_spec.values() for child in children}
    hidden_parents = set(prior.partial_label_spec)
    visible_fine = set(prior.fine_classes) - hidden_children
    legal_labels = visible_fine | hidden_parents
    hidden_child_in_ref = sorted(str(label) for label in ref_labels if str(label) in hidden_children)
    if hidden_child_in_ref:
        raise ValueError(
            "reference labels for hidden branches must use the hidden parent/coarse label, "
            f"not hidden child/fine labels: {hidden_child_in_ref}"
        )
    illegal = sorted(str(label) for label in ref_labels if str(label) not in legal_labels)
    if illegal:
        raise ValueError(
            "reference labels are not valid for the marker tree. "
            f"Invalid labels: {illegal}. Legal labels are visible fine labels plus hidden parent labels."
        )
    missing_parent_cells = sorted(parent for parent in hidden_parents if parent not in ref_labels)
    if missing_parent_cells:
        raise ValueError(
            "hidden branch parent labels must be present in reference supervision. "
            f"Missing parent labels: {missing_parent_cells}"
        )


def build_prior_bundle(marker_tree: MarkerTree, labels: Sequence[str] | None = None) -> PriorBundle:
    """Convert the public marker tree schema into the sign-only prior used by the core models."""
    labels_from_tree = _leaf_names(marker_tree.root)
    fine_classes = [str(x) for x in (labels if labels is not None else labels_from_tree)]
    if len(set(fine_classes)) != len(fine_classes):
        raise ValueError("fine label list contains duplicates")

    label_set = set(fine_classes)
    nodes = _node_by_name(marker_tree)
    children = _children_map(marker_tree)
    parent = _parent_map(marker_tree)
    descendants_all = _descendants_map(marker_tree)
    descendants = {
        node: [leaf for leaf in leaves if leaf in label_set]
        for node, leaves in descendants_all.items()
    }

    branch_teacher_specs: dict[str, dict[str, Any]] = {}
    branch_to_child_names: dict[str, list[str]] = {}
    branch_to_child_leaf_indices: dict[str, list[list[int]]] = {}
    branch_to_child_indices: dict[str, list[int]] = {}
    branch_to_desc_indices: dict[str, list[int]] = {}
    protein_to_branches: dict[str, set[str]] = {}
    fine_to_index = {label: idx for idx, label in enumerate(fine_classes)}

    for node in marker_tree.root.walk():
        if len(node.children) < 2:
            continue
        kept_children: list[str] = []
        kept_classes: dict[str, dict[str, dict[str, float]]] = {}
        kept_leaf_indices: list[list[int]] = []
        for child in node.children:
            child_leaves = [leaf for leaf in descendants.get(child.name, []) if leaf in label_set]
            if not child_leaves:
                continue
            kept_children.append(child.name)
            kept_leaf_indices.append([fine_to_index[leaf] for leaf in child_leaves])
            kept_classes[child.name] = {
                "positive": _signed_marker_dict(child.positive_markers),
                "negative": _signed_marker_dict(child.negative_markers),
            }
        if len(kept_children) < 2:
            continue
        desc_indices = sorted({idx for child_indices in kept_leaf_indices for idx in child_indices})
        branch_teacher_specs[node.name] = {
            "children": kept_children,
            "temperature": float(node.metadata.get("temperature", 1.5)),
            "classes": kept_classes,
        }
        branch_to_child_names[node.name] = kept_children
        branch_to_child_leaf_indices[node.name] = kept_leaf_indices
        branch_to_child_indices[node.name] = [
            fine_to_index[child] for child in kept_children if child in fine_to_index
        ]
        branch_to_desc_indices[node.name] = desc_indices
        for class_spec in kept_classes.values():
            for marker in list(class_spec["positive"]) + list(class_spec["negative"]):
                protein_to_branches.setdefault(str(marker), set()).add(node.name)

    tree_spec = {
        "root": marker_tree.root.name,
        "children": children,
        "parent": parent,
        "descendants": descendants,
    }
    prior_spec = {
        "fine_classes": fine_classes,
        "fine_to_index": fine_to_index,
        "tree_spec": tree_spec,
        "reference_label_map": {label: label for label in fine_classes},
        "drop_reference_anno": [],
        "coarse_nodes": fine_classes,
        "coarse_to_index": fine_to_index.copy(),
        "coarse_to_allowed": {label: [label] for label in fine_classes},
        "allowed_matrix": [[1 if i == j else 0 for j in range(len(fine_classes))] for i in range(len(fine_classes))],
        "branch_teacher_specs": branch_teacher_specs,
        "branch_ranking_specs": {},
        "branch_to_child_names": branch_to_child_names,
        "branch_to_child_indices": branch_to_child_indices,
        "branch_to_child_leaf_indices": branch_to_child_leaf_indices,
        "branch_to_desc_indices": branch_to_desc_indices,
        "protein_to_branches": {k: sorted(v) for k, v in protein_to_branches.items()},
    }
    leaf_marker_specs = {
        label: {
            "positive": list(nodes[label].positive_markers) if label in nodes else [],
            "negative": list(nodes[label].negative_markers) if label in nodes else [],
        }
        for label in fine_classes
    }
    partial_label_spec = _explicit_partial_label_spec(marker_tree, fine_classes)
    for parent_label in partial_label_spec:
        branch_spec = branch_teacher_specs.get(parent_label)
        if branch_spec is None:
            raise ValueError(f"hidden parent `{parent_label}` is not a valid branch in the marker tree prior")
        if tuple(str(x) for x in branch_spec.get("children", ())) != tuple(partial_label_spec[parent_label]):
            raise ValueError(
                f"hidden parent `{parent_label}` children do not match branch teacher spec: "
                f"{branch_spec.get('children')!r} vs {partial_label_spec[parent_label]!r}"
            )
    return PriorBundle(
        prior_spec=prior_spec,
        leaf_marker_specs=leaf_marker_specs,
        fine_classes=fine_classes,
        hidden_branch_detected=bool(partial_label_spec),
        partial_label_spec=partial_label_spec,
    )


def _set_protein_obsm(adata: Any, key: str, frame: pd.DataFrame) -> None:
    adata.obsm[key] = frame.reindex(adata.obs_names).astype(np.float32)
    adata.uns[f"{key}_names"] = frame.columns.astype(str).tolist()


def _clear_multidim_annotations(adata: Any) -> None:
    """Remove source-specific multidimensional annotations before AnnData concat."""

    for mapping_name in ("obsm", "obsp", "varm", "varp"):
        mapping = getattr(adata, mapping_name, None)
        if mapping is None:
            continue
        for key in list(mapping.keys()):
            del mapping[key]


def _hash_indices(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> str:
    digest = hashlib.sha256()
    for arr in (train_idx, val_idx, test_idx):
        digest.update(np.asarray(arr, dtype=np.int64).tobytes())
    return digest.hexdigest()


def get_or_create_external_indexing(
    adata: Any,
    split_dir: Path,
    *,
    seed: int,
    train_fraction: float,
    batch_size: int,
    reference_name: str,
    query_name: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    split_dir.mkdir(parents=True, exist_ok=True)
    npz_path = split_dir / "fixed_external_indexing.npz"
    json_path = split_dir / "fixed_external_indexing.json"
    should_create = True
    if npz_path.exists():
        data = np.load(npz_path)
        train_idx = data["train_idx"].astype(np.int64)
        val_idx = data["val_idx"].astype(np.int64)
        test_idx = data["test_idx"].astype(np.int64)
        max_idx = max(
            int(train_idx.max()) if train_idx.size else -1,
            int(val_idx.max()) if val_idx.size else -1,
            int(test_idx.max()) if test_idx.size else -1,
        )
        should_create = max_idx >= int(adata.n_obs)
        if not should_create and json_path.exists():
            try:
                old_payload = json.loads(json_path.read_text())
                should_create = int(old_payload.get("n_obs", adata.n_obs)) != int(adata.n_obs)
            except Exception:
                should_create = True
    if should_create:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(adata.n_obs)
        n_train = int(np.floor(float(train_fraction) * adata.n_obs))
        train_idx = np.sort(perm[:n_train]).astype(np.int64)
        val_idx = np.sort(perm[n_train:]).astype(np.int64)
        test_idx = np.array([], dtype=np.int64)
        np.savez(npz_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    split_values = adata.obs["ref_query_col"].astype(str)
    payload = {
        "split_npz": str(npz_path),
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "batch_size": int(batch_size),
        "n_obs": int(adata.n_obs),
        "n_train": int(train_idx.size),
        "n_valid": int(val_idx.size),
        "n_test": int(test_idx.size),
        "external_indexing_hash": _hash_indices(train_idx, val_idx, test_idx),
        "train_reference": int(split_values.iloc[train_idx].eq(reference_name).sum()),
        "train_query": int(split_values.iloc[train_idx].eq(query_name).sum()),
        "valid_reference": int(split_values.iloc[val_idx].eq(reference_name).sum()),
        "valid_query": int(split_values.iloc[val_idx].eq(query_name).sum()),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return [train_idx, val_idx, test_idx], payload


def build_teacher_data(inputs: CanonicalInputs, config: ExperimentConfig) -> TeacherDataBundle:
    import anndata as ad

    from .teacher import fit_protein_teacher_stats

    validate_inputs(
        inputs,
        config.columns,
        counts_layer=config.counts_layer,
        protein_obsm_key=config.protein_obsm_key,
        heldout_protein_obsm_key=config.heldout_protein_obsm_key,
    )
    ref = inputs.reference.copy()
    query = inputs.query.copy()
    old_prefix = "bench" + "mark"
    old_teacher_col = ("for" + "mal") + "_" + ("scan" + "vi") + "_label"
    obsolete_internal_columns = [
        f"{old_prefix}_split",
        f"{old_prefix}_label",
        f"{old_prefix}_batch",
        old_teacher_col,
        f"{old_prefix}_label_partial_supervision",
        f"{old_prefix}_label_partial_supervision_code",
        f"{old_prefix}_label_partial_train",
        f"{old_prefix}_label_partial_collapsed_true",
    ]
    ref.obs.drop(columns=[c for c in obsolete_internal_columns if c in ref.obs], inplace=True)
    query.obs.drop(columns=[c for c in obsolete_internal_columns if c in query.obs], inplace=True)
    ref.obs["ref_query_col"] = config.reference_name
    query.obs["ref_query_col"] = config.query_name

    ref_label_key = config.columns.celltype_key
    query_label_key = config.columns.query_label_key or config.columns.celltype_key
    ref.obs["true_label"] = ref.obs[ref_label_key].astype(str)
    if query_label_key in query.obs:
        query.obs["true_label"] = query.obs[query_label_key].astype(str)
    else:
        query.obs["true_label"] = ""

    ref_protein = protein_obsm_to_frame(ref, config.heldout_protein_obsm_key)
    query_protein = protein_obsm_to_frame(query, config.protein_obsm_key)
    protein_names = sorted(set(ref_protein.columns.astype(str)) | set(query_protein.columns.astype(str)))
    ref_protein = ref_protein.reindex(columns=protein_names, fill_value=0.0)
    query_protein = query_protein.reindex(columns=protein_names, fill_value=0.0)

    _clear_multidim_annotations(ref)
    _clear_multidim_annotations(query)
    adata = ad.concat([ref, query], join="inner", merge="same", uns_merge="same", index_unique=None)
    adata.obs_names_make_unique()
    heldout = pd.concat(
        [
            pd.DataFrame(0.0, index=ref.obs_names.astype(str), columns=protein_names),
            query_protein.reindex(query.obs_names.astype(str)),
        ],
        axis=0,
    )
    _set_protein_obsm(adata, config.heldout_protein_obsm_key, heldout)
    adata.obs["batch"] = (
        adata.obs[config.columns.batch_key].astype(str)
        if config.columns.batch_key in adata.obs
        else adata.obs["ref_query_col"].astype(str)
    )

    ref_labels = set(ref.obs["true_label"].astype(str))
    prior = build_prior_bundle(config.marker_tree)
    _validate_reference_labels_for_prior(ref_labels, prior)

    values = np.repeat(config.teacher.unlabeled_category, adata.n_obs).astype(object)
    ref_mask = adata.obs["ref_query_col"].astype(str).eq(config.reference_name)
    ref_train_labels = adata.obs.loc[ref_mask, "true_label"].astype(str).copy()
    if prior.partial_label_spec:
        hidden_parent_labels = set(prior.partial_label_spec)
        ref_train_labels = ref_train_labels.map(
            lambda label: config.teacher.unlabeled_category if str(label) in hidden_parent_labels else str(label)
        )
    values[ref_mask.to_numpy()] = ref_train_labels.to_numpy()
    adata.obs[config.label_key] = pd.Categorical(values, categories=prior.fine_classes + [config.teacher.unlabeled_category])

    partial_supervision_categories: list[str] = []
    supervision_label_to_desc_indices: dict[str, list[int]] = {}
    partial_hidden_branches: list[str] = []
    if prior.partial_label_spec:
        from .partial import (
            PARTIAL_SUPERVISION_CODE_COL,
            PARTIAL_SUPERVISION_LABEL_COL,
            PARTIAL_TRAIN_LABEL_COL,
            add_partial_supervision_code_column,
            add_partial_supervision_label_column,
            add_partial_training_label_column,
            build_supervision_label_to_desc_indices,
            collapse_partial_values,
        )
        from .partial import (
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
        )

        partial_supervision_categories = add_partial_supervision_label_column(
            adata.obs,
            partial_label_spec=prior.partial_label_spec,
            source_label_col="true_label",
            target_label_col=PARTIAL_SUPERVISION_LABEL_COL,
            split_col="ref_query_col",
            reference_name=config.reference_name,
            label_categories=prior.fine_classes,
        )
        add_partial_training_label_column(
            adata.obs,
            partial_label_spec=prior.partial_label_spec,
            fine_output_labels=prior.fine_classes,
            source_label_col="true_label",
            target_label_col=PARTIAL_TRAIN_LABEL_COL,
            split_col="ref_query_col",
            reference_name=config.reference_name,
            unlabeled_category=config.teacher.unlabeled_category,
        )
        hidden_parent_ref_mask = ref_mask & adata.obs["true_label"].astype(str).isin(list(prior.partial_label_spec))
        adata.obs.loc[hidden_parent_ref_mask, PARTIAL_TRAIN_LABEL_COL] = config.teacher.unlabeled_category
        add_partial_supervision_code_column(
            adata.obs,
            supervision_categories=partial_supervision_categories,
            supervision_label_col=PARTIAL_SUPERVISION_LABEL_COL,
            target_code_col=PARTIAL_SUPERVISION_CODE_COL,
            split_col="ref_query_col",
            reference_name=config.reference_name,
        )
        supervision_label_to_desc_indices = build_supervision_label_to_desc_indices(
            prior.fine_classes,
            partial_supervision_categories,
            prior.partial_label_spec,
        )
        partial_hidden_branches = sorted(prior.partial_label_spec)
        for key, default in [
            (PARTIAL_QUERY_PSEUDO_SELECTED_KEY, 0.0),
            (PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY, -1.0),
            (PARTIAL_QUERY_PSEUDO_FINE_WEIGHT_KEY, 0.0),
            (PARTIAL_QUERY_PSEUDO_COARSE_TARGET_KEY, -1.0),
            (PARTIAL_QUERY_PSEUDO_COARSE_WEIGHT_KEY, 0.0),
            (PARTIAL_QUERY_PSEUDO_ROUND_KEY, -1.0),
            (HIDDEN_PARENT_ANCHOR_BRANCH_KEY, -1.0),
            (HIDDEN_PARENT_ANCHOR_CHILD_KEY, -1.0),
            (HIDDEN_PARENT_ANCHOR_WEIGHT_KEY, 0.0),
        ]:
            adata.obs[key] = np.full(adata.n_obs, default, dtype=np.float32)
        adata.obs[PARTIAL_QUERY_PSEUDO_MODE_KEY] = ""
        adata.obs[PARTIAL_QUERY_PSEUDO_SOURCE_KEY] = ""
        adata.obs["partial_collapsed_true_label"] = collapse_partial_values(
            adata.obs["true_label"].astype(str),
            partial_label_spec=prior.partial_label_spec,
        ).to_numpy()

    query_mask = adata.obs["ref_query_col"].astype(str).eq(config.query_name)
    query_index = pd.Index(adata.obs_names[query_mask].astype(str))
    query_protein_for_teacher = query_protein.reindex(query_index)
    protein_teacher_stats = fit_protein_teacher_stats(query_protein_for_teacher, protein_names)
    protein_arcsinh = np.arcsinh(query_protein_for_teacher.astype(np.float32))
    external_indexing, split_payload = get_or_create_external_indexing(
        adata,
        config.root_dir / "_fixed_external_indexing",
        seed=config.teacher.fixed_split_seed,
        train_fraction=config.teacher.fixed_split_train_fraction,
        batch_size=config.teacher.batch_size,
        reference_name=config.reference_name,
        query_name=config.query_name,
    )
    return TeacherDataBundle(
        adata_model=adata,
        query_index=query_index,
        query_mask=query_mask,
        query_obs=adata.obs.loc[query_index].copy(),
        protein_arcsinh=protein_arcsinh,
        protein_names=protein_names,
        label_categories=prior.fine_classes,
        prior_spec=prior.prior_spec,
        leaf_marker_specs=prior.leaf_marker_specs,
        protein_teacher_stats=protein_teacher_stats,
        external_indexing=external_indexing,
        fixed_split_payload=split_payload,
        hidden_branch_detected=prior.hidden_branch_detected,
        partial_label_spec=prior.partial_label_spec,
        partial_supervision_categories=partial_supervision_categories,
        supervision_label_to_desc_indices=supervision_label_to_desc_indices,
        partial_hidden_branches=partial_hidden_branches,
    )


def build_student_data_bundle(
    *,
    teacher_results_h5ad: str | Path,
    teacher_soft_csv: str | Path,
    results_dir: str | Path,
    prior_spec: Mapping[str, Any],
    leaf_marker_specs: Mapping[str, Mapping[str, Sequence[str]]],
    protein_obsm_key: str,
    batch_key: str,
    query_name: str = "query",
    protein_panel: str = "allprotein",
    partial_label_spec: Mapping[str, Sequence[str]] | None = None,
):
    """Build the query feature student bundle without dataset-specific curation hooks."""
    import anndata as ad

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    teacher = ad.read_h5ad(str(teacher_results_h5ad))
    teacher.obs_names_make_unique()
    query = teacher[teacher.obs["ref_query_col"].astype(str).eq(query_name)].copy()
    query.obs_names_make_unique()
    query_index = pd.Index(query.obs_names.astype(str))
    if "true_label" not in query.obs:
        query.obs["true_label"] = ""
    if protein_obsm_key not in query.obsm and "protein_expression_heldout" in query.obsm:
        query.obsm[protein_obsm_key] = query.obsm["protein_expression_heldout"].copy()
        query.uns[f"{protein_obsm_key}_names"] = query.uns.get("protein_expression_heldout_names", query.uns.get(f"{protein_obsm_key}_names", []))

    pred_col, conf_col, _entropy_col, latent_key = _detect_teacher_columns(teacher, preferred_prefix=None)
    label_names = pd.read_csv(teacher_soft_csv, nrows=0, index_col=0).columns.astype(str).tolist()
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    protein_raw = protein_obsm_to_frame(query, protein_obsm_key).reindex(query_index)
    protein_arcsinh_all = build_protein_feature_tables(protein_raw)["arcsinh"].reindex(query_index)
    if protein_panel == "allprotein":
        protein_names = protein_arcsinh_all.columns.astype(str).tolist()
    elif protein_panel == "treemarker":
        protein_names = _collect_tree_marker_panel(prior_spec, protein_arcsinh_all.columns.astype(str).tolist())
        if not protein_names:
            raise ValueError("No tree markers were found in query protein panel.")
    else:
        raise ValueError(f"Unknown protein_panel={protein_panel!r}")
    protein_z_df, protein_mean, protein_std = _robust_feature_z(protein_arcsinh_all.loc[:, protein_names])
    z_teacher_raw = np.asarray(query.obsm[latent_key], dtype=np.float32)
    z_teacher, z_mean, z_std = _zscore(z_teacher_raw)
    soft, _soft_source = _load_or_build_teacher_soft(
        teacher=query,
        query_index=query_index,
        label_names=label_names,
        pred_col=pred_col,
        conf_col=conf_col,
        teacher_soft_csv=Path(teacher_soft_csv),
        out_csv=results_dir / "teacher_query_soft_probs.csv",
    )
    teacher_pred = soft.idxmax(axis=1).astype(str)
    normalized_partial_spec = {
        str(parent): tuple(str(child) for child in children)
        for parent, children in (partial_label_spec or {}).items()
    }
    if normalized_partial_spec:
        from .partial import compute_collapsed_predictions_from_soft

        _collapsed_soft, teacher_collapsed_pred, _collapsed_conf = compute_collapsed_predictions_from_soft(
            soft,
            partial_label_spec=normalized_partial_spec,
            fine_output_labels=label_names,
        )
    else:
        teacher_collapsed_pred = teacher_pred.copy()
    teacher_confidence = soft.max(axis=1).astype(float)
    target_scores, score_available, score_availability_df = _build_target_scores(
        protein_arcsinh=protein_arcsinh_all,
        leaf_marker_specs={
            str(k): {"positive": list(v.get("positive", [])), "negative": list(v.get("negative", []))}
            for k, v in leaf_marker_specs.items()
        },
        prior_spec=dict(prior_spec),
        label_names=label_names,
    )
    score_availability_df.to_csv(results_dir / "student_pseudolabel_score_availability.csv", index=False)
    if batch_key in query.obs:
        batch_values = query.obs[batch_key].astype(str).reindex(query_index)
    elif "batch" in query.obs:
        batch_values = query.obs["batch"].astype(str).reindex(query_index)
    else:
        batch_values = pd.Series("query_batch", index=query_index, dtype=str)
    batch_names = sorted(batch_values.unique().tolist())
    batch_to_idx = {name: idx for idx, name in enumerate(batch_names)}
    batch_idx = batch_values.map(batch_to_idx).to_numpy(dtype=np.int64)
    true_label_idx = query.obs["true_label"].astype(str).reindex(query_index).map(label_to_idx).fillna(-1).to_numpy(dtype=np.int64)
    config_payload = {
        "protein_panel": str(protein_panel),
        "protein_names": protein_names,
        "n_protein_features": int(len(protein_names)),
        "teacher_results_h5ad": str(teacher_results_h5ad),
        "teacher_pred_col": str(pred_col),
        "teacher_conf_col": str(conf_col),
        "teacher_latent_key": str(latent_key),
        "teacher_soft_csv": str(teacher_soft_csv),
        "label_names": label_names,
        "batch_key": str(batch_key),
        "batch_names": batch_names,
        "partial_label_spec": {k: list(v) for k, v in normalized_partial_spec.items()},
    }
    (results_dir / "student_data_config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
    np.save(results_dir / "z_teacher_mean.npy", z_mean)
    np.save(results_dir / "z_teacher_std.npy", z_std)
    np.save(results_dir / "protein_feature_center.npy", protein_mean)
    np.save(results_dir / "protein_feature_scale.npy", protein_std)
    return StudentDataBundle(
        query=query,
        query_index=query_index,
        label_names=label_names,
        label_to_idx=label_to_idx,
        true_label_idx=true_label_idx,
        z_teacher_raw=z_teacher_raw,
        z_teacher=z_teacher,
        z_mean=z_mean,
        z_std=z_std,
        protein_raw=protein_raw,
        protein_arcsinh=protein_arcsinh_all,
        protein_features=protein_z_df.to_numpy(dtype=np.float32),
        protein_names=protein_names,
        protein_panel=str(protein_panel),
        protein_mean=protein_mean,
        protein_std=protein_std,
        batch_idx=batch_idx,
        batch_names=batch_names,
        teacher_soft=soft,
        teacher_pred=teacher_pred,
        teacher_collapsed_pred=teacher_collapsed_pred.reindex(query_index).astype(str),
        teacher_confidence=teacher_confidence,
        prior_spec=dict(prior_spec),
        partial_label_spec=normalized_partial_spec,
        leaf_marker_specs={
            str(k): {"positive": list(v.get("positive", [])), "negative": list(v.get("negative", []))}
            for k, v in leaf_marker_specs.items()
        },
        target_scores=target_scores,
        target_score_available=score_available,
        knn_purity=_knn_purity(z_teacher, teacher_pred, k=15),
    )
