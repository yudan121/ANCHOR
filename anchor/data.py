from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import ColumnMap
from .markers import MarkerTree, load_marker_tree


@dataclass(frozen=True)
class CanonicalInputs:
    reference: Any
    query: Any
    marker_tree: MarkerTree
    reference_path: Path | None = None
    query_path: Path | None = None


def load_inputs(
    reference: str | Path | Any,
    query: str | Path | Any,
    marker_tree: str | Path | Mapping,
) -> CanonicalInputs:
    import anndata as ad

    reference_is_path = isinstance(reference, (str, Path))
    query_is_path = isinstance(query, (str, Path))
    return CanonicalInputs(
        reference=ad.read_h5ad(str(reference)) if reference_is_path else reference,
        query=ad.read_h5ad(str(query)) if query_is_path else query,
        marker_tree=load_marker_tree(marker_tree),
        reference_path=Path(reference) if reference_is_path else None,
        query_path=Path(query) if query_is_path else None,
    )


def _require_obs(adata: Any, key: str | None, label: str) -> None:
    if key is not None and key not in adata.obs.columns:
        raise ValueError(f"{label} is missing obs column `{key}`")


def validate_inputs(
    inputs: CanonicalInputs,
    columns: ColumnMap,
    *,
    counts_layer: str = "counts",
    protein_obsm_key: str = "protein_expression",
    heldout_protein_obsm_key: str = "protein_expression_heldout",
) -> None:
    if counts_layer not in inputs.reference.layers:
        raise ValueError(f"reference is missing layer `{counts_layer}`")
    if counts_layer not in inputs.query.layers:
        raise ValueError(f"query is missing layer `{counts_layer}`")
    if protein_obsm_key not in inputs.query.obsm:
        raise ValueError(f"query is missing obsm `{protein_obsm_key}`")
    if heldout_protein_obsm_key not in inputs.reference.obsm:
        raise ValueError(f"reference is missing obsm `{heldout_protein_obsm_key}`")
    _require_obs(inputs.reference, columns.batch_key, "reference")
    _require_obs(inputs.reference, columns.celltype_key, "reference")
    _require_obs(inputs.reference, columns.split_key, "reference")
    _require_obs(inputs.reference, columns.hidden_branch_key, "reference")
    _require_obs(inputs.query, columns.batch_key, "query")
    _require_obs(inputs.query, columns.query_label_key, "query")
    _require_obs(inputs.query, columns.split_key, "query")
    _require_obs(inputs.query, columns.hidden_branch_key, "query")


def load_anchor_inputs(
    reference: str | Path | Any,
    query: str | Path | Any,
    marker_tree: str | Path | Mapping,
) -> CanonicalInputs:
    return load_inputs(reference=reference, query=query, marker_tree=marker_tree)


def build_anchor_prior(marker_tree: MarkerTree):
    from .builder import build_prior_bundle

    return build_prior_bundle(marker_tree)


def build_teacher_bundle(inputs: CanonicalInputs, config):
    from .builder import build_teacher_data

    return build_teacher_data(inputs, config)


def build_student_bundle(*args, **kwargs):
    from .builder import build_student_data_bundle

    return build_student_data_bundle(*args, **kwargs)


__all__ = [
    "CanonicalInputs",
    "load_inputs",
    "load_anchor_inputs",
    "validate_inputs",
    "build_anchor_prior",
    "build_teacher_bundle",
    "build_student_bundle",
]
