from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

GENERIC_FLAT_SCORE_MODE = "generic_feature"


def _merge_marker_specs(*specs: Mapping[str, Any]) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    merged: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for spec in specs:
        for node, child_map in spec.items():
            node_out = merged.setdefault(str(node), {})
            for child, signed in child_map.items():
                child_out = node_out.setdefault(str(child), {"positive": {}, "negative": {}})
                for sign in ("positive", "negative"):
                    values = signed.get(sign, {}) if isinstance(signed, Mapping) else {}
                    if isinstance(values, Mapping):
                        child_out[sign].update({str(marker): float(weight) for marker, weight in values.items()})
                    else:
                        child_out[sign].update({str(marker): 1.0 for marker in values})
    return merged


def default_alternative_leaf_marker_specs() -> dict[str, dict[str, list[str]]]:
    """Return no built-in markers; marker knowledge should come from the marker tree."""
    return {}


def _align_index(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index.astype(str)
    return out.reindex(index)


def _score_protein_markers(
    label: str,
    protein_arcsinh: pd.DataFrame,
    leaf_marker_specs: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[pd.Series | None, dict[str, Any]]:
    spec = leaf_marker_specs.get(str(label), {})
    pos = [str(marker) for marker in spec.get("positive", [])]
    neg = [str(marker) for marker in spec.get("negative", [])]
    available = set(protein_arcsinh.columns.astype(str))
    pos_avail = [marker for marker in pos if marker in available]
    neg_avail = [marker for marker in neg if marker in available]
    missing = sorted((set(pos) | set(neg)) - available)
    if not pos_avail and not neg_avail:
        return None, {
            "score_available": False,
            "score_type": "protein_marker_mean_pos_minus_neg",
            "positive_markers": "|".join(pos),
            "negative_markers": "|".join(neg),
            "missing_markers": "|".join(missing),
            "parent": str(spec.get("parent", "")),
        }
    score = pd.Series(0.0, index=protein_arcsinh.index, dtype=float)
    if pos_avail:
        score = score + protein_arcsinh.loc[:, pos_avail].astype(float).mean(axis=1)
    if neg_avail:
        score = score - protein_arcsinh.loc[:, neg_avail].astype(float).mean(axis=1)
    return score, {
        "score_available": True,
        "score_type": "protein_marker_mean_pos_minus_neg",
        "positive_markers": "|".join(pos_avail),
        "negative_markers": "|".join(neg_avail),
        "missing_markers": "|".join(missing),
        "parent": str(spec.get("parent", "")),
    }


def compute_flat_leaf_target_score(
    label: str,
    *,
    protein_arcsinh: pd.DataFrame,
    leaf_marker_specs: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    score_mode: str = GENERIC_FLAT_SCORE_MODE,
) -> tuple[pd.Series | None, dict[str, Any]]:
    query_index = pd.Index(protein_arcsinh.index.astype(str))
    leaf_marker_specs = leaf_marker_specs or default_alternative_leaf_marker_specs()
    score_mode = str(score_mode)
    if score_mode != GENERIC_FLAT_SCORE_MODE:
        raise ValueError(f"Unknown flat leaf score_mode={score_mode!r}")
    score, meta = _score_protein_markers(
        label,
        protein_arcsinh,
        leaf_marker_specs,
    )
    meta = dict(meta)
    meta["score_type"] = "generic_feature_mean_pos_minus_neg"
    if score is None:
        meta.setdefault("score_available", False)
        return None, meta
    return score.reindex(query_index).astype(float), meta


# ---- Partial-label utilities ----
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


PARTIAL_SUPERVISION_LABEL_COL = "partial_supervision_label"
PARTIAL_SUPERVISION_CODE_COL = "partial_supervision_code"
PARTIAL_TRAIN_LABEL_COL = "partial_train_label"
PARTIAL_ANALYSIS_COARSE_FALLBACK_COL = "analysis_partial_coarse_fallback_pred"
PARTIAL_ANALYSIS_BRANCH_CONF_COL = "analysis_partial_branch_top_child_conf"

DEFAULT_PARTIAL_BRANCH_CHILDREN: dict[str, tuple[str, ...]] = {}


@dataclass(frozen=True)
class PartialLabelBranchSpec:
    key: str
    parent_label: str
    children: tuple[str, ...]

    @property
    def label_a(self) -> str | None:
        return str(self.children[0]) if len(self.children) >= 1 else None

    @property
    def label_b(self) -> str | None:
        return str(self.children[1]) if len(self.children) >= 2 else None


DEFAULT_PARTIAL_BRANCH_SPECS: tuple[PartialLabelBranchSpec, ...] = tuple(
    PartialLabelBranchSpec(
        key=str(parent_label),
        parent_label=str(parent_label),
        children=tuple(str(x) for x in children),
    )
    for parent_label, children in DEFAULT_PARTIAL_BRANCH_CHILDREN.items()
)


def build_default_partial_label_spec() -> dict[str, tuple[str, ...]]:
    return {str(spec.parent_label): tuple(str(x) for x in spec.children) for spec in DEFAULT_PARTIAL_BRANCH_SPECS}


def build_default_partial_branch_specs() -> tuple[PartialLabelBranchSpec, ...]:
    return DEFAULT_PARTIAL_BRANCH_SPECS


def normalize_partial_branch_specs(
    partial_label_spec: dict[str, Sequence[str]],
) -> list[PartialLabelBranchSpec]:
    default_by_parent = {str(spec.parent_label): spec for spec in DEFAULT_PARTIAL_BRANCH_SPECS}
    out: list[PartialLabelBranchSpec] = []
    for parent_label, children in partial_label_spec.items():
        parent_label = str(parent_label)
        default = default_by_parent.get(parent_label)
        key = default.key if default is not None else parent_label
        out.append(
            PartialLabelBranchSpec(
                key=str(key),
                parent_label=parent_label,
                children=tuple(str(x) for x in children),
            )
        )
    out.sort(key=lambda spec: (spec.parent_label, spec.key))
    return out


def build_fine_output_labels(base_label_categories: Sequence[str]) -> list[str]:
    return [str(x) for x in base_label_categories]


def build_partial_supervision_categories(
    fine_output_labels: Sequence[str],
    partial_label_spec: dict[str, Sequence[str]],
) -> list[str]:
    fine_output_labels = [str(x) for x in fine_output_labels]
    child_to_parent = {
        str(child): str(parent)
        for parent, children in partial_label_spec.items()
        for child in children
    }
    inserted: set[str] = set()
    categories: list[str] = []
    for label in fine_output_labels:
        parent = child_to_parent.get(str(label))
        if parent is not None and parent not in inserted:
            categories.append(parent)
            inserted.add(parent)
        categories.append(str(label))
    return categories


def build_supervision_label_to_desc_indices(
    fine_output_labels: Sequence[str],
    supervision_categories: Sequence[str],
    partial_label_spec: dict[str, Sequence[str]],
) -> dict[str, list[int]]:
    fine_output_labels = [str(x) for x in fine_output_labels]
    fine_to_index = {label: idx for idx, label in enumerate(fine_output_labels)}
    mapping: dict[str, list[int]] = {}
    for label in supervision_categories:
        label = str(label)
        if label in partial_label_spec:
            mapping[label] = [fine_to_index[str(child)] for child in partial_label_spec[label] if str(child) in fine_to_index]
            continue
        if label in fine_to_index:
            mapping[label] = [fine_to_index[label]]
    return mapping


def add_partial_supervision_label_column(
    obs: pd.DataFrame,
    *,
    partial_label_spec: dict[str, Sequence[str]],
    source_label_col: str = "true_label",
    target_label_col: str = PARTIAL_SUPERVISION_LABEL_COL,
    split_col: str = "ref_query_col",
    reference_name: str = "reference",
    label_categories: Sequence[str] | None = None,
) -> list[str]:
    child_to_parent = {
        str(child): str(parent)
        for parent, children in partial_label_spec.items()
        for child in children
    }
    supervision_values = obs[source_label_col].astype(str).copy()
    ref_mask = obs[split_col].astype(str).eq(str(reference_name))
    supervision_values.loc[ref_mask] = supervision_values.loc[ref_mask].map(
        lambda x: child_to_parent.get(str(x), str(x))
    )
    fine_output_labels = build_fine_output_labels(
        label_categories if label_categories is not None else pd.Index(obs[source_label_col].astype(str).unique()).tolist()
    )
    supervision_categories = build_partial_supervision_categories(fine_output_labels, partial_label_spec)
    obs[target_label_col] = pd.Categorical(
        supervision_values.astype(str),
        categories=supervision_categories,
    )
    return supervision_categories


def add_partial_training_label_column(
    obs: pd.DataFrame,
    *,
    partial_label_spec: dict[str, Sequence[str]],
    fine_output_labels: Sequence[str],
    source_label_col: str = "true_label",
    target_label_col: str = PARTIAL_TRAIN_LABEL_COL,
    split_col: str = "ref_query_col",
    reference_name: str = "reference",
    unlabeled_category: str = "Unknown",
) -> None:
    hidden_fine_labels = {str(child) for children in partial_label_spec.values() for child in children}
    ref_mask = obs[split_col].astype(str).eq(str(reference_name))
    values = np.repeat(str(unlabeled_category), obs.shape[0]).astype(object)
    ref_labels = obs.loc[ref_mask, source_label_col].astype(str)
    visible_ref = ref_labels.map(lambda label: str(label) if str(label) not in hidden_fine_labels else str(unlabeled_category))
    values[ref_mask.to_numpy()] = visible_ref.to_numpy(dtype=object)
    obs[target_label_col] = pd.Categorical(
        values,
        categories=[str(label) for label in fine_output_labels] + [str(unlabeled_category)],
    )


def add_partial_supervision_code_column(
    obs: pd.DataFrame,
    *,
    supervision_categories: Sequence[str],
    supervision_label_col: str = PARTIAL_SUPERVISION_LABEL_COL,
    target_code_col: str = PARTIAL_SUPERVISION_CODE_COL,
    split_col: str = "ref_query_col",
    reference_name: str = "reference",
) -> None:
    supervision_categories = [str(x) for x in supervision_categories]
    label_to_code = {label: idx for idx, label in enumerate(supervision_categories)}
    codes = np.full(obs.shape[0], -1, dtype=np.int64)
    ref_mask = obs[split_col].astype(str).eq(str(reference_name))
    codes[ref_mask.to_numpy()] = (
        obs.loc[ref_mask, supervision_label_col].astype(str).map(label_to_code).astype(np.int64).to_numpy()
    )
    obs[target_code_col] = codes


def collapse_partial_values(
    values: Sequence[str] | pd.Series,
    *,
    partial_label_spec: dict[str, Sequence[str]],
) -> pd.Series:
    child_to_parent = {
        str(child): str(parent)
        for parent, children in partial_label_spec.items()
        for child in children
    }
    series = pd.Series(values).astype(str)
    collapsed = series.map(lambda x: child_to_parent.get(str(x), str(x)))
    collapsed.index = getattr(values, "index", series.index)
    return collapsed


def build_collapsed_eval_categories(
    fine_output_labels: Sequence[str],
    *,
    partial_label_spec: dict[str, Sequence[str]],
) -> list[str]:
    fine_output_labels = [str(x) for x in fine_output_labels]
    child_to_parent = {
        str(child): str(parent)
        for parent, children in partial_label_spec.items()
        for child in children
    }
    hidden_fine_labels = set(child_to_parent)
    inserted: set[str] = set()
    categories: list[str] = []
    for label in fine_output_labels:
        parent = child_to_parent.get(str(label))
        if parent is not None and parent not in inserted:
            categories.append(parent)
            inserted.add(parent)
        if str(label) not in hidden_fine_labels:
            categories.append(str(label))
    return categories


def collapse_partial_soft(
    soft: pd.DataFrame,
    *,
    partial_label_spec: dict[str, Sequence[str]],
    fine_output_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    soft = soft.copy()
    if fine_output_labels is None:
        fine_output_labels = [str(x) for x in soft.columns]
    merged_categories = build_collapsed_eval_categories(fine_output_labels, partial_label_spec=partial_label_spec)
    out = pd.DataFrame(index=soft.index)
    for label in merged_categories:
        if str(label) in partial_label_spec:
            out[str(label)] = soft.loc[:, list(partial_label_spec[str(label)])].sum(axis=1).astype(float)
        else:
            out[str(label)] = soft.loc[:, str(label)].astype(float)
    return out


def compute_collapsed_predictions_from_soft(
    soft: pd.DataFrame,
    *,
    partial_label_spec: dict[str, Sequence[str]],
    fine_output_labels: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    collapsed_soft = collapse_partial_soft(
        soft,
        partial_label_spec=partial_label_spec,
        fine_output_labels=fine_output_labels,
    )
    collapsed_pred = collapsed_soft.idxmax(axis=1).astype(str)
    collapsed_confidence = collapsed_soft.max(axis=1).astype(float)
    return collapsed_soft, collapsed_pred, collapsed_confidence




PARTIAL_QUERY_PSEUDO_SELECTED_KEY = "partial_hier_pseudo_selected"
PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY = "partial_hier_pseudo_fine_target"
PARTIAL_QUERY_PSEUDO_FINE_WEIGHT_KEY = "partial_hier_pseudo_fine_weight"
PARTIAL_QUERY_PSEUDO_COARSE_TARGET_KEY = "partial_hier_pseudo_coarse_target"
PARTIAL_QUERY_PSEUDO_COARSE_WEIGHT_KEY = "partial_hier_pseudo_coarse_weight"
PARTIAL_QUERY_PSEUDO_MODE_KEY = "partial_hier_pseudo_mode"
PARTIAL_QUERY_PSEUDO_ROUND_KEY = "partial_hier_pseudo_round"
PARTIAL_QUERY_PSEUDO_SOURCE_KEY = "partial_hier_pseudo_source"
HIDDEN_PARENT_ANCHOR_BRANCH_KEY = "hidden_parent_anchor_branch_code"
HIDDEN_PARENT_ANCHOR_CHILD_KEY = "hidden_parent_anchor_child_code"
HIDDEN_PARENT_ANCHOR_WEIGHT_KEY = "hidden_parent_anchor_weight"
HIDDEN_BALANCE_MODE_MSE = "mse"
HIDDEN_BALANCE_MODE_KL_PBAR_UNIFORM = "kl_pbar_uniform"
HIDDEN_MARKER_RANK_POOL_COLLAPSED_ARGMAX_PARENT = "collapsed_argmax_parent"
HIDDEN_MARKER_RANK_SCORE_FULL = "full"
HIDDEN_MARKER_RANK_SCORE_SIBLING_UNIQUE = "sibling_unique"
