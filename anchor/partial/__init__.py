from __future__ import annotations

from .labels import *
from .metrics import (
    build_analysis_coarse_fallback_predictions,
    compute_partial_collapsed_overall_metrics,
    compute_partial_fine_overall_metrics,
    compute_partial_hidden_pair_fine_accuracy,
)
from .pseudolabels import (
    DEFAULT_HIDDEN_PARENT_ANCHOR_METHOD,
    DEFAULT_HIDDEN_PARENT_ANCHOR_STRATEGY,
    DEFAULT_PARTIAL_FLAT_SELECTION_METHOD,
    DEFAULT_PARTIAL_FLAT_SELECTION_STRATEGY,
    DEFAULT_PARTIAL_HIDDEN_SELECTION_MODE,
    PARENT_POOL_PARTIAL_HIDDEN_SELECTION_MODE,
    PREDSAME_TREE_PARENT_RESCUE_ADAPTIVE_SELECTION_MODE,
    PREDSAME_TREE_PARENT_RESCUE_SELECTION_MODE,
    PREDSAME_TREE_PARENT_RESCUE_TOP30_CAP50_SELECTION_MODE,
    HiddenParentAnchorSelection,
    PartialFlatLeafPseudoLabelSelection,
    PartialHierarchicalPseudoLabelBundle,
    apply_hidden_parent_anchor_obs,
    apply_partial_flat_leaf_pseudolabel_obs,
    apply_partial_hierarchical_pseudolabel_obs,
    build_partial_hierarchical_pseudolabel_bundle,
    select_hidden_parent_anchor_cells,
    select_partial_flat_leaf_pseudolabels,
)

_LAZY_EXPORTS = {
    "AnchorPartialTeacherModel": (".model", "AnchorPartialTeacherModel"),
    "_AnchorPartialTeacherBaseModel": (".model", "_AnchorPartialTeacherBaseModel"),
    "_AnchorPartialTeacherModule": (".module", "_AnchorPartialTeacherModule"),
    "_AnchorPartialTeacherTrainingPlan": (".module", "_AnchorPartialTeacherTrainingPlan"),
    "SMALLCLASS_CE_MODE_OFF": (".module", "SMALLCLASS_CE_MODE_OFF"),
    "SMALLCLASS_CE_MODE_OVERSAMPLE": (".module", "SMALLCLASS_CE_MODE_OVERSAMPLE"),
    "HIDDEN_BALANCE_MODE_KL_PBAR_UNIFORM": (".module", "HIDDEN_BALANCE_MODE_KL_PBAR_UNIFORM"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [name for name in globals() if not name.startswith("_")] + list(_LAZY_EXPORTS)
