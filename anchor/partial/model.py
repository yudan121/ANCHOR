from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import anndata as ad
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager, fields
from scvi.data.fields import LabelsWithUnlabeledObsField

from .labels import (
    HIDDEN_PARENT_ANCHOR_BRANCH_KEY,
    HIDDEN_PARENT_ANCHOR_CHILD_KEY,
    HIDDEN_PARENT_ANCHOR_WEIGHT_KEY,
    PARTIAL_QUERY_PSEUDO_COARSE_TARGET_KEY,
    PARTIAL_QUERY_PSEUDO_COARSE_WEIGHT_KEY,
    PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY,
    PARTIAL_QUERY_PSEUDO_FINE_WEIGHT_KEY,
    PARTIAL_QUERY_PSEUDO_SELECTED_KEY,
    PARTIAL_SUPERVISION_CODE_COL,
)
from .module import _AnchorPartialTeacherModule, _AnchorPartialTeacherTrainingPlan
from ..teacher import _AnchorTeacherBaseModel


class _AnchorPartialTeacherBaseModel(_AnchorTeacherBaseModel):
    """Teacher wrapper for partial-label settings.

    This layer keeps the same totalVI-style teacher backbone as the full-label
    model, but registers partial supervision codes and partial pseudo-label
    fields so the module can compute set-valued supervision losses.
    """

    _module_cls = _AnchorPartialTeacherModule
    _training_plan_cls = _AnchorPartialTeacherTrainingPlan

    def __init__(
        self,
        adata: ad.AnnData,
        *,
        fine_output_labels: Sequence[str],
        supervision_categories: Sequence[str],
        supervision_label_to_desc_indices: dict[str, Sequence[int]],
        n_labels: int | None = None,
        **model_kwargs,
    ):
        self.fine_output_labels = [str(x) for x in fine_output_labels]
        self.supervision_categories = [str(x) for x in supervision_categories]
        self.supervision_label_to_desc_indices = {
            str(label): tuple(int(idx) for idx in desc_indices)
            for label, desc_indices in supervision_label_to_desc_indices.items()
        }
        super().__init__(
            adata,
            n_labels=len(self.fine_output_labels) if n_labels is None else int(n_labels),
            fine_output_labels=self.fine_output_labels,
            supervision_categories=self.supervision_categories,
            supervision_label_to_desc_indices=self.supervision_label_to_desc_indices,
            **model_kwargs,
        )
        self.n_labels = len(self.fine_output_labels)

    def _set_indices_and_labels(self, datamodule=None):
        super()._set_indices_and_labels(datamodule)
        # scvi's label mapping tracks the observed supervision categories, but
        # prediction is still over fine output labels.
        self._supervision_label_mapping = list(self._label_mapping)
        self._label_mapping = list(self.fine_output_labels)
        self._code_to_label = dict(enumerate(self._label_mapping))

    @classmethod
    def setup_anndata(
        cls,
        adata: ad.AnnData,
        protein_expression_obsm_key: str,
        labels_key: str,
        unlabeled_category: str,
        partial_supervision_code_key: str = PARTIAL_SUPERVISION_CODE_COL,
        pseudo_selected_key: str = PARTIAL_QUERY_PSEUDO_SELECTED_KEY,
        pseudo_fine_target_key: str = PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY,
        pseudo_fine_weight_key: str = PARTIAL_QUERY_PSEUDO_FINE_WEIGHT_KEY,
        pseudo_coarse_target_key: str = PARTIAL_QUERY_PSEUDO_COARSE_TARGET_KEY,
        pseudo_coarse_weight_key: str = PARTIAL_QUERY_PSEUDO_COARSE_WEIGHT_KEY,
        hidden_parent_anchor_branch_key: str = HIDDEN_PARENT_ANCHOR_BRANCH_KEY,
        hidden_parent_anchor_child_key: str = HIDDEN_PARENT_ANCHOR_CHILD_KEY,
        hidden_parent_anchor_weight_key: str = HIDDEN_PARENT_ANCHOR_WEIGHT_KEY,
        protein_names_uns_key: str | None = None,
        batch_key: str | None = None,
        panel_key: str | None = None,
        layer: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        if panel_key is not None:
            batch_field = fields.CategoricalObsField("panel", panel_key)
        else:
            batch_field = fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        # Partial-label training needs the usual teacher tensors plus fields
        # for coarse supervision, fine/coarse query pseudo-labels, and optional
        # hidden-parent anchors used by the round0 hidden-branch stage.
        anndata_fields = [
            fields.LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            LabelsWithUnlabeledObsField(REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category),
            fields.CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            fields.NumericalObsField(PARTIAL_SUPERVISION_CODE_COL, partial_supervision_code_key),
            fields.NumericalObsField(PARTIAL_QUERY_PSEUDO_SELECTED_KEY, pseudo_selected_key),
            fields.NumericalObsField(PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY, pseudo_fine_target_key),
            fields.NumericalObsField(PARTIAL_QUERY_PSEUDO_FINE_WEIGHT_KEY, pseudo_fine_weight_key),
            fields.NumericalObsField(PARTIAL_QUERY_PSEUDO_COARSE_TARGET_KEY, pseudo_coarse_target_key),
            fields.NumericalObsField(PARTIAL_QUERY_PSEUDO_COARSE_WEIGHT_KEY, pseudo_coarse_weight_key),
            fields.NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            fields.CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            fields.NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            fields.ProteinObsmField(
                REGISTRY_KEYS.PROTEIN_EXP_KEY,
                protein_expression_obsm_key,
                use_batch_mask=True,
                batch_field=batch_field,
                colnames_uns_key=protein_names_uns_key,
                is_count_data=True,
            ),
        ]
        for registry_key, obs_key in [
            (HIDDEN_PARENT_ANCHOR_BRANCH_KEY, hidden_parent_anchor_branch_key),
            (HIDDEN_PARENT_ANCHOR_CHILD_KEY, hidden_parent_anchor_child_key),
            (HIDDEN_PARENT_ANCHOR_WEIGHT_KEY, hidden_parent_anchor_weight_key),
        ]:
            if obs_key is not None and str(obs_key) in adata.obs:
                anndata_fields.insert(
                    -1,
                    fields.NumericalObsField(registry_key, obs_key),
                )
        if panel_key is not None:
            anndata_fields.insert(0, fields.CategoricalObsField("panel", panel_key))
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)


class AnchorPartialTeacherModel(_AnchorPartialTeacherBaseModel):
    """Public teacher model used for partial-label and hidden-branch stages."""

    _module_cls = _AnchorPartialTeacherModule
    _training_plan_cls = _AnchorPartialTeacherTrainingPlan
