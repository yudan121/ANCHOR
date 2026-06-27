from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager, fields
from scvi.data.fields import LabelsWithUnlabeledObsField
from scvi.module.base import LossOutput

from ..teacher import (
    SMALLCLASS_CE_MODE_OFF,
    SMALLCLASS_CE_MODE_OVERSAMPLE,
    _AnchorTeacherBaseModel,
    _AnchorTeacherModule,
    _AnchorTeacherTrainingPlan,
)
from .labels import (
    HIDDEN_BALANCE_MODE_KL_PBAR_UNIFORM,
    HIDDEN_BALANCE_MODE_MSE,
    HIDDEN_MARKER_RANK_POOL_COLLAPSED_ARGMAX_PARENT,
    HIDDEN_MARKER_RANK_SCORE_FULL,
    HIDDEN_MARKER_RANK_SCORE_SIBLING_UNIQUE,
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


class _AnchorPartialTeacherModule(_AnchorTeacherModule):
    """Teacher module for partial-label and hidden-branch training.

    The backbone remains the same teacher VAE/classifier module.  This module
    replaces full-label reference CE with allowed-descendant supervision, then
    adds optional query pseudo-label and hidden-branch losses depending on the
    current training stage.
    """

    def __init__(
        self,
        *args,
        fine_output_labels: Sequence[str],
        supervision_categories: Sequence[str],
        supervision_label_to_desc_indices: dict[str, Sequence[int]],
        query_pseudolabel_fine_ratio: float = 5.0,
        query_pseudolabel_coarse_ratio: float = 5.0,
        hidden_balance_enable: bool = False,
        hidden_balance_lambda: float = 0.0,
        hidden_balance_branches: Sequence[str] | None = None,
        hidden_balance_mode: str = HIDDEN_BALANCE_MODE_MSE,
        hidden_balance_min_parent_mass: float = 0.0,
        hidden_parent_anchor_ce_enable: bool = False,
        hidden_parent_anchor_ce_lambda: float = 0.0,
        hidden_parent_anchor_branches: Sequence[str] | None = None,
        hidden_marker_rank_enable: bool = False,
        hidden_marker_rank_lambda: float = 0.0,
        hidden_marker_rank_margin: float = 0.1,
        hidden_marker_rank_pool_mode: str = HIDDEN_MARKER_RANK_POOL_COLLAPSED_ARGMAX_PARENT,
        hidden_marker_rank_score_mode: str = HIDDEN_MARKER_RANK_SCORE_FULL,
        hidden_marker_rank_max_pairs_per_child: int = 256,
        hidden_marker_rank_min_pool_size: int = 4,
        hidden_marker_rank_branches: Sequence[str] | None = None,
        smallclass_ce_mode: str = SMALLCLASS_CE_MODE_OFF,
        smallclass_repeat_by_label: Sequence[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fine_output_labels = [str(x) for x in fine_output_labels]
        self.supervision_categories = [str(x) for x in supervision_categories]
        self.supervision_label_to_desc_indices = {
            str(label): tuple(int(idx) for idx in desc_indices)
            for label, desc_indices in supervision_label_to_desc_indices.items()
        }
        self.supervision_code_to_desc_indices = {
            int(code): tuple(self.supervision_label_to_desc_indices[str(label)])
            for code, label in enumerate(self.supervision_categories)
            if str(label) in self.supervision_label_to_desc_indices
        }
        self.query_pseudolabel_fine_ratio = float(query_pseudolabel_fine_ratio)
        self.query_pseudolabel_coarse_ratio = float(query_pseudolabel_coarse_ratio)
        self.hidden_balance_enable = bool(hidden_balance_enable)
        self.hidden_balance_lambda = float(hidden_balance_lambda)
        balance_mode = str(hidden_balance_mode)
        if balance_mode not in {HIDDEN_BALANCE_MODE_MSE, HIDDEN_BALANCE_MODE_KL_PBAR_UNIFORM}:
            raise ValueError(f"Unknown hidden_balance_mode={balance_mode!r}")
        self.hidden_balance_mode = balance_mode
        self.hidden_balance_min_parent_mass = float(hidden_balance_min_parent_mass)
        self.hidden_balance_branches = {
            str(x)
            for x in (
                hidden_balance_branches
                if hidden_balance_branches is not None
                else self.prior_spec.get("branch_teacher_specs", {}).keys()
            )
        }
        anchor_branches = (
            hidden_parent_anchor_branches
            if hidden_parent_anchor_branches is not None
            else self.prior_spec.get("branch_teacher_specs", {}).keys()
        )
        self.hidden_parent_anchor_ce_enable = bool(hidden_parent_anchor_ce_enable)
        self.hidden_parent_anchor_ce_lambda = float(hidden_parent_anchor_ce_lambda)
        self.hidden_parent_anchor_branches = [str(x) for x in anchor_branches]
        self.hidden_parent_anchor_branch_to_code = {
            str(branch): int(code)
            for code, branch in enumerate(self.hidden_parent_anchor_branches)
        }
        marker_rank_branches = (
            hidden_marker_rank_branches
            if hidden_marker_rank_branches is not None
            else self.prior_spec.get("branch_teacher_specs", {}).keys()
        )
        marker_rank_pool_mode = str(hidden_marker_rank_pool_mode)
        if marker_rank_pool_mode != HIDDEN_MARKER_RANK_POOL_COLLAPSED_ARGMAX_PARENT:
            raise ValueError(f"Unknown hidden_marker_rank_pool_mode={marker_rank_pool_mode!r}")
        marker_rank_score_mode = str(hidden_marker_rank_score_mode)
        if marker_rank_score_mode not in {
            HIDDEN_MARKER_RANK_SCORE_FULL,
            HIDDEN_MARKER_RANK_SCORE_SIBLING_UNIQUE,
        }:
            raise ValueError(f"Unknown hidden_marker_rank_score_mode={marker_rank_score_mode!r}")
        self.hidden_marker_rank_enable = bool(hidden_marker_rank_enable)
        self.hidden_marker_rank_lambda = float(hidden_marker_rank_lambda)
        self.hidden_marker_rank_margin = float(hidden_marker_rank_margin)
        self.hidden_marker_rank_pool_mode = marker_rank_pool_mode
        self.hidden_marker_rank_score_mode = marker_rank_score_mode
        self.hidden_marker_rank_max_pairs_per_child = int(hidden_marker_rank_max_pairs_per_child)
        self.hidden_marker_rank_min_pool_size = int(hidden_marker_rank_min_pool_size)
        self.hidden_marker_rank_branches = [str(x) for x in marker_rank_branches]
        mode = str(smallclass_ce_mode)
        if mode not in {SMALLCLASS_CE_MODE_OFF, SMALLCLASS_CE_MODE_OVERSAMPLE}:
            raise ValueError(f"Unknown smallclass_ce_mode={mode!r}")
        self.smallclass_ce_mode = mode
        repeat = list(smallclass_repeat_by_label or [0] * int(self.n_labels))
        if len(repeat) != int(self.n_labels):
            raise ValueError("smallclass_repeat_by_label length must match n_labels")
        self.smallclass_repeat_by_label = torch.as_tensor(repeat, dtype=torch.long)
        self._last_hidden_balance_query_count = 0
        self._last_hidden_parent_anchor_count = 0
        self._last_hidden_parent_anchor_mean_target_prob = 0.0
        self._last_hidden_marker_rank_pair_count = 0
        self._last_hidden_marker_rank_mean_violation = 0.0
        self._last_hidden_marker_rank_mean_score_gap = 0.0
        self._last_query_pseudolabel_smallclass_count = 0

    def _smallclass_repeat_tensor(self, device: torch.device) -> torch.Tensor:
        return self.smallclass_repeat_by_label.to(device=device)

    @staticmethod
    def _zero(device: torch.device) -> torch.Tensor:
        return torch.zeros((), device=device)

    def _hidden_branch_balance_loss(
        self,
        q_c: torch.Tensor,
        tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encourage query mass inside each hidden branch to cover its children.

        This is used in the second part of partial round0.  It is a batch-level
        branch distribution penalty, not a per-cell uniform target.
        """
        zero = self._zero(q_c.device)
        self._last_hidden_balance_query_count = 0
        if not self.hidden_balance_enable or self.hidden_balance_lambda <= 0:
            return zero
        supervision_codes = tensors[PARTIAL_SUPERVISION_CODE_COL].reshape(-1).long()
        query_mask = supervision_codes.lt(0)
        if not bool(query_mask.any()):
            return zero
        eps = 1e-8
        branch_losses: list[torch.Tensor] = []
        total_query_rows = 0
        for branch_name in self.hidden_balance_branches:
            child_leaf_indices = self.prior_spec["branch_to_child_leaf_indices"].get(str(branch_name))
            if not child_leaf_indices or len(child_leaf_indices) <= 1:
                continue
            child_masses = torch.stack(
                [q_c[:, indices].sum(dim=-1) for indices in child_leaf_indices],
                dim=-1,
            )
            gate = child_masses.sum(dim=-1)
            valid = query_mask & gate.gt(eps)
            if not bool(valid.any()):
                continue

            # Batch-level pbar should be parent-gated, but computing it as
            # gate * (child_mass / gate) creates unstable gradients when a cell
            # has near-zero parent mass. Sum child masses directly instead; this
            # is algebraically equivalent to the gated average and avoids the
            # tiny per-cell denominator in the backward pass.
            child_mass_sum = child_masses[valid].sum(dim=0)
            gate_weight_sum = child_mass_sum.sum()
            if gate_weight_sum.detach().item() < float(self.hidden_balance_min_parent_mass):
                continue
            mean_probs = child_mass_sum / gate_weight_sum.clamp_min(eps)
            uniform = torch.full_like(mean_probs, 1.0 / float(mean_probs.numel()))
            if self.hidden_balance_mode == HIDDEN_BALANCE_MODE_KL_PBAR_UNIFORM:
                mean_probs_safe = mean_probs.clamp_min(eps)
                branch_loss = torch.sum(mean_probs_safe * (torch.log(mean_probs_safe) - torch.log(uniform)))
            else:
                branch_loss = torch.sum((mean_probs - uniform) ** 2)
            branch_losses.append(branch_loss)
            total_query_rows += int(valid.sum().item())
        self._last_hidden_balance_query_count = total_query_rows
        if not branch_losses:
            return zero
        return float(self.hidden_balance_lambda) * torch.stack(branch_losses).mean()

    def _hidden_parent_anchor_ce_loss(
        self,
        logits: torch.Tensor,
        tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply child CE within a hidden parent branch for selected anchors.

        Hidden-parent anchors are marker-selected query cells from partial
        round0.  The loss uses P(target child | hidden parent), so it sharpens
        the split inside the parent branch without forcing a global fine label.
        """
        zero = self._zero(logits.device)
        self._last_hidden_parent_anchor_count = 0
        self._last_hidden_parent_anchor_mean_target_prob = 0.0
        if not self.hidden_parent_anchor_ce_enable or self.hidden_parent_anchor_ce_lambda <= 0:
            return zero
        required = [
            HIDDEN_PARENT_ANCHOR_BRANCH_KEY,
            HIDDEN_PARENT_ANCHOR_CHILD_KEY,
            HIDDEN_PARENT_ANCHOR_WEIGHT_KEY,
        ]
        missing = [key for key in required if key not in tensors]
        if missing:
            raise KeyError(
                "hidden_parent_anchor_ce_enable=True, but missing registered tensor keys: "
                f"{missing}. Add the anchor obs columns before setup_anndata."
            )

        branch_codes = tensors[HIDDEN_PARENT_ANCHOR_BRANCH_KEY].reshape(-1).long()
        child_codes = tensors[HIDDEN_PARENT_ANCHOR_CHILD_KEY].reshape(-1).long()
        weights = tensors[HIDDEN_PARENT_ANCHOR_WEIGHT_KEY].reshape(-1).float()
        supervision_codes = tensors[PARTIAL_SUPERVISION_CODE_COL].reshape(-1).long()
        valid_base = branch_codes.ge(0) & child_codes.ge(0) & weights.gt(0.0) & supervision_codes.lt(0)
        if not bool(valid_base.any()):
            return zero

        all_losses: list[torch.Tensor] = []
        all_weights: list[torch.Tensor] = []
        all_target_probs: list[torch.Tensor] = []
        for branch_code, branch_name in enumerate(self.hidden_parent_anchor_branches):
            child_leaf_indices = self.prior_spec["branch_to_child_leaf_indices"].get(str(branch_name))
            if not child_leaf_indices or len(child_leaf_indices) <= 1:
                continue
            row_mask = valid_base & branch_codes.eq(int(branch_code))
            if not bool(row_mask.any()):
                continue
            child_log_masses = torch.stack(
                [torch.logsumexp(logits[:, indices], dim=-1) for indices in child_leaf_indices],
                dim=-1,
            )
            n_children = int(child_log_masses.shape[-1])
            valid = row_mask & child_codes.lt(n_children)
            if not bool(valid.any()):
                continue
            parent_log_mass = torch.logsumexp(child_log_masses[valid], dim=-1)
            target = child_codes[valid]
            target_log_mass = child_log_masses[valid].gather(1, target.unsqueeze(-1)).squeeze(-1)
            # Stable equivalent of -log(sum P(target child leaves) / sum P(parent leaves)).
            # Computing this in logit space avoids NaN gradients from saturated softmax ratios.
            target_log_prob = target_log_mass - parent_log_mass
            all_losses.append(-target_log_prob)
            all_weights.append(weights[valid])
            all_target_probs.append(target_log_prob.detach().exp())

        if not all_losses:
            return zero
        losses = torch.cat(all_losses, dim=0)
        loss_weights = torch.cat(all_weights, dim=0)
        target_probs = torch.cat(all_target_probs, dim=0)
        self._last_hidden_parent_anchor_count = int(losses.numel())
        self._last_hidden_parent_anchor_mean_target_prob = float(target_probs.mean().item())
        return float(self.hidden_parent_anchor_ce_lambda) * self._weighted_mean_from_count(losses, loss_weights)

    def _node_descendant_indices(self, node_name: str) -> tuple[int, ...]:
        node_name = str(node_name)
        tree_spec = self.prior_spec.get("tree_spec", {})
        descendants = tree_spec.get("descendants", {})
        fine_to_index = self.prior_spec.get("fine_to_index", {})
        leaf_names = descendants.get(node_name, [node_name])
        return tuple(
            int(fine_to_index[str(leaf)])
            for leaf in leaf_names
            if str(leaf) in fine_to_index
        )

    def _collapsed_argmax_parent_pool(
        self,
        q_c: torch.Tensor,
        branch_name: str,
    ) -> torch.Tensor:
        tree_spec = self.prior_spec.get("tree_spec", {})
        parent_map = tree_spec.get("parent", {})
        children_map = tree_spec.get("children", {})
        branch_name = str(branch_name)
        parent_name = str(parent_map.get(branch_name, ""))
        if not parent_name:
            return torch.zeros(q_c.shape[0], dtype=torch.bool, device=q_c.device)
        siblings = [str(child) for child in children_map.get(parent_name, [])]
        if branch_name not in siblings:
            return torch.zeros(q_c.shape[0], dtype=torch.bool, device=q_c.device)

        masses: list[torch.Tensor] = []
        kept_siblings: list[str] = []
        for sibling in siblings:
            indices = self._node_descendant_indices(sibling)
            if not indices:
                continue
            masses.append(q_c[:, list(indices)].sum(dim=-1))
            kept_siblings.append(sibling)
        if branch_name not in kept_siblings or len(masses) <= 1:
            return torch.zeros(q_c.shape[0], dtype=torch.bool, device=q_c.device)
        sibling_masses = torch.stack(masses, dim=-1)
        branch_pos = kept_siblings.index(branch_name)
        return sibling_masses.argmax(dim=-1).eq(int(branch_pos))

    def _hidden_marker_rank_child_score(
        self,
        teacher_values: torch.Tensor,
        class_spec: dict[str, Any],
        sibling_class_specs: Sequence[dict[str, Any]] | None = None,
    ) -> torch.Tensor | None:
        pos = [
            str(marker)
            for marker in class_spec.get("positive", {}).keys()
            if str(marker) in self.protein_to_index
        ]
        neg = [
            str(marker)
            for marker in class_spec.get("negative", {}).keys()
            if str(marker) in self.protein_to_index
        ]
        if (
            self.hidden_marker_rank_score_mode == HIDDEN_MARKER_RANK_SCORE_SIBLING_UNIQUE
            and sibling_class_specs
        ):
            sibling_pos = {
                str(marker)
                for sibling_spec in sibling_class_specs
                for marker in sibling_spec.get("positive", {}).keys()
            }
            sibling_neg = {
                str(marker)
                for sibling_spec in sibling_class_specs
                for marker in sibling_spec.get("negative", {}).keys()
            }
            pos = [marker for marker in pos if marker not in sibling_pos]
            neg = [marker for marker in neg if marker not in sibling_neg]
        if not pos and not neg:
            return None
        score = torch.zeros(teacher_values.shape[0], device=teacher_values.device, dtype=teacher_values.dtype)
        if pos:
            pos_idx = [self.protein_to_index[marker] for marker in pos]
            score = score + teacher_values[:, pos_idx].mean(dim=-1)
        if neg:
            neg_idx = [self.protein_to_index[marker] for marker in neg]
            score = score - teacher_values[:, neg_idx].mean(dim=-1)
        return score

    @staticmethod
    def _quantile_rank_1d(values: torch.Tensor) -> torch.Tensor:
        n = int(values.numel())
        if n <= 1:
            return torch.zeros_like(values)
        order = torch.argsort(values)
        ranks = torch.empty_like(values)
        ranks[order] = torch.linspace(0.0, 1.0, n, device=values.device, dtype=values.dtype)
        return ranks

    def _hidden_marker_rank_loss(
        self,
        q_c: torch.Tensor,
        tensors: dict[str, torch.Tensor],
        y: torch.Tensor,
        protein_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        zero = self._zero(q_c.device)
        self._last_hidden_marker_rank_pair_count = 0
        self._last_hidden_marker_rank_mean_violation = 0.0
        self._last_hidden_marker_rank_mean_score_gap = 0.0
        if not self.hidden_marker_rank_enable or self.hidden_marker_rank_lambda <= 0:
            return zero

        supervision_codes = tensors[PARTIAL_SUPERVISION_CODE_COL].reshape(-1).long()
        query_mask = supervision_codes.lt(0)
        if protein_mask is not None:
            observed_mask = protein_mask.sum(dim=-1).gt(0)
        else:
            observed_mask = torch.ones_like(query_mask, dtype=torch.bool)
        base_valid = query_mask & observed_mask
        if not bool(base_valid.any()):
            return zero

        teacher_values = self._protein_teacher_values(y)
        eps = 1e-8
        all_losses: list[torch.Tensor] = []
        all_score_gaps: list[torch.Tensor] = []
        all_violations: list[torch.Tensor] = []
        pair_count = 0
        for branch_name in self.hidden_marker_rank_branches:
            branch_spec = self.prior_spec.get("branch_teacher_specs", {}).get(str(branch_name), {})
            child_leaf_indices = self.prior_spec.get("branch_to_child_leaf_indices", {}).get(str(branch_name))
            children = [str(child) for child in branch_spec.get("children", [])]
            if not child_leaf_indices or len(child_leaf_indices) <= 1 or not children:
                continue

            child_masses = torch.stack(
                [q_c[:, indices].sum(dim=-1) for indices in child_leaf_indices],
                dim=-1,
            )
            parent_mass = child_masses.sum(dim=-1)
            branch_probs = child_masses / parent_mass.unsqueeze(-1).clamp_min(eps)
            branch_pool = (
                base_valid
                & parent_mass.gt(eps)
                & self._collapsed_argmax_parent_pool(q_c, str(branch_name))
            )
            pool_idx = torch.where(branch_pool)[0]
            if int(pool_idx.numel()) < max(2, int(self.hidden_marker_rank_min_pool_size)):
                continue

            for child_pos, child_name in enumerate(children):
                class_spec = branch_spec.get("classes", {}).get(str(child_name), {})
                sibling_specs = [
                    branch_spec.get("classes", {}).get(str(other_child), {})
                    for other_child in children
                    if str(other_child) != str(child_name)
                ]
                score = self._hidden_marker_rank_child_score(
                    teacher_values,
                    class_spec,
                    sibling_class_specs=sibling_specs,
                )
                if score is None:
                    continue
                pool_score = score[pool_idx]
                if int(torch.unique(pool_score.detach()).numel()) <= 1:
                    continue
                score_rank = self._quantile_rank_1d(pool_score.detach())
                n_pool = int(pool_idx.numel())
                n_pairs = min(
                    int(self.hidden_marker_rank_max_pairs_per_child),
                    max(0, n_pool * (n_pool - 1) // 2),
                )
                if n_pairs <= 0:
                    continue
                # Oversample random candidates slightly, then keep ordered non-tie pairs.
                n_draw = max(int(n_pairs * 4), int(n_pairs + 8))
                a_local = torch.randint(0, n_pool, (n_draw,), device=q_c.device)
                b_local = torch.randint(0, n_pool, (n_draw,), device=q_c.device)
                valid_pair = a_local.ne(b_local) & score_rank[a_local].ne(score_rank[b_local])
                if not bool(valid_pair.any()):
                    continue
                a_local = a_local[valid_pair]
                b_local = b_local[valid_pair]
                a_higher = score_rank[a_local].gt(score_rank[b_local])
                high_local = torch.where(a_higher, a_local, b_local)
                low_local = torch.where(a_higher, b_local, a_local)
                if int(high_local.numel()) > n_pairs:
                    high_local = high_local[:n_pairs]
                    low_local = low_local[:n_pairs]
                high_idx = pool_idx[high_local]
                low_idx = pool_idx[low_local]
                prob_high = branch_probs[high_idx, child_pos]
                prob_low = branch_probs[low_idx, child_pos]
                raw_violation = prob_low - prob_high + float(self.hidden_marker_rank_margin)
                pair_loss = F.relu(raw_violation)
                if int(pair_loss.numel()) <= 0:
                    continue
                all_losses.append(pair_loss)
                all_violations.append(pair_loss.detach())
                all_score_gaps.append((score_rank[high_local] - score_rank[low_local]).detach())
                pair_count += int(pair_loss.numel())

        if not all_losses:
            return zero
        losses = torch.cat(all_losses, dim=0)
        score_gaps = torch.cat(all_score_gaps, dim=0)
        violations = torch.cat(all_violations, dim=0)
        self._last_hidden_marker_rank_pair_count = int(pair_count)
        self._last_hidden_marker_rank_mean_violation = float(violations.mean().item())
        self._last_hidden_marker_rank_mean_score_gap = float(score_gaps.mean().item())
        return float(self.hidden_marker_rank_lambda) * losses.mean()

    def _build_allowed_mask(
        self,
        supervision_codes: torch.Tensor,
    ) -> torch.Tensor:
        n_obs = int(supervision_codes.shape[0])
        mask = torch.ones((n_obs, int(self.n_labels)), dtype=torch.bool, device=supervision_codes.device)
        valid = supervision_codes.ge(0)
        if not bool(valid.any()):
            return mask
        mask[valid] = False
        for code in torch.unique(supervision_codes[valid]):
            code_int = int(code.item())
            desc_indices = self.supervision_code_to_desc_indices.get(code_int)
            if desc_indices is None:
                continue
            row_mask = supervision_codes.eq(code_int)
            row_idx = torch.where(row_mask)[0]
            if row_idx.numel() == 0:
                continue
            col_idx = torch.as_tensor(desc_indices, dtype=torch.long, device=mask.device)
            if col_idx.numel() == 0:
                continue
            mask[row_idx.unsqueeze(1), col_idx.unsqueeze(0)] = True
        empty_rows = ~mask.any(dim=1)
        mask[empty_rows] = True
        return mask

    def _restricted_probabilities_from_logits(
        self,
        logits: torch.Tensor,
        allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = self.classifier(logits) if logits.shape[-1] != int(self.n_labels) else logits
        if logits.shape[-1] == int(self.n_labels) and self.classifier.logits:
            probs = F.softmax(logits, dim=-1)
        if allowed_mask is None:
            return probs
        masked = probs * allowed_mask.to(dtype=probs.dtype)
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return masked / denom

    def _standard_normal_prior_terms_restricted(
        self,
        qz1,
        z1: torch.Tensor,
        allowed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._standard_normal_prior_terms(qz1, z1)

    def _supervision_nll_per_sample(
        self,
        logits: torch.Tensor,
        supervision_codes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Negative log-likelihood for observed fine or coarse supervision.

        A supervision code can map to one fine label or to a set of descendant
        fine labels.  Multi-descendant codes use -log P(label in descendants).
        """
        if logits.numel() == 0 or supervision_codes.numel() == 0:
            return torch.zeros((0,), device=logits.device), None, None, None

        nll = torch.zeros((logits.shape[0],), dtype=logits.dtype, device=logits.device)
        fine_metric_logits: list[torch.Tensor] = []
        fine_metric_targets: list[torch.Tensor] = []
        for code in torch.unique(supervision_codes):
            code_int = int(code.item())
            desc_indices = self.supervision_code_to_desc_indices.get(code_int)
            if desc_indices is None:
                continue
            mask = supervision_codes.eq(code_int)
            if not bool(mask.any()):
                continue
            logits_subset = logits[mask]
            if len(desc_indices) == 1:
                target_index = int(desc_indices[0])
                targets = torch.full(
                    (logits_subset.shape[0],),
                    target_index,
                    dtype=torch.long,
                    device=logits.device,
                )
                nll[mask] = F.cross_entropy(logits_subset, targets, reduction="none")
                fine_metric_logits.append(logits_subset)
                fine_metric_targets.append(targets)
            else:
                desc_logits = logits_subset[:, list(desc_indices)]
                nll[mask] = torch.logsumexp(logits_subset, dim=-1) - torch.logsumexp(desc_logits, dim=-1)
        if not fine_metric_targets:
            return nll, None, None, None
        metric_logits = torch.cat(fine_metric_logits, dim=0)
        metric_true = torch.cat(fine_metric_targets, dim=0)
        metric_ce = F.cross_entropy(metric_logits, metric_true, reduction="mean")
        return nll, metric_ce, metric_true, metric_logits

    def _reference_supervision_loss(
        self,
        logits: torch.Tensor,
        tensors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """Reference supervision loss for partial labels.

        This replaces ordinary full-label CE in partial settings.  Query cells
        have code -1 and are excluded from this term.
        """
        supervision_codes = tensors[PARTIAL_SUPERVISION_CODE_COL].reshape(-1).long()
        valid = supervision_codes.ge(0)
        if not bool(valid.any()):
            zero = self._zero(logits.device)
            return zero, None, None, None, torch.zeros((), device=logits.device, dtype=torch.long)
        nll, metric_ce, metric_true, metric_logits = self._supervision_nll_per_sample(
            logits[valid],
            supervision_codes[valid],
        )
        loss = nll.mean() if nll.numel() else self._zero(logits.device)
        return loss, metric_ce, metric_true, metric_logits, valid.sum()

    def _weighted_mean_from_count(
        self,
        losses: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if losses.numel() == 0:
            return self._zero(weights.device)
        denom = torch.tensor(float(losses.shape[0]), device=weights.device, dtype=weights.dtype).clamp_min(1.0)
        return (losses * weights).sum() / denom

    def _query_pseudolabel_losses(
        self,
        logits: torch.Tensor,
        tensors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Weighted query pseudo-label CE used in partial round1/round2."""
        zero = self._zero(logits.device)

        fine_targets = tensors[PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY].reshape(-1).long()
        fine_weights = tensors[PARTIAL_QUERY_PSEUDO_FINE_WEIGHT_KEY].reshape(-1).float()
        fine_valid = fine_targets.ge(0) & fine_targets.lt(int(self.n_labels)) & fine_weights.gt(0.0)
        fine_loss = zero
        if bool(fine_valid.any()):
            fine_ce = F.cross_entropy(logits[fine_valid], fine_targets[fine_valid], reduction="none")
            fine_loss = self._weighted_mean_from_count(fine_ce, fine_weights[fine_valid])

        coarse_targets = tensors[PARTIAL_QUERY_PSEUDO_COARSE_TARGET_KEY].reshape(-1).long()
        coarse_weights = tensors[PARTIAL_QUERY_PSEUDO_COARSE_WEIGHT_KEY].reshape(-1).float()
        coarse_valid = coarse_targets.ge(0) & coarse_weights.gt(0.0)
        coarse_loss = zero
        if bool(coarse_valid.any()):
            coarse_nll, _, _, _ = self._supervision_nll_per_sample(logits[coarse_valid], coarse_targets[coarse_valid])
            coarse_loss = self._weighted_mean_from_count(coarse_nll, coarse_weights[coarse_valid])

        return fine_loss, coarse_loss, fine_valid.sum(), coarse_valid.sum()

    def _query_pseudolabel_smallclass_loss(
        self,
        logits: torch.Tensor,
        tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Extra CE for pseudo-labeled classes with too few selected anchors."""
        zero = self._zero(logits.device)
        self._last_query_pseudolabel_smallclass_count = 0
        if self.smallclass_ce_mode != SMALLCLASS_CE_MODE_OVERSAMPLE:
            return zero
        selected = tensors[PARTIAL_QUERY_PSEUDO_SELECTED_KEY].reshape(-1) > 0.5
        targets = tensors[PARTIAL_QUERY_PSEUDO_FINE_TARGET_KEY].reshape(-1).long()
        valid = selected & targets.ge(0) & targets.lt(int(self.n_labels))
        if not valid.any():
            return zero
        repeat_by_label = self._smallclass_repeat_tensor(logits.device)
        repeats = repeat_by_label[targets.clamp(0, int(self.n_labels) - 1)]
        valid = valid & repeats.gt(0)
        if not valid.any():
            return zero
        indices = torch.where(valid)[0]
        self._last_query_pseudolabel_smallclass_count = int(indices.numel())
        ce = F.cross_entropy(logits[indices], targets[indices], reduction="none")
        repeat_weights = repeats[indices].to(dtype=ce.dtype)
        loss = (ce * repeat_weights).sum() / repeat_weights.sum().clamp_min(1.0)
        return float(self.query_pseudolabel_fine_ratio) * loss

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        pro_recons_weight=1.0,
        kl_weight=1.0,
        labelled_tensors: dict[str, torch.Tensor] | None = None,
        classification_ratio: float | None = None,
        epoch: int | None = None,
    ):
        """Compute the partial teacher objective for the active training stage."""
        qz = inference_outputs["qz"]
        ql = inference_outputs["ql"]
        z = inference_outputs["z"]
        px_ = generative_outputs["px_"]
        py_ = generative_outputs["py_"]
        per_batch_efficiency = generative_outputs["per_batch_efficiency"]

        x = tensors[REGISTRY_KEYS.X_KEY]
        y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        panel_index = tensors[self.panel_key]

        protein_mask = self._protein_mask_for_minibatch(y, panel_index)
        reconst_loss_gene, reconst_loss_protein = self.get_reconstruction_loss(
            x,
            y,
            px_,
            py_,
            protein_mask,
            per_batch_efficiency,
        )

        supervision_codes = tensors[PARTIAL_SUPERVISION_CODE_COL].reshape(-1).long()
        allowed_mask = self._build_allowed_mask(supervision_codes)
        standard_normal_prior_weight = self._standard_normal_prior_weight()
        if standard_normal_prior_weight > 0:
            standard_normal_z_reconstruction, standard_normal_kl = self._standard_normal_prior_terms_restricted(
                qz,
                z,
                allowed_mask=allowed_mask,
            )
        else:
            standard_normal_z_reconstruction = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            standard_normal_kl = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        if not self.use_observed_lib_size:
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                F.one_hot(batch_index.squeeze(-1), n_batch).float(),
                self.library_log_means,
            )
            local_library_log_vars = F.linear(
                F.one_hot(batch_index.squeeze(-1), n_batch).float(),
                self.library_log_vars,
            )
            kl_div_l_gene = torch.distributions.kl_divergence(
                ql,
                torch.distributions.Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_div_l_gene = torch.zeros_like(standard_normal_kl)

        kl_div_back_pro_full = torch.distributions.kl_divergence(
            torch.distributions.Normal(py_["back_alpha"], py_["back_beta"]),
            inference_outputs["back_mean_prior"],
        )
        lkl_back_pro_full = -torch.distributions.LogNormal(
            torch.tensor([0.0], device=x.device),
            torch.tensor([1.0], device=x.device),
        ).log_prob(per_batch_efficiency)
        lkl_protein_expressed = -1e-3 * torch.distributions.Bernoulli(
            logits=py_["mixing"]
        ).log_prob(torch.ones_like(py_["mixing"]))
        if protein_mask is not None:
            kl_div_back_pro = protein_mask.bool() * kl_div_back_pro_full
            kl_div_back_pro = (
                kl_div_back_pro.sum(dim=1)
                + lkl_back_pro_full.sum(dim=1)
                + lkl_protein_expressed.sum(dim=1)
            )
        else:
            kl_div_back_pro = (
                kl_div_back_pro_full.sum(dim=1)
                + lkl_back_pro_full.sum(dim=1)
                + lkl_protein_expressed.sum(dim=1)
            )

        q_c = F.softmax(self.classifier(z), dim=-1)
        reconstruction_prior_loss = torch.mean(
            reconst_loss_gene
            + kl_weight * pro_recons_weight * reconst_loss_protein
            + float(standard_normal_prior_weight) * standard_normal_z_reconstruction
            + kl_weight * float(standard_normal_prior_weight) * standard_normal_kl
            + kl_div_l_gene
            + kl_weight * kl_div_back_pro
        )
        loss = reconstruction_prior_loss

        logits_full = self.classifier(z)
        reference_supervision_loss, metric_ce, metric_true, metric_logits, n_reference_supervised = (
            self._reference_supervision_loss(logits_full, tensors)
        )
        loss = loss + float(classification_ratio or 0.0) * reference_supervision_loss

        pseudo_fine_loss, pseudo_coarse_loss, n_pseudo_fine, n_pseudo_coarse = self._query_pseudolabel_losses(
            logits_full,
            tensors,
        )
        smallclass_loss = self._query_pseudolabel_smallclass_loss(logits_full, tensors)
        loss = (
            loss
            + float(self.query_pseudolabel_fine_ratio) * pseudo_fine_loss
            + float(self.query_pseudolabel_coarse_ratio) * pseudo_coarse_loss
            + smallclass_loss
        )
        hidden_balance_loss = self._hidden_branch_balance_loss(q_c, tensors)
        loss = loss + hidden_balance_loss
        hidden_parent_anchor_ce_loss = self._hidden_parent_anchor_ce_loss(logits_full, tensors)
        loss = loss + hidden_parent_anchor_ce_loss
        hidden_marker_rank_loss = self._hidden_marker_rank_loss(q_c, tensors, y, protein_mask)
        loss = loss + hidden_marker_rank_loss

        return LossOutput(
            loss=loss,
            reconstruction_loss={
                "reconst_loss_gene": reconst_loss_gene,
                "reconst_loss_protein": reconst_loss_protein,
                "standard_normal_z_reconstruction": standard_normal_z_reconstruction,
            },
            kl_local={
                "standard_normal_kl": standard_normal_kl,
                "kl_div_l_gene": kl_div_l_gene,
                "kl_div_back_pro": kl_div_back_pro,
            },
            classification_loss=metric_ce,
            true_labels=metric_true,
            logits=metric_logits,
            extra_metrics={
                "reconstruction_prior_loss": reconstruction_prior_loss.detach(),
                "reference_partial_supervision_loss": reference_supervision_loss.detach(),
                "query_partial_pseudo_fine_loss": pseudo_fine_loss.detach(),
                "query_partial_pseudo_coarse_loss": pseudo_coarse_loss.detach(),
                "query_partial_pseudo_smallclass_ce_loss": smallclass_loss.detach(),
                "hidden_balance_loss": hidden_balance_loss.detach(),
                "hidden_parent_anchor_ce_loss": hidden_parent_anchor_ce_loss.detach(),
                "hidden_marker_rank_loss": hidden_marker_rank_loss.detach(),
                "n_reference_supervised_batch": n_reference_supervised.detach(),
                "n_partial_pseudo_fine_batch": n_pseudo_fine.detach(),
                "n_partial_pseudo_coarse_batch": n_pseudo_coarse.detach(),
                "n_partial_pseudo_smallclass_batch": torch.tensor(
                    float(self._last_query_pseudolabel_smallclass_count),
                    device=loss.device,
                ),
                "n_hidden_balance_query_batch": torch.tensor(
                    float(self._last_hidden_balance_query_count),
                    device=loss.device,
                ),
                "n_hidden_parent_anchor_ce_batch": torch.tensor(
                    float(self._last_hidden_parent_anchor_count),
                    device=loss.device,
                ),
                "hidden_parent_anchor_mean_target_prob": torch.tensor(
                    float(self._last_hidden_parent_anchor_mean_target_prob),
                    device=loss.device,
                ),
                "n_hidden_marker_rank_pairs_batch": torch.tensor(
                    float(self._last_hidden_marker_rank_pair_count),
                    device=loss.device,
                ),
                "hidden_marker_rank_mean_violation": torch.tensor(
                    float(self._last_hidden_marker_rank_mean_violation),
                    device=loss.device,
                ),
                "hidden_marker_rank_mean_score_gap": torch.tensor(
                    float(self._last_hidden_marker_rank_mean_score_gap),
                    device=loss.device,
                ),
            },
        )


class _AnchorPartialTeacherTrainingPlan(_AnchorTeacherTrainingPlan):
    def compute_and_log_metrics(self, loss_output: LossOutput, metrics: dict, mode: str):
        super().compute_and_log_metrics(loss_output, metrics, mode)
        if loss_output.extra_metrics is None:
            return
        for key in [
            "reference_partial_supervision_loss",
            "query_partial_pseudo_fine_loss",
            "query_partial_pseudo_coarse_loss",
            "query_partial_pseudo_smallclass_ce_loss",
            "hidden_balance_loss",
            "hidden_parent_anchor_ce_loss",
            "hidden_marker_rank_loss",
            "n_reference_supervised_batch",
            "n_partial_pseudo_fine_batch",
            "n_partial_pseudo_coarse_batch",
            "n_partial_pseudo_smallclass_batch",
            "n_hidden_balance_query_batch",
            "n_hidden_parent_anchor_ce_batch",
            "hidden_parent_anchor_mean_target_prob",
            "n_hidden_marker_rank_pairs_batch",
            "hidden_marker_rank_mean_violation",
            "hidden_marker_rank_mean_score_gap",
        ]:
            if key in loss_output.extra_metrics:
                self.log_with_mode(
                    key,
                    loss_output.extra_metrics[key],
                    mode,
                    on_step=self.on_step,
                    on_epoch=self.on_epoch,
                    batch_size=loss_output.n_obs_minibatch,
                )
