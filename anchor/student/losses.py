from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from .bundle import (
    StudentDataBundle,
    _build_target_scores,
    _child_conditional,
    _collect_tree_marker_panel,
    _descendant_leaves,
    _direct_parent_map,
    _node_posterior,
)

DEFAULT_STUDENT_LOSS_WEIGHTS: dict[str, float] = {
    "z_recon_huber": 1.0,
    "protein_recon_huber": 0.5,
    "weighted_pseudo_ce": 5.0,
    "teacher_soft_kl": 0.5,
    "hard_anchor_supcon": 0.2,
    "agreement_self_supcon": 0.1,
    "axis_rank_loss": 1.0,
}

def _student_loss_weights(loss_weights: Mapping[str, float] | None = None) -> dict[str, float]:
    weights = dict(DEFAULT_STUDENT_LOSS_WEIGHTS)
    if loss_weights:
        for key, value in loss_weights.items():
            if key not in weights:
                raise KeyError(f"Unknown student loss weight: {key}")
            weights[key] = float(value)
    return weights


def _scheduled_teacher_soft_kl_weight(
    *,
    epoch: int,
    base_weight: float,
    schedule: Mapping[str, Any] | None = None,
) -> float:
    if not schedule:
        return float(base_weight)
    mode = str(schedule.get("mode", "linear")).lower()
    if mode not in {"linear", "constant"}:
        raise ValueError(f"Unsupported teacher_soft_kl_schedule mode={mode!r}")
    start = float(schedule.get("start", schedule.get("teacher_soft_kl_start", base_weight)))
    end = float(schedule.get("end", schedule.get("teacher_soft_kl_end", base_weight)))
    if mode == "constant":
        return start
    hold_epochs = int(schedule.get("hold_epochs", schedule.get("teacher_soft_kl_hold_epochs", 0)))
    decay_end_epoch = int(schedule.get("decay_end_epoch", schedule.get("teacher_soft_kl_decay_end_epoch", hold_epochs)))
    epoch = int(epoch)
    if epoch < hold_epochs:
        return start
    if decay_end_epoch <= hold_epochs:
        return end
    if epoch >= decay_end_epoch:
        return end
    frac = float(epoch - hold_epochs) / float(decay_end_epoch - hold_epochs)
    return float(start + frac * (end - start))


def _supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    if embeddings.shape[0] < 4:
        return torch.zeros((), device=embeddings.device)
    unique, counts = torch.unique(labels, return_counts=True)
    if unique.numel() < 2 or (counts >= 2).sum() < 2:
        return torch.zeros((), device=embeddings.device)
    z = F.normalize(embeddings, dim=-1)
    logits = z @ z.t() / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(logits.shape[0], device=embeddings.device, dtype=torch.bool)
    same = labels[:, None].eq(labels[None, :]) & ~eye
    valid = same.any(dim=1)
    if not valid.any():
        return torch.zeros((), device=embeddings.device)
    exp_logits = torch.exp(logits) * (~eye).float()
    pos_exp = exp_logits * same.float()
    log_prob = torch.log(pos_exp.sum(dim=1).clamp_min(1e-8)) - torch.log(exp_logits.sum(dim=1).clamp_min(1e-8))
    return -log_prob[valid].mean()


def _weighted_ce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.zeros((), device=logits.device)
    per = F.cross_entropy(logits, targets, reduction="none")
    return (per * weights).sum() / weights.sum().clamp_min(1e-8)


def _axis_rank_loss(
    probs: torch.Tensor,
    scores: torch.Tensor,
    *,
    branch_rank_specs: list[dict[str, Any]],
    teacher_probs: torch.Tensor | None = None,
    batch_indices: torch.Tensor | None = None,
    max_pairs_per_child: int = 256,
    margin: float = 0.1,
) -> tuple[torch.Tensor, int]:
    losses: list[torch.Tensor] = []
    n_pairs_total = 0
    device = probs.device
    for spec in branch_rank_specs:
        spec_weight = float(spec.get("rank_weight", 1.0))
        if spec_weight <= 0:
            continue
        desc = torch.as_tensor(spec["parent_desc_indices"], dtype=torch.long, device=device)
        if desc.numel() == 0:
            continue
        parent_source = teacher_probs if teacher_probs is not None and bool(spec.get("use_teacher_parent_pool", False)) else probs
        parent_mass = parent_source[:, desc].sum(dim=1)
        pool = parent_mass.gt(0.20)
        child_parent_mass = probs[:, desc].sum(dim=1)
        if int(pool.sum().item()) < 4:
            continue
        for child in spec["children"]:
            child_desc = torch.as_tensor(child["desc_indices"], dtype=torch.long, device=device)
            if child_desc.numel() == 0:
                continue
            child_prob = probs[:, child_desc].sum(dim=1) / child_parent_mass.clamp_min(1e-8)
            if "score_values" in child and batch_indices is not None:
                score_values = torch.as_tensor(child["score_values"], dtype=scores.dtype, device=device)
                child_score = score_values[batch_indices.long()]
            else:
                child_score = scores[:, int(child["label_idx"])]
            valid = pool & torch.isfinite(child_score) & child_score.gt(-1e5)
            idx = torch.nonzero(valid, as_tuple=False).flatten()
            if idx.numel() < 4:
                continue
            if idx.numel() > max_pairs_per_child * 2:
                perm = torch.randperm(idx.numel(), device=device)[: max_pairs_per_child * 2]
                idx = idx[perm]
            a = idx[0::2]
            b = idx[1::2]
            n = min(a.numel(), b.numel())
            if n == 0:
                continue
            a = a[:n]
            b = b[:n]
            swap = child_score[a] < child_score[b]
            hi = torch.where(swap, b, a)
            lo = torch.where(swap, a, b)
            gap = child_score[hi] - child_score[lo]
            keep = gap.gt(1e-6)
            if not keep.any():
                continue
            pair_loss = F.relu(child_prob[lo][keep] - child_prob[hi][keep] + float(margin))
            losses.append(pair_loss.mean() * spec_weight)
            n_pairs_total += int(keep.sum().item())
    if not losses:
        return torch.zeros((), device=device), 0
    return torch.stack(losses).mean(), n_pairs_total


def _marker_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [str(x) for x in value.keys()]
    if isinstance(value, str):
        return [str(value)]
    return [str(x) for x in value]


def _build_branch_rank_specs(bundle: StudentDataBundle) -> list[dict[str, Any]]:
    def _child_has_rank_evidence(branch: str, child: str, desc_labels: list[str]) -> bool:
        if str(child) in bundle.target_score_available.index and bool(bundle.target_score_available.get(str(child), False)):
            return True
        for label in desc_labels:
            if str(label) in bundle.target_score_available.index and bool(bundle.target_score_available.get(str(label), False)):
                return True
        class_spec = (
            bundle.prior_spec.get("branch_teacher_specs", {})
            .get(str(branch), {})
            .get("classes", {})
            .get(str(child), {})
        )
        return bool(_marker_list(class_spec.get("positive", [])) or _marker_list(class_spec.get("negative", [])))

    def _label_idx_for_child(child: str, desc_labels: list[str]) -> int | None:
        if str(child) in bundle.label_to_idx:
            return int(bundle.label_to_idx[str(child)])
        for label in desc_labels:
            if str(label) in bundle.target_score_available.index and bool(bundle.target_score_available.get(str(label), False)):
                return int(bundle.label_to_idx[str(label)])
        for label in desc_labels:
            if str(label) in bundle.label_to_idx:
                return int(bundle.label_to_idx[str(label)])
        return None

    specs: list[dict[str, Any]] = []
    tree_children = _children_map(bundle.prior_spec)
    branch_specs = bundle.prior_spec.get("branch_teacher_specs", {})
    candidate_branches = sorted(set(tree_children) | {str(x) for x in branch_specs})
    for branch in candidate_branches:
        raw_children = list(branch_specs.get(str(branch), {}).get("children", tree_children.get(str(branch), [])))
        raw_children = [str(child) for child in raw_children]
        if len(raw_children) < 2:
            continue
        branch_spec = bundle.prior_spec.get("branch_teacher_specs", {}).get(branch)
        parent_desc = [
            bundle.label_to_idx[x]
            for x in _descendant_leaves(branch, bundle.prior_spec, bundle.label_names)
            if x in bundle.label_to_idx
        ]
        children = []
        for child in raw_children:
            desc_labels = [x for x in _descendant_leaves(str(child), bundle.prior_spec, bundle.label_names) if x in bundle.label_to_idx]
            desc = [bundle.label_to_idx[x] for x in desc_labels]
            label_idx = _label_idx_for_child(str(child), desc_labels)
            if label_idx is None or not _child_has_rank_evidence(str(branch), str(child), desc_labels):
                continue
            children.append({"child": str(child), "desc_indices": desc, "label_idx": int(label_idx)})
        if parent_desc and len(children) >= 2:
            specs.append({"branch": branch, "parent_desc_indices": parent_desc, "children": children})
    return specs


def _parent_id_for_label(bundle: StudentDataBundle) -> np.ndarray:
    parent_map = _direct_parent_map(bundle.prior_spec)
    parents = sorted({parent_map.get(label, label) for label in bundle.label_names})
    parent_to_idx = {parent: idx for idx, parent in enumerate(parents)}
    return np.asarray([parent_to_idx[parent_map.get(label, label)] for label in bundle.label_names], dtype=np.int64)


def _children_map(prior_spec: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        str(parent): [str(child) for child in children]
        for parent, children in prior_spec.get("tree_spec", {}).get("children", {}).items()
    }


def _tree_leaf_indices(
    prior_spec: Mapping[str, Any],
    node: str,
    label_to_idx: Mapping[str, int],
) -> list[int]:
    if str(node) in label_to_idx:
        return [int(label_to_idx[str(node)])]
    leaves = prior_spec.get("tree_spec", {}).get("descendants", {}).get(str(node), [])
    return [int(label_to_idx[str(leaf)]) for leaf in leaves if str(leaf) in label_to_idx]


def build_rho_policy_kl_specs_from_table(
    *,
    prior_spec: Mapping[str, Any],
    label_names: Sequence[str],
    rho_table: pd.DataFrame,
    default_rho: float = 1.0,
    rho_col: str = "rho_discrete_recommended",
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    label_names = [str(label) for label in label_names]
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    rho_lookup: dict[str, Mapping[str, Any]] = {}
    if rho_table is not None and not rho_table.empty and "node" in rho_table.columns:
        for _, row in rho_table.iterrows():
            rho_lookup[str(row["node"])] = row.to_dict()

    specs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for node, raw_children in sorted(_children_map(prior_spec).items()):
        children: list[str] = []
        child_indices: list[list[int]] = []
        for child in raw_children:
            indices = _tree_leaf_indices(prior_spec, child, label_to_idx)
            if indices:
                children.append(str(child))
                child_indices.append(indices)
        parent_indices = _tree_leaf_indices(prior_spec, node, label_to_idx)
        if len(children) < 2 or not parent_indices:
            continue

        audit = rho_lookup.get(str(node), {})
        raw_rho = audit.get(rho_col, audit.get("rho_v", default_rho))
        try:
            alpha = float(raw_rho)
        except (TypeError, ValueError):
            alpha = float(default_rho)
        if not np.isfinite(alpha):
            alpha = float(default_rho)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        specs.append(
            {
                "node": str(node),
                "alpha": alpha,
                "parent_indices": [int(x) for x in parent_indices],
                "children": [str(x) for x in children],
                "child_indices": [[int(y) for y in indices] for indices in child_indices],
            }
        )
        rows.append(
            {
                "node": str(node),
                "alpha": alpha,
                "rho_source_col": str(rho_col) if str(node) in rho_lookup else "default_rho_missing_audit",
                "rho_policy": audit.get("rho_policy", "keep_teacher_missing_audit"),
                "rho_policy_reason": audit.get("rho_policy_reason", "missing_audit_default_keep_teacher"),
                "n_children": int(len(children)),
                "n_descendant_leaves": int(len(parent_indices)),
                "children": "|".join(children),
                "marker_complete": audit.get("marker_complete", np.nan),
                "protein_power_v": audit.get("protein_power_v", np.nan),
                "policy_challenge_v": audit.get("policy_challenge_v", np.nan),
                "rna_protection_v": audit.get("rna_protection_v", np.nan),
                "query_rna_structure_trust_v": audit.get("query_rna_structure_trust_v", np.nan),
                "reference_child_accuracy": audit.get("reference_child_accuracy", np.nan),
                "n_parent_pool": audit.get("n_parent_pool", np.nan),
                "partial_hidden_node": audit.get("partial_hidden_node", np.nan),
            }
        )
    return specs, pd.DataFrame(rows).sort_values("node", kind="mergesort").reset_index(drop=True)


def _rho_policy_conditional_teacher_kl(
    probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    *,
    specs: Sequence[Mapping[str, Any]],
    teacher_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per = torch.zeros(probs.shape[0], dtype=probs.dtype, device=probs.device)
    for spec in specs:
        alpha = float(spec.get("alpha", 1.0))
        if alpha <= 0.0:
            continue
        parent_idx = torch.as_tensor(spec["parent_indices"], dtype=torch.long, device=probs.device)
        teacher_parent = teacher_probs[:, parent_idx].sum(dim=1)
        student_parent = probs[:, parent_idx].sum(dim=1)
        teacher_parts: list[torch.Tensor] = []
        student_parts: list[torch.Tensor] = []
        for child_indices in spec["child_indices"]:
            idx = torch.as_tensor(child_indices, dtype=torch.long, device=probs.device)
            teacher_parts.append(teacher_probs[:, idx].sum(dim=1))
            student_parts.append(probs[:, idx].sum(dim=1))
        teacher_child = torch.stack(teacher_parts, dim=1) / teacher_parent[:, None].clamp_min(1e-8)
        student_child = torch.stack(student_parts, dim=1) / student_parent[:, None].clamp_min(1e-8)
        node_kl = torch.sum(
            teacher_child.clamp_min(1e-8)
            * (torch.log(teacher_child.clamp_min(1e-8)) - torch.log(student_child.clamp_min(1e-8))),
            dim=1,
        )
        per = per + float(alpha) * teacher_parent * node_kl
    if teacher_weights is None:
        return per.mean()
    return (teacher_weights * per).sum() / teacher_weights.sum().clamp_min(1e-8)


def _student_prototype_logits(u: torch.Tensor, prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    u_norm = F.normalize(u, dim=-1)
    proto_norm = F.normalize(prototypes, dim=-1)
    return (u_norm @ proto_norm.t()) / max(float(temperature), 1e-6)


def _compute_student_prototypes(
    u: np.ndarray,
    pseudo_target: np.ndarray,
    selected: np.ndarray,
    *,
    n_labels: int,
    pseudo_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    prototypes = np.zeros((int(n_labels), int(u.shape[1])), dtype=np.float32)
    active = np.zeros(int(n_labels), dtype=bool)
    u_norm = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-8)
    weights = np.ones(int(u.shape[0]), dtype=np.float32) if pseudo_weight is None else np.asarray(pseudo_weight, dtype=np.float32)
    for label_idx in range(int(n_labels)):
        mask = selected & (pseudo_target == label_idx)
        if not np.any(mask):
            continue
        w = np.clip(weights[mask], 0.0, None)
        if float(w.sum()) <= 0.0:
            continue
        proto = (u_norm[mask] * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-8)
        proto = proto / max(float(np.linalg.norm(proto)), 1e-8)
        prototypes[label_idx] = proto.astype(np.float32)
        active[label_idx] = True
    return prototypes, active


def _graph_consistency_loss(probs: torch.Tensor, neighbor_target: torch.Tensor, graph_mask: torch.Tensor) -> torch.Tensor:
    if not graph_mask.any():
        return torch.zeros((), device=probs.device)
    p = probs[graph_mask].clamp_min(1e-8)
    target = neighbor_target[graph_mask].clamp_min(1e-8)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    kl_pt = torch.sum(p * (torch.log(p) - torch.log(target.detach())), dim=1)
    kl_tp = torch.sum(target.detach() * (torch.log(target.detach()) - torch.log(p)), dim=1)
    return 0.5 * (kl_pt + kl_tp).mean()


def _train_val_indices(n: int, *, seed: int, val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    n = int(n)
    if n <= 1:
        return np.arange(n, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)
    n_val = max(1, int(round(float(val_fraction) * n)))
    n_val = min(n - 1, n_val)
    val_idx = np.sort(indices[:n_val])
    train_idx = np.sort(indices[n_val:])
    return train_idx, val_idx
