"""Student training loop.

The student is trained on query cells only.  Its objective combines teacher
latent reconstruction, protein reconstruction, weighted anchor pseudo-label CE,
marker-rank supervision, node-wise conditional KL and auxiliary regularizers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .anchors import (
    DEFAULT_BOTTOMUP_CONFIG,
    _build_generic_branch_rank_specs,
    _rank_specs_summary,
    _write_student_safety_guard_report,
    select_bottomup_treeguard_pseudolabels,
)
from .bundle import QueryTeacherStudentDataset, StudentDataBundle, _build_label_score_arrays
from .evaluation import evaluate_and_write_student, _student_knn_purity_summary
from .losses import (
    DEFAULT_STUDENT_LOSS_WEIGHTS,
    _axis_rank_loss,
    _build_branch_rank_specs,
    _compute_student_prototypes,
    _graph_consistency_loss,
    _parent_id_for_label,
    _rho_policy_conditional_teacher_kl,
    _scheduled_teacher_soft_kl_weight,
    _student_loss_weights,
    _student_prototype_logits,
    _supervised_contrastive_loss,
    _train_val_indices,
    _weighted_ce,
)
from .model import QueryTeacherFeatureStudent
from ..output import write_student_confusion_heatmap

def _train_val_indices(n: int, *, seed: int, val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(n, dtype=np.int64)
    rng.shuffle(order)
    n_val = max(1, int(round(float(val_fraction) * n)))
    val = np.sort(order[:n_val])
    train = np.sort(order[n_val:])
    return train, val


def _predict_student_protograph_arrays(
    model: QueryTeacherFeatureStudent,
    dataset: QueryTeacherStudentDataset,
    *,
    batch_size: int,
    device: torch.device,
    prototypes: torch.Tensor | None = None,
    prototype_temperature: float = 0.15,
    prototype_logit_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logits_all: list[np.ndarray] = []
    u_all: list[np.ndarray] = []
    z_recon_all: list[np.ndarray] = []
    p_recon_all: list[np.ndarray] = []
    mlp_logits_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            protein = batch["protein"].to(device)
            batch_id = batch["batch_idx"].to(device)
            out = model(z, protein, batch_id)
            logits = out["logits"]
            mlp_logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
            if prototypes is not None:
                logits = logits + float(prototype_logit_weight) * _student_prototype_logits(
                    out["u"],
                    prototypes.to(device),
                    prototype_temperature,
                )
            logits_all.append(logits.detach().cpu().numpy().astype(np.float32))
            u_all.append(out["u"].detach().cpu().numpy().astype(np.float32))
            z_recon_all.append(out["z_recon"].detach().cpu().numpy().astype(np.float32))
            p_recon_all.append(out["protein_recon"].detach().cpu().numpy().astype(np.float32))
    return (
        np.concatenate(logits_all, axis=0),
        np.concatenate(u_all, axis=0),
        np.concatenate(z_recon_all, axis=0),
        np.concatenate(p_recon_all, axis=0),
        np.concatenate(mlp_logits_all, axis=0),
    )


def _refresh_student_protograph_state(
    model: QueryTeacherFeatureStudent,
    dataset: QueryTeacherStudentDataset,
    *,
    batch_size: int,
    device: torch.device,
    pseudo_target: np.ndarray,
    selected: np.ndarray,
    pseudo_weight: np.ndarray | None = None,
    teacher_parent: np.ndarray,
    teacher_confidence: np.ndarray,
    parent_id_for_label: np.ndarray,
    n_labels: int,
    graph_k: int = 15,
    prototype_temperature: float = 0.15,
    prototype_logit_weight: float = 1.0,
    student_confidence_threshold: float = 0.90,
    teacher_confidence_threshold: float = 0.75,
) -> dict[str, Any]:
    _, u_np, _, _, _ = _predict_student_protograph_arrays(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
    )
    prototypes_np, active = _compute_student_prototypes(u_np, pseudo_target, selected, n_labels=n_labels, pseudo_weight=pseudo_weight)
    proto_tensor = torch.as_tensor(prototypes_np, dtype=torch.float32, device=device)
    logits_np, u_np, _, _, _ = _predict_student_protograph_arrays(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        prototypes=proto_tensor,
        prototype_temperature=prototype_temperature,
        prototype_logit_weight=prototype_logit_weight,
    )
    probs_np = F.softmax(torch.as_tensor(logits_np), dim=-1).numpy().astype(np.float32)
    pred_idx = probs_np.argmax(axis=1).astype(np.int64)
    pred_parent = parent_id_for_label[pred_idx]
    confidence = probs_np.max(axis=1)
    nn = NearestNeighbors(n_neighbors=min(int(graph_k) + 1, int(u_np.shape[0]))).fit(u_np)
    neighbors = nn.kneighbors(u_np, return_distance=False)[:, 1:]
    neighbor_target = np.zeros_like(probs_np, dtype=np.float32)
    graph_mask = np.zeros(u_np.shape[0], dtype=bool)
    n_edges = 0
    for i in range(u_np.shape[0]):
        nbr = neighbors[i]
        keep = (teacher_parent[nbr] == teacher_parent[i]) | (pred_parent[nbr] == pred_parent[i])
        nbr = nbr[keep]
        if nbr.size == 0:
            continue
        neighbor_target[i] = probs_np[nbr].mean(axis=0)
        graph_mask[i] = True
        n_edges += int(nbr.size)
    weak_mask = (
        (~selected)
        & (teacher_parent == pred_parent)
        & (confidence >= float(student_confidence_threshold))
        & (teacher_confidence >= float(teacher_confidence_threshold))
    )
    return {
        "prototypes": prototypes_np,
        "prototype_active": active,
        "neighbor_target": neighbor_target,
        "graph_mask": graph_mask,
        "n_graph_edges": int(n_edges),
        "weak_mask": weak_mask,
        "weak_target": pred_idx,
    }


def train_student_model_protograph(
    bundle: StudentDataBundle,
    pseudo_df: pd.DataFrame,
    *,
    results_dir: Path,
    max_epochs: int = 100,
    early_stopping_patience: int = 12,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    random_seed: int = 2026,
    u_dim: int = 32,
    loss_weights: Mapping[str, float] | None = None,
    prototype_temperature: float = 0.15,
    prototype_logit_weight: float = 1.0,
    prototype_ce_lambda: float = 0.5,
    graph_consistency_lambda: float = 0.5,
    graph_refresh_every: int = 5,
    graph_k: int = 15,
    branch_rank_specs_override: list[dict[str, Any]] | None = None,
    rank_loss_use_global_child_scores: bool = False,
    conditional_kl_specs_override: list[dict[str, Any]] | None = None,
    conditional_kl_table_override: pd.DataFrame | None = None,
    teacher_soft_kl_schedule: Mapping[str, Any] | None = None,
    class_balanced_pseudo_pool: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> tuple[QueryTeacherFeatureStudent, pd.DataFrame, dict[str, Any], QueryTeacherStudentDataset]:
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    weights = _student_loss_weights(loss_weights)
    if conditional_kl_specs_override is None:
        raise ValueError("rho-policy conditional KL requires conditional_kl_specs_override")

    n = len(bundle.query_index)
    pseudo_target = np.full(n, -1, dtype=np.int64)
    pseudo_weight = np.zeros(n, dtype=np.float32)
    selected_mask = np.zeros(n, dtype=bool)
    if not pseudo_df.empty:
        used = pseudo_df.loc[pseudo_df["used_for_training"].astype(bool)].copy()
        used = used.drop_duplicates("cell_id", keep="first")
        loc = pd.Series(np.arange(n), index=bundle.query_index)
        for row in used.itertuples(index=False):
            cell_id = str(row.cell_id)
            label = str(row.target_label)
            if cell_id in loc.index and label in bundle.label_to_idx:
                i = int(loc.loc[cell_id])
                pseudo_target[i] = int(bundle.label_to_idx[label])
                pseudo_weight[i] = float(getattr(row, "pseudo_weight", 1.0))
                selected_mask[i] = True
    class_balanced_cfg = dict(class_balanced_pseudo_pool or {})
    class_balanced_enabled = bool(class_balanced_cfg.get("enabled", False))
    class_balanced_samples_per_label = int(class_balanced_cfg.get("samples_per_label", 50))
    class_balanced_sample_by_weight = bool(class_balanced_cfg.get("sample_by_weight", False))
    pseudo_pool_by_label: dict[int, np.ndarray] = {}
    pseudo_pool_prob_by_label: dict[int, np.ndarray] = {}
    if class_balanced_enabled:
        for label_idx in sorted(set(pseudo_target[selected_mask].astype(int).tolist())):
            if label_idx >= 0:
                idxs = np.where(selected_mask & (pseudo_target == int(label_idx)))[0].astype(np.int64)
                pseudo_pool_by_label[int(label_idx)] = idxs
                if class_balanced_sample_by_weight and idxs.size:
                    w = np.asarray(pseudo_weight[idxs], dtype=np.float64)
                    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
                    total = float(w.sum())
                    if total > 0.0:
                        pseudo_pool_prob_by_label[int(label_idx)] = w / total

    label_score, score_ok = _build_label_score_arrays(bundle)
    dataset = QueryTeacherStudentDataset(
        z=bundle.z_teacher,
        protein=bundle.protein_features,
        batch_idx=bundle.batch_idx,
        teacher_probs=bundle.teacher_soft.loc[bundle.query_index, bundle.label_names].to_numpy(dtype=np.float32),
        teacher_confidence=bundle.teacher_confidence.loc[bundle.query_index].to_numpy(dtype=np.float32),
        pseudo_target=pseudo_target,
        pseudo_weight=pseudo_weight,
        selected_mask=selected_mask,
        label_score=label_score,
        score_ok=score_ok,
        indices=np.arange(n, dtype=np.int64),
    )
    train_idx, val_idx = _train_val_indices(n, seed=random_seed, val_fraction=0.1)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QueryTeacherFeatureStudent(
        z_dim=bundle.z_teacher.shape[1],
        protein_dim=bundle.protein_features.shape[1],
        n_batches=len(bundle.batch_names),
        n_labels=len(bundle.label_names),
        u_dim=u_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    parent_id_np = _parent_id_for_label(bundle)
    parent_id = torch.as_tensor(parent_id_np, dtype=torch.long, device=device)
    teacher_prob_np = bundle.teacher_soft.loc[bundle.query_index, bundle.label_names].to_numpy(dtype=np.float32)
    teacher_parent_np = parent_id_np[teacher_prob_np.argmax(axis=1)]
    teacher_confidence_np = bundle.teacher_confidence.loc[bundle.query_index].to_numpy(dtype=np.float32)
    branch_rank_specs = branch_rank_specs_override if branch_rank_specs_override is not None else _build_branch_rank_specs(bundle)
    conditional_kl_specs = list(conditional_kl_specs_override)
    conditional_kl_table = conditional_kl_table_override.copy() if conditional_kl_table_override is not None else pd.DataFrame()
    conditional_kl_table.to_csv(results_dir / "rho_policy_kl_node_table.csv", index=False)

    def _refresh() -> dict[str, Any]:
        return _refresh_student_protograph_state(
            model,
            dataset,
            batch_size=batch_size,
            device=device,
            pseudo_target=pseudo_target,
            selected=selected_mask,
            pseudo_weight=pseudo_weight,
            teacher_parent=teacher_parent_np,
            teacher_confidence=teacher_confidence_np,
            parent_id_for_label=parent_id_np,
            n_labels=len(bundle.label_names),
            graph_k=graph_k,
            prototype_temperature=prototype_temperature,
            prototype_logit_weight=prototype_logit_weight,
        )

    state = _refresh()
    prototypes_t = torch.as_tensor(state["prototypes"], dtype=torch.float32, device=device)
    neighbor_target_t = torch.as_tensor(state["neighbor_target"], dtype=torch.float32, device=device)
    graph_mask_t = torch.as_tensor(state["graph_mask"], dtype=torch.bool, device=device)
    weak_target_t = torch.as_tensor(state["weak_target"], dtype=torch.long, device=device)
    weak_mask_t = torch.as_tensor(state["weak_mask"], dtype=torch.bool, device=device)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    bad_epochs = 0
    pseudo_rng = np.random.default_rng(int(random_seed) + 7919)

    def _sample_balanced_pseudo_indices() -> np.ndarray:
        if not class_balanced_enabled or class_balanced_samples_per_label <= 0 or not pseudo_pool_by_label:
            return np.array([], dtype=np.int64)
        chunks: list[np.ndarray] = []
        for label_idx, idxs in pseudo_pool_by_label.items():
            if idxs.size == 0:
                continue
            replace = idxs.size < int(class_balanced_samples_per_label)
            probs = pseudo_pool_prob_by_label.get(int(label_idx)) if class_balanced_sample_by_weight else None
            chunks.append(
                pseudo_rng.choice(
                    idxs,
                    size=int(class_balanced_samples_per_label),
                    replace=replace,
                    p=probs,
                ).astype(np.int64)
            )
        if not chunks:
            return np.array([], dtype=np.int64)
        return np.concatenate(chunks).astype(np.int64)

    def _run_epoch(loader: DataLoader, *, train: bool, teacher_soft_kl_weight: float) -> dict[str, float]:
        model.train(train)
        totals: dict[str, float] = {}
        n_batches = 0
        for batch in loader:
            z = batch["z"].to(device)
            protein = batch["protein"].to(device)
            batch_id = batch["batch_idx"].to(device)
            teacher_probs = batch["teacher_probs"].to(device)
            teacher_conf = batch["teacher_confidence"].to(device)
            selected = batch["selected_mask"].to(device)
            pseudo_target_t = batch["pseudo_target"].to(device)
            pseudo_weight_t = batch["pseudo_weight"].to(device)
            label_score_t = batch["label_score"].to(device)
            score_ok_t = batch["score_ok"].to(device)
            batch_index = batch["index"].to(device)

            with torch.set_grad_enabled(train):
                out = model(z, protein, batch_id)
                proto_logits = _student_prototype_logits(out["u"], prototypes_t, prototype_temperature)
                logits = out["logits"] + float(prototype_logit_weight) * proto_logits
                probs = F.softmax(logits, dim=-1)
                pseudo_out = out
                pseudo_logits = logits
                pseudo_proto_logits = proto_logits
                pseudo_target_loss = pseudo_target_t
                pseudo_weight_loss = pseudo_weight_t
                pseudo_selected = selected
                if class_balanced_enabled:
                    pseudo_indices_np = _sample_balanced_pseudo_indices()
                    if pseudo_indices_np.size:
                        # Dataset tensors live on CPU; index them on CPU, then move the sampled
                        # pseudo mini-batch to the model device.
                        pseudo_indices = torch.as_tensor(pseudo_indices_np, dtype=torch.long)
                        pseudo_out = model(
                            dataset.z[pseudo_indices].to(device),
                            dataset.protein[pseudo_indices].to(device),
                            dataset.batch_idx[pseudo_indices].to(device),
                        )
                        pseudo_proto_logits = _student_prototype_logits(pseudo_out["u"], prototypes_t, prototype_temperature)
                        pseudo_logits = pseudo_out["logits"] + float(prototype_logit_weight) * pseudo_proto_logits
                        pseudo_target_loss = dataset.pseudo_target[pseudo_indices].to(device)
                        pseudo_weight_loss = dataset.pseudo_weight[pseudo_indices].to(device)
                        pseudo_selected = torch.ones_like(pseudo_target_loss, dtype=torch.bool, device=device)
                    else:
                        pseudo_target_loss = torch.empty(0, dtype=torch.long, device=device)
                        pseudo_weight_loss = torch.empty(0, dtype=torch.float32, device=device)
                        pseudo_selected = torch.empty(0, dtype=torch.bool, device=device)
                z_recon_loss = F.huber_loss(out["z_recon"], z, delta=1.0)
                protein_recon_loss = F.huber_loss(out["protein_recon"], protein, delta=1.0)
                pseudo_ce = (
                    _weighted_ce(pseudo_logits[pseudo_selected], pseudo_target_loss[pseudo_selected], pseudo_weight_loss[pseudo_selected])
                    if pseudo_selected.any()
                    else torch.zeros((), device=device)
                )
                prototype_ce = (
                    _weighted_ce(pseudo_proto_logits[pseudo_selected], pseudo_target_loss[pseudo_selected], pseudo_weight_loss[pseudo_selected])
                    if pseudo_selected.any()
                    else torch.zeros((), device=device)
                )
                teacher_weights = teacher_probs.max(dim=-1).values.square()
                teacher_kl = _rho_policy_conditional_teacher_kl(
                    probs,
                    teacher_probs,
                    specs=conditional_kl_specs,
                    teacher_weights=teacher_weights,
                )
                hard_supcon = (
                    _supervised_contrastive_loss(pseudo_out["u"][pseudo_selected], pseudo_target_loss[pseudo_selected], temperature=0.2)
                    if pseudo_selected.any() and weights["hard_anchor_supcon"] != 0.0
                    else torch.zeros((), device=device)
                )
                student_conf, student_pred = probs.max(dim=-1)
                teacher_pred = teacher_probs.argmax(dim=-1)
                parent_agree = parent_id[student_pred].eq(parent_id[teacher_pred])
                score_ok_pred = score_ok_t.gather(1, student_pred[:, None]).squeeze(1)
                self_mask = torch.zeros_like(selected, dtype=torch.bool)
                if weights["agreement_self_supcon"] != 0.0:
                    self_mask = (
                        (~selected)
                        & parent_agree
                        & student_conf.ge(0.90)
                        & teacher_conf.ge(0.75)
                        & score_ok_pred
                    )
                    self_mask = self_mask | weak_mask_t[batch_index]
                self_target = torch.where(weak_mask_t[batch_index], weak_target_t[batch_index], student_pred)
                self_supcon = (
                    _supervised_contrastive_loss(out["u"][self_mask], self_target[self_mask], temperature=0.2)
                    if self_mask.any()
                    else torch.zeros((), device=device)
                )
                graph_loss = _graph_consistency_loss(probs, neighbor_target_t[batch_index], graph_mask_t[batch_index])
                if weights["axis_rank_loss"] != 0.0:
                    rank_loss, n_rank_pairs = _axis_rank_loss(
                        probs,
                        label_score_t,
                        branch_rank_specs=branch_rank_specs,
                        teacher_probs=teacher_probs,
                        batch_indices=batch_index if bool(rank_loss_use_global_child_scores) else None,
                        max_pairs_per_child=256,
                        margin=0.1,
                    )
                else:
                    rank_loss, n_rank_pairs = torch.zeros((), device=device), 0
                loss = (
                    weights["z_recon_huber"] * z_recon_loss
                    + weights["protein_recon_huber"] * protein_recon_loss
                    + weights["weighted_pseudo_ce"] * pseudo_ce
                    + float(teacher_soft_kl_weight) * teacher_kl
                    + float(prototype_ce_lambda) * prototype_ce
                    + weights["hard_anchor_supcon"] * hard_supcon
                    + weights["agreement_self_supcon"] * self_supcon
                    + float(graph_consistency_lambda) * graph_loss
                    + weights["axis_rank_loss"] * rank_loss
                )
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

            row = {
                "loss": float(loss.detach().cpu()),
                "z_recon_loss": float(z_recon_loss.detach().cpu()),
                "protein_recon_loss": float(protein_recon_loss.detach().cpu()),
                "pseudo_ce_loss": float(pseudo_ce.detach().cpu()),
                "teacher_parent_kl_loss": float(teacher_kl.detach().cpu()),
                "prototype_ce_loss": float(prototype_ce.detach().cpu()),
                "hard_anchor_supcon_loss": float(hard_supcon.detach().cpu()),
                "agreement_self_supcon_loss": float(self_supcon.detach().cpu()),
                "graph_consistency_loss": float(graph_loss.detach().cpu()),
                "axis_rank_loss": float(rank_loss.detach().cpu()),
                "n_axis_rank_pairs_batch": float(n_rank_pairs),
                "n_pseudo_batch": float(pseudo_selected.sum().detach().cpu()),
                "n_self_supcon_batch": float(self_mask.sum().detach().cpu()),
                "n_graph_edges_batch": float(graph_mask_t[batch_index].sum().detach().cpu()),
                "prototype_logit_weight": float(prototype_logit_weight),
                "effective_teacher_soft_kl_weight": float(teacher_soft_kl_weight),
            }
            for key, value in row.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in totals.items()}

    epoch_iter = range(int(max_epochs))
    if verbose and tqdm is not None:
        epoch_iter = tqdm(epoch_iter, total=int(max_epochs), desc="protograph student training", leave=True)
    for epoch in epoch_iter:
        teacher_soft_kl_weight = _scheduled_teacher_soft_kl_weight(
            epoch=int(epoch),
            base_weight=float(weights["teacher_soft_kl"]),
            schedule=teacher_soft_kl_schedule,
        )
        if epoch > 0 and int(graph_refresh_every) > 0 and epoch % int(graph_refresh_every) == 0:
            state = _refresh()
            prototypes_t = torch.as_tensor(state["prototypes"], dtype=torch.float32, device=device)
            neighbor_target_t = torch.as_tensor(state["neighbor_target"], dtype=torch.float32, device=device)
            graph_mask_t = torch.as_tensor(state["graph_mask"], dtype=torch.bool, device=device)
            weak_target_t = torch.as_tensor(state["weak_target"], dtype=torch.long, device=device)
            weak_mask_t = torch.as_tensor(state["weak_mask"], dtype=torch.bool, device=device)
        train_stats = _run_epoch(train_loader, train=True, teacher_soft_kl_weight=teacher_soft_kl_weight)
        val_stats = _run_epoch(val_loader, train=False, teacher_soft_kl_weight=teacher_soft_kl_weight)
        row = {"epoch": int(epoch), "n_graph_edges_full": float(state["n_graph_edges"])}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"validation_{k}": v for k, v in val_stats.items()})
        history.append(row)
        progress_values = {
            "train": f"{row.get('train_loss', np.nan):.4f}",
            "val": f"{row.get('validation_loss', np.nan):.4f}",
            "ce": f"{row.get('train_pseudo_ce_loss', np.nan):.4f}",
            "pce": f"{row.get('train_prototype_ce_loss', np.nan):.4f}",
            "pkl": f"{row.get('train_teacher_parent_kl_loss', np.nan):.4f}",
            "klw": f"{row.get('train_effective_teacher_soft_kl_weight', np.nan):.3f}",
            "graph": f"{row.get('train_graph_consistency_loss', np.nan):.4f}",
            "rank": f"{row.get('train_axis_rank_loss', np.nan):.4f}",
        }
        if verbose and tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(progress_values)
        elif verbose:
            print(
                "epoch "
                f"{epoch + 1:03d}/{int(max_epochs):03d} "
                + " ".join(f"{key}={value}" for key, value in progress_values.items()),
                flush=True,
            )
        val_loss = float(val_stats.get("loss", np.inf))
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(early_stopping_patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    state = _refresh()
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / "student_training_history.csv", index=False)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "label_names": bundle.label_names,
            "protein_names": bundle.protein_names,
            "batch_names": bundle.batch_names,
            "u_dim": int(u_dim),
            "history": history,
            "loss_weights": weights,
            "teacher_soft_kl_schedule": dict(teacher_soft_kl_schedule or {}),
            "student_variant": "protograph",
            "prototypes": state["prototypes"],
            "prototype_active": state["prototype_active"],
            "prototype_temperature": float(prototype_temperature),
            "prototype_logit_weight": float(prototype_logit_weight),
            "conditional_kl_policy": "rho_policy_conditional",
            "conditional_kl_nodes": conditional_kl_table.to_dict(orient="records"),
            "rho_policy_kl_nodes": conditional_kl_table.to_dict(orient="records"),
        },
        results_dir / "student_model.pt",
    )
    model.to(device)
    state["conditional_kl_table"] = conditional_kl_table
    state["rho_policy_kl_table"] = conditional_kl_table
    state["pseudo_target"] = pseudo_target
    state["selected_mask"] = selected_mask
    return model, history_df, state, dataset


def predict_student(bundle: StudentDataBundle, model: QueryTeacherFeatureStudent, *, batch_size: int = 1024) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    logits_all: list[np.ndarray] = []
    u_all: list[np.ndarray] = []
    z_recon_all: list[np.ndarray] = []
    p_recon_all: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(bundle.query_index), int(batch_size)):
            stop = min(start + int(batch_size), len(bundle.query_index))
            z = torch.as_tensor(bundle.z_teacher[start:stop], dtype=torch.float32, device=device)
            protein = torch.as_tensor(bundle.protein_features[start:stop], dtype=torch.float32, device=device)
            batch_idx = torch.as_tensor(bundle.batch_idx[start:stop], dtype=torch.long, device=device)
            out = model(z, protein, batch_idx)
            logits_all.append(out["logits"].detach().cpu().numpy().astype(np.float32))
            u_all.append(out["u"].detach().cpu().numpy().astype(np.float32))
            z_recon_all.append(out["z_recon"].detach().cpu().numpy().astype(np.float32))
            p_recon_all.append(out["protein_recon"].detach().cpu().numpy().astype(np.float32))
    logits = np.concatenate(logits_all, axis=0)
    probs = F.softmax(torch.as_tensor(logits), dim=-1).numpy()
    soft = pd.DataFrame(probs, index=bundle.query_index, columns=bundle.label_names)
    return soft, np.concatenate(u_all, axis=0), np.concatenate(z_recon_all, axis=0), np.concatenate(p_recon_all, axis=0)


def run_bottomup_treeguard_student_from_bundle(
    *,
    bundle: StudentDataBundle,
    results_dir: Path,
    random_seed: int = 2026,
    max_epochs: int = 100,
    early_stopping_patience: int = 12,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    verbose: bool = True,
    run_label: str | None = "bottomup_treeguard",
    config_overrides: Mapping[str, Any] | None = None,
    conditional_kl_specs_override: list[dict[str, Any]] | None = None,
    conditional_kl_table_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_BOTTOMUP_CONFIG)
    if config_overrides:
        cfg.update({str(k): v for k, v in config_overrides.items()})

    pseudo_df, pseudo_by_class, vetoed, evidence = select_bottomup_treeguard_pseudolabels(
        bundle,
        max_marker_pseudo_per_class=int(cfg["max_marker_pseudo_per_class"]),
        max_no_marker_pseudo_per_class=int(cfg["max_no_marker_pseudo_per_class"]),
        max_hidden_rescue_per_child=int(cfg["max_hidden_rescue_per_child"]),
        posterior_threshold=float(cfg["posterior_threshold"]),
        wide_candidate_multiplier=int(cfg["wide_candidate_multiplier"]),
        parent_pool_threshold=float(cfg["parent_pool_threshold"]),
        child_conditional_threshold=float(cfg["child_conditional_threshold"]),
        enable_hidden_rescue=bool(cfg["enable_hidden_rescue"]),
        hard_contradiction_quantile=float(cfg["hard_contradiction_quantile"]),
        soft_contradiction_quantile=float(cfg["soft_contradiction_quantile"]),
        soft_contradiction_penalty=float(cfg["soft_contradiction_penalty"]),
        pseudo_selection_mode="adaptive_tail_robust_elbow",
        adaptive_config=cfg,
    )
    pseudo_df.to_csv(results_dir / "bottomup_treeguard_pseudolabel_cell_level.csv", index=False)
    pseudo_by_class.to_csv(results_dir / "bottomup_treeguard_pseudolabel_by_class.csv", index=False)
    vetoed.to_csv(results_dir / "bottomup_treeguard_vetoed_candidates.csv", index=False)
    evidence.to_csv(results_dir / "bottomup_treeguard_anchor_evidence_summary.csv", index=False)
    pseudo_df.to_csv(results_dir / "student_pseudolabel_cell_level.csv", index=False)
    pseudo_by_class.to_csv(results_dir / "student_pseudolabel_by_class.csv", index=False)
    pseudo_df.to_csv(results_dir / "student_adaptive_tail_pseudolabel_cell_level.csv", index=False)
    pseudo_by_class.to_csv(results_dir / "student_adaptive_tail_pseudolabel_by_class.csv", index=False)
    pseudo_df.attrs.get("adaptive_candidate_audit", pd.DataFrame()).to_csv(results_dir / "student_adaptive_tail_candidate_audit.csv", index=False)
    pseudo_df.attrs.get("adaptive_missing_or_low_coverage", pd.DataFrame()).to_csv(results_dir / "student_adaptive_tail_missing_or_low_coverage.csv", index=False)
    pseudo_df.attrs.get("adaptive_reliability_by_class", pseudo_by_class.copy()).to_csv(results_dir / "student_adaptive_tail_reliability_by_class.csv", index=False)
    pseudo_df.attrs.get("adaptive_anchor_floor_topup", pd.DataFrame()).to_csv(results_dir / "student_adaptive_tail_anchor_floor_topup.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_selected_rows": int(pseudo_df.shape[0]),
                "n_used_for_training": int(pseudo_df["used_for_training"].sum()) if not pseudo_df.empty else 0,
                "pseudo_precision": float(pseudo_df.loc[pseudo_df["used_for_training"].astype(bool), "is_correct_pseudolabel"].mean())
                if (not pseudo_df.empty and pseudo_df["used_for_training"].any())
                else np.nan,
                "n_vetoed_candidates": int(vetoed.shape[0]),
            }
        ]
    ).to_csv(results_dir / "student_pseudolabel_overall.csv", index=False)

    if cfg.get("branch_rank_specs_override") is None and bool(cfg.get("auto_tree_rank_specs", True)):
        cfg["branch_rank_specs_override"] = _build_generic_branch_rank_specs(
            bundle,
            branches=None,
            use_teacher_parent_pool=True,
            min_locality_weight=float(cfg.get("rank_min_locality_weight", 0.05)),
        )
    rank_specs_for_training = cfg.get("branch_rank_specs_override")
    _rank_specs_summary(rank_specs_for_training or []).to_csv(results_dir / "bottomup_treeguard_rank_specs.csv", index=False)
    if conditional_kl_table_override is not None and not conditional_kl_table_override.empty:
        conditional_kl_table_override.to_csv(results_dir / "rho_policy_kl_node_table.csv", index=False)

    loss_weights = {"axis_rank_loss": float(cfg["rank_loss_weight"])}
    loss_weight_overrides = dict(cfg.get("student_loss_weight_overrides") or {})
    loss_weights.update({str(k): float(v) for k, v in loss_weight_overrides.items()})
    model, history, proto_state, dataset = train_student_model_protograph(
        bundle,
        pseudo_df,
        results_dir=results_dir,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        random_seed=random_seed,
        loss_weights=loss_weights,
        prototype_logit_weight=float(cfg["prototype_logit_weight"]),
        prototype_ce_lambda=float(cfg["prototype_ce_lambda"]),
        graph_consistency_lambda=float(cfg["graph_consistency_lambda"]),
        graph_refresh_every=int(cfg["graph_refresh_every"]),
        graph_k=int(cfg["graph_k"]),
        branch_rank_specs_override=rank_specs_for_training,
        rank_loss_use_global_child_scores=bool(cfg.get("rank_loss_use_global_child_scores", False)),
        conditional_kl_specs_override=conditional_kl_specs_override,
        conditional_kl_table_override=conditional_kl_table_override,
        teacher_soft_kl_schedule=cfg.get("teacher_soft_kl_schedule"),
        class_balanced_pseudo_pool=None,
        verbose=verbose,
    )
    device = next(model.parameters()).device
    prototypes_t = torch.as_tensor(proto_state["prototypes"], dtype=torch.float32, device=device)
    logits, u_student, z_recon, protein_recon, mlp_logits = _predict_student_protograph_arrays(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        prototypes=prototypes_t,
        prototype_temperature=0.15,
        prototype_logit_weight=float(cfg["prototype_logit_weight"]),
    )
    soft = pd.DataFrame(F.softmax(torch.as_tensor(logits), dim=-1).numpy(), index=bundle.query_index, columns=bundle.label_names)
    mlp_soft = pd.DataFrame(F.softmax(torch.as_tensor(mlp_logits), dim=-1).numpy(), index=bundle.query_index, columns=bundle.label_names)
    evals = evaluate_and_write_student(
        bundle,
        soft=soft,
        u_student=u_student,
        z_recon=z_recon,
        protein_recon=protein_recon,
        pseudo_df=pseudo_df,
        results_dir=results_dir,
    )
    pred = soft.idxmax(axis=1).astype(str)
    mlp_pred = mlp_soft.idxmax(axis=1).astype(str)
    safety_guard = _write_student_safety_guard_report(
        bundle=bundle,
        pred=pred,
        pseudo_by_class=pseudo_by_class,
        results_dir=results_dir,
    )
    schedule = cfg.get("teacher_soft_kl_schedule") or {}
    first_kl_weight = (
        float(history["train_effective_teacher_soft_kl_weight"].iloc[0])
        if not history.empty and "train_effective_teacher_soft_kl_weight" in history.columns
        else float(_student_loss_weights(None)["teacher_soft_kl"])
    )
    final_kl_weight = (
        float(history["train_effective_teacher_soft_kl_weight"].iloc[-1])
        if not history.empty and "train_effective_teacher_soft_kl_weight" in history.columns
        else first_kl_weight
    )
    pd.DataFrame(
        [
            {
                "label": label,
                "prototype_active": bool(proto_state["prototype_active"][idx]),
                "n_anchor": int(np.sum(proto_state["selected_mask"] & (proto_state["pseudo_target"] == idx))),
                "prototype_norm": float(np.linalg.norm(proto_state["prototypes"][idx])),
            }
            for idx, label in enumerate(bundle.label_names)
        ]
    ).to_csv(results_dir / "student_prototype_by_class.csv", index=False)
    pd.DataFrame(
        [
            {
                "rank_loss_weight": float(cfg["rank_loss_weight"]),
                "rank_loss_use_global_child_scores": bool(cfg.get("rank_loss_use_global_child_scores", False)),
                "rank_loss_source": "auto_tree_direct_child_marker_rank_low_weight" if rank_specs_for_training is not None else "base_student_axis_rank_specs_low_weight",
                "n_rank_specs": int(len(rank_specs_for_training or [])),
                "min_rank_spec_weight": float(_rank_specs_summary(rank_specs_for_training or [])["rank_weight"].min()) if rank_specs_for_training else np.nan,
                "max_rank_spec_weight": float(_rank_specs_summary(rank_specs_for_training or [])["rank_weight"].max()) if rank_specs_for_training else np.nan,
                "conditional_kl_policy": "rho_policy_conditional",
                "conditional_kl_specs_active": conditional_kl_specs_override is not None,
                "conditional_kl_table": "rho_policy_kl_node_table.csv",
                "teacher_soft_kl": float(_student_loss_weights(None)["teacher_soft_kl"]),
                "teacher_soft_kl_schedule_active": bool(schedule),
                "teacher_soft_kl_schedule": json.dumps(schedule, sort_keys=True) if schedule else "",
                "initial_effective_teacher_soft_kl_weight": float(first_kl_weight),
                "final_effective_teacher_soft_kl_weight": float(final_kl_weight),
                "effective_teacher_soft_kl_weight": float(final_kl_weight),
                "student_loss_weight_overrides": json.dumps(loss_weight_overrides, sort_keys=True),
                "hard_anchor_supcon_weight": float(loss_weights.get("hard_anchor_supcon", DEFAULT_STUDENT_LOSS_WEIGHTS["hard_anchor_supcon"])),
                "agreement_self_supcon_weight": float(loss_weights.get("agreement_self_supcon", DEFAULT_STUDENT_LOSS_WEIGHTS["agreement_self_supcon"])),
                "parentlocal_marker_kl_loss_active": False,
                "parentlocal_node_prototype_active": False,
                "class_level_prototype_from_used_for_training_anchors": True,
                "final_train_axis_rank_loss": float(history.filter(like="train_axis_rank_loss").tail(1).iloc[0, 0]) if not history.empty and any("train_axis_rank_loss" in c for c in history.columns) else np.nan,
                "final_train_n_axis_rank_pairs_batch": float(history.filter(like="train_n_axis_rank_pairs_batch").tail(1).iloc[0, 0]) if not history.empty and any("train_n_axis_rank_pairs_batch" in c for c in history.columns) else np.nan,
            }
        ]
    ).to_csv(results_dir / "bottomup_treeguard_rank_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "teacher_student_agreement": float(bundle.teacher_pred.reindex(bundle.query_index).astype(str).eq(pred).mean()),
                "mlp_final_agreement": float(mlp_pred.eq(pred).mean()),
                "n_changed_by_prototype": int((~mlp_pred.eq(pred)).sum()),
                "mean_teacher_confidence": float(bundle.teacher_confidence.reindex(bundle.query_index).astype(float).mean()),
                "mean_mlp_confidence": float(mlp_soft.max(axis=1).mean()),
                "mean_final_confidence": float(soft.max(axis=1).mean()),
            }
        ]
    ).to_csv(results_dir / "student_teacher_vs_final_logit_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "graph_k": int(cfg["graph_k"]),
                "graph_refresh_every": int(cfg["graph_refresh_every"]),
                "n_graph_edges": int(proto_state["n_graph_edges"]),
                "n_graph_rows": int(np.sum(proto_state["graph_mask"])),
                "n_weak_self_supcon": int(np.sum(proto_state["weak_mask"])),
            }
        ]
    ).to_csv(results_dir / "student_graph_consistency_summary.csv", index=False)
    knn_summary = _student_knn_purity_summary(bundle, u_student=u_student, pred=pred, results_dir=results_dir, label_col="true_label")
    confusion_paths = write_student_confusion_heatmap(
        evals["obs"],
        results_dir=results_dir,
        pred_col="student_pred_label",
        label_col="true_label",
        run_label=run_label,
    )
    config_for_json = dict(cfg)
    if config_for_json.get("branch_rank_specs_override") is not None:
        config_for_json["branch_rank_specs_override"] = {
            "n_specs": int(len(rank_specs_for_training or [])),
            "summary_csv": "bottomup_treeguard_rank_specs.csv",
        }
    (results_dir / "bottomup_treeguard_student_config.json").write_text(
        json.dumps(
            {
                "run_type": "bottomup_treeguard_student",
                "config": config_for_json,
                "conditional_kl_policy": "rho_policy_conditional",
                "conditional_kl_specs_active": conditional_kl_specs_override is not None,
                "conditional_kl_table": "rho_policy_kl_node_table.csv",
                "teacher_soft_kl": float(_student_loss_weights(None)["teacher_soft_kl"]),
                "teacher_soft_kl_schedule": schedule,
                "initial_effective_teacher_soft_kl_weight": float(first_kl_weight),
                "final_effective_teacher_soft_kl_weight": float(final_kl_weight),
                "effective_teacher_soft_kl_weight": float(final_kl_weight),
                "student_loss_weight_overrides": loss_weight_overrides,
                "hard_anchor_supcon_weight": float(loss_weights.get("hard_anchor_supcon", DEFAULT_STUDENT_LOSS_WEIGHTS["hard_anchor_supcon"])),
                "agreement_self_supcon_weight": float(loss_weights.get("agreement_self_supcon", DEFAULT_STUDENT_LOSS_WEIGHTS["agreement_self_supcon"])),
                "parentlocal_marker_kl_loss_active": False,
                "parentlocal_node_prototype_active": False,
                "enable_hidden_rescue": bool(cfg["enable_hidden_rescue"]),
                "custom_branch_rank_specs_active": cfg.get("branch_rank_specs_override") is not None,
                "rank_locality_weight_rule": "clip(2 / n_descendant_leaves, rank_min_locality_weight, 1.0)",
                "selection_note": "leaf-local candidate ranking first; ancestor markers only veto/penalize contradiction.",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "bundle": bundle,
        "pseudo_cell_level": pseudo_df,
        "pseudo_by_class": pseudo_by_class,
        "vetoed_candidates": vetoed,
        "anchor_evidence": evidence,
        "history": history,
        "evaluation": evals,
                "confusion_paths": confusion_paths,
        "knn_purity_summary": knn_summary,
        "safety_guard": safety_guard,
        "results_dir": results_dir,
    }


def train_student(*args, **kwargs):
    return run_bottomup_treeguard_student_from_bundle(*args, **kwargs)
