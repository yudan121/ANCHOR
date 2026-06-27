#!/usr/bin/env python
"""Generate the synthetic mechanism v3 inputs used by ANCHOR.

The simulation is intentionally abstract: labels are Lineage_A/B/C/D leaves
and features are numbered genes/proteins. It is designed to test marker-guided
anchor transfer without relying on biological marker assumptions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CELLTYPE_COL = "cell_type"
ORIGINAL_CELLTYPE_COL = "original_cell_type"
BATCH_COL = "batch"
SPLIT_COL = "split"
REFERENCE_NAME = "reference"
QUERY_NAME = "query"
PROTEIN_KEY = "protein_expression"
HELDOUT_PROTEIN_KEY = "protein_expression_heldout"

FINE_LABELS = (
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "C1",
    "C2",
    "D1",
    "D2",
    "D3",
)

LINEAGE_CHILDREN: dict[str, list[str]] = {
    "Lineage_A": ["A1", "A2", "A3"],
    "Lineage_B": ["B1", "B2"],
    "Lineage_C": ["C1", "C2"],
    "Lineage_D": ["D1", "D2", "D3"],
}

PARTIAL_LABEL_HIDDEN_BRANCHES = ("Lineage_B", "Lineage_C", "Lineage_D")
PROTEIN_NAMES = tuple(f"P{i:02d}" for i in range(1, 25))


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 2027
    n_genes: int = 1000
    n_reference: int = 5600
    n_query: int = 6400
    gene_nb_theta: float = 8.0
    protein_nb_theta: float = 24.0
    parent_log_shift: float = 1.05
    rna_easy_leaf_log_shift: float = 1.35
    rna_simple_ambiguous_leaf_log_shift: float = 0.08
    rna_gradient_ambiguous_leaf_log_shift: float = 0.06
    rna_mixed_residual_log_shift: float = 0.04
    rna_shared_jitter_sd: float = 0.035
    rna_cell_noise_sd: float = 0.13
    gene_dropout_base: float = 0.040
    rna_leaf_signal_attenuation_fraction: float = 0.06
    rna_leaf_signal_attenuation_min: float = 0.08
    rna_leaf_signal_attenuation_max: float = 0.35
    rna_easy_offtarget_leak_fraction: float = 0.02
    rna_easy_offtarget_leak_log_shift: float = 0.18
    reference_batch_effect_sd: float = 0.08
    query_batch_effect_sd: float = 0.13
    reference_query_leaf_shift_sd: float = 0.045
    protein_lineage_log_shift: float = 0.75
    protein_simple_log_shift: float = 1.35
    protein_simple_negative_log_shift: float = -0.35
    protein_gradient_log_shift: float = 1.15
    protein_mixed_log_shift: float = 1.38
    protein_hidden_pattern_strength: float = 1.20
    protein_noise_sd: float = 0.16
    protein_dropout_base: float = 0.038
    protein_batch_effect_sd: float = 0.16
    protein_cell_scale_sd: float = 0.18


def marker_node(
    name: str,
    pos: Sequence[str] = (),
    neg: Sequence[str] = (),
    children: Sequence[dict[str, Any]] = (),
    *,
    hidden_branch: bool = False,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "name": str(name),
        "positive_markers": list(pos),
        "negative_markers": list(neg),
        "children": list(children),
    }
    if hidden_branch:
        node["hidden_branch"] = True
    return node


def build_simulation_marker_tree(*, partial_label_mode: bool = False) -> dict[str, Any]:
    """Build the protein marker tree for full or partial-label mode."""

    return marker_node(
        "root",
        children=[
            marker_node(
                "Lineage_A",
                pos=["P01"],
                neg=["P05", "P08", "P12"],
                children=[marker_node("A1"), marker_node("A2"), marker_node("A3")],
            ),
            marker_node(
                "Lineage_B",
                pos=["P05"],
                neg=["P01", "P08", "P12"],
                hidden_branch=partial_label_mode,
                children=[
                    marker_node("B1", pos=["P06"], neg=["P07"]),
                    marker_node("B2", pos=["P07"], neg=["P06"]),
                ],
            ),
            marker_node(
                "Lineage_C",
                pos=["P08"],
                neg=["P01", "P05", "P12"],
                hidden_branch=partial_label_mode,
                children=[
                    marker_node("C1", pos=["P09"], neg=["P10"]),
                    marker_node("C2", pos=["P10"], neg=["P09"]),
                ],
            ),
            marker_node(
                "Lineage_D",
                pos=["P12"],
                neg=["P01", "P05", "P08"],
                hidden_branch=partial_label_mode,
                children=[
                    marker_node("D1", pos=["P13"], neg=["P14", "P15"]),
                    marker_node("D2", pos=["P14"], neg=["P13", "P15"]),
                    marker_node("D3", pos=["P15"], neg=["P13", "P14"]),
                ],
            ),
        ],
    )


def walk_tree(node: Mapping[str, Any]):
    yield node
    for child in node.get("children", []) or []:
        yield from walk_tree(child)


def children_map(tree: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        str(node["name"]): [str(child["name"]) for child in node.get("children", []) or []]
        for node in walk_tree(tree)
        if node.get("children")
    }


MARKER_TREE_FULL = build_simulation_marker_tree(partial_label_mode=False)
CHILDREN = children_map(MARKER_TREE_FULL)
PARENT = {child: parent for parent, values in CHILDREN.items() for child in values}


def top_lineage(label: str) -> str:
    for lineage, labels in LINEAGE_CHILDREN.items():
        if str(label) in labels:
            return lineage
    return str(label)


def label_weights() -> dict[str, float]:
    return {
        "A1": 1.00,
        "A2": 0.95,
        "A3": 0.85,
        "B1": 0.75,
        "B2": 0.65,
        "C1": 0.65,
        "C2": 0.55,
        "D1": 0.55,
        "D2": 0.50,
        "D3": 0.45,
    }


def allocation_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    labels = list(weights)
    values = np.asarray([float(weights[x]) for x in labels], dtype=float)
    values = values / values.sum()
    raw = values * int(total)
    counts = np.floor(raw).astype(int)
    remainder = int(total) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return {label: int(count) for label, count in zip(labels, counts)}


def sample_negative_binomial(rng: np.random.Generator, mu: np.ndarray, theta: float) -> np.ndarray:
    mu = np.clip(mu, 1e-8, None)
    p = theta / (theta + mu)
    return rng.negative_binomial(theta, p).astype(np.int32)


def build_expression_templates(
    config: SimulationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, np.ndarray]]:
    rng = np.random.default_rng(config.seed + 17)
    gene_names = [f"G{i:03d}" for i in range(1, config.n_genes + 1)]
    labels = list(FINE_LABELS)
    log_gene = pd.DataFrame(
        rng.normal(loc=-1.30, scale=0.34, size=(len(labels), config.n_genes)),
        index=labels,
        columns=gene_names,
    )
    lineage_blocks = {
        "Lineage_A": np.arange(0, 80),
        "Lineage_B": np.arange(80, 160),
        "Lineage_C": np.arange(160, 240),
        "Lineage_D": np.arange(240, 320),
    }
    leaf_block_size = 16
    leaf_blocks = {
        label: np.arange(360 + i * leaf_block_size, 360 + (i + 1) * leaf_block_size)
        for i, label in enumerate(labels)
    }
    for label in labels:
        lineage = top_lineage(label)
        log_gene.loc[label, log_gene.columns[lineage_blocks[lineage]]] += config.parent_log_shift
        if lineage == "Lineage_A":
            shift = config.rna_easy_leaf_log_shift
        elif lineage == "Lineage_B":
            shift = config.rna_simple_ambiguous_leaf_log_shift
        elif lineage == "Lineage_C":
            shift = config.rna_gradient_ambiguous_leaf_log_shift
        else:
            shift = config.rna_mixed_residual_log_shift
        log_gene.loc[label, log_gene.columns[leaf_blocks[label]]] += shift

    # Keep the easy branch focused on compact leaf RNA blocks, not accidental
    # whole-transcriptome differences.
    a_children = LINEAGE_CHILDREN["Lineage_A"]
    a_leaf_union = np.concatenate([leaf_blocks[label] for label in a_children])
    a_nonleaf = np.setdiff1d(np.arange(config.n_genes), a_leaf_union)
    a_mean = log_gene.loc[a_children].mean(axis=0).to_numpy()
    a_shared = a_mean + rng.normal(0.0, config.rna_shared_jitter_sd, size=config.n_genes)
    for label in a_children:
        residual = log_gene.loc[label].to_numpy() - a_mean
        values = log_gene.loc[label].to_numpy()
        values[a_nonleaf] = a_shared[a_nonleaf] + 0.05 * residual[a_nonleaf]
        log_gene.loc[label] = values

    for lineage in ["Lineage_B", "Lineage_C", "Lineage_D"]:
        child_labels = LINEAGE_CHILDREN[lineage]
        mean_profile = log_gene.loc[child_labels].mean(axis=0).to_numpy()
        shared_jitter = rng.normal(0.0, config.rna_shared_jitter_sd, size=config.n_genes)
        shared_profile = mean_profile + shared_jitter
        for label in child_labels:
            residual = log_gene.loc[label].to_numpy() - mean_profile
            scale = 0.10 if lineage == "Lineage_D" else 0.08
            log_gene.loc[label] = shared_profile + scale * residual

    protein_log = pd.DataFrame(
        rng.normal(loc=0.90, scale=0.14, size=(len(labels), len(PROTEIN_NAMES))),
        index=labels,
        columns=list(PROTEIN_NAMES),
    )
    lineage_marker = {
        "Lineage_A": "P01",
        "Lineage_B": "P05",
        "Lineage_C": "P08",
        "Lineage_D": "P12",
    }
    for label in labels:
        protein_log.loc[label, lineage_marker[top_lineage(label)]] += config.protein_lineage_log_shift
    return log_gene, protein_log, gene_names, leaf_blocks


def add_label_protein_pattern(
    label: str,
    protein_base: np.ndarray,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cells = protein_base.shape[0]
    axis_score = np.full(n_cells, np.nan, dtype=np.float32)
    transition = np.zeros(n_cells, dtype=bool)

    def idx(marker: str) -> int:
        return PROTEIN_NAMES.index(marker)

    if label in {"A1", "A2", "A3"}:
        marker = {"A1": "P02", "A2": "P03", "A3": "P04"}[label]
        neg = [m for m in ["P02", "P03", "P04"] if m != marker]
        protein_base[:, idx(marker)] += 0.70
        for marker_name in neg:
            protein_base[:, idx(marker_name)] -= 0.15

    elif label in {"B1", "B2"}:
        pos, neg = ("P06", "P07") if label == "B1" else ("P07", "P06")
        strength = rng.normal(config.protein_simple_log_shift, 0.34, size=n_cells)
        protein_base[:, idx(pos)] += strength
        protein_base[:, idx(neg)] += config.protein_simple_negative_log_shift + rng.normal(0.0, 0.12, size=n_cells)
        axis_score = strength.astype(np.float32)
        transition = np.abs(strength - np.median(strength)) < 0.18

    elif label in {"C1", "C2"}:
        center = -0.68 if label == "C1" else 0.68
        t = rng.normal(center, 0.60, size=n_cells)
        t += 0.20 * np.sin(2.2 * t) + rng.normal(0.0, 0.12, size=n_cells)
        high = np.tanh(t)
        protein_base[:, idx("P09")] += config.protein_gradient_log_shift * (-high)
        protein_base[:, idx("P10")] += config.protein_gradient_log_shift * high
        protein_base[:, idx("P11")] += 0.45 * np.sin(2.0 * t)
        axis_score = t.astype(np.float32)
        transition = np.abs(t) < 0.28

    elif label in {"D1", "D2", "D3"}:
        cls = {"D1": 0, "D2": 1, "D3": 2}[label]
        centers = np.asarray([0.0, 2.10, -2.10])
        theta = centers[cls] + rng.normal(0.0, 0.56, size=n_cells)
        radius = rng.normal(1.0 + 0.045 * cls, 0.17, size=n_cells)
        x = radius * np.cos(theta) + rng.normal(0.0, 0.13, size=n_cells)
        y = radius * np.sin(theta) + rng.normal(0.0, 0.13, size=n_cells)
        marker_logits = np.vstack(
            [
                0.72 * np.cos(theta - center)
                + 0.21 * np.sin(2.0 * theta + 0.45 * j)
                + rng.normal(0.0, 0.15, size=n_cells)
                for j, center in enumerate(centers)
            ]
        ).T
        for j, marker in enumerate(["P13", "P14", "P15"]):
            value = 0.22 + 0.50 * np.tanh(marker_logits[:, j])
            if j == cls:
                value = value + 0.38
            else:
                value = value - 0.14
            protein_base[:, idx(marker)] += config.protein_mixed_log_shift * value
        protein_base[:, idx("P16")] += config.protein_hidden_pattern_strength * (
            0.41 * np.sin(1.6 * x) + 0.18 * np.cos(y)
        )
        protein_base[:, idx("P17")] += config.protein_hidden_pattern_strength * (
            0.37 * np.cos(1.4 * y - 0.3 * cls)
        )
        protein_base[:, idx("P18")] += config.protein_hidden_pattern_strength * (
            0.33 * np.sin(x - y + 0.5 * cls)
        )
        sorted_logits = np.sort(marker_logits, axis=1)
        margin = sorted_logits[:, -1] - sorted_logits[:, -2]
        axis_score = (marker_logits[:, cls] - np.mean(marker_logits, axis=1)).astype(np.float32)
        transition = margin < 0.17

    return protein_base, axis_score, transition


def make_split(
    *,
    split_name: str,
    counts_by_label: Mapping[str, int],
    batch_names: Sequence[str],
    log_gene: pd.DataFrame,
    protein_log: pd.DataFrame,
    gene_names: Sequence[str],
    leaf_blocks: Mapping[str, np.ndarray],
    config: SimulationConfig,
    rng: np.random.Generator,
) -> ad.AnnData:
    xs: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    obs_rows: list[dict[str, Any]] = []
    for label, n_cells in counts_by_label.items():
        if n_cells <= 0:
            continue
        batches = rng.choice(list(batch_names), size=n_cells, replace=True)
        lib = rng.lognormal(mean=8.1, sigma=0.28, size=n_cells)
        gene_log = np.repeat(log_gene.loc[label].to_numpy(dtype=float)[None, :], n_cells, axis=0)
        if split_name == QUERY_NAME and top_lineage(label) != "Lineage_A":
            block = leaf_blocks[label]
            gene_log[:, block] += rng.normal(0.0, config.reference_query_leaf_shift_sd, size=(n_cells, len(block)))

        own_leaf_block = leaf_blocks[label]
        attenuated = rng.random(n_cells) < config.rna_leaf_signal_attenuation_fraction
        if attenuated.any():
            attenuation = rng.uniform(
                config.rna_leaf_signal_attenuation_min,
                config.rna_leaf_signal_attenuation_max,
                size=int(attenuated.sum()),
            )
            gene_log[np.ix_(np.where(attenuated)[0], own_leaf_block)] -= attenuation[:, None]
        if top_lineage(label) == "Lineage_A":
            leakage = rng.random(n_cells) < config.rna_easy_offtarget_leak_fraction
            if leakage.any():
                sibling_labels = [x for x in LINEAGE_CHILDREN["Lineage_A"] if x != label]
                leak_targets = rng.choice(sibling_labels, size=int(leakage.sum()), replace=True)
                leakage_rows = np.where(leakage)[0]
                for target in sibling_labels:
                    rows = leakage_rows[leak_targets == target]
                    if len(rows):
                        gene_log[np.ix_(rows, leaf_blocks[target])] += config.rna_easy_offtarget_leak_log_shift
        gene_log += rng.normal(0.0, config.rna_cell_noise_sd, size=gene_log.shape)
        gene_mu = np.exp(gene_log) * (lib[:, None] / np.median(lib))
        for batch in sorted(set(batches)):
            pos = np.where(batches == batch)[0]
            sd = config.query_batch_effect_sd if split_name == QUERY_NAME else config.reference_batch_effect_sd
            gene_mu[pos] *= np.exp(rng.normal(0.0, sd, size=gene_mu.shape[1]))[None, :]
        gene_counts = sample_negative_binomial(rng, gene_mu, config.gene_nb_theta)
        gene_counts[rng.random(gene_counts.shape) < config.gene_dropout_base] = 0

        protein_base = np.repeat(protein_log.loc[label].to_numpy(dtype=float)[None, :], n_cells, axis=0)
        protein_base, axis_score, transition = add_label_protein_pattern(label, protein_base, rng, config)
        protein_base += rng.normal(0.0, config.protein_noise_sd, size=protein_base.shape)
        for batch in sorted(set(batches)):
            pos = np.where(batches == batch)[0]
            protein_base[pos] += rng.normal(0.0, config.protein_batch_effect_sd, size=protein_base.shape[1])
        protein_mu = np.exp(protein_base) * rng.lognormal(
            mean=0.0,
            sigma=config.protein_cell_scale_sd,
            size=(n_cells, 1),
        )
        protein_counts = sample_negative_binomial(rng, protein_mu, config.protein_nb_theta)
        protein_counts[rng.random(protein_counts.shape) < config.protein_dropout_base] = 0

        xs.append(gene_counts)
        ps.append(protein_counts)
        for i in range(n_cells):
            obs_rows.append(
                {
                    CELLTYPE_COL: label,
                    ORIGINAL_CELLTYPE_COL: label,
                    BATCH_COL: str(batches[i]),
                    SPLIT_COL: split_name,
                    "top_lineage": top_lineage(label),
                    "axis_parent": top_lineage(label),
                    "axis_continuum_t": float(axis_score[i]) if np.isfinite(axis_score[i]) else np.nan,
                    "is_transition_cell": bool(transition[i]),
                }
            )
    x_all = np.vstack(xs).astype(np.int32)
    p_all = np.vstack(ps).astype(np.int32)
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"{split_name}_cell_{i:05d}" for i in range(obs.shape[0])]
    adata = ad.AnnData(X=sp.csr_matrix(x_all), obs=obs, var=pd.DataFrame(index=list(gene_names)))
    adata.layers["counts"] = adata.X.copy()
    adata.obsm[PROTEIN_KEY] = pd.DataFrame(p_all, index=adata.obs_names, columns=list(PROTEIN_NAMES))
    adata.uns[f"{PROTEIN_KEY}_names"] = list(PROTEIN_NAMES)
    adata.uns["protein_names"] = list(PROTEIN_NAMES)
    return adata


def collapse_partial_reference_labels(adata: ad.AnnData) -> ad.AnnData:
    out = adata.copy()
    labels = out.obs[CELLTYPE_COL].astype(str).copy()
    collapse = {child: parent for parent in PARTIAL_LABEL_HIDDEN_BRANCHES for child in LINEAGE_CHILDREN[parent]}
    out.obs[CELLTYPE_COL] = labels.map(lambda x: collapse.get(str(x), str(x))).astype(str).values
    return out


def remove_reference_observed_protein(adata: ad.AnnData) -> ad.AnnData:
    out = adata.copy()
    if PROTEIN_KEY in out.obsm:
        protein_names = list(PROTEIN_NAMES)
        out.obsm[HELDOUT_PROTEIN_KEY] = pd.DataFrame(
            0.0,
            index=out.obs_names.astype(str),
            columns=protein_names,
            dtype=np.float32,
        )
        out.uns[f"{HELDOUT_PROTEIN_KEY}_names"] = protein_names
        del out.obsm[PROTEIN_KEY]
        out.uns.pop(f"{PROTEIN_KEY}_names", None)
        out.uns.pop("protein_names", None)
    return out


def protein_arcsinh_frame(adata: ad.AnnData) -> pd.DataFrame:
    value = adata.obsm[PROTEIN_KEY]
    if isinstance(value, pd.DataFrame):
        raw = value.copy()
    else:
        raw = pd.DataFrame(np.asarray(value), index=adata.obs_names, columns=list(PROTEIN_NAMES))
    return np.arcsinh(raw.astype(float) / 5.0)


def _marker_list(node: Mapping[str, Any], key: str) -> list[str]:
    return [str(x) for x in (node.get(key, []) or [])]


def marker_design_audit(tree: Mapping[str, Any], config: SimulationConfig) -> pd.DataFrame:
    rows = []
    rna_blocks = {
        "Lineage_A": "G001-G080 parent; A leaves G361-G408, attenuated/leaky in a subset of cells",
        "Lineage_B": "G081-G160 parent; weak sibling residual",
        "Lineage_C": "G161-G240 parent; weak sibling residual",
        "Lineage_D": "G241-G320 parent; weak nonlinear/complement residual",
    }
    axis_type = {
        "Lineage_A": "RNA-easy positive control",
        "Lineage_B": "RNA-ambiguous simple protein markers",
        "Lineage_C": "RNA-ambiguous continuous protein gradient",
        "Lineage_D": "RNA-weak nonlinear/complementary protein pattern",
    }
    expected = {
        "Lineage_A": "RNA-only should be high",
        "Lineage_B": "protein should exceed RNA",
        "Lineage_C": "protein gradient helps but middle cells remain ambiguous",
        "Lineage_D": "combined protein pattern needed; no single marker perfectly solves",
    }
    for node in walk_tree(tree):
        name = str(node["name"])
        lineage = name if name.startswith("Lineage_") else top_lineage(name)
        note = {
            "Lineage_A": "no A sibling marker specs in tree; RNA/teacher should define A leaves",
            "Lineage_B": "P06 vs P07 simple noisy opposing markers",
            "Lineage_C": "P09/P10 continuous opposing gradient plus P11 auxiliary",
            "Lineage_D": "intermediate-overlap P13/P14/P15 mixed-complement pattern plus P16-P18 auxiliary proteins",
        }.get(lineage, "")
        rows.append(
            {
                "node": name,
                "parent": PARENT.get(name, ""),
                "is_leaf": name in FINE_LABELS,
                "hidden_branch": bool(node.get("hidden_branch", False)),
                "positive_proteins": "|".join(_marker_list(node, "positive_markers")),
                "negative_proteins": "|".join(_marker_list(node, "negative_markers")),
                "rna_blocks": rna_blocks.get(lineage, ""),
                "axis_type": axis_type.get(lineage, ""),
                "expected_difficulty": expected.get(lineage, ""),
                "protein_generation_note": note,
                "config_seed": config.seed,
                "simulation_version": "v3",
            }
        )
    return pd.DataFrame(rows)


def format_tree(node: Mapping[str, Any], depth: int = 0) -> list[str]:
    flags: list[str] = []
    if node.get("hidden_branch", False):
        flags.append("hidden")
    if _marker_list(node, "positive_markers"):
        flags.append("pos=" + ",".join(_marker_list(node, "positive_markers")))
    if _marker_list(node, "negative_markers"):
        flags.append("neg=" + ",".join(_marker_list(node, "negative_markers")))
    line = "  " * depth + f"- {node['name']}"
    if flags:
        line += " [" + "; ".join(flags) + "]"
    out = [line]
    for child in node.get("children", []) or []:
        out.extend(format_tree(child, depth + 1))
    return out


def marker_score(label: str, protein: pd.DataFrame, tree: Mapping[str, Any]) -> pd.Series | None:
    node_by_name = {str(node["name"]): node for node in walk_tree(tree)}
    if label not in node_by_name:
        return None
    node = node_by_name[label]
    pos = [x for x in _marker_list(node, "positive_markers") if x in protein.columns]
    neg = [x for x in _marker_list(node, "negative_markers") if x in protein.columns]
    if not pos and not neg:
        return None
    score = pd.Series(0.0, index=protein.index, dtype=float)
    if pos:
        score = score + protein[pos].mean(axis=1)
    if neg:
        score = score - protein[neg].mean(axis=1)
    return score


def cv_accuracy(x: np.ndarray, y: Sequence[str], seed: int = 2027) -> tuple[float, float]:
    y_arr = np.asarray(y).astype(str)
    if len(np.unique(y_arr)) < 2:
        return np.nan, np.nan
    counts = np.bincount(pd.Categorical(y_arr).codes)
    n_splits = min(5, int(counts[counts > 0].min()))
    if n_splits < 2:
        return np.nan, np.nan
    clf = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, x, y_arr, cv=cv, scoring="accuracy")
    return float(np.mean(scores)), float(np.std(scores))


def write_audits(reference_audit: ad.AnnData, query_audit: ad.AnnData, out_dir: Path, tree: Mapping[str, Any]) -> None:
    combined = ad.concat([reference_audit, query_audit], join="inner", merge="same", uns_merge="same", index_unique=None)
    protein = protein_arcsinh_frame(combined)
    rna = np.log1p(combined.X.toarray() if sp.issparse(combined.X) else np.asarray(combined.X))
    labels = combined.obs[ORIGINAL_CELLTYPE_COL].astype(str).to_numpy()

    marker_rows = []
    for label in FINE_LABELS:
        score = marker_score(label, protein, tree)
        if score is None:
            continue
        y = combined.obs[ORIGINAL_CELLTYPE_COL].astype(str).eq(label).astype(int)
        auc = np.nan if y.nunique() < 2 else float(roc_auc_score(y, score))
        if np.isfinite(auc):
            auc = max(auc, 1.0 - auc)
        marker_rows.append(
            {
                "label": label,
                "lineage": top_lineage(label),
                "marker_auc_one_vs_rest_or_flipped": auc,
                "n_cells": int(y.sum()),
            }
        )
    pd.DataFrame(marker_rows).to_csv(out_dir / "synthetic_marker_separability_audit.csv", index=False)

    axis_rows = []
    similarity_rows = []
    protein_values = protein.to_numpy(dtype=np.float32)
    for axis, child_labels in LINEAGE_CHILDREN.items():
        mask = np.isin(labels, child_labels)
        y = labels[mask]
        rna_acc, rna_sd = cv_accuracy(rna[mask], y)
        protein_acc, protein_sd = cv_accuracy(protein_values[mask], y)
        combo = np.hstack([rna[mask], protein_values[mask]])
        combo_acc, combo_sd = cv_accuracy(combo, y)
        marker_argmax_acc = np.nan
        scores = []
        for child in child_labels:
            score = marker_score(child, protein.loc[mask], tree)
            if score is not None:
                scores.append(score.to_numpy(dtype=float))
        if len(scores) == len(child_labels):
            pred = np.asarray(child_labels, dtype=object)[np.argmax(np.vstack(scores).T, axis=1)]
            marker_argmax_acc = float(np.mean(pred.astype(str) == y.astype(str)))
        axis_rows.append(
            {
                "axis": axis,
                "child_labels": "|".join(child_labels),
                "n_cells": int(mask.sum()),
                "n_classes": int(len(np.unique(y))),
                "rna_only_logreg_cv_accuracy_mean": rna_acc,
                "rna_only_logreg_cv_accuracy_sd": rna_sd,
                "protein_only_logreg_cv_accuracy_mean": protein_acc,
                "protein_only_logreg_cv_accuracy_sd": protein_sd,
                "rna_protein_logreg_cv_accuracy_mean": combo_acc,
                "rna_protein_logreg_cv_accuracy_sd": combo_sd,
                "marker_score_argmax_accuracy": marker_argmax_acc,
                "transition_fraction": float(combined.obs.loc[mask, "is_transition_cell"].astype(bool).mean()),
            }
        )
        if len(child_labels) == 2:
            m0 = rna[labels == child_labels[0]].mean(axis=0)
            m1 = rna[labels == child_labels[1]].mean(axis=0)
            similarity_rows.append(
                {
                    "axis": axis,
                    "child1": child_labels[0],
                    "child2": child_labels[1],
                    "mean_log1p_profile_pearson": float(np.corrcoef(m0, m1)[0, 1]),
                    "mean_abs_log1p_diff": float(np.mean(np.abs(m0 - m1))),
                    "rms_log1p_diff": float(np.sqrt(np.mean((m0 - m1) ** 2))),
                }
            )
    pd.DataFrame(axis_rows).to_csv(out_dir / "synthetic_axis_oracle_difficulty_audit.csv", index=False)
    pd.DataFrame(similarity_rows).to_csv(out_dir / "synthetic_axis_rna_similarity_audit.csv", index=False)


def write_label_counts(reference: ad.AnnData, query: ad.AnnData, out_dir: Path) -> None:
    rows = []
    for role, adata in [(REFERENCE_NAME, reference), (QUERY_NAME, query)]:
        for col in [CELLTYPE_COL, ORIGINAL_CELLTYPE_COL]:
            counts = adata.obs[col].astype(str).value_counts()
            for label, n_cells in counts.items():
                rows.append({"role": role, "column": col, "label": label, "n_cells": int(n_cells)})
    pd.DataFrame(rows).to_csv(out_dir / "label_counts.csv", index=False)


def write_simulation_inputs(
    *,
    output_dir: Path,
    reference: ad.AnnData,
    query: ad.AnnData,
    marker_tree: Mapping[str, Any],
    reference_audit: ad.AnnData,
    query_audit: ad.AnnData,
    partial_label_mode: bool,
    config: SimulationConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "reference.h5ad"
    query_path = output_dir / "query.h5ad"
    marker_tree_path = output_dir / "marker_tree.json"

    reference.write_h5ad(reference_path)
    query.write_h5ad(query_path)
    marker_tree_path.write_text(json.dumps(marker_tree, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "data_generation_config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "marker_tree_readable.txt").write_text("\n".join(format_tree(marker_tree)) + "\n", encoding="utf-8")
    marker_design_audit(marker_tree, config).to_csv(output_dir / "marker_design_audit.csv", index=False)
    write_label_counts(reference, query, output_dir)
    write_audits(reference_audit, query_audit, output_dir, marker_tree)
    pd.DataFrame({"protein": list(PROTEIN_NAMES)}).to_csv(output_dir / "protein_panel.csv", index=False)

    hidden_parents = list(PARTIAL_LABEL_HIDDEN_BRANCHES) if partial_label_mode else []
    if partial_label_mode:
        pd.DataFrame(
            [
                {"hidden_parent": parent, "hidden_child": child}
                for parent in PARTIAL_LABEL_HIDDEN_BRANCHES
                for child in LINEAGE_CHILDREN[parent]
            ]
        ).to_csv(output_dir / "hidden_branch_children.csv", index=False)

    manifest = {
        "reference_h5ad": "reference.h5ad",
        "query_h5ad": "query.h5ad",
        "marker_tree_json": "marker_tree.json",
        "n_reference": int(reference.n_obs),
        "n_query": int(query.n_obs),
        "n_genes": int(query.n_vars),
        "n_proteins": len(PROTEIN_NAMES),
        "fine_labels": list(FINE_LABELS),
        "hidden_parents": hidden_parents,
        "simulation_version": "v3",
    }
    (output_dir / "input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_simulation_splits(config: SimulationConfig) -> tuple[ad.AnnData, ad.AnnData]:
    rng = np.random.default_rng(config.seed)
    log_gene, protein_log, gene_names, leaf_blocks = build_expression_templates(config)
    reference = make_split(
        split_name=REFERENCE_NAME,
        counts_by_label=allocation_counts(config.n_reference, label_weights()),
        batch_names=["ref_batch0", "ref_batch1"],
        log_gene=log_gene,
        protein_log=protein_log,
        gene_names=gene_names,
        leaf_blocks=leaf_blocks,
        config=config,
        rng=rng,
    )
    query = make_split(
        split_name=QUERY_NAME,
        counts_by_label=allocation_counts(config.n_query, label_weights()),
        batch_names=["query_batch0", "query_batch1", "query_batch2"],
        log_gene=log_gene,
        protein_log=protein_log,
        gene_names=gene_names,
        leaf_blocks=leaf_blocks,
        config=config,
        rng=rng,
    )
    return reference, query


def generate_full_mode_inputs(output_dir: Path, config: SimulationConfig = SimulationConfig()) -> None:
    reference_audit, query_audit = build_simulation_splits(config)
    reference = remove_reference_observed_protein(reference_audit)
    query = query_audit.copy()
    write_simulation_inputs(
        output_dir=output_dir,
        reference=reference,
        query=query,
        marker_tree=build_simulation_marker_tree(partial_label_mode=False),
        reference_audit=reference_audit,
        query_audit=query_audit,
        partial_label_mode=False,
        config=config,
    )


def generate_partial_label_mode_inputs(output_dir: Path, config: SimulationConfig = SimulationConfig()) -> None:
    reference_audit, query_audit = build_simulation_splits(config)
    reference = remove_reference_observed_protein(collapse_partial_reference_labels(reference_audit))
    query = query_audit.copy()
    write_simulation_inputs(
        output_dir=output_dir,
        reference=reference,
        query=query,
        marker_tree=build_simulation_marker_tree(partial_label_mode=True),
        reference_audit=reference_audit,
        query_audit=query_audit,
        partial_label_mode=True,
        config=config,
    )
