# ANCHOR

ANCHOR (**AN**notating **C**ells with **H**ierarchical marker-**O**riented **R**egularization) is a Python package for marker-guided cell-type annotation in paired RNA-protein single-cell or spatial data. ANCHOR combines a labeled scRNA-seq reference, an RNA-protein query dataset, and a user-provided hierarchical protein marker tree through a teacher-student training framework.

The thesis experiments were generated with the v1 thesis-compatible release. The v2 release keeps the same teacher, anchor selection, student training, and safety-guard pipeline, and updates only the node-wise rho policy used for the student conditional KL loss.

## Method overview

![ANCHOR method overview](docs/method.png)

**Overview of the ANCHOR framework.** **(A)** ANCHOR takes as input a labeled scRNA-seq reference, an unlabeled RNA-protein query, and a hierarchical marker tree whose nodes carry positive (`+`) and negative (`-`) protein markers. Solid branches denote reference-labeled types; dashed branches denote user-defined subtypes resolved through markers alone. **(B)** The teacher model jointly trains on reference and query via a VAE-based encoder-decoder. Reference labels supervise the classifier, augmented by anchor pseudo-labels for query cells. **(C)** Anchors are query cells for which classifier prediction, protein marker score, and KNN purity agree. Selected anchors occupy unambiguous positions in the embedding. **(D)** The student model trains on query cells only, using teacher latent features and protein measurements as input. It is guided by anchor pseudo-labels, a rank loss that aligns predicted probabilities with marker-score ordering, and a KL term with node-wise adaptive weighting. **(E)** Final annotations integrate RNA-derived structure with protein-resolved cell states.

If the PNG preview does not render in your environment, see `docs/method.pdf`.

## Installation

```bash
git clone <ANCHOR_REPOSITORY_URL>
cd ANCHOR
python -m pip install -e .
```

ANCHOR is developed for Python 3.10 or newer. A CUDA-capable GPU is strongly recommended for the full teacher-student workflow.

## Quick start

```python
from anchor import run_anchor

result = run_anchor(
    reference="path/to/reference.h5ad",
    query="path/to/query.h5ad",
    marker_tree="path/to/marker_tree.json",
    results_dir="path/to/anchor_results",
    run_name="my_anchor_run",
    batch_key="batch",
    celltype_key="cell_type",
    query_label_key=None,
)
print(result.final_dir)
```

`reference` and `query` can be `.h5ad` paths or loaded `AnnData` objects. `marker_tree` can be a JSON path, a nested Python dictionary, or a `MarkerTree` object.

To train only the student from existing teacher outputs under `results_dir / run_name / round2`, use:

```python
from anchor import run_student

run_student(
    reference="path/to/reference.h5ad",
    query="path/to/query.h5ad",
    marker_tree="path/to/marker_tree.json",
    results_dir="path/to/anchor_results",
    run_name="my_anchor_run",
    batch_key="batch",
    celltype_key="cell_type",
)
```

## Input data

ANCHOR expects:

- A labeled reference `.h5ad` containing RNA counts and reference cell-type labels.
- A query `.h5ad` containing RNA and protein measurements.
- A marker tree describing positive and negative protein markers for cell states or internal branches.

The default example scripts assume these column names:

- `batch`: batch, donor, sample, or processing-site label.
- `cell_type`: reference cell-type label, and query ground truth when available.

Use the `batch_key`, `celltype_key`, and `query_label_key` arguments, or the environment variables documented in `reproduce/README.md`, if your files use different names.

## Marker tree schema

Each marker-tree node is a JSON object with:

- `name`: node or cell-type name.
- `positive_markers`: optional list of protein markers expected to be high.
- `negative_markers`: optional list of protein markers expected to be low.
- `children`: optional list of child nodes.

Minimal example:

```json
{
  "name": "root",
  "children": [
    {
      "name": "T cell",
      "positive_markers": ["CD3"],
      "negative_markers": ["CD19"]
    },
    {
      "name": "B cell",
      "positive_markers": ["CD19"],
      "negative_markers": ["CD3"]
    }
  ]
}
```

## Expected output

Each run writes a run directory containing teacher rounds, student outputs, node-wise rho audit tables, selected anchors, metrics when query labels are available, and a final decision report. If the student safety guard detects excessive loss of teacher-supported rare classes, ANCHOR falls back to the round-2 teacher output for the final annotation.

## Reproducibility

The `reproduce/` directory contains scripts for the ANCHOR experiments described in the thesis:

- `simulation_full_mode`
- `simulation_partial_label_mode`
- `bmmc_full_mode`
- `bmmc_partial_label_mode`
- `marrowatlas`
- `bm_abseq`
- `pbmc`
- `xenium_ccrcc_full_mode`
- `xenium_ccrcc_b_plasma_partial_label_mode`

The scripts use portable placeholders and environment variables rather than local machine paths. See `reproduce/README.md` for details.

## Data availability

The processed input files for reproducing the ANCHOR experiments will be deposited on Zenodo. Zenodo DOI: **coming soon**.

The GitHub repository intentionally does not track `.h5ad` files, trained model checkpoints, logs, or result tables.

## System requirements

ANCHOR depends on `anndata`, `scanpy`, `scvi-tools`, `torch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, and `seaborn`. Full benchmark-scale runs were tested on Linux servers with NVIDIA GPUs. Small input-validation and student-only checks can be run on CPU, but full teacher training is expected to be slow without GPU acceleration.

## Version history

- `v1.0.0-formal-compatible`: thesis-compatible ANCHOR release used for the primary reported experiments.
- `v2.0.0-hybrid-rho`: same training pipeline with the hybrid rho policy, which simplifies the protein-power and teacher-challenge components while retaining the RNA-protection term.

## Citation

Citation information will be added after manuscript release.

## License

This project is licensed under the GNU General Public License v3.0. See `LICENSE` for details.
