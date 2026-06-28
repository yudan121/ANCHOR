# Reproducing ANCHOR experiments

This folder contains portable scripts for running ANCHOR on the datasets used in our experiments. The scripts are organized by dataset and mode names.

## Data layout

Each dataset folder expects:

```text
<dataset>/data/reference.h5ad
<dataset>/data/query.h5ad
<dataset>/data/marker_tree.json
```

The marker trees are included in this repository. The `.h5ad` files are distributed separately through Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18872086.svg)](https://doi.org/10.5281/zenodo.20965819)

## Common environment variables

- `ANCHOR_DATA_DIR`: directory containing `reference.h5ad`, `query.h5ad`, and `marker_tree.json` for one dataset. Defaults to `<dataset>/data`.
- `ANCHOR_RESULTS_DIR`: output root. Defaults to `reproduce/results/<version>`.
- `ANCHOR_RUN_NAME`: run name. Defaults to a stable dataset-specific name.
- `ANCHOR_TOTALVI_INIT_DIR`: optional path to an existing `totalvi_init_model` directory. If unset, ANCHOR trains the initialization from scratch.
- `ANCHOR_SCRATCH=1`: force training totalVI initialization from scratch, even if `ANCHOR_TOTALVI_INIT_DIR` is set.
- `ANCHOR_BATCH_KEY`: batch column in `.obs`; default `batch`.
- `ANCHOR_CELLTYPE_KEY`: reference label column in `.obs`; default `cell_type`.
- `ANCHOR_QUERY_LABEL_KEY`: optional query label column for metric calculation. For benchmark datasets the default is `cell_type`; for Xenium case studies the default is unset.

## Example

```bash
cd reproduce/bmmc_full_mode
ANCHOR_DATA_DIR=path/to/bmmc_full_mode/data \
ANCHOR_RESULTS_DIR=path/to/results \
ANCHOR_TOTALVI_INIT_DIR=path/to/totalvi_init_model \
python run_anchor_v1.py
```
