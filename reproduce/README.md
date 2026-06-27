# Reproducing ANCHOR experiments

This folder contains portable scripts for running ANCHOR on the datasets used in the thesis. The scripts are organized by thesis-facing dataset and mode names rather than internal experiment names.

## Data layout

Each dataset folder expects:

```text
<dataset>/data/reference.h5ad
<dataset>/data/query.h5ad
<dataset>/data/marker_tree.json
```

The marker trees are included in this repository. The `.h5ad` files and optional totalVI initialization models are not tracked by GitHub and will be distributed separately through Zenodo. Zenodo DOI: **coming soon**.

## Common environment variables

- `ANCHOR_DATA_DIR`: directory containing `reference.h5ad`, `query.h5ad`, and `marker_tree.json` for one dataset. Defaults to `<dataset>/data`.
- `ANCHOR_RESULTS_DIR`: output root. Defaults to `reproduce/results/<version>`.
- `ANCHOR_RUN_NAME`: run name. Defaults to a stable thesis-style name.
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

To reproduce v1 after the repository has moved to v2, either check out `v1.0.0-formal-compatible` or set `ANCHOR_PACKAGE_DIR` to a separate v1 checkout.

To run v2 from the main branch, use `python run_anchor_v2.py` in the same dataset folder.
