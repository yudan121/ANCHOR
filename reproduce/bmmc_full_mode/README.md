# BMMC full mode

NeurIPS 2021 BMMC CITE-seq benchmark with the full shared fine-label space.

Expected local data files:

```text
data/reference.h5ad
data/query.h5ad
data/marker_tree.json
```

The marker tree is included. The `.h5ad` files and optional `totalvi_init_model` directory will be provided through Zenodo. Zenodo DOI: **coming soon**.

Example full run:

```bash
ANCHOR_DATA_DIR=path/to/bmmc_full_mode/data \
ANCHOR_RESULTS_DIR=path/to/results \
python run_anchor_v1.py
```

To reuse an existing totalVI initialization, set `ANCHOR_TOTALVI_INIT_DIR=path/to/totalvi_init_model`.
