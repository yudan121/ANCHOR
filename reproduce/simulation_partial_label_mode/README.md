# Simulation partial-label mode

Controlled simulation with selected fine labels collapsed to coarse parent labels in the reference.

Expected local data files:

```text
data/reference.h5ad
data/query.h5ad
data/marker_tree.json
```

The marker tree is included. The `.h5ad` files and optional `totalvi_init_model` directory will be provided through Zenodo. Zenodo DOI: **coming soon**.

You can also regenerate the synthetic inputs locally:

```bash
python generate_inputs.py
```

By default this writes to `data/`. To write elsewhere:

```bash
ANCHOR_SIMULATION_OUTPUT_DIR=path/to/output python generate_inputs.py
```

Example ANCHOR run:

```bash
ANCHOR_DATA_DIR=path/to/simulation_partial_label_mode/data \
ANCHOR_RESULTS_DIR=path/to/results \
python run_anchor_v1.py
```

To reuse an existing totalVI initialization, set `ANCHOR_TOTALVI_INIT_DIR=path/to/totalvi_init_model`.
