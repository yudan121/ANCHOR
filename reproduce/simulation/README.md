# Simulation input generator

This directory contains the release-clean generator for the synthetic mechanism
v3 simulation used in the ANCHOR thesis experiments.

The generator keeps the original synthetic mechanism v3 numerical design:
seed `2027`, 5,600 reference cells, 6,400 query cells, 1,000 genes, 24 proteins,
the same RNA/protein signal strengths, and the same marker tree structure. The
release version removes local paths and old experiment scaffolding, and writes
the public ANCHOR schema with `cell_type`, `batch`, and `split` columns.

Use the thin wrappers in:

```text
../simulation_full_mode/generate_inputs.py
../simulation_partial_label_mode/generate_inputs.py
```

Each wrapper writes `reference.h5ad`, `query.h5ad`, `marker_tree.json`, and
diagnostic audit files to its local `data/` directory by default. Set
`ANCHOR_SIMULATION_OUTPUT_DIR=/path/to/output` to write elsewhere.
