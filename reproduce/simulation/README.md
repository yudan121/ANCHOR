# Simulation input generator

This directory contains the generator for the synthetic data used in the ANCHOR simulation.

Use the thin wrappers in:

```text
../simulation_full_mode/generate_inputs.py
../simulation_partial_label_mode/generate_inputs.py
```

Each wrapper writes `reference.h5ad`, `query.h5ad`, `marker_tree.json`, and
diagnostic audit files to its local `data/` directory by default. Set
`ANCHOR_SIMULATION_OUTPUT_DIR=/path/to/output` to write elsewhere.
