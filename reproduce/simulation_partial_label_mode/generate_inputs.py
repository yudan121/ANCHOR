#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
GENERATOR_DIR = DATASET_DIR.parent / "simulation"
if str(GENERATOR_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATOR_DIR))

from generate_inputs import generate_partial_label_mode_inputs  # noqa: E402


def main() -> None:
    output_dir = Path(os.environ.get("ANCHOR_SIMULATION_OUTPUT_DIR", str(DATASET_DIR / "data")))
    generate_partial_label_mode_inputs(output_dir)
    print(f"Wrote simulation partial-label mode inputs to {output_dir}")


if __name__ == "__main__":
    main()
