from __future__ import annotations

import os
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
REPRODUCE_DIR = DATASET_DIR.parent
REPO_DIR = REPRODUCE_DIR.parent
PACKAGE_DIR = Path(os.environ.get("ANCHOR_PACKAGE_DIR", str(REPO_DIR))).resolve()
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from anchor import __version__, run_student  # noqa: E402

if not str(__version__).startswith("1."):
    raise RuntimeError("This script expects ANCHOR v1. Check out the matching release tag or set ANCHOR_PACKAGE_DIR to a v1 checkout.")

DEFAULT_RUN_NAME = 'anchor_xenium_ccrcc_full_mode_v1'
DEFAULT_RESULTS_DIR = REPRODUCE_DIR / "results" / "v1"
STUDENT_OVERRIDES = {'batch_size': 4096, 'max_epochs': 100, 'random_seed': 2026, 'rank_loss_weight': 1.0, 'safety_guard_fallback_to_teacher_round2': True}
RHO_OVERRIDES = {}

def _optional_env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    return value or None

def main() -> None:
    data_dir = Path(os.environ.get("ANCHOR_DATA_DIR", str(DATASET_DIR / "data")))
    result = run_student(
        reference=data_dir / "reference.h5ad",
        query=data_dir / "query.h5ad",
        marker_tree=data_dir / "marker_tree.json",
        results_dir=Path(os.environ.get("ANCHOR_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))),
        run_name=os.environ.get("ANCHOR_RUN_NAME", DEFAULT_RUN_NAME),
        batch_key=os.environ.get("ANCHOR_BATCH_KEY", "batch"),
        celltype_key=os.environ.get("ANCHOR_CELLTYPE_KEY", "cell_type"),
        query_label_key=_optional_env_value("ANCHOR_QUERY_LABEL_KEY", '' or None),
        sample_key=_optional_env_value("ANCHOR_SAMPLE_KEY", 'batch' or None),
        hidden_branch_key=_optional_env_value("ANCHOR_HIDDEN_BRANCH_KEY"),
        student_overrides=STUDENT_OVERRIDES,
        rho_overrides=RHO_OVERRIDES,
    )
    print(result)

if __name__ == "__main__":
    main()
