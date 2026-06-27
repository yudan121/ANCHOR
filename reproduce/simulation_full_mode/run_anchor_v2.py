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

from anchor import __version__, run_anchor  # noqa: E402

if not str(__version__).startswith("2."):
    raise RuntimeError("This script expects ANCHOR v2. Check out the matching release tag or set ANCHOR_PACKAGE_DIR to a v2 checkout.")

DEFAULT_RUN_NAME = 'anchor_simulation_full_mode_v2'
DEFAULT_RESULTS_DIR = REPRODUCE_DIR / "results" / "v2"
TEACHER_OVERRIDES = {'hard_ref_sampling_correct_weight': 1.0, 'hard_ref_sampling_enable': True, 'hard_ref_sampling_max_wrong_fraction': 0.3, 'hard_ref_sampling_wrong_weight': 10.0, 'n_latent': 30, 'n_layers': 2, 'n_samples_per_label': 100, 'round0_epochs': 20, 'round1_epochs': 10, 'round2_epochs': 10}
STUDENT_OVERRIDES = {'graph_consistency_lambda': 0.5, 'max_epochs': 100, 'prototype_ce_lambda': 0.5, 'prototype_logit_weight': 1.0, 'rank_loss_weight': 1.0, 'safety_guard_fallback_to_teacher_round2': True, 'teacher_soft_kl_schedule': {'end': 0.5, 'mode': 'constant', 'start': 0.5}}
SELECTION_OVERRIDES = {}
RHO_OVERRIDES = {'strong_protein_power_threshold': 0.65, 'strong_release_rho': 0.0}

def _optional_env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None

def _optional_env_value(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    return value or None

def main() -> None:
    data_dir = Path(os.environ.get("ANCHOR_DATA_DIR", str(DATASET_DIR / "data")))
    scratch = os.environ.get("ANCHOR_SCRATCH", "0").lower() in {"1", "true", "yes", "y"}
    result = run_anchor(
        reference=data_dir / "reference.h5ad",
        query=data_dir / "query.h5ad",
        marker_tree=data_dir / "marker_tree.json",
        results_dir=Path(os.environ.get("ANCHOR_RESULTS_DIR", str(DEFAULT_RESULTS_DIR))),
        run_name=os.environ.get("ANCHOR_RUN_NAME", DEFAULT_RUN_NAME),
        batch_key=os.environ.get("ANCHOR_BATCH_KEY", "batch"),
        celltype_key=os.environ.get("ANCHOR_CELLTYPE_KEY", "cell_type"),
        query_label_key=_optional_env_value("ANCHOR_QUERY_LABEL_KEY", 'cell_type' or None),
        sample_key=_optional_env_value("ANCHOR_SAMPLE_KEY", '' or None),
        hidden_branch_key=_optional_env_value("ANCHOR_HIDDEN_BRANCH_KEY"),
        teacher_overrides=TEACHER_OVERRIDES,
        student_overrides=STUDENT_OVERRIDES,
        selection_overrides=SELECTION_OVERRIDES,
        rho_overrides=RHO_OVERRIDES,
        source_totalvi_init_dir=None if scratch else _optional_env_path("ANCHOR_TOTALVI_INIT_DIR"),
        force_retrain=os.environ.get("ANCHOR_FORCE_RETRAIN", "0").lower() in {"1", "true", "yes", "y"},
        allow_resume_from_existing=os.environ.get("ANCHOR_ALLOW_RESUME", "0").lower() in {"1", "true", "yes", "y"},
    )
    print(result)

if __name__ == "__main__":
    main()
