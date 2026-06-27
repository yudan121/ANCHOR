from __future__ import annotations

import argparse

from anchor import run_anchor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ANCHOR teacher-student pipeline.")
    parser.add_argument("--reference", required=True, help="Reference .h5ad file.")
    parser.add_argument("--query", required=True, help="Query .h5ad file.")
    parser.add_argument("--marker-tree", required=True, help="Marker-tree JSON file.")
    parser.add_argument("--results-dir", required=True, help="Directory where run outputs will be written.")
    parser.add_argument("--run-name", required=True, help="Name for this ANCHOR run.")
    parser.add_argument("--batch-key", required=True, help="obs column containing batch labels.")
    parser.add_argument("--celltype-key", required=True, help="reference obs column containing cell-type labels.")
    parser.add_argument("--query-label-key", default=None, help="optional query obs label column for evaluation.")
    parser.add_argument("--source-totalvi-init-dir", default=None, help="optional existing totalVI initialization directory.")
    parser.add_argument("--batch-size", type=int, default=None, help="optional shared teacher/student batch size.")
    parser.add_argument("--student-max-epochs", type=int, default=None, help="optional student epoch override.")
    parser.add_argument("--allow-resume", action="store_true", help="allow existing checkpoints in the target run.")
    args = parser.parse_args()

    result = run_anchor(
        reference=args.reference,
        query=args.query,
        marker_tree=args.marker_tree,
        results_dir=args.results_dir,
        run_name=args.run_name,
        batch_key=args.batch_key,
        celltype_key=args.celltype_key,
        query_label_key=args.query_label_key,
        source_totalvi_init_dir=args.source_totalvi_init_dir,
        batch_size=args.batch_size,
        student_max_epochs=args.student_max_epochs,
        allow_resume_from_existing=args.allow_resume,
    )
    print(result)


if __name__ == "__main__":
    main()
