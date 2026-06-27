from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class SafetyGuardDecision:
    triggered: bool
    final_source: str
    reason: str | None = None


def read_guard_report(student_dir: str | Path) -> dict[str, Any] | None:
    path = Path(student_dir) / "student_safety_guard_report.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def guard_triggered(report: Mapping[str, Any] | None) -> bool:
    if report is None:
        return False
    value = report.get("guard_trigger", False)
    if isinstance(value, str):
        value = value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def choose_final_stage(
    *,
    student_dir: str | Path,
    teacher_round2_dir: str | Path,
    fallback_to_teacher_round2: bool = True,
) -> tuple[Path, SafetyGuardDecision]:
    report = read_guard_report(student_dir)
    triggered = guard_triggered(report)
    if triggered and fallback_to_teacher_round2:
        return Path(teacher_round2_dir), SafetyGuardDecision(
            triggered=True,
            final_source="teacher_round2",
            reason="student safety guard triggered",
        )
    return Path(student_dir), SafetyGuardDecision(triggered=triggered, final_source="student")


def write_final_decision(out_dir: str | Path, decision: SafetyGuardDecision) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "final_decision.json"
    path.write_text(json.dumps(decision.__dict__, indent=2, sort_keys=True))
    return path
