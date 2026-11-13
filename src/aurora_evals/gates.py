"""Quality gates: turn metric scores into a ship / don't-ship decision.

    gates = load_gates()["rag"]
    results = check_gates(run.df, gates, baseline_summary=EvalRun.load_summary("results/baseline/summary.json"))
    print(format_gate_report(results))
    assert all(r.passed for r in results)

The gate logic is plain Python with no LLM calls, so it is unit-tested in tests/test_gates.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .config import CONFIG_DIR


@dataclass
class GateResult:
    gate: str  # "overall", "slice", "regression"
    scope: str  # "all rows" or "question_type=adversarial" ...
    metric: str
    actual: float
    threshold: float
    passed: bool
    note: str = ""


def load_gates(path: str | Path = CONFIG_DIR / "quality_gates.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mean(df: pd.DataFrame, metric: str) -> float:
    if metric not in df:
        return math.nan
    return float(df[metric].mean())


def check_gates(df: pd.DataFrame, gates: dict, baseline_summary: dict | None = None) -> list[GateResult]:
    results: list[GateResult] = []

    for metric, rule in (gates.get("overall") or {}).items():
        actual = _mean(df, metric)
        results.append(
            GateResult("overall", "all rows", metric, actual, rule["min"], _ok(actual, rule["min"]),
                       "metric missing from run" if math.isnan(actual) else "")
        )

    for selector, rules in (gates.get("slices") or {}).items():
        column, value = selector.split("=", 1)
        subset = df[df[column].astype(str) == value] if column in df else df.iloc[0:0]
        for metric, rule in rules.items():
            actual = _mean(subset, metric)
            results.append(
                GateResult("slice", selector, metric, actual, rule["min"], _ok(actual, rule["min"]),
                           f"{len(subset)} rows" if len(subset) else "no rows matched")
            )

    regression = gates.get("regression")
    if regression and baseline_summary:
        max_drop = regression["max_drop"]
        for metric in regression.get("metrics", []):
            actual, base = _mean(df, metric), baseline_summary.get(metric, math.nan)
            floor = base - max_drop
            results.append(
                GateResult("regression", "vs baseline", metric, actual, round(floor, 3), _ok(actual, floor),
                           f"baseline={base:.3f}")
            )
    return results


def _ok(actual: float, threshold: float) -> bool:
    # A metric that could not be computed must never silently pass a gate.
    # The tiny epsilon stops float noise (0.8499999) from failing a 0.85 gate.
    return not math.isnan(actual) and actual >= threshold - 1e-9


def format_gate_report(results: list[GateResult]) -> str:
    lines = [f"{'':2} {'gate':<10} {'scope':<28} {'metric':<20} {'actual':>7} {'min':>7}  note"]
    for r in results:
        mark = "✅" if r.passed else "❌"
        actual = "n/a" if math.isnan(r.actual) else f"{r.actual:.3f}"
        lines.append(f"{mark} {r.gate:<10} {r.scope:<28} {r.metric:<20} {actual:>7} {r.threshold:>7.3f}  {r.note}")
    passed = sum(r.passed for r in results)
    lines.append(f"\n{passed}/{len(results)} gates passed → {'SHIP ✅' if passed == len(results) else 'BLOCK ❌'}")
    return "\n".join(lines)
