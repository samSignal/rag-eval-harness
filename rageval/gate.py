"""Quality gate: compare a run against thresholds and an optional baseline run.

Thresholds file (JSON):
{
  "fact_recall":        {"min": 0.80},
  "faithfulness":       {"min": 0.90, "max_drop": 0.02},
  "citation_validity":  {"min": 1.0},
  "latency_p95_ms":     {"max": 2000, "max_increase_pct": 25}
}
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Violation:
    metric: str
    rule: str
    value: float
    limit: float

    def __str__(self) -> str:
        return f"{self.metric}: {self.value:g} violates {self.rule} {self.limit:g}"


def check(summary: dict, thresholds: dict, baseline: dict | None = None) -> list[Violation]:
    out: list[Violation] = []
    for metric, rules in thresholds.items():
        if metric not in summary:
            out.append(Violation(metric, "missing from run", float("nan"), float("nan")))
            continue
        v = float(summary[metric])
        if "min" in rules and v < rules["min"]:
            out.append(Violation(metric, "min", v, rules["min"]))
        if "max" in rules and v > rules["max"]:
            out.append(Violation(metric, "max", v, rules["max"]))
        if baseline and metric in baseline:
            b = float(baseline[metric])
            if "max_drop" in rules and b - v > rules["max_drop"]:
                out.append(Violation(metric, f"max_drop vs baseline {b:g} ->", v, rules["max_drop"]))
            if "max_increase_pct" in rules and b > 0 and (v - b) / b * 100 > rules["max_increase_pct"]:
                out.append(Violation(metric, f"max_increase_pct vs baseline {b:g} ->", v, rules["max_increase_pct"]))
    return out
