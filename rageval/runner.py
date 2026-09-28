"""Run a system over an eval set and aggregate the results."""
from __future__ import annotations

import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .judges import HeuristicJudge, Judge
from .metrics import score_case
from .types import CaseResult, EvalCase, SystemOutput


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = (len(s) - 1) * p / 100
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


@dataclass
class RunResult:
    system: str
    judge: str
    summary: dict[str, float]
    cases: list[CaseResult]
    meta: dict = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        data = {"system": self.system, "judge": self.judge, "summary": self.summary, "meta": self.meta,
                "cases": [{"id": r.case.id, "question": r.case.question, "answer": r.output.answer,
                           "citations": r.output.citations, "retrieved": [c.id for c in r.output.contexts],
                           "scores": r.scores, "issues": r.issues, "latency_ms": round(r.output.latency_ms, 2)}
                          for r in self.cases]}
        Path(path).write_text(json.dumps(data, indent=2))


def summarise(results: list[CaseResult], price_in: float = 0.0, price_out: float = 0.0) -> dict[str, float]:
    """Mean of each metric over the cases where it applies, plus latency, tokens and cost.
    Prices are per million tokens; pass your provider's current prices."""
    keys = sorted({k for r in results for k in r.scores})
    summary = {}
    for k in keys:
        vals = [r.scores[k] for r in results if r.scores.get(k) is not None]
        if vals:
            summary[k] = round(mean(vals), 4)
    lat = [r.output.latency_ms for r in results]
    pt = sum(r.output.prompt_tokens for r in results)
    ct = sum(r.output.completion_tokens for r in results)
    summary.update({
        "cases": len(results),
        "errors": sum(1 for r in results if r.output.error),
        "latency_p50_ms": round(percentile(lat, 50), 2),
        "latency_p95_ms": round(percentile(lat, 95), 2),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cost_usd": round((pt * price_in + ct * price_out) / 1e6, 6),
    })
    return summary


def run_eval(system, cases: list[EvalCase], judge: Judge | None = None, concurrency: int = 4,
             price_in: float = 0.0, price_out: float = 0.0) -> RunResult:
    judge = judge or HeuristicJudge()

    def one(case: EvalCase) -> CaseResult:
        start = time.perf_counter()
        try:
            out = system.answer(case.question)
        except Exception as e:  # a crashing system is a result to report, not a reason to stop the run
            out = SystemOutput("", error=f"{type(e).__name__}: {e}")
        if not out.latency_ms:
            out.latency_ms = (time.perf_counter() - start) * 1000
        scores, issues = score_case(case, out, judge)
        return CaseResult(case, out, scores, issues)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(one, cases))  # map keeps input order
    meta = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "concurrency": concurrency}
    return RunResult(system.name, judge.name, summarise(results, price_in, price_out), results, meta)


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())

