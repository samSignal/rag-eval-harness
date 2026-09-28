"""Command-line interface.

    rageval run --out runs/current.json --html runs/report.html          # reference pipeline, offline
    rageval run --llm openai:gpt-4o-mini --judge openai:gpt-4o-mini       # real LLM generator + LLM judge
    rageval run --system http --url http://localhost:8000/ask             # any RAG service
    rageval gate runs/current.json --thresholds thresholds.json --baseline runs/baseline.json
    rageval compare runs/baseline.json runs/current.json
    rageval calibrate --judge heuristic                                   # how trustworthy is the judge?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .calibrate import calibrate
from .gate import check
from .judges import HeuristicJudge, LLMJudge
from .llm import get_client, text_only
from .report import compare_markdown, failures_markdown, to_html
from .runner import load_run, run_eval
from .systems import FaultInjector, HTTPSystem, LegalRAG, LLMGenerator
from .types import load_cases

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = ROOT / "data" / "cases.jsonl"
DEFAULT_CORPUS = ROOT / "data" / "corpus.jsonl"


def cmd_run(a) -> int:
    cases = load_cases(a.cases)
    if a.system == "http":
        system = HTTPSystem(a.url, name=f"http:{a.url}")
    else:
        gen = LLMGenerator(get_client(a.llm), name=a.llm) if a.llm else None
        system = LegalRAG(a.corpus, generator=gen, mode=a.mode, top_k=a.k)
    if a.inject_faults:
        system = FaultInjector(system, rate=a.inject_faults)
    judge = LLMJudge(text_only(get_client(a.judge)), name=a.judge) if a.judge != "heuristic" else HeuristicJudge()

    run = run_eval(system, cases, judge, a.concurrency, a.price_in, a.price_out)
    print(compare_markdown([run]))
    print()
    print(failures_markdown(run))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        run.to_json(a.out)
    if a.html:
        Path(a.html).parent.mkdir(parents=True, exist_ok=True)
        Path(a.html).write_text(to_html(run), encoding="utf-8")
    return 0


def cmd_gate(a) -> int:
    run = load_run(a.run)
    thresholds = json.loads(Path(a.thresholds).read_text())
    baseline = load_run(a.baseline)["summary"] if a.baseline else None
    violations = check(run["summary"], thresholds, baseline)
    if violations:
        print("QUALITY GATE FAILED")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(f"Quality gate passed ({len(thresholds)} metrics checked{' against baseline' if baseline else ''}).")
    return 0


def cmd_compare(a) -> int:
    base, cur = load_run(a.baseline), load_run(a.current)
    print(f"| Metric | {base['system']} | {cur['system']} | Change |\n|---|---|---|---|")
    for k in sorted(set(base["summary"]) | set(cur["summary"])):
        b, c = base["summary"].get(k), cur["summary"].get(k)
        delta = f"{c - b:+.4g}" if isinstance(b, (int, float)) and isinstance(c, (int, float)) else ""
        print(f"| {k} | {b} | {c} | {delta} |")
    return 0


def cmd_calibrate(a) -> int:
    judge = LLMJudge(text_only(get_client(a.judge)), name=a.judge) if a.judge != "heuristic" else HeuristicJudge()
    c = calibrate(judge, a.labels, a.corpus)
    print(f"Judge {c.judge} on {c.n} labelled claims: accuracy {c.accuracy:.3f}, "
          f"hallucination precision {c.hallucination_precision:.3f}, recall {c.hallucination_recall:.3f}")
    for d in c.disagreements:
        print(f"  - {d}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rageval", description="Evaluate and regression-test RAG systems")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", default="legal-rag", choices=["legal-rag", "http"])
    r.add_argument("--url", default="http://localhost:8000/ask")
    r.add_argument("--cases", default=str(DEFAULT_CASES))
    r.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    r.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    r.add_argument("--k", type=int, default=4)
    r.add_argument("--llm", help="generator, e.g. openai:gpt-4o-mini or ollama:llama3.1 (default: extractive)")
    r.add_argument("--judge", default="heuristic", help="heuristic | openai:<model> | ollama:<model>")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--price-in", type=float, default=0.0, help="USD per 1M prompt tokens")
    r.add_argument("--price-out", type=float, default=0.0, help="USD per 1M completion tokens")
    r.add_argument("--inject-faults", type=float, default=0.0, help="corrupt numbers in this share of answers")
    r.add_argument("--out")
    r.add_argument("--html")
    g = sub.add_parser("gate")
    g.add_argument("run")
    g.add_argument("--thresholds", required=True)
    g.add_argument("--baseline")
    c = sub.add_parser("compare")
    c.add_argument("baseline")
    c.add_argument("current")
    k = sub.add_parser("calibrate", help="measure a judge against human-labelled claims")
    k.add_argument("--judge", default="heuristic")
    k.add_argument("--labels", default=str(ROOT / "data" / "judge_calibration.jsonl"))
    k.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    a = p.parse_args(argv)
    return {"run": cmd_run, "gate": cmd_gate, "compare": cmd_compare, "calibrate": cmd_calibrate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
