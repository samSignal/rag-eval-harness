"""Markdown and self-contained HTML reports."""
from __future__ import annotations

import html

from .runner import RunResult

QUALITY = [("context_recall", "Context recall"), ("context_precision", "Context precision"),
           ("citation_precision", "Citation precision"), ("citation_recall", "Citation recall"),
           ("citation_validity", "Citation validity"), ("fact_recall", "Key-fact recall"),
           ("faithfulness", "Faithfulness"), ("refusal_accuracy", "Refusal accuracy"),
           ("false_refusal", "False refusals")]


def compare_markdown(runs: list[RunResult]) -> str:
    cols = [(k, label) for k, label in QUALITY if any(k in r.summary for r in runs)]
    head = "| System | " + " | ".join(l for _, l in cols) + " | p95 ms | Cost $ |"
    sep = "|---" * (len(cols) + 3) + "|"
    rows = []
    for r in runs:
        vals = [f"{r.summary[k]:.3f}" if k in r.summary else "-" for k, _ in cols]
        rows.append(f"| {r.system} | " + " | ".join(vals) + f" | {r.summary['latency_p95_ms']:.1f} | {r.summary['cost_usd']:.4f} |")
    return "\n".join([head, sep, *rows])


def failures_markdown(run: RunResult, limit: int = 10) -> str:
    bad = [r for r in run.cases if r.issues]
    lines = [f"**{len(bad)} of {len(run.cases)} cases have issues** ({run.system}, judge: {run.judge})", ""]
    for r in bad[:limit]:
        lines.append(f"- `{r.case.id}` {r.case.question}")
        lines += [f"  - {i}" for i in r.issues]
    return "\n".join(lines)


def to_html(run: RunResult) -> str:
    e = html.escape
    summary = "".join(f"<tr><td>{e(k)}</td><td>{v}</td></tr>" for k, v in run.summary.items())
    rows = []
    for r in run.cases:
        ok = not r.issues
        rows.append(
            f"<tr class={'ok' if ok else 'bad'}><td>{e(r.case.id)}</td><td>{e(r.case.question)}</td>"
            f"<td>{e(r.output.answer)}</td><td>{e(', '.join(r.output.citations))}</td>"
            f"<td>{e(', '.join(c.id for c in r.output.contexts))}</td>"
            f"<td>{'<br>'.join(e(i) for i in r.issues) or 'OK'}</td></tr>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>RAG eval: {e(run.system)}</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#1d2330}}table{{border-collapse:collapse;width:100%;margin:12px 0}}
td,th{{border:1px solid #e4e7ec;padding:6px;vertical-align:top;text-align:left}}tr.bad td:last-child{{color:#b42318}}
tr.ok td:last-child{{color:#067647}}h1{{font-size:20px}}</style></head><body>
<h1>RAG evaluation: {e(run.system)}</h1><p>Judge: {e(run.judge)} &middot; {e(run.meta.get('timestamp', ''))}</p>
<table><tr><th>Metric</th><th>Value</th></tr>{summary}</table>
<table><tr><th>ID</th><th>Question</th><th>Answer</th><th>Cited</th><th>Retrieved</th><th>Issues</th></tr>{''.join(rows)}</table>
</body></html>"""
