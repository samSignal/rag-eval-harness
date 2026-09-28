"""Per-case metrics. Each returns a value in [0, 1], or None when it does not apply to the case."""
from __future__ import annotations

from .judges import Judge
from .text import is_refusal, normalise, sentences
from .types import EvalCase, SystemOutput


def score_case(case: EvalCase, out: SystemOutput, judge: Judge) -> tuple[dict[str, float | None], list[str]]:
    s: dict[str, float | None] = {}
    issues: list[str] = []
    retrieved = [c.id for c in out.contexts]
    refused = is_refusal(out.answer)

    if out.error:
        issues.append(f"system error: {out.error}")

    # Hallucinated citations: the answer cites a source that was never retrieved
    invalid = [c for c in out.citations if c not in retrieved]
    s["citation_validity"] = (1 - len(invalid) / len(out.citations)) if out.citations else None
    if invalid:
        issues.append(f"cites sources that were not retrieved: {', '.join(invalid)}")

    if not case.answerable:
        s["refusal_accuracy"] = float(refused)
        if not refused:
            issues.append("answered a question the documents cannot answer")
        return s, issues

    gold = set(case.gold_sources)
    s["false_refusal"] = float(refused)
    if refused:
        issues.append("declined an answerable question")

    # Retrieval
    s["context_recall"] = len(gold & set(retrieved)) / len(gold) if gold else None
    s["context_precision"] = len(gold & set(retrieved)) / len(retrieved) if retrieved else 0.0
    if gold and not gold & set(retrieved):
        issues.append(f"retrieval missed {', '.join(sorted(gold))}")

    # Citations
    cited = set(out.citations)
    s["citation_precision"] = len(cited & gold) / len(cited) if cited else 0.0
    s["citation_recall"] = len(cited & gold) / len(gold) if gold else None
    if not cited and not refused:
        issues.append("answer has no citations")

    # Correctness: are the key facts stated?
    if case.must_include:
        ans = f" {normalise(out.answer)} "
        missing = [f for f in case.must_include if f" {normalise(f)} " not in ans]
        s["fact_recall"] = 1 - len(missing) / len(case.must_include)
        if missing:
            issues.append(f"missing key fact(s): {', '.join(missing)}")

    # Faithfulness: share of answer sentences supported by the retrieved context
    claims = [] if refused else sentences(out.answer)
    if claims:
        unsupported = [c for c in claims if not judge.supported(c, out.contexts)]
        s["faithfulness"] = 1 - len(unsupported) / len(claims)
        issues += [f"unsupported claim: {c[:120]}" for c in unsupported]
    else:
        s["faithfulness"] = None
    return s, issues
