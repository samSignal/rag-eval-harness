"""Judges decide whether a claim is supported by the retrieved context.

HeuristicJudge is deterministic and free: a claim is supported only if every number in it appears in
the context (the most common and most harmful legal hallucination is a wrong number of days, months
or percent) and most of its content words appear in one context sentence.

LLMJudge asks a model to verify each claim, which handles paraphrase better. It takes any
prompt -> text function, so it works with OpenAI, Ollama or a fake in tests.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Protocol

from .text import numbers, sentences, terms
from .types import Context


class Judge(Protocol):
    name: str

    def supported(self, claim: str, contexts: list[Context]) -> bool: ...


class HeuristicJudge:
    name = "heuristic"

    def __init__(self, min_overlap: float = 0.5):
        self.min_overlap = min_overlap

    def supported(self, claim: str, contexts: list[Context]) -> bool:
        if not contexts:
            return False
        ctx_text = " ".join(c.text for c in contexts)
        if not numbers(claim) <= numbers(ctx_text):
            return False
        claim_terms = terms(claim)
        if not claim_terms:
            return True
        best = max(len(claim_terms & terms(s)) / len(claim_terms)
                   for c in contexts for s in sentences(c.text) or [c.text])
        return best >= self.min_overlap


JUDGE_PROMPT = """You check answers for hallucinations. Is the CLAIM fully supported by the CONTEXT?
Numbers, durations and percentages must match exactly. Reply with JSON only: {{"supported": true}} or {{"supported": false}}.

CONTEXT:
{context}

CLAIM: {claim}"""


class LLMJudge:
    def __init__(self, complete: Callable[[str], str], name: str = "llm"):
        self.complete, self.name = complete, name

    def supported(self, claim: str, contexts: list[Context]) -> bool:
        context = "\n\n".join(f"[{c.id}] {c.text}" for c in contexts)
        raw = self.complete(JUDGE_PROMPT.format(context=context, claim=claim))
        m = re.search(r"\{.*?\}", raw, re.S)
        try:
            return bool(json.loads(m.group(0))["supported"]) if m else False
        except (ValueError, KeyError, TypeError):
            return False  # unparseable verdicts count as unsupported: fail safe, never inflate scores
