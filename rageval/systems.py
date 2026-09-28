"""Systems under test. Anything with `answer(question) -> SystemOutput` can be evaluated.

* HTTPSystem: any RAG service with a JSON API (preset for rag-docs-chatbot).
* LegalRAG: a reference pipeline (legal-hybrid-search retrieval + extractive or LLM generator).
* FaultInjector: wraps a system and corrupts numbers in its answers, to prove the metrics catch it.
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path
from typing import Callable, Protocol

import requests

from .text import CITATION, STOP, is_refusal, normalise
from .types import Context, SystemOutput

REFUSAL_TEXT = "I couldn't find this in the provided documents."


class System(Protocol):
    name: str

    def answer(self, question: str) -> SystemOutput: ...


class HTTPSystem:
    """POSTs {question_field: question} to `url` and maps the JSON response with dotted paths."""

    def __init__(self, url: str, name: str = "http", question_field: str = "question", answer_path: str = "answer",
                 sources_path: str = "sources", source_id_field: str = "source", source_text_field: str = "text",
                 timeout: float = 120):
        self.url, self.name, self.timeout = url, name, timeout
        self.qf, self.ap, self.sp, self.sid, self.stext = (question_field, answer_path, sources_path,
                                                           source_id_field, source_text_field)

    @staticmethod
    def _get(obj, path: str):
        for part in path.split("."):
            obj = obj.get(part) if isinstance(obj, dict) else None
        return obj

    def answer(self, question: str) -> SystemOutput:
        start = time.perf_counter()
        try:
            r = requests.post(self.url, json={self.qf: question}, timeout=self.timeout)
            r.raise_for_status()
            body = r.json()
        except (requests.RequestException, ValueError) as e:
            return SystemOutput("", error=str(e), latency_ms=(time.perf_counter() - start) * 1000)
        sources = self._get(body, self.sp) or []
        contexts = [Context(str(s.get(self.sid)), str(s.get(self.stext, ""))) for s in sources]
        answer = str(self._get(body, self.ap) or "")
        cited = CITATION.findall(answer)
        # Services that cite by number ([1], [2]) are mapped to the source ids they refer to
        ids = [contexts[int(c) - 1].id if c.isdigit() and 0 < int(c) <= len(contexts) else c for c in cited]
        return SystemOutput(answer, contexts, list(dict.fromkeys(ids)), latency_ms=(time.perf_counter() - start) * 1000)

    @classmethod
    def rag_docs_chatbot(cls, base_url: str = "http://localhost:8000") -> "HTTPSystem":
        return cls(f"{base_url.rstrip('/')}/ask", name="rag-docs-chatbot")


# ---------- reference pipeline ----------

def _content(text: str) -> set[str]:
    return {t[:6] for t in normalise(text).split() if t not in STOP}


class ExtractiveGenerator:
    """Offline generator: quotes the context sentences that best match the question and cites them."""

    name = "extractive"

    def __init__(self, min_overlap: float = 0.34, max_sentences: int = 2):
        self.min_overlap, self.max_sentences = min_overlap, max_sentences

    def __call__(self, question: str, contexts: list[Context]) -> tuple[str, int, int]:
        q = _content(question)
        scored = []
        for c in contexts:
            for s in re.split(r"(?<=[.!?])\s+", c.text):
                if q:
                    scored.append((len(q & _content(s)) / len(q), c.id, s.strip()))
        scored.sort(key=lambda x: -x[0])
        if not scored or scored[0][0] < self.min_overlap:
            return REFUSAL_TEXT, 0, 0
        best = [x for x in scored if x[0] >= scored[0][0] * 0.8][: self.max_sentences]
        return " ".join(f"{s} [{cid}]" for _, cid, s in best), 0, 0


GEN_PROMPT = """Answer the question using ONLY the sources below. Cite every statement with the source id in
square brackets, e.g. [LC-4]. Be concise. If the sources do not contain the answer, reply exactly:
"{refusal}"

Sources:
{sources}

Question: {question}
Answer:"""


class LLMGenerator:
    def __init__(self, client: Callable[[str], tuple[str, int, int]], name: str = "llm"):
        self.client, self.name = client, name

    def __call__(self, question: str, contexts: list[Context]) -> tuple[str, int, int]:
        sources = "\n\n".join(f"[{c.id}] {c.title}\n{c.text}" for c in contexts)
        return self.client(GEN_PROMPT.format(refusal=REFUSAL_TEXT, sources=sources, question=question))


class LegalRAG:
    """Retrieval from legal-hybrid-search + a generator. Documents are indexed once at start-up."""

    def __init__(self, corpus: str | Path, generator=None, mode: str = "hybrid", top_k: int = 4,
                 min_score: float | None = None):
        from legalsearch.corpus import Document
        from legalsearch.embeddings import WordLlamaEncoder
        from legalsearch.search import HybridSearcher

        with open(corpus, encoding="utf-8") as f:
            docs = [Document(str(r["_id"]), r.get("title", ""), r["text"], r.get("metadata", {}))
                    for r in map(json.loads, f) if r]
        self.searcher = HybridSearcher(WordLlamaEncoder())
        self.searcher.index(docs)
        self.generator = generator or ExtractiveGenerator()
        self.mode, self.top_k = mode, top_k
        self.name = f"legal-rag[{mode}, k={top_k}, {getattr(self.generator, 'name', 'gen')}]"

    def answer(self, question: str) -> SystemOutput:
        start = time.perf_counter()
        hits = self.searcher.search(question, k=self.top_k, mode=self.mode).hits
        contexts = [Context(h.doc.id, h.doc.text, h.doc.title) for h in hits]
        text, pt, ct = self.generator(question, contexts)
        cited = [] if is_refusal(text) else list(dict.fromkeys(CITATION.findall(text)))
        return SystemOutput(text, contexts, cited, pt, ct, (time.perf_counter() - start) * 1000)


class FaultInjector:
    """Changes one number in a fraction of answers (e.g. '14 days' -> '21 days'), seeded for reproducibility.
    Used to verify that the faithfulness and fact metrics detect numeric hallucinations."""

    def __init__(self, system: System, rate: float = 0.3, seed: int = 1):
        self.system, self.rate, self.seed = system, rate, seed
        self.name = f"{system.name} + {int(rate * 100)}% numeric faults"

    def answer(self, question: str) -> SystemOutput:
        out = self.system.answer(question)
        # decided per question (not by call order), so results are identical with parallel runs
        if random.Random(f"{self.seed}:{question}").random() < self.rate:
            out.answer = re.sub(r"\b(\d+)\b", lambda m: str(int(m.group(1)) + 7), out.answer, count=1)
        return out
