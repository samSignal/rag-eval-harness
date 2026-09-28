"""Core data types."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    id: str
    question: str
    reference: str = ""
    gold_sources: list[str] = field(default_factory=list)  # ids of documents that contain the answer
    must_include: list[str] = field(default_factory=list)  # key facts the answer must state, e.g. "72 hours"
    answerable: bool = True  # False -> the correct behaviour is to decline


@dataclass
class Context:
    id: str
    text: str
    title: str = ""


@dataclass
class SystemOutput:
    answer: str
    contexts: list[Context] = field(default_factory=list)  # what retrieval handed to the generator
    citations: list[str] = field(default_factory=list)  # source ids the answer cites
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class CaseResult:
    case: EvalCase
    output: SystemOutput
    scores: dict[str, float | None]
    issues: list[str] = field(default_factory=list)  # human-readable reasons a case lost points


def load_cases(path: str | Path) -> list[EvalCase]:
    with open(path, encoding="utf-8") as f:
        return [EvalCase(**json.loads(line)) for line in f if line.strip()]
