"""Evaluate the evaluator: how well does a judge agree with human labels?

`data/judge_calibration.jsonl` holds claims labelled supported/unsupported against one source article,
including paraphrases (hard for lexical judges) and changed numbers (the main hallucination risk).
Reported from the point of view of catching hallucinations (the "unsupported" class).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .judges import Judge
from .types import Context


@dataclass
class Calibration:
    judge: str
    n: int
    accuracy: float
    hallucination_precision: float  # of claims flagged unsupported, share that really are
    hallucination_recall: float  # of truly unsupported claims, share that were flagged
    disagreements: list[str]


def calibrate(judge: Judge, labels_path: str | Path, corpus_path: str | Path) -> Calibration:
    with open(corpus_path, encoding="utf-8") as f:
        corpus = {r["_id"]: r["text"] for r in map(json.loads, f)}
    with open(labels_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    tp = fp = fn = correct = 0
    disagreements = []
    for r in rows:
        predicted = judge.supported(r["claim"], [Context(r["source"], corpus[r["source"]])])
        correct += predicted == r["supported"]
        if not predicted and not r["supported"]:
            tp += 1
        elif not predicted and r["supported"]:
            fp += 1
            disagreements.append(f"{r['id']}: flagged a supported claim: {r['claim']}")
        elif predicted and not r["supported"]:
            fn += 1
            disagreements.append(f"{r['id']}: missed a hallucination: {r['claim']}")
    return Calibration(judge.name, len(rows), round(correct / len(rows), 3),
                       round(tp / (tp + fp), 3) if tp + fp else 0.0, round(tp / (tp + fn), 3) if tp + fn else 0.0,
                       disagreements)
