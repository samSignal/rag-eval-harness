# RAG Eval Harness

![tests](https://github.com/samSignal/rag-eval-harness/actions/workflows/eval.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

An evaluation and regression-testing framework for RAG systems. It answers the question that separates a demo from a production feature: **is this change actually better, and would we notice if it got worse?**

Each question is scored on retrieval, citations, faithfulness, key facts, refusals, latency and cost. The results go into a report that lists the specific problem behind every failed case, and a **quality gate in CI fails the build when a metric regresses**.

## Metrics

| Stage | Metric | What it catches |
|---|---|---|
| Retrieval | **Context recall / precision** | The right source never reached the LLM, or it was buried in noise |
| Citations | **Citation precision / recall** | The answer cites the wrong article, or not the one that answers the question |
| Citations | **Citation validity** | The answer cites a source that was never retrieved (a hallucinated citation) |
| Answer | **Key-fact recall** | The answer omits the fact that matters ("72 hours", "two months' rent"); number words and digits are treated as equal |
| Answer | **Faithfulness** | Share of answer sentences supported by the retrieved context, checked by a pluggable judge |
| Behaviour | **Refusal accuracy / false refusals** | Answering what the documents don't cover, or declining what they do |
| Operations | **p50/p95 latency, tokens, cost** | Slow or expensive changes (prices are passed in, so they stay current) |

**Judges.** `HeuristicJudge` is deterministic and free. It requires every number in a claim to appear in the context, since a wrong number of days, months or percent is the most damaging legal hallucination, and it requires most content words to match one context sentence. `LLMJudge` asks a model (OpenAI or Ollama) to verify each claim. It fails safe: an unparseable verdict counts as "unsupported" and never inflates the score.

**Evaluating the evaluator.** `rageval calibrate` measures a judge against 30 human-labelled claims (paraphrases plus claims with changed numbers):

```text
Judge heuristic on 30 labelled claims: accuracy 0.767, hallucination precision 0.778, recall 0.583
  - j02: missed a hallucination: A probation period can last up to six months.
  - j13: flagged a supported claim: The regulator has to be told about a leak ... generally inside three days.
```

The heuristic judge catches most changed numbers, but it misses hallucinations that reuse a number found elsewhere in the article ("six months" does appear in LC-1, in a different rule), and it rejects correct paraphrases. That is the measured case for using `--judge openai:<model>` on important runs. The same command measures the LLM judge's agreement.

## Results

Evaluated on 36 cases (32 answerable legal questions with gold sources and key facts, plus 4 questions the documents cannot answer) over the Norland collection from [legal-hybrid-search](https://github.com/samSignal/legal-hybrid-search). System: the reference pipeline with hybrid retrieval and an offline extractive generator. Judge: heuristic.

| System | Context recall | Citation precision | Citation validity | Key-fact recall | Faithfulness | Refusal accuracy | False refusals | p95 ms |
|---|---|---|---|---|---|---|---|---|
| Hybrid retrieval, k=4 | 1.000 | 0.734 | 1.000 | 0.750 | 1.000 | 1.000 | 0.188 | 7.3 |
| BM25 only, k=4 | 1.000 | 0.734 | 1.000 | 0.750 | 1.000 | 1.000 | 0.188 | 7.7 |
| Dense only, k=4 | 0.969 | 0.734 | 1.000 | 0.750 | 1.000 | 1.000 | 0.188 | 7.1 |
| Hybrid, k=2 | 0.969 | 0.734 | 1.000 | 0.750 | 1.000 | 1.000 | 0.188 | 9.6 |
| Hybrid, k=4 **+ 30% injected numeric faults** | 1.000 | 0.734 | 1.000 | **0.562** | **0.788** | 1.000 | 0.188 | 7.3 |

**What this shows**

- **The bottleneck is the generator, not retrieval.** Context recall is 1.0, so the right article was always retrieved, yet key-fact recall is only 0.75 and 19% of answerable questions were declined. Per-stage metrics separate these, which tells you where to spend effort: here, on the generator (an LLM instead of sentence extraction), not on retrieval.
- **Faithful is not the same as correct.** Faithfulness is 1.0 (every sentence is quoted from the law) while some answers quote the wrong rule, e.g. the employer's notice period instead of the employee's (`c08`). You need both metrics.
- **Fault injection proves the harness works.** `FaultInjector` silently changes a number in 30% of answers ("20 working days" becomes "27"). Faithfulness drops by 0.21 and key-fact recall by 0.19, while retrieval metrics stay unchanged, and the report names each corrupted sentence. The CI gate fails this run:

```text
$ rageval gate runs/faulty.json --thresholds thresholds.json --baseline runs/baseline.json
QUALITY GATE FAILED
  - fact_recall: 0.5625 violates min 0.7
  - fact_recall: 0.5625 violates max_drop vs baseline 0.75 -> 0.03
  - faithfulness: 0.7885 violates min 0.95
  - faithfulness: 0.7885 violates max_drop vs baseline 1 -> 0.02
```

All numbers here come from the offline reference system, so they need no API key. Run with `--llm openai:gpt-4o-mini` to evaluate a real LLM generator, which also records token usage and cost.

## Quick start

```bash
git clone https://github.com/samSignal/rag-eval-harness.git
cd rag-eval-harness
pip install -e ".[dev]"

rageval run --out runs/current.json --html runs/report.html       # offline reference pipeline
rageval gate runs/current.json --thresholds thresholds.json --baseline runs/baseline.json
rageval compare runs/baseline.json runs/current.json
rageval calibrate                                                  # judge vs human labels

# with real models (pip install -e ".[openai]", set OPENAI_API_KEY)
rageval run --llm openai:gpt-4o-mini --judge openai:gpt-4o-mini --price-in <usd/1M> --price-out <usd/1M>
rageval run --llm ollama:llama3.1 --judge ollama:llama3.1          # free, local

# any RAG service with a JSON API, e.g. rag-docs-chatbot
rageval run --system http --url http://localhost:8000/ask --cases my_cases.jsonl
```

The HTML report lists every case with the answer, cited and retrieved sources, and the exact reason it lost points.

## Writing eval cases

```json
{"id": "c11", "question": "Within how many hours must a data breach be reported?",
 "reference": "Within 72 hours of becoming aware of it.", "gold_sources": ["DP-6"], "must_include": ["72 hours"]}
{"id": "u1", "question": "What is the corporate income tax rate?", "answerable": false}
```

## Design

```
rageval/
  types.py      EvalCase, Context, SystemOutput, CaseResult
  systems.py    HTTPSystem, LegalRAG reference pipeline, Extractive/LLM generators, FaultInjector
  judges.py     HeuristicJudge, LLMJudge
  metrics.py    per-case scoring with human-readable issues
  runner.py     concurrent runs (order-preserving), aggregation, latency percentiles, cost
  gate.py       thresholds + baseline comparison
  calibrate.py  judge agreement with human labels
  report.py     Markdown and HTML reports
  llm.py        OpenAI / Ollama clients with token usage
```

- Any object with `answer(question) -> SystemOutput` can be evaluated, so the harness is not tied to one framework.
- A system that crashes or times out is recorded as an error for that case; it does not stop the run.
- Fault injection is seeded per question, so results are identical whether the run is sequential or parallel.

## Tests and CI

`pytest -q` runs 14 tests covering metric maths, judge behaviour (including fail-safe parsing), refusals, hallucinated citations, cost accounting, the HTTP adapter against a local fake service, detection of injected faults, and the CLI exit codes. On every push, GitHub Actions runs the tests, then the full evaluation (posted to the run summary and uploaded as an HTML artifact), then the quality gate against the committed baseline.

## License

MIT. The legal texts are fictional (see legal-hybrid-search).
