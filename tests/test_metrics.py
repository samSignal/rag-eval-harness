import pytest

from rageval.gate import check
from rageval.judges import HeuristicJudge, LLMJudge
from rageval.metrics import score_case
from rageval.runner import percentile, run_eval
from rageval.text import is_refusal, normalise, numbers, sentences
from rageval.types import Context, EvalCase, SystemOutput

CTX = [Context("RT-2", "The security deposit may not exceed two months' rent. The landlord shall return the "
                       "deposit within 30 days after the end of the lease.")]
CASE = EvalCase("c1", "Max deposit?", "Two months' rent.", ["RT-2"], ["two months"])


def test_normalisation():
    assert normalise("Two months' rent [RT-2]") == "2 months  rent"
    assert numbers("1,000 euros or 4% within thirty days") == {"1000", "4", "30"}
    assert sentences("It is 2 months [RT-2]. Returned in 30 days.") == ["It is 2 months .", "Returned in 30 days."]
    assert is_refusal("I couldn't find this in the provided documents.") and not is_refusal("It is 30 days.")


def test_heuristic_judge_catches_changed_numbers():
    j = HeuristicJudge()
    assert j.supported("The deposit may not exceed two months' rent.", CTX)
    assert j.supported("The landlord must return the deposit within thirty days.", CTX)  # words vs digits
    assert not j.supported("The deposit may not exceed three months' rent.", CTX)
    assert not j.supported("Tenants may keep pets.", CTX)
    assert not j.supported("anything", [])


def test_llm_judge_parses_and_fails_safe():
    assert LLMJudge(lambda p: 'Sure: {"supported": true}').supported("x", CTX)
    assert not LLMJudge(lambda p: '{"supported": false}').supported("x", CTX)
    assert not LLMJudge(lambda p: "I think so").supported("x", CTX)  # unparseable -> unsupported
    seen = {}
    LLMJudge(lambda p: seen.setdefault("p", p) and '{"supported": true}').supported("claim X", CTX)
    assert "claim X" in seen["p"] and "[RT-2]" in seen["p"]


def test_perfect_answer_scores_perfectly():
    out = SystemOutput("The deposit may not exceed two months' rent. [RT-2]", CTX, ["RT-2"])
    s, issues = score_case(CASE, out, HeuristicJudge())
    assert issues == []
    assert s["context_recall"] == s["citation_precision"] == s["fact_recall"] == s["faithfulness"] == 1.0
    assert s["citation_validity"] == 1.0 and s["false_refusal"] == 0.0


def test_hallucinated_number_and_fake_citation_are_reported():
    out = SystemOutput("The deposit may not exceed three months' rent. [RT-2] [RT-9]", CTX, ["RT-2", "RT-9"])
    s, issues = score_case(CASE, out, HeuristicJudge())
    assert s["faithfulness"] == 0.0 and s["fact_recall"] == 0.0
    assert s["citation_validity"] == 0.5 and s["citation_precision"] == 0.5
    assert any("RT-9" in i for i in issues) and any("unsupported claim" in i for i in issues)


def test_refusals():
    unanswerable = EvalCase("u1", "Speed limit?", answerable=False)
    refusal = SystemOutput("I couldn't find this in the provided documents.", CTX)
    assert score_case(unanswerable, refusal, HeuristicJudge())[0] == {"citation_validity": None, "refusal_accuracy": 1.0}
    s, issues = score_case(unanswerable, SystemOutput("It is 130 km/h.", CTX), HeuristicJudge())
    assert s["refusal_accuracy"] == 0.0 and issues
    s, _ = score_case(CASE, refusal, HeuristicJudge())
    assert s["false_refusal"] == 1.0 and s["faithfulness"] is None


class Echo:
    name = "echo"

    def answer(self, q):
        if q == "boom":
            raise RuntimeError("model down")
        return SystemOutput(f"{q} [RT-2]", CTX, ["RT-2"], prompt_tokens=1000, completion_tokens=500)


def test_runner_keeps_order_records_errors_and_costs():
    cases = [EvalCase(f"c{i}", q, gold_sources=["RT-2"]) for i, q in enumerate(["a deposit", "boom", "b deposit"])]
    run = run_eval(Echo(), cases, concurrency=3, price_in=1.0, price_out=2.0)
    assert [r.case.id for r in run.cases] == ["c0", "c1", "c2"]
    assert run.summary["errors"] == 1 and "model down" in run.cases[1].issues[0]
    assert run.summary["cost_usd"] == pytest.approx((2000 * 1 + 1000 * 2) / 1e6)
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)


def test_gate_rules():
    t = {"faithfulness": {"min": 0.9, "max_drop": 0.02}, "latency_p95_ms": {"max": 100, "max_increase_pct": 20}}
    assert check({"faithfulness": 0.95, "latency_p95_ms": 50}, t) == []
    v = check({"faithfulness": 0.92, "latency_p95_ms": 70}, t, baseline={"faithfulness": 0.96, "latency_p95_ms": 50})
    assert {x.rule.split()[0] for x in v} == {"max_drop", "max_increase_pct"}
    assert check({}, {"faithfulness": {"min": 0.5}})[0].rule == "missing from run"
