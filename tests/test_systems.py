import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from rageval.calibrate import calibrate
from rageval.cli import main
from rageval.judges import HeuristicJudge
from rageval.runner import run_eval
from rageval.systems import FaultInjector, HTTPSystem, LegalRAG, LLMGenerator
from rageval.types import load_cases

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def rag():
    return LegalRAG(DATA / "corpus.jsonl")


@pytest.fixture(scope="module")
def cases():
    return load_cases(DATA / "cases.jsonl")


def test_reference_pipeline_meets_baseline(rag, cases):
    s = run_eval(rag, cases).summary
    assert s["context_recall"] >= 0.95 and s["citation_validity"] == 1.0
    assert s["faithfulness"] >= 0.95 and s["refusal_accuracy"] >= 0.75 and s["fact_recall"] >= 0.7


def test_fault_injection_is_detected(rag, cases):
    clean = run_eval(rag, cases).summary
    faulty = run_eval(FaultInjector(rag, rate=0.3), cases).summary
    assert faulty["faithfulness"] < clean["faithfulness"] - 0.1
    assert faulty["fact_recall"] < clean["fact_recall"] - 0.1
    assert faulty["context_recall"] == clean["context_recall"]  # retrieval unaffected: metrics isolate the stage


def test_llm_generator_prompt_citations_and_tokens(rag):
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return "Breaches must be reported within 72 hours. [DP-6]", 900, 20

    rag_llm = LegalRAG(DATA / "corpus.jsonl", generator=LLMGenerator(fake_llm))
    out = rag_llm.answer("When must a data breach be reported?")
    assert "[DP-6]" in prompts[0] and "Question: When must a data breach" in prompts[0]
    assert out.citations == ["DP-6"] and out.prompt_tokens == 900 and out.completion_tokens == 20


def test_http_system_maps_numbered_citations():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = json.dumps({"answer": f"Deposit is two months' rent [2]. ({body['question']})",
                              "sources": [{"source": "a.md", "text": "x"}, {"source": "lease.pdf", "text": "y"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = HTTPSystem.rag_docs_chatbot(f"http://127.0.0.1:{srv.server_port}").answer("deposit?")
        assert out.citations == ["lease.pdf"] and [c.id for c in out.contexts] == ["a.md", "lease.pdf"]
        down = HTTPSystem("http://127.0.0.1:1/ask", timeout=2).answer("x")
        assert down.error and down.answer == ""
    finally:
        srv.shutdown()


def test_heuristic_judge_calibration_is_reported():
    c = calibrate(HeuristicJudge(), DATA / "judge_calibration.jsonl", DATA / "corpus.jsonl")
    assert c.n == 30 and c.accuracy >= 0.7 and len(c.disagreements) == round(c.n * (1 - c.accuracy))


def test_cli_run_and_gate(tmp_path):
    out = tmp_path / "run.json"
    assert main(["run", "--out", str(out), "--html", str(tmp_path / "r.html")]) == 0
    assert json.loads(out.read_text())["summary"]["cases"] == 36
    assert "<table>" in (tmp_path / "r.html").read_text()
    th = tmp_path / "t.json"
    th.write_text(json.dumps({"faithfulness": {"min": 0.95}}))
    assert main(["gate", str(out), "--thresholds", str(th)]) == 0
    th.write_text(json.dumps({"faithfulness": {"min": 1.01}}))
    assert main(["gate", str(out), "--thresholds", str(th)]) == 1
