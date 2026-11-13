"""Offline tests: no API key, no network, run in seconds.

They protect the *evaluation harness itself*. A bug in the harness quietly corrupts every score you
compute with it (see `test_converter_keeps_langchain_1x_tool_calls`).
"""

import asyncio
import json
import math

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aurora_evals.config import GOLDEN_DIR, KB_DIR
from aurora_evals.gates import check_gates, load_gates
from aurora_evals.judge_alignment import alignment_report, cohen_kappa
from aurora_evals.knowledge_base import load_documents, split_documents
from aurora_evals.messages import PLACEHOLDER, to_ragas_messages, to_ragas_tool_calls
from aurora_evals.metrics import MetricSpec, cites_sources, state_check


# ----------------------------------------------------------------------------- datasets
def _jsonl(name):
    return [json.loads(line) for line in (GOLDEN_DIR / name).read_text().splitlines() if line.strip()]


def test_rag_golden_schema():
    rows = _jsonl("rag_golden.jsonl")
    kb_files = {p.name for p in KB_DIR.glob("*.md")}
    assert len({r["id"] for r in rows}) == len(rows), "duplicate ids"
    for r in rows:
        assert {"id", "user_input", "reference", "reference_sources", "question_type"} <= r.keys()
        assert set(r["reference_sources"]) <= kb_files, f"{r['id']} cites a missing document"


def test_agent_scenarios_schema():
    for r in _jsonl("agent_scenarios.jsonl"):
        assert {"id", "user_input", "reference_tool_calls", "reference_topics", "expected_state"} <= r.keys()
        for call in r["reference_tool_calls"]:
            assert {"name", "args"} <= call.keys()


def test_calibration_labels_are_valid():
    labels = {r["human_label"] for r in _jsonl("judge_calibration.jsonl")}
    assert labels <= {"pass", "fail"} and len(labels) == 2


# ----------------------------------------------------------------------------- knowledge base
def test_header_aware_chunks_keep_their_section():
    chunks = split_documents(load_documents(), chunk_size=800, chunk_overlap=100, header_aware=True)
    savings = [c for c in chunks if "4.25%" in c.page_content]
    assert savings and all("High-Yield Savings" in c.page_content for c in savings)
    assert all("source" in c.metadata for c in chunks)


def test_naive_chunks_are_smaller():
    docs = load_documents()
    naive = split_documents(docs, chunk_size=200, chunk_overlap=0, header_aware=False)
    smart = split_documents(docs, chunk_size=800, chunk_overlap=100, header_aware=True)
    assert len(naive) > len(smart)


# ----------------------------------------------------------------------------- converter
def _trace():
    return [
        HumanMessage("Freeze card 4242 for C-1001, and what's the wire limit?"),
        AIMessage(
            "",
            tool_calls=[
                {"name": "freeze_card", "args": {"customer_id": "C-1001", "card_last4": "4242"}, "id": "1"},
                {"name": "search_policy", "args": {"query": "wire limit"}, "id": "2"},
            ],
        ),
        ToolMessage("frozen", tool_call_id="1"),
        ToolMessage("limit is $50,000", tool_call_id="2"),
        AIMessage("Done. The domestic limit is $50,000."),
    ]


def test_converter_keeps_langchain_1x_tool_calls():
    """ragas' own convert_to_ragas_messages drops these (it reads additional_kwargs)."""
    msgs = to_ragas_messages(_trace())
    calls = [tc for m in msgs if getattr(m, "tool_calls", None) for tc in m.tool_calls]
    assert [c.name for c in calls] == ["freeze_card", "search_policy"]


def test_free_text_args_are_masked_on_both_sides():
    msgs = to_ragas_messages(_trace())
    pred = [tc for m in msgs if getattr(m, "tool_calls", None) for tc in m.tool_calls]
    ref = to_ragas_tool_calls([{"name": "search_policy", "args": {"query": "anything at all"}}])
    assert pred[1].args == ref[0].args == {"query": PLACEHOLDER}
    assert pred[0].args["card_last4"] == "4242"  # IDs are never masked


def test_tool_call_accuracy_end_to_end_without_llm():
    from ragas.metrics.collections import ToolCallAccuracy

    ref = to_ragas_tool_calls(
        [
            {"name": "freeze_card", "args": {"customer_id": "C-1001", "card_last4": "4242"}},
            {"name": "search_policy", "args": {"query": "international wire limit"}},
        ]
    )
    result = asyncio.run(ToolCallAccuracy().ascore(user_input=to_ragas_messages(_trace()), reference_tool_calls=ref))
    assert result.value == 1.0


# ----------------------------------------------------------------------------- code-based metrics
def test_cites_sources():
    ok = cites_sources.score(response="... [Sources: credit_cards.md]", retrieved_sources=["credit_cards.md"])
    missing = cites_sources.score(response="No citation here.", retrieved_sources=["credit_cards.md"])
    wrong = cites_sources.score(response="[Sources: made_up.md]", retrieved_sources=["credit_cards.md"])
    assert (ok.value, missing.value, wrong.value) == ("pass", "fail", "fail")


def test_state_check():
    expected = {"frozen_cards": ["4242"], "disputes_opened": []}
    assert state_check.score(expected_state=expected, frozen_cards=["4242"], disputes_opened=[]).value == "pass"
    assert state_check.score(expected_state=expected, frozen_cards=[], disputes_opened=[]).value == "fail"


def test_metric_spec_returns_nan_when_input_missing():
    spec = MetricSpec("cites_sources", cites_sources, ("response", "retrieved_sources"))
    value, reason = asyncio.run(spec.ascore({"response": "x", "retrieved_sources": None}))
    assert math.isnan(value) and reason.startswith("n/a")


# ----------------------------------------------------------------------------- gates
def _df(**cols):
    return pd.DataFrame(cols)


def test_overall_gate_pass_and_fail():
    gates = {"overall": {"faithfulness": {"min": 0.8}}}
    assert check_gates(_df(faithfulness=[0.9, 0.8]), gates)[0].passed
    assert not check_gates(_df(faithfulness=[0.9, 0.5]), gates)[0].passed


def test_slice_gate_catches_what_the_average_hides():
    df = _df(policy_compliance=[1.0] * 9 + [0.0], question_type=["factual"] * 9 + ["adversarial"])
    gates = {
        "overall": {"policy_compliance": {"min": 0.9}},
        "slices": {"question_type=adversarial": {"policy_compliance": {"min": 1.0}}},
    }
    overall, adversarial = check_gates(df, gates)
    assert overall.passed and not adversarial.passed


def test_regression_gate():
    gates = {"regression": {"max_drop": 0.05, "metrics": ["context_recall"]}}
    assert check_gates(_df(context_recall=[0.86]), gates, {"context_recall": 0.90})[0].passed
    assert not check_gates(_df(context_recall=[0.80]), gates, {"context_recall": 0.90})[0].passed


def test_missing_metric_never_passes():
    assert not check_gates(_df(other=[1.0]), {"overall": {"faithfulness": {"min": 0.0}}})[0].passed


def test_project_gate_file_is_valid():
    gates = load_gates()
    assert {"rag", "agent"} <= gates.keys()
    for section in gates.values():
        for rule in section.get("overall", {}).values():
            assert 0 <= rule["min"] <= 5


# ----------------------------------------------------------------------------- judge alignment
def test_cohen_kappa_known_values():
    assert cohen_kappa(["pass", "fail"] * 5, ["pass", "fail"] * 5) == pytest.approx(1.0)
    # agreement no better than chance → kappa 0
    assert cohen_kappa(["pass", "pass", "fail", "fail"], ["pass", "fail", "pass", "fail"]) == pytest.approx(0.0)


def test_alignment_report_counts_dangerous_errors():
    report = alignment_report(["fail", "fail", "pass"], ["pass", "fail", "fail"])
    assert report["false_pass"] == 1 and report["false_fail"] == 1
