"""Live smoke test: a tiny real evaluation (2 questions, 2 metrics). Costs a fraction of a cent.

Skipped automatically when OPENAI_API_KEY is not set:
    pytest -m live          # run only live tests
    pytest -m "not live"    # skip them
"""

import asyncio
import os

import pytest

from aurora_evals.evaluation import evaluate_rag, load_rag_golden
from aurora_evals.metrics import cites_sources, rag_quality_suite, MetricSpec

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"),
]


def test_rag_pipeline_end_to_end():
    faithfulness = next(m for m in rag_quality_suite() if m.name == "faithfulness")
    metrics = [faithfulness, MetricSpec("cites_sources", cites_sources, ("response", "retrieved_sources"))]
    run = asyncio.run(evaluate_rag("baseline", metrics, samples=load_rag_golden()[:2], concurrency=2))

    assert len(run.df) == 2
    assert run.errors().empty, run.errors().to_string()
    assert run.df["faithfulness"].between(0, 1).all()
    assert run.summary()["cites_sources"] >= 0.5
