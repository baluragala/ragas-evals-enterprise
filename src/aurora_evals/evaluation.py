"""The evaluation runner: run the system on a golden dataset, then grade every output.

The pipeline has two separate stages, and keeping them separate is what makes it "enterprise grade":

    1. GENERATE   run the system under test on every golden question → records
    2. SCORE      run every metric on every record                     → a tidy DataFrame

Because the stages are separate you can re-score saved outputs with new metrics without paying to
regenerate them, score production logs that never went through stage 1, and swap judge models
to check that your conclusions don't depend on one judge.

Both stages are async with a concurrency limit (rate limits are real), and a failing metric call
becomes NaN plus an error message instead of crashing a 2-hour evaluation run.
"""

from __future__ import annotations

import asyncio
import json
import math
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Awaitable, Callable, Iterable

import pandas as pd
from tqdm.auto import tqdm

from .agent_app import AuroraAgent
from .config import APP_MODEL, GOLDEN_DIR, JUDGE_MODEL, RESULTS_DIR
from .messages import to_ragas_messages, to_ragas_tool_calls
from .metrics import MetricSpec
from .rag_app import AuroraRAG, RAGConfig

# Approximate list prices in USD per 1M tokens. Check your provider's pricing page; these are
# only used to give a *relative* cost signal between configurations.
PRICE_PER_1M = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-4.1-mini": (0.40, 1.60)}


# --------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------
def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rag_golden() -> list[dict]:
    return load_jsonl(GOLDEN_DIR / "rag_golden.jsonl")


def load_agent_scenarios() -> list[dict]:
    return load_jsonl(GOLDEN_DIR / "agent_scenarios.jsonl")


def load_calibration_set() -> list[dict]:
    return load_jsonl(GOLDEN_DIR / "judge_calibration.jsonl")


# --------------------------------------------------------------------------------------
# Concurrency helper
# --------------------------------------------------------------------------------------
async def gather_limited(
    factories: Iterable[Callable[[], Awaitable]], limit: int, desc: str | None = None
) -> list:
    """Run coroutine factories with at most `limit` in flight; preserve input order."""
    factories = list(factories)
    sem = asyncio.Semaphore(limit)
    bar = tqdm(total=len(factories), desc=desc, disable=desc is None, leave=False)

    async def run(i, factory):
        async with sem:
            try:
                return i, await factory()
            finally:
                bar.update(1)

    results = await asyncio.gather(*(run(i, f) for i, f in enumerate(factories)))
    bar.close()
    return [r for _, r in sorted(results, key=lambda x: x[0])]


# --------------------------------------------------------------------------------------
# Stage 1: generate
# --------------------------------------------------------------------------------------
async def generate_rag_records(app: AuroraRAG, samples: list[dict], concurrency: int = 8) -> list[dict]:
    """Run the RAG app on each golden sample. A record = golden fields + system outputs."""

    async def one(sample):
        result = await app.ainvoke(sample["user_input"])
        return {**sample, **result.to_dict()}

    return await gather_limited(
        [lambda s=s: one(s) for s in samples], concurrency, desc=f"generate[{app.config.name}]"
    )


async def generate_agent_records(agent: AuroraAgent, scenarios: list[dict], concurrency: int = 4) -> list[dict]:
    """Run the agent on each scenario and convert its trace to Ragas messages."""

    async def one(sc):
        run = await agent.arun(sc["user_input"])
        return {
            **sc,
            "messages": to_ragas_messages(run.messages),
            "reference_tool_calls": to_ragas_tool_calls(sc["reference_tool_calls"]),
            "predicted_tool_calls": run.tool_calls,
            "final_response": run.final_response,
            "frozen_cards": sorted(run.backend.frozen_cards),
            "disputes_opened": [d["id"] for d in run.backend.disputes],
            "latency_s": run.latency_s,
            "input_tokens": run.input_tokens,
            "output_tokens": run.output_tokens,
            "lc_messages": run.messages,
        }

    return await gather_limited([lambda s=s: one(s) for s in scenarios], concurrency, desc="generate[agent]")


# --------------------------------------------------------------------------------------
# Stage 2: score
# --------------------------------------------------------------------------------------
async def score_records(
    records: list[dict], metrics: list[MetricSpec], concurrency: int = 8, keep_reasons: bool = True
) -> pd.DataFrame:
    """Score every (record, metric) pair. Returns one row per record, one column per metric."""

    async def one(rec, spec):
        try:
            value, reason = await spec.ascore(rec)
            return value, reason, None
        except Exception as exc:  # judge timeouts, malformed JSON, rate limits after retries...
            return math.nan, None, f"{type(exc).__name__}: {exc}"[:300]

    pairs = [(i, spec) for i in range(len(records)) for spec in metrics]
    outputs = await gather_limited(
        [lambda i=i, s=s: one(records[i], s) for i, s in pairs], concurrency, desc="score"
    )

    rows = [dict(_flatten(r)) for r in records]
    for (i, spec), (value, reason, error) in zip(pairs, outputs):
        rows[i][spec.name] = value
        if keep_reasons and reason:
            rows[i][f"{spec.name}__reason"] = reason
        if error:
            rows[i][f"{spec.name}__error"] = error
    return pd.DataFrame(rows)


def _flatten(record: dict):
    """Keep DataFrame-friendly fields; drop raw message objects."""
    for k, v in record.items():
        if k in ("lc_messages", "messages", "reference_tool_calls", "extra"):
            continue
        yield k, v


# --------------------------------------------------------------------------------------
# A whole evaluation run
# --------------------------------------------------------------------------------------
@dataclass
class EvalRun:
    name: str
    df: pd.DataFrame
    metric_names: list[str]
    metadata: dict = field(default_factory=dict)

    def summary(self) -> pd.Series:
        """Mean of every metric across the dataset (NaNs ignored), plus ops stats."""
        s = self.df[self.metric_names].mean(numeric_only=True)
        if "latency_s" in self.df:
            s["latency_p50_s"] = self.df["latency_s"].median()
            s["latency_p95_s"] = self.df["latency_s"].quantile(0.95)
        if "input_tokens" in self.df:
            s["cost_per_1k_queries_usd"] = self.cost_per_1k_queries()
        s["n"] = len(self.df)
        return s.round(3)

    def by_slice(self, column: str) -> pd.DataFrame:
        """Metric means per value of a tag column (e.g. question_type). Averages hide failures."""
        return self.df.groupby(column)[self.metric_names].mean().round(3)

    def worst(self, metric: str, n: int = 5) -> pd.DataFrame:
        cols = [c for c in ("id", "user_input", "response", "final_response", metric, f"{metric}__reason") if c in self.df]
        return self.df.sort_values(metric)[cols].head(n)

    def cost_per_1k_queries(self) -> float:
        model = self.metadata.get("app_model", APP_MODEL)
        p_in, p_out = PRICE_PER_1M.get(model, (math.nan, math.nan))
        per_query = (self.df["input_tokens"].mean() * p_in + self.df["output_tokens"].mean() * p_out) / 1e6
        return round(per_query * 1000, 4)

    def errors(self) -> pd.DataFrame:
        cols = [c for c in self.df.columns if c.endswith("__error")]
        return self.df.loc[self.df[cols].notna().any(axis=1), ["id", *cols]] if cols else pd.DataFrame()

    def save(self, out_dir: str | Path = RESULTS_DIR) -> Path:
        """Persist per-sample scores (CSV) and a summary with full provenance (JSON)."""
        out = Path(out_dir) / self.name
        out.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(out / "scores.csv", index=False)
        payload = {"name": self.name, "summary": self.summary().to_dict(), "metadata": self.metadata}
        (out / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
        return out

    @staticmethod
    def load_summary(path: str | Path) -> dict:
        return json.loads(Path(path).read_text())["summary"]


def _provenance(**extra) -> dict:
    """What you need to reproduce a number six months later."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "app_model": APP_MODEL,
        "judge_model": JUDGE_MODEL,
        "ragas_version": version("ragas"),
        "langchain_version": version("langchain"),
        "python": platform.python_version(),
        **extra,
    }


async def evaluate_rag(
    config: RAGConfig | str,
    metrics: list[MetricSpec],
    samples: list[dict] | None = None,
    concurrency: int = 8,
    run_name: str | None = None,
) -> EvalRun:
    """Generate answers with one RAG configuration and score them with `metrics`."""
    app = AuroraRAG(config)
    samples = samples if samples is not None else load_rag_golden()
    records = await generate_rag_records(app, samples, concurrency)
    df = await score_records(records, metrics, concurrency)
    return EvalRun(
        name=run_name or app.config.name,
        df=df,
        metric_names=[m.name for m in metrics],
        metadata=_provenance(rag_config=app.config.__dict__, dataset_size=len(samples)),
    )


async def evaluate_agent(
    metrics: list[MetricSpec],
    scenarios: list[dict] | None = None,
    agent: AuroraAgent | None = None,
    concurrency: int = 4,
    run_name: str = "agent",
) -> tuple[EvalRun, list[dict]]:
    """Run the agent on each scenario and score it. Also returns raw records for trace inspection."""
    agent = agent or AuroraAgent()
    scenarios = scenarios if scenarios is not None else load_agent_scenarios()
    records = await generate_agent_records(agent, scenarios, concurrency)
    df = await score_records(records, metrics, concurrency)
    run = EvalRun(
        name=run_name,
        df=df,
        metric_names=[m.name for m in metrics],
        metadata=_provenance(dataset_size=len(scenarios)),
    )
    return run, records


def diagnose(row: pd.Series) -> str:
    """Rule-of-thumb root cause for one RAG sample, read off its metric pattern.

        context_recall low                  → RETRIEVAL: the needed facts never reached the LLM
        faithfulness low                    → GENERATION: claims not supported by the context
        factual_correctness low (rest ok)   → GENERATION: right context, wrong or incomplete answer
    Reference-based metrics are less meaningful for unanswerable/adversarial items; check those by hand.
    """
    if row.get("question_type") in ("unanswerable", "adversarial"):
        return "check manually (refusal-type question)"
    if row.get("context_recall", 1) < 0.5:
        return "RETRIEVAL: needed facts not retrieved"
    if row.get("faithfulness", 1) < 0.7:
        return "GENERATION: unsupported claims (hallucination)"
    if row.get("factual_correctness", 1) < 0.5:
        return "GENERATION: wrong/incomplete answer from good context"
    return "ok"


def compare_runs(runs: list[EvalRun]) -> pd.DataFrame:
    """Side-by-side summary table: rows = metrics, columns = runs."""
    return pd.DataFrame({r.name: r.summary() for r in runs})
