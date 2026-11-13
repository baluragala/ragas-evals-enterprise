#!/usr/bin/env python
"""Run an evaluation from the command line, the same way a CI pipeline would.

Examples
--------
    # Evaluate the production RAG config and check it against the quality gates
    python scripts/run_eval.py rag --config baseline

    # Evaluate a proposed change and compare with the saved reference baseline (regression gate)
    python scripts/run_eval.py rag --config candidate_v3

    # Quick smoke test on 4 questions (cheap, for every pull request)
    python scripts/run_eval.py rag --config baseline --limit 4 --no-gates

    # Promote a run to become the new reference baseline (after review!)
    python scripts/run_eval.py rag --config candidate_v3 --promote

    # Evaluate the agent with system prompt v1 or v2
    python scripts/run_eval.py agent --prompt v2

Exit code: 0 if every gate passes, 1 if any gate fails. CI uses this to block a merge.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aurora_evals.agent_app import SYSTEM_PROMPT, SYSTEM_PROMPT_V2, AuroraAgent  # noqa: E402
from aurora_evals.config import RESULTS_DIR, setup_environment  # noqa: E402
from aurora_evals.evaluation import EvalRun, evaluate_agent, evaluate_rag, load_rag_golden  # noqa: E402
from aurora_evals.gates import check_gates, format_gate_report, load_gates  # noqa: E402
from aurora_evals.metrics import agent_suite, business_suite, rag_quality_suite  # noqa: E402
from aurora_evals.rag_app import PRESET_CONFIGS  # noqa: E402

REFERENCE_DIR = RESULTS_DIR / "reference"
warnings.filterwarnings("ignore")  # ragas emits many deprecation/user warnings; keep CI logs readable
pd.set_option("display.width", 160)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=["rag", "agent"])
    parser.add_argument("--config", default="baseline", choices=sorted(PRESET_CONFIGS), help="RAG config")
    parser.add_argument("--prompt", default="v2", choices=["v1", "v2"], help="agent system prompt")
    parser.add_argument("--limit", type=int, default=None, help="only the first N golden samples")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--no-gates", action="store_true", help="report only, always exit 0")
    parser.add_argument("--promote", action="store_true", help="save this run as the reference baseline")
    args = parser.parse_args()

    setup_environment(require_key=True)

    if args.target == "rag":
        samples = load_rag_golden()[: args.limit] if args.limit else None
        run = await evaluate_rag(
            args.config, rag_quality_suite() + business_suite(), samples=samples, concurrency=args.concurrency
        )
        reference = REFERENCE_DIR / "rag_baseline" / "summary.json"
    else:
        agent = AuroraAgent(system_prompt=SYSTEM_PROMPT_V2 if args.prompt == "v2" else SYSTEM_PROMPT)
        run, _ = await evaluate_agent(agent_suite(), agent=agent, run_name=f"agent_{args.prompt}")
        reference = REFERENCE_DIR / "agent_baseline" / "summary.json"

    out = run.save()
    print(f"\n=== {run.name}: summary ===")
    print(run.summary().to_string())
    if "question_type" in run.df:
        print("\n=== by question_type ===")
        print(run.by_slice("question_type").to_string())
    if not run.errors().empty:
        print("\n⚠️  metric errors (scored as NaN):")
        print(run.errors().to_string())
    print(f"\nPer-sample scores and judge reasons: {out / 'scores.csv'}")

    if args.promote:
        target = reference.parent
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "summary.json", target / "summary.json")
        print(f"📌 Promoted to reference baseline: {target / 'summary.json'}")

    if args.no_gates:
        return 0

    baseline = EvalRun.load_summary(reference) if reference.exists() and not args.promote else None
    results = check_gates(run.df, load_gates()[args.target], baseline_summary=baseline)
    print("\n=== quality gates ===")
    print(format_gate_report(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
