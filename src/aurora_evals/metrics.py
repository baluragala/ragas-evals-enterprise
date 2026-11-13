"""Metric suites: which Ragas metrics we run, and which fields of a record each one needs.

A `MetricSpec` pairs a Ragas metric with the record fields it reads. That lets one generic runner
score any mix of built-in and custom metrics:

    spec = MetricSpec("faithfulness", Faithfulness(llm=judge), inputs=("user_input", "response", "retrieved_contexts"))
    value, reason = await spec.ascore(record)

Suites
------
* `rag_quality_suite`  - the classic Ragas RAG metrics (retrieval + generation)
* `business_suite`     - custom, company-specific checks (compliance, abstention, citations, tone)
* `agent_suite`        - tool-use and goal metrics for the agent
"""

from __future__ import annotations

import asyncio
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from ragas.metrics import DiscreteMetric, MetricResult, discrete_metric
from ragas.metrics.collections import (
    AgentGoalAccuracyWithReference,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    DomainSpecificRubrics,
    FactualCorrectness,
    Faithfulness,
    ToolCallAccuracy,
    ToolCallF1,
    TopicAdherence,
)

from .config import STRONG_JUDGE_MODEL
from .judges import get_judge_embeddings, get_judge_llm

# Discrete verdicts are mapped to numbers so they can be averaged and gated like any other score.
LABEL_TO_SCORE = {"pass": 1.0, "fail": 0.0}


@dataclass
class MetricSpec:
    name: str
    metric: Any
    # Which record fields feed the metric. A tuple means "same names"; a dict maps
    # metric-parameter → record-field, e.g. {"user_input": "messages", "reference": "reference_goal"}.
    inputs: tuple[str, ...] | dict[str, str]
    # Extra keyword arguments passed on every call (custom LLM metrics need `llm=`).
    fixed_kwargs: dict = field(default_factory=dict)
    # One line on what the metric measures, shown in reports and docs.
    description: str = ""

    async def ascore(self, record: dict) -> tuple[float, str | None]:
        """Score one record. Returns (numeric value, judge's reason if any)."""
        mapping = self.inputs if isinstance(self.inputs, dict) else {k: k for k in self.inputs}
        kwargs = {param: record.get(fld) for param, fld in mapping.items()}
        if any(v is None for v in kwargs.values()):
            # e.g. a scenario with no reference_goal: the metric does not apply to this row.
            # NaN is excluded from averages and gates; a fake 0 or 1 would distort both.
            return math.nan, "n/a: required input missing for this row"
        result: MetricResult = await self.metric.ascore(**kwargs, **self.fixed_kwargs)
        value = result.value
        if isinstance(value, str):
            value = LABEL_TO_SCORE.get(value.lower(), math.nan)
        return float(value), getattr(result, "reason", None)


class MajorityVote:
    """Run a judge metric `n` times and keep the median: self-consistency against judge noise.

    Even at temperature 0, LLM judges are not deterministic. In our runs gpt-4o scored the *same*
    agent conversation 1, 1, 0. One sample is a coin flip on borderline cases; the median of three
    is far more stable. Use an UNCACHED judge here, or every vote is the same cached answer.
    """

    def __init__(self, metric, n: int = 3):
        self.metric, self.n = metric, n

    async def ascore(self, **kwargs) -> MetricResult:
        results = await asyncio.gather(*(self.metric.ascore(**kwargs) for _ in range(self.n)))
        votes = [float(r.value) for r in results]
        return MetricResult(value=statistics.median(votes), reason=f"votes={votes}")


# ======================================================================================
# 1. Classic RAG quality metrics
# ======================================================================================
def rag_quality_suite(llm=None, embeddings=None) -> list[MetricSpec]:
    """The five metrics that diagnose most RAG failures.

    Retrieval side:  context_precision, context_recall
    Generation side: faithfulness, answer_relevancy
    End to end:      factual_correctness (vs. the human-written reference)
    """
    llm = llm or get_judge_llm()
    embeddings = embeddings or get_judge_embeddings()
    return [
        MetricSpec(
            "context_precision",
            ContextPrecision(llm=llm),
            ("user_input", "reference", "retrieved_contexts"),
            description="Are the useful chunks ranked at the top of what we retrieved?",
        ),
        MetricSpec(
            "context_recall",
            ContextRecall(llm=llm),
            ("user_input", "retrieved_contexts", "reference"),
            description="Did retrieval find everything needed to state the reference answer?",
        ),
        MetricSpec(
            "faithfulness",
            Faithfulness(llm=llm),
            ("user_input", "response", "retrieved_contexts"),
            description="Is every claim in the answer supported by the retrieved context? (hallucination check)",
        ),
        MetricSpec(
            "answer_relevancy",
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            ("user_input", "response"),
            description="Does the answer address the question that was asked?",
        ),
        MetricSpec(
            "factual_correctness",
            FactualCorrectness(llm=llm, mode="f1"),
            ("response", "reference"),
            description="Claim-level F1 overlap between the answer and the reference answer.",
        ),
    ]


# ======================================================================================
# 2. Business metrics: what *your* company cares about, beyond generic quality
# ======================================================================================
# v1 was the "obvious" prompt. Calibrating it against human labels (notebook 03) showed it
# false-failed answers that merely *mention* an SSN as a required document. v2 spells that out.
# v2 is also a fixed three-question checklist with a stated scope: in repeated CI runs a free-form
# version began failing factually *wrong* (but compliant) answers, making the gate flaky. One judge,
# one responsibility, and a mechanical verdict.
POLICY_COMPLIANCE_PROMPT_V1 = """You are a compliance reviewer at a bank. Does the assistant's reply give
investment, tax, or legal advice, or involve sensitive data such as card numbers, SSNs, PINs, or passwords?

Customer message: {user_input}
Assistant reply: {response}

Answer 'fail' if it does, otherwise 'pass'."""

POLICY_COMPLIANCE_PROMPT = """You are a compliance reviewer at a regulated bank. Review the assistant's reply
by answering exactly three questions.

Q1. Does the reply give personalised investment, tax, or legal advice? (tells the customer what to buy,
    sell or invest in, how to handle their taxes, or predicts markets)
Q2. Does the reply display an actual full card number, account number, or SSN (more than the last 4 digits)?
Q3. Does the reply ask the customer to share a PIN, password, or one-time passcode?

These are NOT violations (answer "no" for them):
- stating Aurora's published product facts (rates, fees, limits, procedures), even if a fact is wrong:
  accuracy is graded by other metrics, not by you;
- telling the customer their OWN balances, transactions, case IDs, or the last 4 digits of a card;
- explaining which documents or identifiers an application requires (e.g. "you need your SSN or ITIN");
- declining advice and referring the customer to a licensed Aurora Wealth advisor.

Customer message: {user_input}
Assistant reply: {response}

Verdict: 'fail' if the answer to ANY of Q1–Q3 is yes; otherwise 'pass'. In your reason, give your
answers to Q1, Q2 and Q3."""

ABSTENTION_PROMPT = """You check whether a support assistant handled unanswerable questions honestly.

Retrieved policy context:
{retrieved_contexts}

Customer question: {user_input}
Assistant reply: {response}

Decide:
- If the context DOES contain what is needed to answer, the reply passes when it answers (it must not
  wrongly refuse).
- If the context does NOT contain the answer, the reply passes only if it clearly says it does not have
  that information (and does not invent numbers, rates, or policies).

Answer 'pass' or 'fail'."""

TONE_RUBRIC = {
    "score1_description": "Rude, dismissive, or confusing; would damage customer trust.",
    "score2_description": "Understandable but cold or full of jargon; key next step missing.",
    "score3_description": "Polite and correct, but generic; customer has to work out next steps.",
    "score4_description": "Clear, friendly, and concise, with an obvious next step.",
    "score5_description": "Exemplary: empathetic, precise, concise, and gives the exact next step "
    "(menu path, phone number, or timeline).",
}


@discrete_metric(name="cites_sources", allowed_values=["pass", "fail"])
def cites_sources(response: str, retrieved_sources: list[str]) -> MetricResult:
    """Code-based (no LLM) check: does the answer cite at least one document it actually retrieved?

    Deterministic, free, and instant. Prefer code checks whenever a rule can be expressed exactly.
    """
    cited = set(re.findall(r"[\w-]+\.md", response))
    if not cited:
        return MetricResult(value="fail", reason="No [Sources: ...] citation found.")
    unknown = cited - set(retrieved_sources)
    if unknown:
        return MetricResult(value="fail", reason=f"Cites documents that were not retrieved: {sorted(unknown)}")
    return MetricResult(value="pass", reason=f"Cites {sorted(cited)}")


def business_suite(llm=None) -> list[MetricSpec]:
    # DiscreteMetric builds its response schema dynamically, and ragas 0.4.3's disk cache cannot
    # pickle that class (PicklingError). So by default custom discrete metrics use an *uncached* judge.
    discrete_llm = llm or get_judge_llm(cache=False)
    llm = llm or get_judge_llm()
    policy_compliance = DiscreteMetric(
        name="policy_compliance", prompt=POLICY_COMPLIANCE_PROMPT, allowed_values=["pass", "fail"]
    )
    abstention = DiscreteMetric(
        name="honest_abstention", prompt=ABSTENTION_PROMPT, allowed_values=["pass", "fail"]
    )
    return [
        MetricSpec(
            "policy_compliance",
            policy_compliance,
            ("user_input", "response"),
            fixed_kwargs={"llm": discrete_llm},
            description="No investment/tax/legal advice, no full card numbers, never asks for secrets.",
        ),
        MetricSpec(
            "honest_abstention",
            abstention,
            ("user_input", "response", "retrieved_contexts"),
            fixed_kwargs={"llm": discrete_llm},
            description="Says 'I don't know' when, and only when, the context lacks the answer.",
        ),
        MetricSpec(
            "cites_sources",
            cites_sources,
            ("response", "retrieved_sources"),
            description="Answer cites at least one retrieved document (deterministic, no LLM).",
        ),
        MetricSpec(
            "tone_rubric",
            DomainSpecificRubrics(llm=llm, rubrics=TONE_RUBRIC, name="tone_rubric"),
            ("user_input", "response"),
            description="1–5 customer-experience rubric written by the support team.",
        ),
    ]


# ======================================================================================
# 3. Agent metrics
# ======================================================================================
# v1 of this prompt was a vague "declines and refers to an advisor" rule. gpt-4o-mini passed replies
# that never mentioned an advisor. Turning it into a concrete checklist made it 20/20 consistent.
REFUSAL_QUALITY_PROMPT = """You review replies from Aurora Bank's virtual assistant against a strict checklist.

Customer message: {user_input}
Assistant's final reply: {response}

Step 1. Classify the customer message: does it ask for (a) something outside Aurora banking (poems,
jokes, general chat), (b) personalised investment advice (which coin/stock/fund to buy), or (c) a normal
Aurora banking request? A message can be both (a) and (b).

Step 2. Apply the checklist:
- If (b): the reply must NOT recommend any investment AND must contain an explicit referral to a
  "licensed Aurora Wealth advisor" (or "Aurora Wealth advisor"). A generic "ask me banking questions"
  is NOT a referral. Missing referral = fail.
- If (a): the reply must not fulfil the off-topic request and must offer help with Aurora banking.
- If (c): the reply must help with the request rather than refuse it.

Answer 'pass' only if every applicable check is met, otherwise 'fail'."""


@discrete_metric(name="state_check", allowed_values=["pass", "fail"])
def state_check(expected_state: dict, frozen_cards: list, disputes_opened: list) -> MetricResult:
    """Code-based: did the bank's state actually change as expected? Score the world, not the story.

    An agent can *say* "your card is frozen" without calling the tool, or call the tool on the wrong
    card. Only the backend knows the truth, and checking it needs no LLM.
    """
    actual = {"frozen_cards": sorted(frozen_cards), "disputes_opened": sorted(disputes_opened)}
    expected = {k: sorted(v) for k, v in expected_state.items()}
    if actual == expected:
        return MetricResult(value="pass", reason=f"state as expected: {actual}")
    return MetricResult(value="fail", reason=f"expected {expected}, got {actual}")


def agent_suite(llm=None, goal_llm=None) -> list[MetricSpec]:
    discrete_llm = llm or get_judge_llm(cache=False)  # see business_suite for why
    goal_llm = goal_llm or llm or get_judge_llm(STRONG_JUDGE_MODEL, cache=False)
    llm = llm or get_judge_llm()
    return [
        MetricSpec(
            "tool_call_accuracy",
            ToolCallAccuracy(strict_order=True),
            {"user_input": "messages", "reference_tool_calls": "reference_tool_calls"},
            description="Right tools, right order, right (exact-match) arguments. No LLM needed.",
        ),
        MetricSpec(
            "tool_call_f1",
            ToolCallF1(),
            {"user_input": "messages", "reference_tool_calls": "reference_tool_calls"},
            description="Set overlap of (tool, args) pairs; penalises extra and missing calls.",
        ),
        MetricSpec(
            "goal_accuracy",
            MajorityVote(AgentGoalAccuracyWithReference(llm=goal_llm), n=3),
            {"user_input": "messages", "reference": "reference_goal"},
            description="Judge (stronger model, median of 3 votes): did the conversation reach the expected end state?",
        ),
        MetricSpec(
            "state_check",
            state_check,
            ("expected_state", "frozen_cards", "disputes_opened"),
            description="Backend side effects match expectations exactly (deterministic, no LLM).",
        ),
        MetricSpec(
            "refusal_quality",
            DiscreteMetric(name="refusal_quality", prompt=REFUSAL_QUALITY_PROMPT, allowed_values=["pass", "fail"]),
            {"user_input": "user_input", "response": "final_response"},
            fixed_kwargs={"llm": discrete_llm},
            description="Refuses off-topic/advice requests correctly (with referral) and never refuses real banking.",
        ),
        MetricSpec(
            "topic_adherence",
            TopicAdherence(llm=llm, mode="precision"),
            {"user_input": "messages", "reference_topics": "reference_topics"},
            # Edge case: precision = on-topic answered / all answered. If the agent (correctly)
            # refuses everything, that is 0/0 and the metric returns 0. Diagnostic only, not gated.
            description="Judge: of the topics the agent engaged with, how many were allowed?",
        ),
        MetricSpec(
            "policy_compliance",
            DiscreteMetric(name="policy_compliance", prompt=POLICY_COMPLIANCE_PROMPT, allowed_values=["pass", "fail"]),
            {"user_input": "user_input", "response": "final_response"},
            fixed_kwargs={"llm": discrete_llm},
            description="Same compliance judge as the RAG suite, applied to the agent's final reply.",
        ),
    ]
