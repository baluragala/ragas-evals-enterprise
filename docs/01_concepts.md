# 01 · Concepts: what LLM evaluation is and why it's different

## Why evaluate at all?

Traditional software is tested with exact assertions: `assert add(2, 2) == 4`. LLM applications produce
**open-ended text** that changes between runs and between model versions. "Does it work?" turns into many
questions:

* Is the answer **correct**? Is it **supported** by our documents, or invented?
* Did retrieval find the **right documents**?
* Did it break a **business rule** (compliance, privacy, tone)?
* For agents: did it call the **right tools**, and did the **real system state** change correctly?
* Is it **fast** and **cheap** enough?

An **evaluation (eval)** answers these questions automatically and repeatably, so you can compare versions and
make release decisions with evidence instead of vibes.

```
            ┌──────────────────┐     ┌────────────────────┐     ┌────────────────┐
golden  ──► │ system under test│ ──► │ metrics (+ judge)  │ ──► │ scores, slices │ ──► gates ──► ship / block
dataset     │ (RAG app, agent) │     │ code · LLM · rubric│     │ failures, cost │
            └──────────────────┘     └────────────────────┘     └────────────────┘
```

## The building blocks

### 1. The golden dataset
A curated, **version-controlled** set of test cases. Each case has an input (question or task) and, optionally,
a **reference** (the expert's answer, or the expected tool calls / end state).

Good golden datasets:
* reflect **real traffic** (pull questions from logs, support tickets, SMEs);
* over-represent **hard and dangerous** cases: multi-hop, numeric, unanswerable, adversarial, PII;
* are **tagged** (question type, product, difficulty) so results can be sliced;
* **grow** every time production reveals a new failure (see the playbook).

### 2. Metrics
A metric turns one (input, output, context, reference) record into a number or label. Three families:

| family | example | strengths | weaknesses |
|---|---|---|---|
| **Deterministic / code** | exact match, regex, JSON schema, `state_check`, `cites_sources` | free, instant, 100% reproducible | only for rules you can express exactly |
| **Embedding-based** | semantic similarity, `answer_relevancy` (partly) | cheap, no judge prompt | similarity ≠ correctness |
| **LLM-as-judge** | `faithfulness`, `policy_compliance`, rubrics | can judge meaning, tone, policy | costs money, is noisy, must be **calibrated** |

### 3. Reference-based vs reference-free
* **Reference-based** metrics compare with a gold answer (`context_recall`, `factual_correctness`). They are the
  most informative, but you only have references for your golden dataset.
* **Reference-free** metrics need only the input/output/context (`faithfulness`, `answer_relevancy`,
  `policy_compliance`). They also work on **live production traffic**, which makes them the basis of monitoring.

### 4. The judge
For LLM-as-judge metrics, a model (the *judge*) makes small structured decisions. Ragas breaks big questions
into many small ones. Faithfulness, for example, = "split the answer into claims; for each claim, is it supported
by the context?" Small decisions are far more reliable than "rate this answer 1–10".

A judge is itself a model making predictions, so it needs its own evaluation: **calibration against human labels**
(notebook 03).

## RAG evaluation in one picture

```
question ──► RETRIEVER ──► contexts ──► GENERATOR (LLM) ──► answer
                │                            │
     context_precision                 faithfulness       (answer vs contexts)
     context_recall (needs reference)  answer_relevancy   (answer vs question)
                                       factual_correctness (answer vs reference)
```

The **pattern** of scores localises the fault:

| context_recall | faithfulness | factual_correctness | diagnosis |
|---|---|---|---|
| low | n/a | low | **Retrieval**: the facts never reached the model |
| high | low | low | **Generation**: hallucination |
| high | high | low | **Generation**: reasoning error, or the reference is wrong |
| high | high | high | ✅ |

## Agent evaluation in one picture

| layer | question | example metrics |
|---|---|---|
| Trajectory | right tools, order, arguments? | ToolCallAccuracy, ToolCallF1 |
| Outcome | did the conversation reach the goal? | AgentGoalAccuracy |
| Side effects | did the backend really change? | custom state checks |
| Behaviour | refusals, compliance, staying on topic | custom judges, TopicAdherence |

## Offline vs online evaluation

| | offline (pre-release) | online (production) |
|---|---|---|
| data | golden dataset | sampled live traffic |
| references | yes | usually no |
| metrics | all | reference-free + user feedback + business KPIs |
| purpose | **gate** releases, compare versions | **monitor** drift, discover new failure modes |
| this course | notebooks 01–05 | notebook 05 §7 |

The two connect: production failures become new golden test cases, so the offline suite gets stronger over time.

## Why Ragas?

[Ragas](https://docs.ragas.io) provides research-backed implementations of RAG and agent metrics, tools to build
custom metrics (`DiscreteMetric`, `@discrete_metric`, rubrics), experiment tracking (`Dataset`, `@experiment`),
judge caching, and synthetic test generation, so you don't have to write and validate judge prompts from scratch.
It is framework-agnostic: here the app is LangChain 1.x, but the same metrics work for LlamaIndex, Haystack, or
plain API calls.
