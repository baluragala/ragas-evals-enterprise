# Enterprise LLM Evaluation with Ragas + LangChain 1.x

**Learn to evaluate RAG assistants and tool-using agents the way production teams do:** golden datasets,
research-backed metrics, custom business rules, calibrated LLM judges, quality gates in CI, and production
monitoring. Built around one realistic case study.

![python](https://img.shields.io/badge/python-3.10%2B%20(tested%203.12)-blue) ![ragas](https://img.shields.io/badge/ragas-0.4.3-orange) ![langchain](https://img.shields.io/badge/langchain-1.x-green) ![colab](https://img.shields.io/badge/Google%20Colab-ready-yellow)

---

## 🏦 The case study: Aurora Bank

*Aurora Bank* (fictional) is launching two AI systems for customer support:

1. **A RAG assistant** that answers questions from 8 policy documents (credit cards, disputes & fraud, wires,
   account opening/KYC, savings & CDs, overdraft fees, privacy, app security).
2. **A tool-using agent** that looks up balances, lists transactions, freezes cards, opens disputes and searches
   policy, backed by a mock core-banking system.

In a regulated bank a wrong answer is a complaint, and "you should buy Tesla stock" is a regulatory incident.
So nothing ships until it passes an evaluation. Along the way, three teams propose changes, and you decide with
data which ones ship:

| proposal | what it changes | outcome in our reference run |
|---|---|---|
| `baseline` (production) | header-aware chunks, top-4, grounded prompt | ✅ ships: 9/9 gates |
| `naive_chunking` ("cut cost 40%") | 200-char chunks, top-2 | ❌ **blocked**: context recall ~0.9 → ~0.7, regression gates |
| `loose_prompt` ("make it friendlier") | drops grounding rules | ❌ **blocked**: recommends stocks/crypto (adversarial slice 0.0), faithfulness ~0.88 → ~0.62, citations → 0, **while tone went *up*** |
| `candidate_v3` (failure-driven fix) | prompt rules from failure analysis, top-6 | ✅ ships: 13/13 gates, better multi-hop & advice handling (+~35% cost) |
| agent prompt v1 → v2 | follow-up and referral rules | v1 ❌ blocked on `refusal_quality`, v2 ✅ ships |

*(LLM outputs vary a little between runs; your numbers will be close to these, not identical.)*

## 🎯 What you will learn

* Build a **golden dataset** with tags and the hard cases that matter (multi-hop, unanswerable, adversarial)
* Use Ragas' core RAG metrics (**context precision/recall, faithfulness, answer relevancy, factual correctness**)
  and read their **pattern** to tell retrieval failures from generation failures
* Write **custom business metrics**: code-based, LLM-judge (`DiscreteMetric`), and rubric
* **Calibrate an LLM judge** against human labels (Cohen's κ, false-pass vs false-fail) and fix it
* Evaluate **agents**: tool-call accuracy, goal accuracy, deterministic **state checks**, refusal quality
* Turn evals into **release decisions**: quality gates (overall / slice / regression), a CLI that blocks merges,
  a GitHub Actions workflow, Ragas `@experiment` tracking, and **reference-free production monitoring**
* Avoid the **real pitfalls** we hit building this (see [Lessons learned](#-lessons-learned-from-building-this))

## 📓 The notebooks

Run them in order. Each one opens in Colab from its badge and is self-contained (it clones the repo and installs
everything).

| # | Notebook | You will… | Time | API cost* |
|---|---|---|---:|---:|
| 00 | [Quickstart](notebooks/00_quickstart.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baluragala/ragas-evals-enterprise/blob/main/notebooks/00_quickstart.ipynb) | score one answer with 3 metrics; catch a hallucination | 10 min | <$0.01 |
| 01 | [RAG evaluation](notebooks/01_rag_evaluation.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baluragala/ragas-evals-enterprise/blob/main/notebooks/01_rag_evaluation.ipynb) | run a full suite on the golden set; slice; diagnose failures; latency and cost | 25 min | ~$0.03 |
| 02 | [A/B testing & failure analysis](notebooks/02_ab_testing_and_failure_analysis.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baluragala/ragas-evals-enterprise/blob/main/notebooks/02_ab_testing_and_failure_analysis.ipynb) | compare versions; catch regressions; build a better version from failures | 30 min | ~$0.10 |
| 03 | [Custom metrics & judge calibration](notebooks/03_custom_metrics_and_judge_calibration.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baluragala/ragas-evals-enterprise/blob/main/notebooks/03_custom_metrics_and_judge_calibration.ipynb) | write business metrics; measure judge vs. humans; fix the judge | 30 min | ~$0.02 |
| 04 | [Agent evaluation](notebooks/04_agent_evaluation.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baluragala/ragas-evals-enterprise/blob/main/notebooks/04_agent_evaluation.ipynb) | grade tools, outcomes, side effects and refusals | 30 min | ~$0.05 |
| 05 | [Quality gates & CI](notebooks/05_quality_gates_and_ci.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baluragala/ragas-evals-enterprise/blob/main/notebooks/05_quality_gates_and_ci.ipynb) | automate ship/block; CI; Ragas experiments; production monitoring | 25 min | ~$0.05 |
| 06 | [Bonus: synthetic test sets](notebooks/06_bonus_synthetic_testset.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baluragala/ragas-evals-enterprise/blob/main/notebooks/06_bonus_synthetic_testset.ipynb) | bootstrap golden data with Ragas' generator | 15 min | ~$0.02 |

\* With the default `gpt-4o-mini` (app and judge) plus `gpt-4o` for agent goal accuracy. The whole course costs
well under **$0.50**. Judge calls are cached on disk, so re-running unchanged cells is nearly free.

The notebooks in this repo are **saved with real outputs** from a full run, so you can read the expected results
before (or instead of) running them.

## 🚀 Setup

You need an **OpenAI API key** ([create one](https://platform.openai.com/api-keys); a few dollars of credit is plenty).

### Option A: Google Colab (recommended for students)
1. Click a **Open in Colab** badge above.
2. In Colab, click the 🔑 **Secrets** icon (left sidebar) → **Add new secret** → name `OPENAI_API_KEY`, paste your
   key, and turn on **Notebook access**.
3. Run the first cell (**Setup**). It clones this repo, installs the pinned requirements (~1 min), and loads your key.
   If you skip step 2, it asks you to paste the key instead.
4. Run the remaining cells top to bottom.

### Option B: Local (macOS / Linux / Windows)
```bash
git clone https://github.com/baluragala/ragas-evals-enterprise.git
cd ragas-evals-enterprise

python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                       # or: uv pip install -r requirements-dev.txt

cp .env.example .env                                      # then put your OPENAI_API_KEY in .env
pytest -m "not live"                                      # 18 offline tests, no API key needed, ~2 s
jupyter lab notebooks/                                    # or open them in VS Code
```

### Check everything works (≈ 10 s, < $0.01)
```bash
pytest -m live          # one tiny real evaluation (2 questions)
```

### Optional configuration (`.env`)
| variable | default | purpose |
|---|---|---|
| `AURORA_APP_MODEL` | `gpt-4o-mini` | the assistant being evaluated |
| `AURORA_JUDGE_MODEL` | `gpt-4o-mini` | the model that grades it |
| `AURORA_STRONG_JUDGE_MODEL` | `gpt-4o` | stronger judge for agent goal accuracy |
| `AURORA_EMBED_MODEL` | `text-embedding-3-small` | retrieval + answer-relevancy embeddings |

## 🖥️ Command line (what CI runs)

```bash
python scripts/run_eval.py rag --config baseline            # evaluate + quality gates (exit 1 if any fails)
python scripts/run_eval.py rag --config candidate_v3        # also checks regression vs the reference baseline
python scripts/run_eval.py rag --config baseline --limit 4 --no-gates   # cheap smoke run
python scripts/run_eval.py rag --config candidate_v3 --promote          # make this the new reference baseline
python scripts/run_eval.py agent --prompt v1                # agent gates (v1 is expected to be blocked)
```
Per-sample scores **with judge reasons** are written to `results/<run>/scores.csv`, and averages plus provenance
to `results/<run>/summary.json`.

## 🗂️ Project structure

```
ragas-evals-enterprise/
├── notebooks/                    # the course (Colab-ready, saved with outputs)
├── src/aurora_evals/             # the reusable evaluation harness
│   ├── config.py                 #   paths, model names, API-key loading (.env / Colab Secrets / prompt)
│   ├── knowledge_base.py         #   load + chunk (header-aware or naive) + embed policy docs
│   ├── rag_app.py                #   SYSTEM UNDER TEST #1: LangChain 1.x RAG, configurable variants & prompts
│   ├── agent_app.py              #   SYSTEM UNDER TEST #2: LangChain 1.x create_agent + mock bank backend
│   ├── messages.py               #   LangChain → Ragas message conversion (+ free-text arg canonicalisation)
│   ├── judges.py                 #   Ragas judge LLM/embeddings via llm_factory, with disk cache
│   ├── metrics.py                #   metric suites: RAG quality · business rules · agent
│   ├── evaluation.py             #   generate → score pipeline, EvalRun (summary, slices, cost, save)
│   ├── gates.py                  #   quality gates: overall / slice / regression → ship or block
│   └── judge_alignment.py        #   Cohen's κ, false-pass / false-fail vs human labels
├── data/
│   ├── knowledge_base/*.md       # Aurora Bank's 8 policy documents (fictional)
│   └── golden/
│       ├── rag_golden.jsonl      # 18 tagged Q&A cases incl. multi-hop, unanswerable, adversarial
│       ├── agent_scenarios.jsonl # 6 agent tasks with expected tool calls, end state, bank state
│       └── judge_calibration.jsonl # 17 human-labelled replies for judge calibration
├── config/quality_gates.yaml     # release criteria (versioned, reviewed)
├── results/reference/            # committed reference baseline for regression gates
├── scripts/run_eval.py           # CLI used by CI (exit code = gate result)
├── tests/                        # offline harness tests + a live smoke test
├── .github/workflows/evals.yml   # tiered CI: harness tests → PR gates → nightly drift
└── docs/                         # concepts, metrics guide, enterprise playbook, troubleshooting
```

## 📚 Documentation

| doc | read it when… |
|---|---|
| [docs/01_concepts.md](docs/01_concepts.md) | you're new to LLM evaluation: datasets, metrics, judges, offline vs online |
| [docs/02_metrics_guide.md](docs/02_metrics_guide.md) | you need to know exactly how a metric works, its inputs and its gotchas |
| [docs/03_enterprise_playbook.md](docs/03_enterprise_playbook.md) | you're taking this to a real project: a practical checklist |
| [docs/04_troubleshooting.md](docs/04_troubleshooting.md) | something errors, or a score looks wrong |

## 💡 Lessons learned from building this

These are real findings from building and running this project, and they are a large part of what it teaches:

1. **Pin your eval stack.** `ragas 0.4.3` fails to import with `langchain-community ≥ 0.4.2` (a removed module).
2. **Test the harness.** Ragas' built-in LangChain converter drops LangChain 1.x tool calls, so a *perfect* agent
   scored **0**, with no error.
3. **Telemetry can throttle you.** Ragas' synchronous usage tracking serialised async judge calls; turning it off made
   scoring **~5× faster**.
4. **Judges need specs, not vibes.** The first compliance judge failed answers that merely *mentioned* an SSN
   requirement (κ ≈ 0.4–0.6 vs. human labels). A three-question checklist with explicit ALLOWED cases reached κ = 1.0.
5. **One judge, one job.** The compliance judge later started failing *factually wrong but compliant* answers, so
   the same config shipped on one run and was blocked on the next. Stating the judge's scope fixed the flaky gate.
6. **Sometimes the data is wrong, not the model.** A faithfulness score of 0 on a "correct" answer exposed an
   ambiguity in the policy document itself.
7. **Pick the metric for the scenario.** Goal accuracy can't grade refusals; a checklist `refusal_quality` judge could.
8. **Judges are noisy.** gpt-4o scored the same conversation 1, 1, 0, so hard metrics use a stronger judge
   plus a majority vote.
9. **Optimising one metric hides damage elsewhere.** The friendlier prompt raised tone and broke compliance.
   Its compliance *average* (0.83) looked tolerable; the adversarial *slice* (0.0) told the truth.
10. **Score the world, not the story.** Check the backend state the agent changed, not just what it says it did.

## 🧑‍🏫 Notes for instructors

* **Colab badges and the setup cell point to `github.com/baluragala/ragas-evals-enterprise`.** Push the repo there
  (or change `REPO_URL` in each notebook's setup cell and the badge URLs) before sharing.
* Notebooks are generated with saved outputs. To re-execute them all:
  `jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb` (~20 min, ~$0.30).
* Each notebook ends with **exercises**; `rag-013` (the $30 overdraft) is a deliberately unsolved failure for students.
* Everything here is fictional: Aurora Bank, its policies, customers and phone numbers.

## License
MIT. Use it freely for teaching.
