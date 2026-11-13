"""aurora_evals: an enterprise-style evaluation harness for a LangChain 1.x assistant, using Ragas.

Modules
-------
config          paths, model names, API-key loading (local .env / Colab Secrets / prompt)
knowledge_base  load, chunk, and index Aurora Bank's policy documents
rag_app         the RAG assistant under test (configurable variants)
agent_app       the tool-using agent under test (LangChain `create_agent`)
messages        LangChain → Ragas message conversion for agent metrics
judges          the Ragas judge LLM + embeddings (with on-disk caching)
metrics         metric suites: RAG quality, business rules, agent behaviour
evaluation      the generate → score pipeline, EvalRun results object
gates           quality gates: thresholds → ship / block decision
judge_alignment measure how well an LLM judge agrees with human labels
"""

import os

# Ragas sends anonymous usage analytics by default. In ragas 0.4.3 that is a *synchronous* HTTP
# call made from inside the async event loop, which blocks every concurrent judge call. Measured
# on 8 parallel Faithfulness scores: 13.8 s with tracking, 2.75 s without (≈5× faster).
# Enterprises usually turn telemetry off anyway. Must be set before `ragas` is imported.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

__version__ = "1.0.0"
