"""The "judge" models Ragas uses to grade the assistant.

Ragas 0.4's metric collections need a *modern* Ragas LLM, built by `llm_factory` around an
official provider client (here `AsyncOpenAI`). This is separate from the LangChain ChatOpenAI
that powers the app, and that separation is good practice anyway:

* the judge can be a different (often stronger) model than the app;
* judge settings (temperature, caching) never leak into production code.

Caching: with `cache=True` every judge call is stored on disk under `.cache/`. Re-running an
evaluation on unchanged inputs is then free and instant, which matters a lot in CI and in class.
"""

from __future__ import annotations

from functools import lru_cache

from openai import AsyncOpenAI
from ragas.cache import DiskCacheBackend
from ragas.embeddings import embedding_factory
from ragas.llms import llm_factory

from .config import CACHE_DIR, EMBED_MODEL, JUDGE_MODEL


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    # Generous retries/timeouts: evaluation fans out many parallel calls and will meet rate limits.
    return AsyncOpenAI(max_retries=6, timeout=90)


@lru_cache(maxsize=4)
def get_judge_llm(model: str = JUDGE_MODEL, cache: bool = True):
    """An Instructor-backed Ragas LLM that returns structured (Pydantic) verdicts."""
    backend = DiskCacheBackend(cache_dir=str(CACHE_DIR / "judge")) if cache else None
    # max_tokens: ragas' default (1024) truncates the judge's structured output on long answers
    # (IncompleteOutputException). 4096 leaves room for statement-by-statement verdicts.
    return llm_factory(model, client=_client(), cache=backend, temperature=0, max_tokens=4096)


@lru_cache(maxsize=2)
def get_judge_embeddings(model: str = EMBED_MODEL, cache: bool = True):
    """Embeddings used by similarity-based metrics such as AnswerRelevancy."""
    backend = DiskCacheBackend(cache_dir=str(CACHE_DIR / "embeddings")) if cache else None
    return embedding_factory("openai", model=model, client=_client(), cache=backend)
