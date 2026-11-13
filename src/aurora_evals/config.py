"""Central configuration: paths, model names, and API-key loading.

Everything that might differ between a laptop, Google Colab and a CI runner lives here,
so the rest of the code never has to care where it is running.

Model names can be overridden with environment variables (or a `.env` file):

    AURORA_APP_MODEL     the LLM that powers the Aurora assistant   (default: gpt-4o-mini)
    AURORA_JUDGE_MODEL   the LLM that grades the assistant          (default: gpt-4o-mini)
    AURORA_STRONG_JUDGE_MODEL  a stronger judge for the hardest metric (default: gpt-4o)
    AURORA_EMBED_MODEL   embeddings for retrieval and some metrics  (default: text-embedding-3-small)
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"
GOLDEN_DIR = DATA_DIR / "golden"
CONFIG_DIR = PROJECT_ROOT / "config"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / ".cache"

# Load a local .env (if any) *before* reading model names, so overrides in it take effect.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
APP_MODEL = os.getenv("AURORA_APP_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("AURORA_JUDGE_MODEL", "gpt-4o-mini")
# Agent goal accuracy asks the judge to summarise a whole multi-step conversation. gpt-4o-mini
# sometimes hallucinates the end state there (it once "saw" a poem the agent had refused to write),
# so that one metric uses a stronger model by default. See notebook 04.
STRONG_JUDGE_MODEL = os.getenv("AURORA_STRONG_JUDGE_MODEL", "gpt-4o")
EMBED_MODEL = os.getenv("AURORA_EMBED_MODEL", "text-embedding-3-small")


def in_colab() -> bool:
    """True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def setup_environment(require_key: bool = True) -> bool:
    """Make sure OPENAI_API_KEY is available. Returns True if a key is set.

    Lookup order (first hit wins):
      1. an already-exported environment variable
      2. a `.env` file in the project root (local development)
      3. Colab "Secrets" (the key icon in Colab's left sidebar), secret name OPENAI_API_KEY
      4. an interactive, hidden prompt (getpass), so you can paste it once per session
    """
    # (2) the .env file was already loaded when this module was imported.
    if not os.getenv("OPENAI_API_KEY") and in_colab():
        try:
            from google.colab import userdata

            key = userdata.get("OPENAI_API_KEY")
            if key:
                os.environ["OPENAI_API_KEY"] = key
        except Exception:
            # SecretNotFoundError / NotebookAccessError: fall through to the prompt.
            pass

    if not os.getenv("OPENAI_API_KEY") and require_key:
        from getpass import getpass

        key = getpass("Paste your OpenAI API key (input is hidden): ").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key

    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if require_key and not has_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file (local), to Colab Secrets "
            "(Colab), or export it in your shell. See README → Setup."
        )
    return has_key
