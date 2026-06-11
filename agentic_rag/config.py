"""
Agentic RAG — Configuration.

PURPOSE:
  Central config loader. Reads .env and provides typed parameters for
  the agent pipeline: model names, RAG store paths, search defaults.

AGENT USAGE:
  from config import RAG_SRC_DIR, DATA_DIR, EMBED_MODEL, MODEL_NAME

ENVIRONMENT VARIABLES:
  GOOGLE_API_KEY            — Gemini API key (simpler, no Vertex)
  GOOGLE_GENAI_USE_VERTEXAI — Set "1" to use Vertex AI instead
  GOOGLE_CLOUD_PROJECT      — GCP project (required for Vertex AI)
  RAG_PROJECT_DIR           — Path to the rag_pipeline project root
  AGENTIC_RAG_DATA_DIR      — Path to the RAG vector store
  AGENTIC_RAG_EMBED_MODEL   — Embedding model name
  AGENTIC_RAG_INDEX_TYPE    — Index type (hnsw / ivf / brute_force)
  AGENTIC_RAG_TOP_K         — Default top-K for RAG search
  AGENTIC_RAG_MODEL         — Gemini model name for sub-agents
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_RAG_DIR = Path.home() / "Documents" / "Miraclesoft" / "rag_pipeline"

for candidate in [_THIS_DIR / ".env", _THIS_DIR.parent / ".env"]:
    if candidate.exists():
        load_dotenv(candidate, override=False)

RAG_PROJECT_DIR = Path(os.getenv("RAG_PROJECT_DIR", str(_DEFAULT_RAG_DIR)))
RAG_SRC_DIR = RAG_PROJECT_DIR / "src"
RAG_SRC_STR = str(RAG_SRC_DIR.resolve())
if RAG_SRC_STR not in sys.path:
    sys.path.insert(0, RAG_SRC_STR)

_DEFAULT_STORE = str(RAG_PROJECT_DIR / "data" / "dintta_kb" / "store_e5")
DATA_DIR = os.getenv("AGENTIC_RAG_DATA_DIR", _DEFAULT_STORE)
EMBED_MODEL = os.getenv("AGENTIC_RAG_EMBED_MODEL", "e5-base")
INDEX_TYPE = os.getenv("AGENTIC_RAG_INDEX_TYPE", "hnsw")
TOP_K = int(os.getenv("AGENTIC_RAG_TOP_K", "5"))
MODEL_NAME = os.getenv("AGENTIC_RAG_MODEL", "gemini-2.5-flash")

# Google auth detection
_USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip() == "1"
_HAS_KEY = bool(os.getenv("GOOGLE_CLOUD_PROJECT", "").strip())

if _USE_VERTEX and not _HAS_KEY:
    print("WARNING: GOOGLE_GENAI_USE_VERTEXAI=1 but GOOGLE_CLOUD_PROJECT not set")
elif not _USE_VERTEX:
    _HAS_KEY = bool(os.getenv("GOOGLE_API_KEY", "").strip())
    if not _HAS_KEY:
        print("WARNING: No Google auth. Set GOOGLE_API_KEY or GOOGLE_GENAI_USE_VERTEXAI=1")


def to_dict() -> dict:
    return {
        "rag_project_dir": str(RAG_PROJECT_DIR),
        "data_dir": DATA_DIR,
        "embed_model": EMBED_MODEL,
        "index_type": INDEX_TYPE,
        "top_k": TOP_K,
        "model_name": MODEL_NAME,
        "use_vertex": _USE_VERTEX,
    }
