"""
RAG Pipeline Search Tool.

PURPOSE:
  Connects to the existing RAG pipeline via subprocess call to the CLI.
  This avoids environment conflicts — the pipeline runs in its own
  .venv311 with all ML dependencies, while the agent runs in system Python.

AGENT USAGE:
  from tools.rag_tool import rag_search

  results = await rag_search("What is the three-plane architecture?")
  # Returns: {"status": "ok", "query": "...", "results": [...], "total": 5}

AGENT NOTES:
  - Calls `rag_cli.py search` as a subprocess using .venv311 Python.
  - Caches the pipeline configuration path for fast invocation.
  - Returns error dict on failure — LLM handles gracefully.
  - First call is slower (process startup), subsequent calls are faster.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

config_cache = {}

logger = logging.getLogger("rag_tool")


def _resolve_config():
    """Resolve RAG project paths once and cache them."""
    if "rag_bin" in config_cache:
        return config_cache

    # Find the RAG pipeline project
    rag_project = Path.home() / "Documents" / "Miraclesoft" / "rag_pipeline"
    venv_python = rag_project / ".venv311" / "bin" / "python3"
    cli_script = rag_project / "scripts" / "rag_cli.py"
    data_dir = rag_project / "data" / "dintta_kb" / "store_e5"

    # Fall back to just `python3` if venv doesn't exist
    if not venv_python.exists():
        venv_python = Path(sys.executable)
        logger.warning(".venv311 not found, using system python for RAG CLI")

    config_cache.update({
        "rag_project": str(rag_project),
        "venv_python": str(venv_python),
        "cli_script": str(cli_script),
        "data_dir": str(data_dir),
    })
    return config_cache


def _call_rag_cli(query: str, top_k: int) -> list[dict]:
    """Run rag_cli.py search as a subprocess and parse results."""
    cfg = _resolve_config()
    cmd = [
        cfg["venv_python"],
        cfg["cli_script"],
        "search",
        query,
        "--data-dir", cfg["data_dir"],
        "--embed-model", "e5-base",
        "--index-type", "hnsw",
        "--top-k", str(top_k),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=cfg["rag_project"],
        )
        if result.returncode != 0:
            logger.error("RAG CLI error (exit %d): %s", result.returncode, result.stderr[:500])
            return []
        return _parse_cli_output(result.stdout)
    except subprocess.TimeoutExpired:
        logger.error("RAG CLI timed out (120s)")
        return []
    except Exception as e:
        logger.error("RAG CLI exception: %s", e)
        return []


def _parse_cli_output(output: str) -> list[dict]:
    """Parse RAG CLI search output into structured results.

    Expected format:
      [1] score=0.8453 | doc=vision-architecture
          Dintta — AI ERP build document text...
    """
    import re
    results = []
    current = None
    for line in output.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match result header: [N] score=X.XXXX | doc=DOC_NAME
        header_match = re.match(r"^\[\d+\]\s+score=([\d.]+)\s*\|\s*doc=(.+)$", line)
        if header_match:
            if current:
                results.append(current)
            current = {
                "score": float(header_match.group(1)),
                "doc_id": header_match.group(2).strip(),
                "text": "",
            }
        elif current and not line.startswith("===") and not line.startswith("2026-"):
            # Accumulate text lines
            current["text"] = (current["text"] + " " + line).strip()

    if current:
        results.append(current)
    return results


async def rag_search(query: str, top_k: int = 0) -> dict:
    """Search the document store using the RAG pipeline CLI.

    Args:
        query: Natural language question or search query.
        top_k: Number of results to return (0 = use default 5).

    Returns:
        dict with keys: status, query, results (list), total (int), error (str|None)
    """
    try:
        k = top_k if top_k > 0 else 5
        results = _call_rag_cli(query, k)
        return {
            "status": "ok" if results else "empty",
            "query": query,
            "results": [
                {
                    "doc_id": r.get("doc_id", ""),
                    "score": float(r.get("score", 0)),
                    "text": r.get("text", "")[:500],
                }
                for r in results
            ],
            "total": len(results),
        }
    except Exception as e:
        logger.error("rag_search error: %s", e)
        return {
            "status": "error",
            "query": query,
            "results": [],
            "total": 0,
            "error": str(e),
        }
