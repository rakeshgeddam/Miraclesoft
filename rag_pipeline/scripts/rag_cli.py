#!/usr/bin/env python3
"""
RAG Pipeline CLI — ingest documents and answer queries from the terminal.

Reads defaults from ~/.rag_pipeline/config.json so common flags
(data-dir, embed-model, index-type, top-k) can be set once.
CLI flags still override config values when provided.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_pipeline import RAGPipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag.cli")

CONFIG_PATH = Path.home() / ".rag_pipeline" / "config.json"


def load_defaults() -> dict:
    """Load persistent defaults from config file."""
    cfg_path = Path(os.environ.get("RAG_CONFIG", CONFIG_PATH))
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text())
        except Exception as e:
            logger.warning("Failed to load config %s: %s", cfg_path, e)
    return {}


def save_defaults(defaults: dict) -> None:
    """Write persistent defaults."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(defaults, indent=2))
    print(f"Saved defaults to {CONFIG_PATH}")


def merge_args(args: argparse.Namespace, defaults: dict) -> dict:
    """Overlay CLI args on top of defaults. CLI values win when non-default."""
    merged = dict(defaults)
    for key, val in vars(args).items():
        # Skip argparse-injected defaults (None/False) — they'd erase config values
        if val is None or val is False:
            continue
        if isinstance(val, bool) and not val:
            continue
        merged[key] = val
    return merged


def main():
    defaults = load_defaults()

    parser = argparse.ArgumentParser(prog="rag", description="RAG Pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # defaults subcommand
    def_p = sub.add_parser("defaults", help="Show or set persistent defaults")
    def_p.add_argument("--data-dir", default=None)
    def_p.add_argument("--embed-model", default=None)
    def_p.add_argument("--index-type", default=None)
    def_p.add_argument("--top-k", type=int, default=None)
    def_p.add_argument("--chunk-strategy", default=None)
    def_p.add_argument("--batch-size", type=int, default=None)

    # ingest
    ingest_p = sub.add_parser("ingest", help="Ingest documents")
    ingest_p.add_argument("file", nargs="+", help="JSON / JSONL / .txt file(s)")
    ingest_p.add_argument("--data-dir", default=None)
    ingest_p.add_argument("--embed-model", default=None)
    ingest_p.add_argument("--chunk-strategy", default=None)
    ingest_p.add_argument("--index-type", default=None)
    ingest_p.add_argument("--batch-size", type=int, default=None)

    # query
    query_p = sub.add_parser("query", help="Answer a question")
    query_p.add_argument("query_text", nargs="+", help="Question text")
    query_p.add_argument("--data-dir", default=None)
    query_p.add_argument("--embed-model", default=None)
    query_p.add_argument("--index-type", default=None)
    query_p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    query_p.add_argument("--top-k", type=int, default=None)
    query_p.add_argument("--no-gen", action="store_true", help="Search only, skip generation")

    # search (shortcut for query --no-gen)
    search_p = sub.add_parser("search", help="Search (retrieve only)")
    search_p.add_argument("query_text", nargs="+")
    search_p.add_argument("--data-dir", default=None)
    search_p.add_argument("--embed-model", default=None)
    search_p.add_argument("--index-type", default=None)
    search_p.add_argument("--top-k", type=int, default=None)

    # info
    info_p = sub.add_parser("info", help="Show pipeline status")
    info_p.add_argument("--data-dir", default=None)

    args = parser.parse_args()

    if args.command == "defaults":
        _do_defaults(args, defaults)
        return

    # Merge CLI args with config file defaults
    merged = merge_args(args, defaults)

    if args.command == "ingest":
        _do_ingest(args, merged)
    elif args.command == "query":
        _do_query(args, merged)
    elif args.command == "search":
        _do_search(args, merged)
    elif args.command == "info":
        _do_info(args, merged)


def _do_defaults(args, current_defaults):
    if not any([args.data_dir, args.embed_model, args.index_type,
                args.top_k is not None, args.chunk_strategy, args.batch_size]):
        # Show current defaults
        if current_defaults:
            print("Current defaults:")
            for k, v in current_defaults.items():
                print(f"  {k}: {v}")
        else:
            print("No defaults set yet. Use e.g.:")
            print("  rag defaults --data-dir data/dintta_kb/store_e5 "
                  "--embed-model e5-base --index-type hnsw --top-k 5")
        return

    new = dict(current_defaults)
    updated = []
    for key, alias in [("data_dir", "data-dir"), ("embed_model", "embed-model"),
                       ("index_type", "index-type"), ("top_k", "top-k"),
                       ("chunk_strategy", "chunk-strategy"), ("batch_size", "batch-size")]:
        val = getattr(args, alias.replace("-", "_"), None)
        if val is not None:
            new[key] = val
            updated.append(f"{key}={val}")
    save_defaults(new)
    print("Updated:", ", ".join(updated))


def _load_documents(file_paths):
    docs = []
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            logger.warning("File not found: %s", fp)
            continue

        if path.suffix in (".json", ".jsonl"):
            with open(path) as f:
                if path.suffix == ".jsonl":
                    for line in f:
                        line = line.strip()
                        if line:
                            docs.append(json.loads(line))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        docs.extend(data)
                    else:
                        docs.append(data)
        elif path.suffix == ".txt":
            text = path.read_text()
            docs.append({"id": path.stem, "text": text, "metadata": {"source_file": str(path)}})
        else:
            logger.warning("Unsupported file type: %s", path.suffix)

    return docs


def _do_ingest(args, merged):
    cfg = PipelineConfig(
        data_dir=merged.get("data_dir", args.data_dir or "./data"),
        embedding_model=merged.get("embed_model", args.embed_model or "miniLM-L6-v2"),
        chunk_strategy=merged.get("chunk_strategy", args.chunk_strategy or "recursive"),
        index_type=merged.get("index_type", args.index_type or "hnsw"),
        batch_size=merged.get("batch_size", args.batch_size or 32),
    )
    pipeline = RAGPipeline(cfg)
    pipeline.initialize()

    docs = _load_documents(args.file)
    if not docs:
        print("No documents loaded")
        return

    for d in docs:
        if "id" not in d:
            d["id"] = d.get("title", d.get("name", f"doc_{hash(d['text'][:50]) % 1000000}"))
        if "metadata" not in d:
            d["metadata"] = {}
        if "text" not in d:
            d["text"] = json.dumps(d)

    result = pipeline.ingest(docs)
    print(f"Ingested {result['chunks']} chunks → {result['vectors_added']} vectors "
          f"({result['time']:.2f}s). Total store: {result['total_vectors']} vectors.")


def _do_query(args, merged):
    cfg = PipelineConfig(
        data_dir=merged.get("data_dir") or args.data_dir or "./data",
        embedding_model=merged.get("embed_model") or args.embed_model or "miniLM-L6-v2",
        index_type=merged.get("index_type") or args.index_type or "hnsw",
        rerank=args.rerank if args.rerank is not None else merged.get("rerank", True),
    )
    pipeline = RAGPipeline(cfg)
    pipeline.initialize()

    query = " ".join(args.query_text)
    top_k = merged.get("top_k", 10)

    if args.no_gen:
        results = pipeline.search(query, k=top_k)
        print(f"\n=== Top {len(results)} passages for: {query} ===\n")
        for i, r in enumerate(results):
            print(f"[{i+1}] score={r['score']:.4f} | doc={r['doc_id']} | chunk={r['chunk_id']}")
            print(f"    {r['text'][:300]}...")
            print()
    else:
        result = pipeline.answer(query, top_k_rerank=top_k)
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}\n")
        print(f"Answer: {result.answer}")
        print(f"\n--- Timing ---")
        for phase, t in result.timing.items():
            print(f"  {phase}: {t:.3f}s")
        print(f"\n--- Retrieved ({len(result.retrieved_passages)}) → Reranked ({len(result.reranked_passages)}) ---")
        for i, rp in enumerate(result.reranked_passages[:5]):
            print(f"[{i+1}] score={rp['score']:.4f} | doc={rp['doc_id']}")
            print(f"    {rp['text'][:200]}...")
            print()


def _do_search(args, merged):
    cfg = PipelineConfig(
        data_dir=merged.get("data_dir") or args.data_dir or "./data",
        embedding_model=merged.get("embed_model") or args.embed_model or "miniLM-L6-v2",
        index_type=merged.get("index_type") or args.index_type or "hnsw",
        rerank=False,
    )
    pipeline = RAGPipeline(cfg)
    pipeline.initialize()

    query = " ".join(args.query_text)
    top_k = merged.get("top_k", 10)
    results = pipeline.search(query, k=top_k)

    print(f"\n=== Top {len(results)} passages for: {query} ===\n")
    for i, r in enumerate(results):
        print(f"[{i+1}] score={r['score']:.4f} | doc={r['doc_id']}")
        print(f"    {r['text'][:300]}...")
        print()


def _do_info(args, merged):
    cfg = PipelineConfig(data_dir=merged.get("data_dir") or args.data_dir or "./data")
    pipeline = RAGPipeline(cfg)
    pipeline.initialize()
    print(f"Vector store: {pipeline.num_vectors} vectors")
    print(f"Data dir:     {cfg.data_dir}")
    print(f"Index type:   {cfg.index_type}")
    print(f"Embed model:  {cfg.embedding_model}")
    print(f"Chunk strat:  {cfg.chunk_strategy}")
    print(f"Rerank:       {cfg.rerank}")


if __name__ == "__main__":
    main()
