"""
RAGPipeline — the main orchestrator tying all layers together.

Supports two modes:
  ingest(documents)    → chunk → embed → store + index
  answer(query)        → embed → retrieve → rerank → generate
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator

import numpy as np

from rag_pipeline.embeddings import EmbeddingManager
from rag_pipeline.chunking import DocumentChunker, Chunk
from rag_pipeline.storage import VectorStore, ChunkMetadata
from rag_pipeline.indexing import IndexManager, BaseIndex
from rag_pipeline.reranking import RerankerManager
from rag_pipeline.generation import GenerationManager, GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    data_dir: str = "./data"
    index_type: str = "hnsw"
    embedding_model: str = "miniLM-L6-v2"
    chunk_strategy: str = "recursive"
    top_k_retrieve: int = 50
    top_k_rerank: int = 10
    rerank: bool = True
    generation_strategy: str = "replug"
    batch_size: int = 32


@dataclass
class QueryResult:
    answer: str
    retrieved_passages: List[dict]
    reranked_passages: List[dict]
    timing: dict = field(default_factory=dict)


class RAGPipeline:
    """End-to-end RAG pipeline built entirely from scratch."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        global_config: Optional["Config"] = None,  # noqa: F821
    ):
        self.cfg = config or PipelineConfig()

        from rag_pipeline.config import Config
        self._gc = global_config or Config()

        # Sub-modules (lazy-init)
        self._embedding_mgr: Optional[EmbeddingManager] = None
        self._chunker: Optional[DocumentChunker] = None
        self._store: Optional[VectorStore] = None
        self._index_mgr: Optional[IndexManager] = None
        self._index: Optional[BaseIndex] = None
        self._reranker_mgr: Optional[RerankerManager] = None
        self._generation_mgr: Optional[GenerationManager] = None

        self._initialized = False
        self._chunk_counter = 0

    # ---- Initialization ----

    def initialize(self) -> None:
        """Create or open the vector store; prepare sub-modules."""
        if self._initialized:
            return

        data_dir = Path(self.cfg.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Get dimension from config
        models_config = self._gc.get("embeddings", "models", default=[])
        dim = 384
        for m in models_config:
            if m["name"] == self.cfg.embedding_model:
                dim = m.get("dimension", 384)
                break

        store_path = data_dir / "index.bin"
        meta_path = data_dir / "chunks.jsonl"

        # Vector store (always try to open existing)
        self._store = VectorStore(
            index_path=str(store_path),
            metadata_path=str(meta_path),
            dimension=dim,
            mmap_enabled=True,
            normalize=True,
            readonly=False,
        )

        if store_path.exists() and store_path.stat().st_size > 64:
            try:
                self._store.open()
                logger.info(f"Opened existing store: {self._store.num_vectors} vectors")
            except Exception:
                logger.warning("Failed to open store, creating fresh")
                self._store.create()
        else:
            self._store.create()

        # Embedding manager
        self._embedding_mgr = EmbeddingManager(self._gc)

        # Chunker
        self._chunker = DocumentChunker(self._gc)

        # Index
        self._index_mgr = IndexManager(self._gc)
        self._index = self._index_mgr.create_index(
            index_type=self.cfg.index_type,
            dimension=dim,
        )

        # Rebuild index from existing store vectors (if any)
        if self._store.num_vectors > 0:
            vectors = self._store._vectors
            if vectors is not None and len(vectors) > 0:
                if self.cfg.index_type == "brute_force":
                    self._index.build(vectors)
                else:
                    # Sample for ANN index build if too large
                    sample = vectors
                    if len(vectors) > 500000:
                        idx = np.random.choice(len(vectors), 500000, replace=False)
                        sample = vectors[idx]
                    self._index.build(sample)
                    logger.info(f"Index built from {len(sample)} seed vectors")

        # Reranker
        if self.cfg.rerank:
            self._reranker_mgr = RerankerManager(self._gc)

        # Generator
        self._generation_mgr = GenerationManager(self._gc)

        self._initialized = True

    # ---- Ingestion ----

    def ingest(
        self,
        documents: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> Dict[str, int]:
        """Ingest documents into the vector store.

        Each document format: {"id": str, "text": str, "metadata": dict}
        """
        self.initialize()
        batch_size = batch_size or self.cfg.batch_size
        t0 = time.time()

        # 1. Chunk all documents
        t1 = time.time()
        all_chunks: List[Chunk] = self._chunker.chunk_documents(
            documents, strategy=self.cfg.chunk_strategy
        )
        chunk_time = time.time() - t1

        if not all_chunks:
            logger.warning("No chunks produced from documents")
            return {"chunks": 0, "vectors_added": 0, "time": time.time() - t0}

        chunk_texts = [c.text for c in all_chunks]
        chunk_ids_list = [c.chunk_id for c in all_chunks]
        doc_ids_list = [c.doc_id for c in all_chunks]

        logger.info(
            f"Chunked {len(documents)} docs → {len(all_chunks)} chunks "
            f"(avg {sum(len(t) for t in chunk_texts)//max(len(chunk_texts),1)} chars)"
        )

        # 2. Embed in batches
        t2 = time.time()
        all_vectors = []
        for start in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[start:start + batch_size]
            vecs = self._embedding_mgr.embed(batch, model=self.cfg.embedding_model)
            all_vectors.append(vecs)
        vectors = np.vstack(all_vectors).astype(np.float32)
        embed_time = time.time() - t2
        logger.info(f"Embedded {len(vectors)} chunks ({embed_time:.2f}s)")

        # 3. Add to vector store
        t3 = time.time()
        store_doc_ids = np.arange(
            self._store.num_vectors,
            self._store.num_vectors + len(vectors),
            dtype=np.uint64,
        )
        store_chunk_offsets = np.array(chunk_ids_list, dtype=np.uint64)

        # Build ChunkMetadata list for JSONL
        meta_list = []
        for i, c in enumerate(all_chunks):
            meta_list.append(ChunkMetadata(
                chunk_id=self._chunk_counter + c.chunk_id,
                doc_id=c.doc_id,
                text=c.text,
                token_start=c.token_start,
                token_end=c.token_end,
                source_file=c.metadata.get("source_file", ""),
                extra={
                    k: v for k, v in c.metadata.items() if k != "source_file"
                },
            ))

        self._store.add_vectors(vectors, store_doc_ids, store_chunk_offsets, meta_list)
        store_time = time.time() - t3

        self._chunk_counter += len(all_chunks)

        # 4. Update ANN index
        t4 = time.time()
        if self._store.num_vectors < 10000 or self.cfg.index_type == "brute_force":
            self._index.build(self._store._vectors)
        else:
            # Incremental HNSW doesn't require rebuild
            pass
        index_time = time.time() - t4

        total_time = time.time() - t0
        logger.info(
            f"Ingest complete: {len(all_chunks)} chunks -> {len(vectors)} vectors "
            f"({total_time:.2f}s total: chunk={chunk_time:.2f}s, "
            f"embed={embed_time:.2f}s, store={store_time:.2f}s, index={index_time:.2f}s)"
        )

        return {
            "chunks": len(all_chunks),
            "vectors_added": len(vectors),
            "total_vectors": self._store.num_vectors,
            "time": total_time,
            "chunk_time": chunk_time,
            "embed_time": embed_time,
            "store_time": store_time,
            "index_time": index_time,
        }

    # ---- Query ----

    def answer(
        self,
        query: str,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
        rerank: Optional[bool] = None,
    ) -> QueryResult:
        """Answer a query through the full RAG pipeline."""
        self.initialize()

        if self._store is None or self._store.num_vectors == 0:
            return QueryResult(
                answer="No documents ingested yet. Call ingest() first.",
                retrieved_passages=[],
                reranked_passages=[],
                timing={},
            )

        timings = {}
        t_start = time.time()

        k_retrieve = top_k_retrieve or self.cfg.top_k_retrieve
        k_rerank = top_k_rerank or self.cfg.top_k_rerank
        do_rerank = rerank if rerank is not None else self.cfg.rerank

        # 1. Embed query
        t1 = time.time()
        q_vec = self._embedding_mgr.embed_query(query, model=self.cfg.embedding_model)
        timings["embed"] = time.time() - t1

        # 2. Retrieve from index
        t2 = time.time()
        if self._index is not None and hasattr(self._index, 'search'):
            results = self._index.search(q_vec, k=k_retrieve)
        else:
            results = self._store.search_brute_force(q_vec, k=k_retrieve)
        timings["retrieve"] = time.time() - t2

        if not results:
            return QueryResult(
                answer="No relevant passages found.",
                retrieved_passages=[],
                reranked_passages=[],
                timing=timings,
            )

        retrieved_indices = [r[1] for r in results]
        retrieved_scores = [r[0] for r in results]

        # Load passage text
        meta_list = self._store.load_metadata(retrieved_indices)
        retrieved_passages = [
            {
                "score": retrieved_scores[i],
                "chunk_id": retrieved_indices[i],
                "doc_id": m.doc_id,
                "text": m.text,
                "source": m.source_file,
            }
            for i, m in enumerate(meta_list)
        ]

        # 3. Rerank (optional)
        if do_rerank and self._reranker_mgr is not None:
            t3 = time.time()
            passages_text = [rp["text"] for rp in retrieved_passages]
            reranked = self._reranker_mgr.rerank(query, passages_text, k=k_rerank)
            timings["rerank"] = time.time() - t3

            reranked_passages = [
                {
                    "score": r[0],
                    "rank": i,
                    "chunk_id": retrieved_passages[r[1]]["chunk_id"],
                    "doc_id": retrieved_passages[r[1]]["doc_id"],
                    "text": r[2],
                }
                for i, r in enumerate(reranked)
            ]
            final_passages = [r["text"] for r in reranked_passages]
        else:
            reranked_passages = retrieved_passages[:k_rerank]
            final_passages = [r["text"] for r in reranked_passages]
            timings["rerank"] = 0.0

        # 4. Generate
        t4 = time.time()
        prompt = self._generation_mgr.build_replug_prompt(query, final_passages)
        gen_result = self._generation_mgr.generate(
            prompt, strategy=self.cfg.generation_strategy
        )
        timings["generate"] = time.time() - t4

        timings["total"] = time.time() - t_start

        return QueryResult(
            answer=gen_result.text,
            retrieved_passages=retrieved_passages,
            reranked_passages=reranked_passages,
            timing=timings,
        )

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> List[dict]:
        """Search only (no generation). Returns passages."""
        self.initialize()
        q_vec = self._embedding_mgr.embed_query(query, model=self.cfg.embedding_model)

        if self._index is not None:
            results = self._index.search(q_vec, k=k)
        else:
            results = self._store.search_brute_force(q_vec, k=k)

        meta_list = self._store.load_metadata([r[1] for r in results])
        return [
            {
                "score": r[0],
                "chunk_id": r[1],
                "doc_id": m.doc_id,
                "text": m.text,
            }
            for r, m in zip(results, meta_list)
        ]

    @property
    def num_vectors(self) -> int:
        if self._store is None:
            return 0
        return self._store.num_vectors


__all__ = [
    "RAGPipeline",
    "PipelineConfig",
    "QueryResult",
]
