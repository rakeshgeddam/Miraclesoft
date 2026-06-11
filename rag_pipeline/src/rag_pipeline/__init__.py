"""
RAG Pipeline - A from-scratch Retrieval-Augmented Generation system.

No external frameworks (LlamaIndex, LangChain) - only embedding model APIs.
Supports multiple embedding models, local vector storage, ANN indexing, and re-ranking.
"""

__version__ = "0.1.0"
__author__ = "RAG Pipeline Team"

from rag_pipeline.embeddings import EmbeddingManager, EmbeddingModel
from rag_pipeline.chunking import DocumentChunker, ChunkingStrategy
from rag_pipeline.storage import VectorStore, ChunkMetadata
from rag_pipeline.indexing import IndexManager, BruteForceIndex, IVFIndex, HNSWIndex
from rag_pipeline.reranking import RerankerManager, CrossEncoderReranker
from rag_pipeline.generation import GenerationManager, GenerationStrategy
from rag_pipeline.pipeline import RAGPipeline, PipelineConfig

__all__ = [
    "EmbeddingManager",
    "EmbeddingModel",
    "DocumentChunker",
    "ChunkingStrategy",
    "VectorStore",
    "ChunkMetadata",
    "IndexManager",
    "BruteForceIndex",
    "IVFIndex",
    "HNSWIndex",
    "RerankerManager",
    "CrossEncoderReranker",
    "GenerationManager",
    "GenerationStrategy",
    "RAGPipeline",
    "PipelineConfig",
]