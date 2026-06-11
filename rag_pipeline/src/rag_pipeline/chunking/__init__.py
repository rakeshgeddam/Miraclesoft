"""
Document chunking strategies.

Supports multiple chunking approaches:
- Recursive character splitting (default)
- Fixed-size splitting
- Semantic chunking (based on embeddings similarity)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Iterator
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A document chunk with metadata."""
    text: str
    token_start: int
    token_end: int
    chunk_id: int
    doc_id: str
    metadata: dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.text)


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, doc_id: str, **kwargs) -> List[Chunk]:
        """Split text into chunks."""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token for English)."""
        return len(text) // 4


class RecursiveChunker(BaseChunker):
    """Recursive character-based chunking with configurable separators.
    
    Similar to LangChain's RecursiveCharacterTextSplitter but from scratch.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        length_function: Optional[callable] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.length_function = length_function or len
    
    def chunk(self, text: str, doc_id: str, **kwargs) -> List[Chunk]:
        """Split text recursively using separators."""
        if not text:
            return []
        
        chunks = self._split_text(text)
        return self._create_chunks(chunks, doc_id)
    
    def _split_text(self, text: str) -> List[str]:
        """Recursively split text by separators."""
        return self._recursive_split(text, self.separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Character-level split
            return self._split_by_char(text)
        
        splits = text.split(separator)
        result = []
        
        for split in splits:
            if self.length_function(split) <= self.chunk_size:
                result.append(split)
            else:
                # Recursively split with remaining separators
                sub_splits = self._recursive_split(split, remaining_separators)
                result.extend(sub_splits)
        
        return self._merge_small_chunks(result)
    
    def _split_by_char(self, text: str) -> List[str]:
        """Split text into character-level chunks of chunk_size."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge adjacent small chunks."""
        if not chunks:
            return []
        
        merged = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            if self.length_function(current) + self.length_function(chunk) <= self.chunk_size:
                current += chunk
            else:
                merged.append(current)
                current = chunk
        
        merged.append(current)
        return merged
    
    def _create_chunks(self, texts: List[str], doc_id: str) -> List[Chunk]:
        """Create Chunk objects with token positions."""
        chunks = []
        token_pos = 0
        
        for i, text in enumerate(texts):
            token_count = self.estimate_tokens(text)
            chunk = Chunk(
                text=text,
                token_start=token_pos,
                token_end=token_pos + token_count,
                chunk_id=i,
                doc_id=doc_id,
            )
            chunks.append(chunk)
            token_pos += token_count
        
        return chunks


class FixedSizeChunker(BaseChunker):
    """Simple fixed-size chunking with overlap."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        length_function: Optional[callable] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len
    
    def chunk(self, text: str, doc_id: str, **kwargs) -> List[Chunk]:
        if not text:
            return []
        
        chunks_text = []
        step = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(text), step):
            chunk_text = text[i:i + self.chunk_size]
            if chunk_text:
                chunks_text.append(chunk_text)
        
        return self._create_chunks(chunks_text, doc_id)
    
    def _create_chunks(self, texts: List[str], doc_id: str) -> List[Chunk]:
        chunks = []
        token_pos = 0
        
        for i, text in enumerate(texts):
            token_count = self.estimate_tokens(text)
            chunk = Chunk(
                text=text,
                token_start=token_pos,
                token_end=token_pos + token_count,
                chunk_id=i,
                doc_id=doc_id,
            )
            chunks.append(chunk)
            token_pos += token_count
        
        return chunks


class SemanticChunker(BaseChunker):
    """Semantic chunking based on embedding similarity.
    
    Splits text where semantic similarity between adjacent segments drops
    below a threshold.
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.5,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1000,
        device: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.device = device
        self._embedder = None
    
    def _get_embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
        return self._embedder
    
    def chunk(self, text: str, doc_id: str, **kwargs) -> List[Chunk]:
        if not text:
            return []
        
        # First split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._create_chunks([text], doc_id)
        
        # Get embeddings for each sentence
        embedder = self._get_embedder()
        embeddings = embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
        
        # Find split points based on similarity
        split_points = [0]
        for i in range(1, len(sentences)):
            # Cosine similarity (dot product since normalized)
            sim = float(embeddings[i] @ embeddings[i-1])
            if sim < self.threshold:
                split_points.append(i)
        
        split_points.append(len(sentences))
        
        # Build chunks from split points
        chunks_text = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)
            
            # Enforce min/max chunk sizes
            if len(chunk_text) > self.max_chunk_size:
                # Sub-split large chunks
                sub_chunks = self._split_by_size(chunk_text)
                chunks_text.extend(sub_chunks)
            elif len(chunk_text) < self.min_chunk_size and chunks_text:
                # Merge small chunks with previous
                chunks_text[-1] += " " + chunk_text
            else:
                chunks_text.append(chunk_text)
        
        return self._create_chunks(chunks_text, doc_id)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with spaCy/nltk
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_by_size(self, text: str) -> List[str]:
        """Split text by character size."""
        chunks = []
        for i in range(0, len(text), self.max_chunk_size - self.min_chunk_size):
            chunk = text[i:i + self.max_chunk_size]
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def _create_chunks(self, texts: List[str], doc_id: str) -> List[Chunk]:
        chunks = []
        token_pos = 0
        
        for i, text in enumerate(texts):
            token_count = self.estimate_tokens(text)
            chunk = Chunk(
                text=text,
                token_start=token_pos,
                token_end=token_pos + token_count,
                chunk_id=i,
                doc_id=doc_id,
            )
            chunks.append(chunk)
            token_pos += token_count
        
        return chunks


class ChunkingStrategy:
    """Enum-like class for chunking strategies."""
    RECURSIVE = "recursive"
    FIXED = "fixed"
    SEMANTIC = "semantic"


class DocumentChunker:
    """Main interface for document chunking."""
    
    def __init__(self, config=None):
        from rag_pipeline.config import config as global_config
        self.config = config or global_config
        self._chunkers = {}
    
    def get_chunker(self, strategy: Optional[str] = None) -> BaseChunker:
        """Get chunker for strategy."""
        if strategy is None:
            strategy = self.config.get("chunking", "default_strategy", default="recursive")
        
        if strategy not in self._chunkers:
            self._chunkers[strategy] = self._create_chunker(strategy)
        
        return self._chunkers[strategy]
    
    def _create_chunker(self, strategy: str) -> BaseChunker:
        """Create chunker from config."""
        strategies = self.config.get("chunking", "strategies", default={})
        config = strategies.get(strategy, {})
        
        if strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(
                chunk_size=config.get("chunk_size", 512),
                chunk_overlap=config.get("chunk_overlap", 50),
                separators=config.get("separators", ["\n\n", "\n", ". ", " ", ""]),
            )
        elif strategy == ChunkingStrategy.FIXED:
            return FixedSizeChunker(
                chunk_size=config.get("chunk_size", 512),
                chunk_overlap=config.get("chunk_overlap", 50),
            )
        elif strategy == ChunkingStrategy.SEMANTIC:
            return SemanticChunker(
                embedding_model=config.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
                threshold=config.get("threshold", 0.5),
                min_chunk_size=config.get("min_chunk_size", 100),
                max_chunk_size=config.get("max_chunk_size", 1000),
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def chunk(self, text: str, doc_id: str, strategy: Optional[str] = None) -> List[Chunk]:
        """Chunk text using specified strategy."""
        chunker = self.get_chunker(strategy)
        return chunker.chunk(text, doc_id)
    
    def chunk_documents(
        self,
        documents: List[dict],  # [{"id": "doc1", "text": "...", "metadata": {...}}]
        strategy: Optional[str] = None,
    ) -> List[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            doc_id = doc.get("id", f"doc_{len(all_chunks)}")
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            chunks = self.chunk(text, doc_id, strategy)
            for chunk in chunks:
                chunk.metadata.update(metadata)
            all_chunks.extend(chunks)
        return all_chunks