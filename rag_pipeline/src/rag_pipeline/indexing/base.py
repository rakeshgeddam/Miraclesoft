"""
ANN Index implementations.

Provides multiple index types:
- BruteForceIndex: Exact search (baseline)
- IVFIndex: Inverted File Index with k-means
- HNSWIndex: Hierarchical Navigable Small World graph
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import numpy as np
import struct
import mmap
import os
import logging
import heapq

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a search operation."""
    score: float
    index: int
    doc_id: Optional[int] = None
    
    def __lt__(self, other):
        return self.score > other.score  # For max-heap behavior


class BaseIndex(ABC):
    """Abstract base class for ANN indices."""
    
    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Build index from vectors."""
        pass
    
    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10, ef: Optional[int] = None) -> List[Tuple[float, int]]:
        """Search for top-k nearest neighbors."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        pass
    
    @abstractmethod
    def add(self, vectors: np.ndarray) -> int:
        """Add vectors to index (optional, for incremental updates)."""
        pass


class BruteForceIndex(BaseIndex):
    """Brute-force exact nearest neighbor search.
    
    Uses heap-based top-k selection for efficiency.
    Complexity: O(N * D) per query.
    """
    
    def __init__(self, normalized: bool = True):
        self.normalized = normalized
        self._vectors: Optional[np.ndarray] = None
        self._norms: Optional[np.ndarray] = None
    
    def build(self, vectors: np.ndarray) -> None:
        """Build index - just stores vectors."""
        self._vectors = vectors.astype(np.float32)
        if not self.normalized:
            self._norms = np.linalg.norm(self._vectors, axis=1).astype(np.float32)
        logger.info(f"Built BruteForceIndex: {len(self._vectors)} vectors, {self._vectors.shape[1]} dims")
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 10, 
        ef: Optional[int] = None,
        exclude: Optional[Set[int]] = None,
    ) -> List[Tuple[float, int]]:
        """Exact top-k search using dot product."""
        if self._vectors is None:
            raise RuntimeError("Index not built")
        
        query = query.astype(np.float32)
        
        if self.normalized:
            # Cosine = dot product for normalized vectors
            scores = self._vectors @ query
        else:
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
            valid = self._norms > 0
            scores = np.zeros(len(self._vectors))
            scores[valid] = (self._vectors[valid] @ query) / self._norms[valid]
        
        if exclude:
            scores[list(exclude)] = -np.inf
        
        n = len(self._vectors)
        if k >= n:
            top_indices = np.argsort(scores)[::-1]
        else:
            # Use argpartition for O(N) selection + O(k log k) sort
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        return [(float(scores[i]), int(i)) for i in top_indices[:k]]
    
    def save(self, path: str) -> None:
        """Save index (just the vectors)."""
        import json
        meta = {
            "type": "brute_force",
            "normalized": self.normalized,
            "num_vectors": len(self._vectors) if self._vectors is not None else 0,
            "dimension": self._vectors.shape[1] if self._vectors is not None else 0,
        }
        with open(path + ".meta.json", "w") as f:
            json.dump(meta, f)
        
        if self._vectors is not None:
            np.save(path + ".vectors.npy", self._vectors)
        
        logger.info(f"Saved BruteForceIndex to {path}")
    
    def load(self, path: str) -> None:
        """Load index."""
        import json
        with open(path + ".meta.json", "r") as f:
            meta = json.load(f)
        
        self.normalized = meta["normalized"]
        self._vectors = np.load(path + ".vectors.npy")
        
        if not self.normalized:
            self._norms = np.linalg.norm(self._vectors, axis=1).astype(np.float32)
        
        logger.info(f"Loaded BruteForceIndex from {path}: {meta['num_vectors']} vectors")
    
    def add(self, vectors: np.ndarray) -> int:
        """Add vectors (rebuilds index)."""
        if self._vectors is None:
            self.build(vectors)
        else:
            self._vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
            if not self.normalized:
                self._norms = np.linalg.norm(self._vectors, axis=1).astype(np.float32)
        return len(self._vectors) - len(vectors)