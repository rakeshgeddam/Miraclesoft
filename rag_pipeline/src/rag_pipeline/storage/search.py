"""
Search methods for VectorStore.
"""

import numpy as np
import logging
from typing import Optional, List, Tuple, Set
from pathlib import Path

from .metadata import ChunkMetadata

logger = logging.getLogger(__name__)


def search_brute_force(
    vectors: np.ndarray,
    query: np.ndarray,
    k: int = 10,
    exclude_indices: Optional[Set[int]] = None,
    normalized: bool = True,
) -> List[Tuple[float, int]]:
    """Brute-force top-k search using dot product."""
    if normalized:
        # Cosine similarity = dot product for normalized vectors
        scores = vectors @ query
    else:
        # Need to normalize query and compute cosine
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        doc_norms = np.linalg.norm(vectors, axis=1)
        valid = doc_norms > 0
        scores = np.zeros(vectors.shape[0])
        scores[valid] = (vectors[valid] @ query) / doc_norms[valid]
    
    # Exclude indices if provided
    if exclude_indices:
        scores[list(exclude_indices)] = -np.inf
    
    # Get top-k using argpartition (faster than full sort)
    n = vectors.shape[0]
    if k >= n:
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    
    return [(float(scores[i]), int(i)) for i in top_indices[:k]]


def load_metadata(metadata_path: Path, chunk_ids: List[int], cache: dict) -> List[ChunkMetadata]:
    """Load metadata for given chunk IDs."""
    from .metadata import ChunkMetadata
    results = []
    for cid in chunk_ids:
        if cid in cache:
            results.append(cache[cid])
        else:
            with open(metadata_path, "r") as f:
                for i, line in enumerate(f):
                    if i == cid:
                        meta = ChunkMetadata.from_json(line.strip())
                        cache[cid] = meta
                        results.append(meta)
                        break
    return results
