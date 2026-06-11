"""
Index manager — factory + unified interface over BruteForce, IVF, HNSW.
"""

import logging
from typing import Optional

from .base import BaseIndex, BruteForceIndex
from .ivf import IVFIndex
from .hnsw import HNSWIndex

logger = logging.getLogger(__name__)

INDEX_REGISTRY = {
    "brute_force": BruteForceIndex,
    "ivf": IVFIndex,
    "hnsw": HNSWIndex,
}


class IndexManager:
    """Factory and container for ANN indices."""

    def __init__(self, config=None):
        from rag_pipeline.config import config as global_config
        self.config = config or global_config
        self._index: Optional[BaseIndex] = None
        self._index_type: str = ""

    def create_index(
        self,
        index_type: Optional[str] = None,
        dimension: int = 768,
        **overrides,
    ) -> BaseIndex:
        """Build an index from config + optional overrides."""
        if index_type is None:
            index_type = self.config.get("indexing", "default_index", default="hnsw")

        index_cfg = self.config.get("indexing", "indices", index_type, default={})
        normalized = self.config.get("storage", "normalize_vectors", default=True)

        params = dict(normalized=normalized)

        if index_type == "ivf":
            params["n_centroids"] = index_cfg.get("n_centroids", 4096)
            params["n_probe"] = index_cfg.get("n_probe", 16)
            params["kmeans_iterations"] = index_cfg.get("kmeans_iterations", 20)
        elif index_type == "hnsw":
            params["M"] = index_cfg.get("M", 16)
            params["ef_construction"] = index_cfg.get("ef_construction", 200)
            params["ef_search"] = index_cfg.get("ef_search", 100)
        elif index_type == "brute_force":
            pass

        # Apply runtime overrides
        params.update(overrides)

        cls = INDEX_REGISTRY.get(index_type)
        if cls is None:
            raise ValueError(f"Unknown index type '{index_type}'. Available: {list(INDEX_REGISTRY)}")

        self._index = cls(**params)
        self._index_type = index_type
        logger.info(f"Created {index_type} index (dim={dimension})")
        return self._index

    @property
    def index(self) -> Optional[BaseIndex]:
        return self._index

    @property
    def index_type(self) -> str:
        return self._index_type

    def search(self, query_emb, k: int = 10, **kw):
        if self._index is None:
            raise RuntimeError("No index built yet. Call create_index() + build() first.")
        return self._index.search(query_emb, k=k, **kw)

    def build(self, vectors):
        if self._index is None:
            raise RuntimeError("No index built yet. Call create_index() first.")
        return self._index.build(vectors)


__all__ = [
    "BaseIndex",
    "BruteForceIndex",
    "IVFIndex",
    "HNSWIndex",
    "IndexManager",
]
