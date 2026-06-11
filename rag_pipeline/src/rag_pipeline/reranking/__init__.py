"""
Cross-encoder re-ranker built from scratch on top of sentence-transformers.

No pipeline framework dependency — direct model inference only.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RerankerModel:
    """Configuration for a reranker model."""
    name: str
    provider: str
    model_id: str
    max_seq_length: int = 512
    batch_size: int = 32
    device: Optional[str] = None
    extra_config: dict = None

    def __post_init__(self):
        if self.extra_config is None:
            self.extra_config = {}


class CrossEncoderReranker:
    """Cross-encoder reranker that scores (query, passage) pairs jointly.

    Built on sentence-transformers' CrossEncoder, but exposed through
    our own interface so no LangChain/LlamaIndex coupling.
    """

    def __init__(self, model: RerankerModel):
        self.model = model
        self._model = None

    def load(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self._model = CrossEncoder(
            self.model.model_id,
            device=self.model.device,
            max_length=self.model.max_seq_length,
        )
        logger.info(f"Loaded cross-encoder: {self.model.model_id}")

    def rerank(
        self,
        query: str,
        passages: List[str],
        k: Optional[int] = 10,
        batch_size: Optional[int] = None,
    ) -> List[Tuple[float, int, str]]:
        """Score and re-rank passages for a given query.

        Returns list of (score, idx, passage_text) sorted descending.
        """
        if self._model is None:
            self.load()

        batch_size = batch_size or self.model.batch_size
        pairs = [(query, p) for p in passages]

        scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        if scores.ndim == 2:
            scores = scores[:, 1]  # [score_pos, score_neg] from NLI models

        scored = [(float(scores[i]), i, passages[i]) for i in range(len(passages))]
        scored.sort(key=lambda x: x[0], reverse=True)

        if k is not None:
            scored = scored[:k]

        return scored

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._model = None


class RerankerManager:
    """Manager for multiple reranker models, backed by config."""

    def __init__(self, config=None):
        from rag_pipeline.config import config as global_config
        self.config = config or global_config
        self._rerankers: dict = {}
        self._current: Optional[CrossEncoderReranker] = None

    def _model_config(self, name: Optional[str] = None) -> RerankerModel:
        if name is None:
            name = self.config.get("reranking", "default_model")
        models = self.config.get("reranking", "models", default=[])
        for m in models:
            if m["name"] == name:
                return RerankerModel(**m)
        raise ValueError(f"Reranker model '{name}' not found in config")

    def get_reranker(self, name: Optional[str] = None) -> CrossEncoderReranker:
        if name is None:
            name = self.config.get("reranking", "default_model")
        if name not in self._rerankers:
            cfg = self._model_config(name)
            self._rerankers[name] = CrossEncoderReranker(cfg)
        self._current = self._rerankers[name]
        return self._current

    def rerank(
        self,
        query: str,
        passages: List[str],
        k: Optional[int] = None,
        model: Optional[str] = None,
    ) -> List[Tuple[float, int, str]]:
        r = self.get_reranker(model)
        if k is None:
            k = self.config.get("reranking", "final_k", default=10)
        return r.rerank(query, passages, k=k)

    @property
    def final_k(self) -> int:
        return self.config.get("reranking", "final_k", default=10)

    @property
    def top_k_retrieve(self) -> int:
        return self.config.get("reranking", "top_k", default=100)

    def list_models(self) -> List[RerankerModel]:
        models = self.config.get("reranking", "models", default=[])
        return [RerankerModel(**m) for m in models]


__all__ = [
    "CrossEncoderReranker",
    "RerankerModel",
    "RerankerManager",
]
