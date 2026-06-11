"""
Embedding models abstraction layer.

Supports multiple providers:
- sentence_transformers (SentenceTransformers library)
- huggingface (direct transformers)
- openai (OpenAI API)
- ollama (local Ollama)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModel:
    """Configuration for an embedding model."""
    name: str
    provider: str
    model_id: str
    dimension: int
    max_seq_length: int = 512
    normalize: bool = True
    batch_size: int = 32
    device: Optional[str] = None
    # Provider-specific options
    extra_config: dict = None
    
    def __post_init__(self):
        if self.extra_config is None:
            self.extra_config = {}
        # Detect E5 models that need query/passage prefix
        model_id_lower = (self.model_id or self.name or "").lower()
        self.requires_prefix = "e5" in model_id_lower or "intfloat" in model_id_lower


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, model: EmbeddingModel):
        self.model = model
        self._model = None
    
    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of shape (len(texts), dimension)
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        pass
    
    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
    
    def __enter__(self):
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class SentenceTransformersEmbedder(BaseEmbedder):
    """Embeddings using sentence-transformers library."""
    
    def load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self._model = SentenceTransformer(
            self.model.model_id,
            device=self.model.device
        )
        # Set max sequence length
        if hasattr(self._model, 'max_seq_length'):
            self._model.max_seq_length = self.model.max_seq_length
        logger.info(f"Loaded SentenceTransformer model: {self.model.model_id}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            self.load()
        
        # E5 models need "passage: " prefix for documents
        if self.model.requires_prefix:
            texts = [f"passage: {t}" if not t.startswith("passage: ") and not t.startswith("query: ") else t for t in texts]
        
        embeddings = self._model.encode(
            texts,
            batch_size=self.model.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.model.normalize
        )
        return embeddings.astype(np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        if self.model.requires_prefix:
            if not query.startswith("query: ") and not query.startswith("passage: "):
                query = f"query: {query}"
        return self.embed([query])[0]


class HuggingFaceEmbedder(BaseEmbedder):
    """Embeddings using raw HuggingFace transformers."""
    
    def load(self) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers torch"
            )
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model.model_id)
        self._model = AutoModel.from_pretrained(self.model.model_id)
        
        if self.model.device:
            self._model = self._model.to(self.model.device)
        else:
            self._model = self._model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        self._model.eval()
        logger.info(f"Loaded HuggingFace model: {self.model.model_id}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            self.load()
        
        import torch
        
        all_embeddings = []
        for i in range(0, len(texts), self.model.batch_size):
            batch = texts[i:i + self.model.batch_size]
            
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.model.max_seq_length,
                return_tensors="pt"
            ).to(self._model.device)
            
            with torch.no_grad():
                model_output = self._model(**encoded)
                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])
                
                if self.model.normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class OpenAIEmbedder(BaseEmbedder):
    """Embeddings using OpenAI API."""
    
    def load(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. "
                "Install with: pip install openai"
            )
        
        api_key = self.model.extra_config.get("api_key") or self.model.extra_config.get("openai_api_key")
        base_url = self.model.extra_config.get("base_url")
        
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        self._client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI embedder: {self.model.model_id}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        if self._client is None:
            self.load()
        
        # OpenAI API has batch limits
        batch_size = min(self.model.batch_size, 2000)
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._client.embeddings.create(
                model=self.model.model_id,
                input=batch
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        if self.model.normalize:
            embeddings = self.normalize(embeddings)
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


class OllamaEmbedder(BaseEmbedder):
    """Embeddings using local Ollama."""
    
    def load(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests not installed. "
                "Install with: pip install requests"
            )
        
        self._base_url = self.model.extra_config.get("base_url", "http://localhost:11434")
        self._session = requests.Session()
        logger.info(f"Initialized Ollama embedder: {self.model.model_id}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        if self._session is None:
            self.load()
        
        all_embeddings = []
        for i in range(0, len(texts), self.model.batch_size):
            batch = texts[i:i + self.model.batch_size]
            embeddings = []
            for text in batch:
                response = self._session.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self.model.model_id, "prompt": text}
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            
            all_embeddings.extend(embeddings)
        
        embeddings = np.array(all_embeddings, dtype=np.float32)
        
        if self.model.normalize:
            embeddings = self.normalize(embeddings)
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]


PROVIDER_MAP = {
    "sentence_transformers": SentenceTransformersEmbedder,
    "huggingface": HuggingFaceEmbedder,
    "openai": OpenAIEmbedder,
    "ollama": OllamaEmbedder,
}


class EmbeddingManager:
    """Manager for multiple embedding models."""
    
    def __init__(self, config=None):
        from rag_pipeline.config import config as global_config
        self.config = config or global_config
        self._embedders: dict[str, BaseEmbedder] = {}
        self._current_model: Optional[str] = None
    
    def get_model_config(self, name: Optional[str] = None) -> EmbeddingModel:
        """Get model configuration by name."""
        if name is None:
            name = self.config.get("embeddings", "default_model")
        
        models = self.config.get("embeddings", "models", default=[])
        for m in models:
            if m["name"] == name:
                return EmbeddingModel(**m)
        
        # Fallback to default
        default_name = self.config.get("embeddings", "default_model")
        for m in models:
            if m["name"] == default_name:
                return EmbeddingModel(**m)
        
        raise ValueError(f"Model '{name}' not found in config")
    
    def get_embedder(self, name: Optional[str] = None) -> BaseEmbedder:
        """Get or create embedder for model."""
        if name is None:
            name = self.config.get("embeddings", "default_model")
        
        if name not in self._embedders:
            model_config = self.get_model_config(name)
            provider_class = PROVIDER_MAP.get(model_config.provider)
            if provider_class is None:
                raise ValueError(f"Unknown provider: {model_config.provider}")
            self._embedders[name] = provider_class(model_config)
        
        return self._embedders[name]
    
    def embed(self, texts: List[str], model: Optional[str] = None) -> np.ndarray:
        """Embed texts using specified model."""
        embedder = self.get_embedder(model)
        return embedder.embed(texts)
    
    def embed_query(self, query: str, model: Optional[str] = None) -> np.ndarray:
        """Embed a single query."""
        embedder = self.get_embedder(model)
        return embedder.embed_query(query)
    
    def list_models(self) -> List[EmbeddingModel]:
        """List all configured models."""
        models = self.config.get("embeddings", "models", default=[])
        return [EmbeddingModel(**m) for m in models]
    
    def set_default_model(self, name: str) -> None:
        """Set the default model."""
        if name not in [m.name for m in self.list_models()]:
            raise ValueError(f"Model '{name}' not configured")
        self._current_model = name
    
    @property
    def current_model(self) -> str:
        return self._current_model or self.config.get("embeddings", "default_model")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass