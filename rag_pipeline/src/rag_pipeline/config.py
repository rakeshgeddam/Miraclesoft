"""
Configuration management for RAG Pipeline.
Loads YAML config with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration manager with YAML file and env var support."""
    
    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if not self._config:
            self.load()
    
    def load(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = os.environ.get(
                "RAG_CONFIG_PATH",
                str(Path(__file__).parent.parent.parent / "config" / "settings.yaml")
            )
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, "r") as f:
            self._config = yaml.safe_load(f) or {}
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to config."""
        env_mappings = {
            "RAG_EMBEDDING_MODEL": ("embeddings", "default_model"),
            "RAG_CHUNK_SIZE": ("chunking", "strategies", "recursive", "chunk_size"),
            "RAG_CHUNK_OVERLAP": ("chunking", "strategies", "recursive", "chunk_overlap"),
            "RAG_INDEX_TYPE": ("indexing", "default_index"),
            "RAG_RERANK_MODEL": ("reranking", "default_model"),
            "RAG_LLM_PROVIDER": ("llm", "provider"),
            "RAG_LLM_MODEL": ("llm", "model"),
            "RAG_LLM_BASE_URL": ("llm", "base_url"),
            "RAG_INDEX_PATH": ("storage", "index_path"),
            "RAG_METADATA_PATH": ("storage", "metadata_path"),
        }
        
        for env_var, keys in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(self._config, keys, self._parse_value(value))
    
    def _set_nested(self, d: Dict, keys: tuple, value: Any) -> None:
        """Set nested dictionary value."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse string value to appropriate type."""
        # Try bool
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        return value
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """Get nested config value."""
        current = self._config
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return default
            if current is None:
                return default
        return current
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire config section."""
        return self._config.get(section, {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full config dict."""
        return self._config.copy()


# Global config instance
config = Config()