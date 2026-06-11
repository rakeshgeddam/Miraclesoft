"""
Generation strategies for RAG.

No framework — just prompt templates + an OpenAI-compatible HTTP client
so you can swap in any LLM (vLLM, Ollama, OpenAI, Anthropic, etc.).
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)


class GenerationStrategy:
    REPLUG = "replug"
    RAG_SEQUENCE = "rag_sequence"
    FID = "fid"


@dataclass
class GenerationResult:
    text: str
    source_passages: List[dict]
    scores: List[float]


@dataclass
class LLMConfig:
    """Configuration for the generator LLM."""
    provider: str = "openai_compatible"
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 512
    temperature: float = 0.1
    extra_kwargs: dict = field(default_factory=dict)


class GenerationManager:
    """Prompt builder + LLM caller for RAG generation."""

    def __init__(self, config=None):
        from rag_pipeline.config import config as global_config
        self.config = config or global_config
        self._llm_config: Optional[LLMConfig] = None

    def _get_llm_config(self) -> LLMConfig:
        if self._llm_config is None:
            raw = self.config.get_section("llm")
            self._llm_config = LLMConfig(
                provider=raw.get("provider", "openai_compatible"),
                model=raw.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
                base_url=raw.get("base_url", "http://localhost:8000/v1"),
                api_key=raw.get("api_key", "EMPTY"),
                max_tokens=raw.get("max_tokens", 512),
                temperature=raw.get("temperature", 0.1),
                extra_kwargs=raw.get("extra_kwargs", {}),
            )
            # Env overrides
            self._llm_config.base_url = os.environ.get(
                "RAG_LLM_BASE_URL", self._llm_config.base_url
            )
            self._llm_config.api_key = os.environ.get(
                "RAG_LLM_API_KEY", self._llm_config.api_key
            )
        return self._llm_config

    # ---- Prompt templates ----

    def _format_passages(self, passages: List[str], max_chars: int = 15000) -> str:
        """Join passages into a formatted document block."""
        lines = []
        total = 0
        for i, p in enumerate(passages):
            remaining = max_chars - total
            if len(p) > remaining:
                p = p[:remaining]
            lines.append(f"Document {i+1}: {p}")
            total += len(p)
            if total >= max_chars:
                break
        return "\n\n".join(lines)

    def build_replug_prompt(self, query: str, passages: List[str]) -> str:
        """REPLUG-style: simply prepend doc passages before query."""
        template = self.config.get(
            "generation", "strategies", "replug", "prompt_template",
            default="{documents}\n\nQuestion: {query}\nAnswer:",
        )
        docs = self._format_passages(passages)
        return template.format(documents=docs, query=query)

    def build_rag_sequence_prompt(self, query: str, passages: List[str]) -> str:
        """RAG-Sequence-style: context block then query."""
        template = self.config.get(
            "generation", "strategies", "rag_sequence", "prompt_template",
            default="Context:\n{documents}\n\nQuestion: {query}\nAnswer:",
        )
        docs = self._format_passages(passages)
        return template.format(documents=docs, query=query)

    # ---- LLM caller ----

    def generate(
        self,
        prompt: str,
        strategy: str = "replug",
    ) -> GenerationResult:
        """Send prompt to LLM and return answer.
        
        Falls back to returning retrieved passages as answer if LLM is unavailable.
        """
        llm = self._get_llm_config()
        try:
            text = self._call_llm(prompt, llm)
        except Exception as e:
            logger.warning(f"LLM call failed ({e}), using fallback answer from retrieved passages")
            text = self._fallback_answer(prompt)
        return GenerationResult(
            text=text,
            source_passages=[],
            scores=[],
        )
    
    def _fallback_answer(self, prompt: str) -> str:
        """Extract a fallback answer from the prompt's retrieved documents."""
        import re
        # Try to extract Document 1 text as a fallback answer
        match = re.search(r'Document \d+:\s*(.*?)(?:\n\nDocument|\Z)', prompt, re.DOTALL)
        if match:
            text = match.group(1).strip()
            if len(text) > 500:
                text = text[:500] + "..."
            return text
        return "Answer not available (LLM server offline). Passages were retrieved and re-ranked."

    def _call_llm(self, prompt: str, llm: LLMConfig) -> str:
        """Generic OpenAI-compatible HTTP call."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx not installed. `pip install httpx`")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llm.api_key}",
        }
        body = {
            "model": llm.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": llm.max_tokens,
            "temperature": llm.temperature,
            **llm.extra_kwargs,
        }

        resp = httpx.post(
            f"{llm.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_llm_openai(self, prompt: str, llm: LLMConfig) -> str:
        """OpenAI-specific SDK path."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai not installed. `pip install openai`")

        client = OpenAI(
            api_key=llm.api_key,
            base_url=llm.base_url,
        )
        resp = client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=llm.max_tokens,
            temperature=llm.temperature,
        )
        return resp.choices[0].message.content

    def _call_llm_ollama(self, prompt: str, llm: LLMConfig) -> str:
        """Ollama-specific call (no /v1 prefix)."""
        import httpx
        base = llm.base_url.replace("/v1", "").rstrip("/")
        body = {
            "model": llm.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": llm.max_tokens,
                "temperature": llm.temperature,
            },
        }
        resp = httpx.post(f"{base}/api/generate", json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"]

    def supported_strategies(self) -> List[str]:
        return [GenerationStrategy.REPLUG, GenerationStrategy.RAG_SEQUENCE,
                GenerationStrategy.FID]


__all__ = [
    "GenerationManager",
    "GenerationStrategy",
    "GenerationResult",
    "LLMConfig",
]
