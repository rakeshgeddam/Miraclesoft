"""
Chunk metadata for vector store.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: int
    doc_id: str
    text: str
    token_start: int
    token_end: int
    source_file: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps({
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "source_file": self.source_file,
            **self.extra
        })
    
    @classmethod
    def from_json(cls, line: str) -> "ChunkMetadata":
        data = json.loads(line)
        extra = {k: v for k, v in data.items() 
                 if k not in ("chunk_id", "doc_id", "text", "token_start", "token_end", "source_file")}
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            text=data["text"],
            token_start=data["token_start"],
            token_end=data["token_end"],
            source_file=data.get("source_file", ""),
            extra=extra
        )