"""
VectorStore - Main class for memory-mapped vector storage.
"""

import os
import struct
import mmap
from pathlib import Path
from typing import List, Optional, Set, Tuple
import numpy as np
import logging

from .constants import (
    MAGIC_HRID, VERSION, DTYPE_FLOAT32,
    FLAG_NORMALIZED, HEADER_SIZE
)
from .metadata import ChunkMetadata

logger = logging.getLogger(__name__)


class VectorStore:
    """Memory-mapped vector store with binary format.
    
    Format:
    - Header: 64 bytes (magic, version, num_vectors, dimension, dtype, flags, padding)
    - Vectors: N x D x 4 bytes (float32, row-major)
    - Doc IDs: N x 8 bytes (uint64)
    - Chunk offsets: N x 8 bytes (uint64)
    - Norms: N x 4 bytes (float32, optional if not normalized)
    """
    
    def __init__(
        self,
        index_path: str,
        metadata_path: str,
        dimension: int,
        mmap_enabled: bool = True,
        normalize: bool = True,
        readonly: bool = False,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dimension = dimension
        self.mmap_enabled = mmap_enabled
        self.normalize = normalize
        self.readonly = readonly
        
        self._mm: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self._vectors: Optional[np.ndarray] = None
        self._doc_ids: Optional[np.ndarray] = None
        self._chunk_offsets: Optional[np.ndarray] = None
        self._norms: Optional[np.ndarray] = None
        self._num_vectors = 0
        self._metadata_cache = {}
        
        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    def create(self, num_vectors: int = 0) -> None:
        """Create new vector store file and open it for writing.
        
        Pre-allocates space for num_vectors but starts with count=0
        so add_vectors fills from index 0.
        """
        vectors_size = num_vectors * self.dimension * 4
        doc_ids_size = num_vectors * 8
        chunk_offsets_size = num_vectors * 8
        norms_size = num_vectors * 4 if not self.normalize else 0
        total_size = HEADER_SIZE + vectors_size + doc_ids_size + chunk_offsets_size + norms_size
        
        with open(self.index_path, "wb") as f:
            # Write header with num_vectors=0 (we pre-allocate but count starts at 0)
            f.write(MAGIC_HRID)
            f.write(struct.pack("<I", VERSION))
            f.write(struct.pack("<Q", 0))  # num_vectors starts at 0
            f.write(struct.pack("<I", self.dimension))
            f.write(struct.pack("<B", DTYPE_FLOAT32))
            flags = FLAG_NORMALIZED if self.normalize else 0
            f.write(struct.pack("<B", flags))
            f.write(b"\x00" * 42)  # padding
            
            # Write pre-allocated empty data sections
            f.write(b"\x00" * (total_size - HEADER_SIZE))
        
        # Create empty metadata file
        with open(self.metadata_path, "w") as f:
            pass
        
        # Open the mmap so we can write to it immediately
        self.open()
        
        logger.info(f"Created vector store: {self.index_path} (pre-allocated {num_vectors} vectors, {self.dimension} dims)")
    
    def open(self) -> None:
        """Open existing vector store for reading/writing."""
        mode = "rb" if self.readonly else "r+b"
        self._fd = os.open(self.index_path, os.O_RDONLY if self.readonly else os.O_RDWR)
        self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ if self.readonly else mmap.ACCESS_WRITE)
        
        self._parse_header()
        self._map_data_sections()
        
        logger.info(f"Opened vector store: {self._num_vectors} vectors, {self.dimension} dims")
    
    def _parse_header(self) -> None:
        """Parse binary header."""
        if len(self._mm) < HEADER_SIZE:
            raise ValueError("File too small for header")
        
        magic = self._mm[0:4]
        if magic != MAGIC_HRID:
            raise ValueError(f"Invalid magic: {magic}, expected {MAGIC_HRID}")
        
        version = struct.unpack("<I", self._mm[4:8])[0]
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")
        
        self._num_vectors = struct.unpack("<Q", self._mm[8:16])[0]
        dim = struct.unpack("<I", self._mm[16:20])[0]
        if dim != self.dimension:
            raise ValueError(f"Dimension mismatch: file={dim}, expected={self.dimension}")
        
        dtype = self._mm[20]
        if dtype != DTYPE_FLOAT32:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        flags = self._mm[21]
        file_normalized = bool(flags & FLAG_NORMALIZED)
        if file_normalized != self.normalize:
            logger.warning(f"Normalization mismatch: file={file_normalized}, config={self.normalize}")
    
    def _map_data_sections(self) -> None:
        """Map data sections from memory-mapped file."""
        offset = HEADER_SIZE
        
        # Vectors
        self._vectors = np.frombuffer(
            self._mm, dtype=np.float32, 
            count=self._num_vectors * self.dimension, offset=offset
        ).reshape(self._num_vectors, self.dimension)
        
        offset += self._num_vectors * self.dimension * 4
        
        # Doc IDs
        self._doc_ids = np.frombuffer(
            self._mm, dtype=np.uint64, count=self._num_vectors, offset=offset
        )
        
        offset += self._num_vectors * 8
        
        # Chunk offsets
        self._chunk_offsets = np.frombuffer(
            self._mm, dtype=np.uint64, count=self._num_vectors, offset=offset
        )
        
        # Norms (if not normalized)
        if not self.normalize:
            offset += self._num_vectors * 8
            self._norms = np.frombuffer(
                self._mm, dtype=np.float32, count=self._num_vectors, offset=offset
            )
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        doc_ids: np.ndarray,
        chunk_offsets: np.ndarray,
        metadata: List[ChunkMetadata],
    ) -> int:
        """Add vectors to the store (appends to end)."""
        if self.readonly:
            raise RuntimeError("Cannot write to readonly store")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: {vectors.shape[1]} != {self.dimension}")
        
        n_new = len(vectors)
        start_idx = self._num_vectors
        
        # Normalize if needed
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
        
        # Calculate capacity from file size (pre-allocated total slots)
        mmap_len = len(self._mm)
        per_vector_size = self.dimension * 4 + 8 + 8 + (0 if self.normalize else 4)
        capacity = (mmap_len - HEADER_SIZE) // per_vector_size if mmap_len > HEADER_SIZE else 0
        
        # Resize file if needed
        self._ensure_capacity(start_idx + n_new)
        
        # Write vectors (start right after header)
        vec_offset = HEADER_SIZE + start_idx * self.dimension * 4
        self._mm[vec_offset:vec_offset + vectors.nbytes] = vectors.tobytes()
        
        # Write doc_ids (after all vector slots, not just used ones)
        doc_ids_section_start = HEADER_SIZE + capacity * self.dimension * 4
        did_offset = doc_ids_section_start + start_idx * 8
        self._mm[did_offset:did_offset + doc_ids.nbytes] = doc_ids.tobytes()
        
        # Write chunk_offsets (after all doc_id slots)
        chunk_offsets_section_start = doc_ids_section_start + capacity * 8
        co_offset = chunk_offsets_section_start + start_idx * 8
        self._mm[co_offset:co_offset + chunk_offsets.nbytes] = chunk_offsets.tobytes()
        
        # Write norms if not normalized
        if not self.normalize:
            norms_section_start = chunk_offsets_section_start + capacity * 8
            norms_offset = norms_section_start + start_idx * 4
            norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
            self._mm[norms_offset:norms_offset + norms.nbytes] = norms.tobytes()
        
        # Update header num_vectors
        self._mm[8:16] = struct.pack("<Q", start_idx + n_new)
        self._num_vectors = start_idx + n_new
        
        # Write metadata
        with open(self.metadata_path, "a") as f:
            for meta in metadata:
                f.write(meta.to_json() + "\n")
        
        # Refresh memory views
        if self._mm:
            self.open()
        
        return start_idx
    
    def _ensure_capacity(self, required: int) -> None:
        """Ensure file has capacity for required vectors. Grows file if needed."""
        if self._mm is None:
            return
        
        current_count = self._num_vectors
        if required <= current_count:
            return
        
        # Release numpy array views before remapping
        self._vectors = None
        self._doc_ids = None
        self._chunk_offsets = None
        self._norms = None
        
        # Calculate sizes
        add_count = required - current_count
        add_vectors_size = add_count * self.dimension * 4
        add_doc_ids_size = add_count * 8
        add_chunk_offsets_size = add_count * 8
        add_norms_size = add_count * 4 if not self.normalize else 0
        add_total = add_vectors_size + add_doc_ids_size + add_chunk_offsets_size + add_norms_size
        
        # Grow the file and remap
        new_size = len(self._mm) + add_total
        self._mm.close()
        os.ftruncate(self._fd, new_size)
        self._mm = mmap.mmap(self._fd, new_size, access=mmap.ACCESS_WRITE)
        
        # Refresh views from the remapped buffer
        self._map_data_sections()
        
        logger.debug(f"Grew vector store from {current_count} to {required} ({add_total} bytes)")
    
    # ===== Accessor methods =====
    
    def get_vector(self, idx: int) -> np.ndarray:
        """Get vector by index."""
        if idx >= self._num_vectors:
            raise IndexError(f"Index {idx} out of range (max {self._num_vectors})")
        return self._vectors[idx].copy()
    
    def get_vectors(self, indices: List[int]) -> np.ndarray:
        """Get multiple vectors by indices."""
        return self._vectors[indices].copy()
    
    def get_doc_id(self, idx: int) -> int:
        """Get doc_id by index."""
        return int(self._doc_ids[idx])
    
    def get_chunk_offset(self, idx: int) -> int:
        """Get chunk offset by index."""
        return int(self._chunk_offsets[idx])
    
    def get_norm(self, idx: int) -> float:
        """Get precomputed norm by index."""
        if self._norms is not None:
            return float(self._norms[idx])
        # Compute on the fly
        return float(np.linalg.norm(self._vectors[idx]))
    
    # ===== Search methods =====
    
    def search_brute_force(
        self,
        query: np.ndarray,
        k: int = 10,
        exclude_indices: Optional[Set[int]] = None,
    ) -> List[Tuple[float, int]]:
        """Brute-force top-k search using dot product."""
        from .search import search_brute_force as _search_brute_force
        return _search_brute_force(
            self._vectors, query, k, exclude_indices, self.normalize
        )
    
    def load_metadata(self, chunk_ids: List[int]) -> List[ChunkMetadata]:
        """Load metadata for given chunk IDs."""
        from .search import load_metadata as _load_metadata
        return _load_metadata(self.metadata_path, chunk_ids, self._metadata_cache)
    
    # ===== Context manager =====
    
    def close(self) -> None:
        """Close the store, releasing memory-mapped views first."""
        # Release numpy array views before closing mmap
        self._vectors = None
        self._doc_ids = None
        self._chunk_offsets = None
        self._norms = None
        
        if self._mm:
            self._mm.flush()
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @property
    def num_vectors(self) -> int:
        return self._num_vectors
    
    @property
    def dim(self) -> int:
        return self.dimension
        pass