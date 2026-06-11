"""
IVF (Inverted File) Index implementation.

Uses k-means clustering for centroids and inverted lists for fast search.
"""

import numpy as np
import struct
import mmap
import logging
from typing import List, Tuple, Optional, Set
from .base import BaseIndex, SearchResult

logger = logging.getLogger(__name__)


class IVFIndex(BaseIndex):
    """Inverted File Index with k-means clustering.
    
    - Training: k-means on sample vectors to find centroids
    - Search: Find nearest nprobe centroids, scan their lists
    
    Parameters:
    - n_centroids: Number of clusters (e.g., 4096 for 1M vectors)
    - n_probe: Number of centroids to search at query time
    - kmeans_iterations: K-means iterations during training
    """
    
    def __init__(
        self,
        n_centroids: int = 4096,
        n_probe: int = 16,
        kmeans_iterations: int = 20,
        normalized: bool = True,
        seed: int = 42,
    ):
        self.n_centroids = n_centroids
        self.n_probe = n_probe
        self.kmeans_iterations = kmeans_iterations
        self.normalized = normalized
        self.seed = seed
        
        self._vectors: Optional[np.ndarray] = None
        self._centroids: Optional[np.ndarray] = None
        self._inverted_lists: List[np.ndarray] = []  # List of arrays of vector indices
        self._list_sizes: Optional[np.ndarray] = None
        self._is_trained = False
        self._doc_ids: Optional[np.ndarray] = None
    
    def _kmeans(self, vectors: np.ndarray, k: int, iterations: int) -> np.ndarray:
        """Mini-batch k-means clustering."""
        np.random.seed(self.seed)
        n, d = vectors.shape
        
        # Initialize centroids by random sampling
        indices = np.random.choice(n, size=min(k, n), replace=False)
        centroids = vectors[indices].copy().astype(np.float32)
        
        for iteration in range(iterations):
            # Assign each vector to nearest centroid (using dot product for normalized)
            if self.normalized:
                dists = vectors @ centroids.T  # (n, k)
                assignments = np.argmax(dists, axis=1)
            else:
                # L2 distance
                diff = vectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]
                dists = np.sum(diff ** 2, axis=2)
                assignments = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k, dtype=np.int32)
            
            for i in range(n):
                c = assignments[i]
                new_centroids[c] += vectors[i]
                counts[c] += 1
            
            # Avoid division by zero
            counts[counts == 0] = 1
            new_centroids = new_centroids / counts[:, np.newaxis]
            
            # Normalize if needed
            if self.normalized:
                norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                new_centroids = new_centroids / norms
            
            # Check convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            
            if shift < 1e-4:
                logger.debug(f"K-means converged at iteration {iteration}")
                break
        
        return centroids
    
    def build(self, vectors: np.ndarray) -> None:
        """Build IVF index: train centroids and assign vectors to lists."""
        self._vectors = vectors.astype(np.float32)
        n, d = self._vectors.shape
        
        # Clamp centroids to at most n (can't have more clusters than vectors)
        K = min(self.n_centroids, n)
        if K < self.n_centroids:
            logger.warning(f"Clamping centroids from {self.n_centroids} to {K} (only {n} vectors)")
        self.n_centroids = K
        
        logger.info(f"Training IVF index: {n} vectors, {d} dims, {K} centroids")
        
        # Train centroids on a sample (or all vectors if small)
        sample_size = min(100000, n)
        if n > sample_size:
            sample_indices = np.random.choice(n, sample_size, replace=False)
            sample = self._vectors[sample_indices]
        else:
            sample = self._vectors
        
        # Train k-means (maintains float32 internally)
        self._centroids = self._kmeans(sample, self.n_centroids, self.kmeans_iterations).astype(np.float32)
        
        # Assign all vectors to nearest centroid
        if self.normalized:
            dists = self._vectors @ self._centroids.T
            assignments = np.argmax(dists, axis=1)
        else:
            diff = self._vectors[:, np.newaxis, :] - self._centroids[np.newaxis, :, :]
            dists = np.sum(diff ** 2, axis=2)
            assignments = np.argmin(dists, axis=1)
        
        # Build inverted lists
        self._inverted_lists = [np.array([], dtype=np.uint32) for _ in range(self.n_centroids)]
        for i, c in enumerate(assignments):
            self._inverted_lists[c] = np.append(self._inverted_lists[c], i)
        
        self._list_sizes = np.array([len(lst) for lst in self._inverted_lists], dtype=np.uint32)
        self._is_trained = True
        
        logger.info(f"IVF index built: {sum(self._list_sizes)} vectors in {self.n_centroids} lists")
        logger.info(f"List sizes: min={np.min(self._list_sizes)}, max={np.max(self._list_sizes)}, "
                    f"mean={np.mean(self._list_sizes):.1f}")
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 10, 
        ef: Optional[int] = None,
        exclude: Optional[Set[int]] = None,
    ) -> List[Tuple[float, int]]:
        """Search IVF index: probe nearest centroids, scan their lists."""
        if not self._is_trained:
            raise RuntimeError("Index not trained")
        
        query = query.astype(np.float32)
        n_probe = min(self.n_probe, self.n_centroids)
        
        # Find nearest centroids
        if self.normalized:
            centroid_scores = self._centroids @ query
            probe_indices = np.argpartition(centroid_scores, -n_probe)[-n_probe:]
            probe_indices = probe_indices[np.argsort(centroid_scores[probe_indices])[::-1]]
        else:
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
            centroid_norms = np.linalg.norm(self._centroids, axis=1)
            dists = (self._centroids @ query) / centroid_norms
            probe_indices = np.argpartition(dists, -n_probe)[-n_probe:]
            probe_indices = probe_indices[np.argsort(dists[probe_indices])[::-1]]
        
        # Scan inverted lists for probe centroids
        candidates = []
        for centroid_idx in probe_indices:
            lst = self._inverted_lists[centroid_idx]
            if len(lst) == 0:
                continue
            candidates.extend(lst)
        
        candidates = np.array(candidates, dtype=np.uint32)
        
        # Exact scoring on candidates
        if self.normalized:
            scores = self._vectors[candidates] @ query
        else:
            cand_norms = np.linalg.norm(self._vectors[candidates], axis=1)
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                scores = (self._vectors[candidates] @ query) / (cand_norms * query_norm)
            else:
                scores = np.zeros(len(candidates))
        
        if exclude:
            # Filter out excluded indices
            mask = ~np.isin(candidates, list(exclude))
            candidates = candidates[mask]
            scores = scores[mask]
        
        # Top-k from candidates
        if k >= len(candidates):
            top_idx = np.argsort(scores)[::-1]
        else:
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        
        results = [(float(scores[i]), int(candidates[i])) for i in top_idx[:k]]
        return results
    
    def save(self, path: str) -> None:
        """Save IVF index to binary format (compatible with skill spec)."""
        if not self._is_trained:
            raise RuntimeError("Index not trained")
        
        # Binary format:
        # Header: magic="IVF1", num_centroids, dim, num_vectors (64 bytes)
        # Centroids: K x D x 4 bytes (float32)
        # List offsets: K x 8 bytes (uint64)
        # List data: for each centroid [size:uint64, doc_ids:uint64...]
        
        n = len(self._vectors)
        d = self._vectors.shape[1]
        K = self.n_centroids
        
        # Calculate sizes
        header_size = 64
        centroids_size = self._centroids.nbytes  # K * d * 4 (float32)
        list_offsets_size = K * 8
        elem_size = self._centroids.dtype.itemsize  # 4 for float32, 8 for float64
        
        # Calculate list data size
        list_data_size = 0
        for lst in self._inverted_lists:
            list_data_size += 8 + len(lst) * 8  # size + doc_ids
        
        total_size = header_size + centroids_size + list_offsets_size + list_data_size
        
        with open(path, "wb") as f:
            # Header
            f.write(b"IVF1")
            f.write(struct.pack("<I", K))
            f.write(struct.pack("<I", d))
            f.write(struct.pack("<Q", n))
            f.write(struct.pack("<I", self.n_probe))  # bytes 20-23
            f.write(b"\x00" * 40)  # padding to 64
            
            # Centroids
            f.write(self._centroids.tobytes())
            
            # List offsets (will fill in after)
            list_offsets = []
            current_offset = header_size + centroids_size + list_offsets_size
            for lst in self._inverted_lists:
                list_offsets.append(current_offset)
                current_offset += 8 + len(lst) * 8
            
            f.write(np.array(list_offsets, dtype=np.uint64).tobytes())
            
            # List data
            for lst in self._inverted_lists:
                f.write(struct.pack("<Q", len(lst)))
                if len(lst) > 0:
                    f.write(lst.astype(np.uint64).tobytes())
        
        # Save vectors separately
        np.save(path + ".vectors.npy", self._vectors)
        
        logger.info(f"Saved IVFIndex to {path} ({total_size} bytes)")
    
    def load(self, path: str) -> None:
        """Load IVF index from binary format."""
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Parse header
            magic = mm[0:4]
            if magic != b"IVF1":
                raise ValueError(f"Invalid magic: {magic}")
            
            K = struct.unpack("<I", mm[4:8])[0]
            d = struct.unpack("<I", mm[8:12])[0]
            n = struct.unpack("<Q", mm[12:20])[0]
            self.n_probe = struct.unpack("<I", mm[20:24])[0]
            
            self.n_centroids = K
            self._is_trained = True
            
            # Read centroids
            centroids_offset = 64
            centroids_bytes = K * d * 4  # saved as float32
            self._centroids = np.frombuffer(
                mm, dtype=np.float32, count=K * d, offset=centroids_offset
            ).reshape(K, d).copy()
            
            # Read list offsets (need as Python ints before mmap closes)
            list_offsets_offset = centroids_offset + centroids_bytes
            list_offsets = np.frombuffer(
                mm, dtype=np.uint64, count=K, offset=list_offsets_offset
            ).copy()  # copy before mmap closes
            
            # Read list data
            self._inverted_lists = []
            for i in range(K):
                lst_offset = int(list_offsets[i])  # Must be Python int
                if lst_offset + 8 > len(mm):
                    logger.error(
                        f"List {i}: offset {lst_offset} exceeds mmap "
                        f"size {len(mm)} (K={K}, d={d}, n={n})"
                    )
                    # Fill remaining lists as empty
                    self._inverted_lists = [
                        self._inverted_lists[j] if j < i else np.array([], dtype=np.uint32)
                        for j in range(K)
                    ]
                    break
                lst_size = struct.unpack("<Q", mm[lst_offset:lst_offset+8])[0]
                if lst_size > 0:
                    lst = np.frombuffer(
                        mm, dtype=np.uint64, count=lst_size, offset=lst_offset + 8
                    ).astype(np.uint32)
                else:
                    lst = np.array([], dtype=np.uint32)
                self._inverted_lists.append(lst)
            
            mm.close()
        
        # Load vectors
        self._vectors = np.load(path + ".vectors.npy")
        self._list_sizes = np.array([len(lst) for lst in self._inverted_lists], dtype=np.uint32)
        
        logger.info(f"Loaded IVFIndex from {path}: {n} vectors, {K} centroids")
    
    def add(self, vectors: np.ndarray) -> int:
        """Add vectors - requires rebuild (IVF doesn't support incremental easily)."""
        if self._vectors is None:
            self.build(vectors)
        else:
            # Rebuild with new vectors
            all_vectors = np.vstack([self._vectors, vectors.astype(np.float32)])
            self.build(all_vectors)
        return len(self._vectors) - len(vectors)