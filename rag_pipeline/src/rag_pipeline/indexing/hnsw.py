"""
HNSW (Hierarchical Navigable Small World) Index implementation.

Based on Malkov & Yashunin, arXiv:1603.09320
"""

import numpy as np
import struct
import mmap
import os
import logging
import heapq
import math
from typing import List, Tuple, Optional, Set
from collections import defaultdict

from .base import BaseIndex

logger = logging.getLogger(__name__)


class HNSWIndex(BaseIndex):
    """Hierarchical Navigable Small World graph index.
    
    Parameters:
    - M: Max neighbors per node at base layer (typically 16)
    - M_max: Max neighbors for upper layers (typically M * 2)
    - ef_construction: Size of dynamic candidate list during build (200)
    - ef_search: Size of dynamic candidate list during search (100)
    - ml: Level normalization factor = 1 / ln(M)
    """
    
    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        M_max: Optional[int] = None,
        normalized: bool = True,
        seed: int = 42,
    ):
        self.M = M
        self.M_max = M_max or M * 2
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.normalized = normalized
        self.seed = seed
        
        # Derived parameters
        self.ml = 1.0 / math.log(M) if M > 1 else 1.0
        
        # Graph structure
        self._vectors: Optional[np.ndarray] = None
        self._neighbors: List[List[List[int]]] = []  # [layer][node] -> neighbors
        self._levels: List[int] = []  # Max layer for each node
        self._enter_point: int = -1
        self._max_level: int = 0
        self._num_vectors: int = 0
        
        # For search
        self._visited: np.ndarray = np.array([], dtype=np.uint64)
        self._vis_id: int = 0
    
    def _get_random_level(self) -> int:
        """Generate random level for new node (geometric distribution)."""
        # P(level=l) = exp(-l / ml) - exp(-(l+1) / ml)
        # Using inverse transform sampling
        r = np.random.random()
        level = int(-math.log(r) * self.ml)
        return min(level, 32)  # Cap at reasonable max
    
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distance between vectors (negative similarity for max-heap)."""
        if self.normalized:
            # Higher dot product = closer
            return -float(a @ b)
        else:
            # L2 distance
            return float(np.sum((a - b) ** 2))
    
    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int,
    ) -> List[int]:
        """Search single layer, return top-ef nodes closest to query.
        
        Uses heapq-based best-first search.
        candidates: min-heap of (dist, node) — closest-first frontier
        results: dict[node] = dist — maintains ef closest found so far
        """
        visited = set()
        results = {}  # node -> distance
        
        # Initialize with entry points
        for ep in entry_points:
            if ep in visited:
                continue
            dist = self._distance(query, self._vectors[ep])
            results[ep] = dist
            visited.add(ep)
        
        # Build candidate heap from results
        candidates = [(d, n) for n, d in results.items()]
        heapq.heapify(candidates)
        
        def _worst_dist() -> float:
            """Return the worst (largest) distance among top-ef results."""
            if len(results) <= ef:
                return float('inf')
            # Need furthest among ef closest; since we keep up to ef+1,
            # the worst is just the maximum
            return max(results.values())
        
        while candidates:
            dist, node = heapq.heappop(candidates)
            
            # If the closest candidate is worse than the ef-th result, stop
            if dist >= _worst_dist():
                break
            
            for neighbor in self._neighbors[layer][node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                
                n_dist = self._distance(query, self._vectors[neighbor])
                
                # Add if we have room or it's better than the worst
                if len(results) < ef or n_dist < _worst_dist():
                    results[neighbor] = n_dist
                    heapq.heappush(candidates, (n_dist, neighbor))
                    
                    # Keep ef closest
                    if len(results) > ef:
                        furthest_node = max(results, key=results.get)
                        del results[furthest_node]
        
        # Return sorted by distance (closest first)
        return sorted(results, key=results.get)
    
    def _heuristic_select_neighbors(self, candidates: List[int], M: int) -> List[int]:
        """Select diverse neighbors using heuristic from HNSW paper.
        
        Instead of just picking M closest, we try to select neighbors
        that are not too close to each other.
        """
        if len(candidates) <= M:
            return candidates
        
        # Start with closest
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        while len(selected) < M and remaining:
            # Find candidate with maximum minimum distance to selected
            best_idx = -1
            best_min_dist = -1
            
            for i, cand in enumerate(remaining):
                min_dist = min(
                    self._distance(self._vectors[cand], self._vectors[sel])
                    for sel in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break
        
        return selected
    
    def build(self, vectors: np.ndarray) -> None:
        """Build HNSW index incrementally."""
        np.random.seed(self.seed)
        
        self._vectors = vectors.astype(np.float32)
        self._num_vectors = len(vectors)
        self._neighbors = []  # Will grow with layers
        self._levels = []
        self._enter_point = 0
        self._max_level = 0
        
        # Initialize first node
        self._levels.append(0)
        self._neighbors.append([[]])  # layer 0
        
        logger.info(f"Building HNSW index: {self._num_vectors} vectors")
        
        for i in range(1, self._num_vectors):
            if i % 10000 == 0:
                logger.info(f"  Inserted {i}/{self._num_vectors} vectors")
            
            self._add_node(i)
        
        logger.info(f"HNSW index built: {self._num_vectors} nodes, max_level={self._max_level}")
    
    def _add_node(self, node_idx: int) -> None:
        """Add a single node to the HNSW graph."""
        level = self._get_random_level()
        self._levels.append(level)
        
        # Ensure neighbors list has enough layers
        while len(self._neighbors) <= level:
            # New layer needs entries for ALL existing nodes
            self._neighbors.append([[] for _ in range(node_idx)])
        for l in range(len(self._neighbors)):
            self._neighbors[l].append([])
        
        if self._num_vectors == 1:
            # First node
            self._enter_point = node_idx
            self._max_level = level
            return
        
        # Start from enter point at top layer
        current_near = [self._enter_point]
        
        # Descend through layers
        for l in range(self._max_level, level, -1):
            current_near = self._search_layer(
                self._vectors[node_idx], current_near, 1, l
            )
        
        # Search and connect at each layer from level down to 0
        for l in range(min(level, self._max_level), -1, -1):
            # Search for ef_construction nearest neighbors in layer l
            candidates = self._search_layer(
                self._vectors[node_idx], current_near, self.ef_construction, l
            )
            
            # Select M neighbors
            M_l = self.M if l == 0 else self.M_max
            selected = self._heuristic_select_neighbors(candidates, M_l)
            
            # Connect bidirectional
            for neighbor in selected:
                self._neighbors[l][node_idx].append(neighbor)
                self._neighbors[l][neighbor].append(node_idx)
            
            # Prune connections if needed (both directions)
            self._prune_connections(node_idx, l)
            for neighbor in selected:
                self._prune_connections(neighbor, l)
            
            # Update current_near for next layer
            current_near = selected
        
        # Update enter point if new node has higher level
        if level > self._max_level:
            self._enter_point = node_idx
            self._max_level = level
    
    def _prune_connections(self, node_idx: int, layer: int) -> None:
        """Prune connections to keep at most M edges."""
        M_l = self.M if layer == 0 else self.M_max
        neighbors = self._neighbors[layer][node_idx]
        
        if len(neighbors) <= M_l:
            return
        
        # Keep closest M
        dists = [(self._distance(self._vectors[node_idx], self._vectors[n]), n) for n in neighbors]
        dists.sort(key=lambda x: x[0])
        keep = [n for _, n in dists[:M_l]]
        
        # Remove this node from pruned neighbors' lists
        pruned = set(neighbors) - set(keep)
        for p in pruned:
            if node_idx in self._neighbors[layer][p]:
                self._neighbors[layer][p].remove(node_idx)
        
        self._neighbors[layer][node_idx] = keep
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        exclude: Optional[Set[int]] = None,
    ) -> List[Tuple[float, int]]:
        """Search HNSW index."""
        if self._num_vectors == 0:
            return []
        
        ef = ef or self.ef_search
        query = query.astype(np.float32)
        
        # Start from enter point at top layer
        current_near = [self._enter_point]
        
        # Descend through layers
        for l in range(self._max_level, 0, -1):
            current_near = self._search_layer(query, current_near, 1, l)
        
        # Search base layer with ef
        candidates = self._search_layer(query, current_near, ef, 0)
        
        # Score and filter
        results = []
        for node in candidates:
            if exclude and node in exclude:
                continue
            if self.normalized:
                score = float(self._vectors[node] @ query)
            else:
                score = -float(np.sum((self._vectors[node] - query) ** 2))
            results.append((score, node))
        
        # Sort by score (descending for normalized, ascending for distance)
        results.sort(key=lambda x: x[0], reverse=self.normalized)
        
        return results[:k]
    
    def save(self, path: str) -> None:
        """Save HNSW index to binary format."""
        # Format from skill spec:
        # Header: magic="HNS1", dim, N, M, M_max, ef_con, ml, max_level, enter_point (64 bytes)
        # Node blocks: variable length
        
        if self._num_vectors == 0:
            return
        
        d = self._vectors.shape[1]
        
        # Calculate total size
        header_size = 64
        node_data = bytearray()
        
        for i in range(self._num_vectors):
            level = self._levels[i]
            node_data.append(level)  # uint8
            
            for l in range(self._max_level, -1, -1):
                if l <= level:
                    neighbors = self._neighbors[l][i]
                    node_data.extend(struct.pack("<I", len(neighbors)))
                    for n in neighbors:
                        node_data.extend(struct.pack("<Q", n))
                # else: empty, skip
        
        total_size = header_size + len(node_data)
        
        with open(path, "wb") as f:
            # Header
            f.write(b"HNS1")
            f.write(struct.pack("<I", d))
            f.write(struct.pack("<Q", self._num_vectors))
            f.write(struct.pack("<I", self.M))
            f.write(struct.pack("<I", self.M_max))
            f.write(struct.pack("<I", self.ef_construction))
            f.write(struct.pack("<f", self.ml))
            f.write(struct.pack("<I", self._max_level))
            f.write(struct.pack("<Q", self._enter_point))
            f.write(b"\x00" * 20)  # padding to 64
            
            # Node data
            f.write(node_data)
        
        # Save vectors
        np.save(path + ".vectors.npy", self._vectors)
        
        logger.info(f"Saved HNSWIndex to {path} ({total_size} bytes)")
    
    def load(self, path: str) -> None:
        """Load HNSW index from binary format."""
        with open(path, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Parse header
            magic = mm[0:4]
            if magic != b"HNS1":
                raise ValueError(f"Invalid magic: {magic}")
            
            offset = 4
            d = struct.unpack("<I", mm[offset:offset+4])[0]; offset += 4
            n = struct.unpack("<Q", mm[offset:offset+8])[0]; offset += 8
            M = struct.unpack("<I", mm[offset:offset+4])[0]; offset += 4
            M_max = struct.unpack("<I", mm[offset:offset+4])[0]; offset += 4
            ef_construction = struct.unpack("<I", mm[offset:offset+4])[0]; offset += 4
            ml = struct.unpack("<f", mm[offset:offset+4])[0]; offset += 4
            max_level = struct.unpack("<I", mm[offset:offset+4])[0]; offset += 4
            enter_point = struct.unpack("<Q", mm[offset:offset+8])[0]; offset += 8
            offset += 20  # padding
            
            self.M = M
            self.M_max = M_max
            self.ef_construction = ef_construction
            self.ml = ml
            self._max_level = max_level
            self._enter_point = enter_point
            self._num_vectors = n
            
            # Parse node blocks
            self._levels = []
            self._neighbors = [[] for _ in range(max_level + 1)]
            
            for i in range(n):
                level = mm[offset]; offset += 1
                self._levels.append(level)
                
                for l in range(max_level, -1, -1):
                    if l <= level:
                        # Ensure layer exists
                        while len(self._neighbors) <= l:
                            self._neighbors.append([])
                        while len(self._neighbors[l]) <= i:
                            self._neighbors[l].append([])
                        
                        num_neighbors = struct.unpack("<I", mm[offset:offset+4])[0]; offset += 4
                        neighbors = []
                        for _ in range(num_neighbors):
                            n_idx = struct.unpack("<Q", mm[offset:offset+8])[0]; offset += 8
                            neighbors.append(n_idx)
                        self._neighbors[l][i] = neighbors
                    # else: empty
            
            mm.close()
        
        # Load vectors
        self._vectors = np.load(path + ".vectors.npy")
        
        logger.info(f"Loaded HNSWIndex from {path}: {n} vectors, max_level={max_level}")
    
    def add(self, vectors: np.ndarray) -> int:
        """Add vectors incrementally."""
        start_idx = self._num_vectors
        for v in vectors:
            self._add_node(self._num_vectors)
            self._num_vectors += 1
        return start_idx