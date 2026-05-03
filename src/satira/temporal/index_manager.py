from __future__ import annotations

import os
import pickle
import threading
from collections import OrderedDict
from typing import Any

import faiss
import numpy as np


class FAISSIndexManager:
    """Manages the in-process FAISS vector index for temporal news context.

    Design decisions:
    - In-process (not network call to vector DB) for sub-millisecond retrieval
    - Supports hot-reload: new index version can be loaded without restart
    - Maintains a write-ahead segment for streaming ingestion between rebuilds
    - LRU cache on query embeddings for high hit rates during trending events
    """

    def __init__(
        self,
        dim: int = 768,
        index_type: str = "IVFFlat",
        nlist: int = 100,
    ) -> None:
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist

        self._index: faiss.Index | None = None
        self._metadata: list[dict] = []

        self._wal_embeddings: list[np.ndarray] = []
        self._wal_metadata: list[dict] = []

        self._lock = threading.RLock()

    # --- index construction --------------------------------------------
    def _make_index(self, num_vectors: int) -> faiss.Index:
        if self.index_type == "IVFFlat" and num_vectors >= self.nlist * 4:
            effective_nlist = max(1, min(self.nlist, num_vectors // 4))
            quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, effective_nlist)
            # Search all partitions for tests; in prod, callers can lower this.
            index.nprobe = effective_nlist
            return index
        # Fall back for "Flat" or for IVFFlat with too little training data.
        return faiss.IndexFlatL2(self.dim)

    def build_index(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Build or rebuild the FAISS index from a batch of embeddings."""
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"embeddings must be (N, {self.dim}), got shape {tuple(embeddings.shape)}"
            )
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"metadata length {len(metadata)} does not match {embeddings.shape[0]} embeddings"
            )

        index = self._make_index(embeddings.shape[0])
        if not index.is_trained:
            index.train(embeddings)
        index.add(embeddings)

        with self._lock:
            self._index = index
            self._metadata = list(metadata)

    # --- search ---------------------------------------------------------
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        """Return the top-k nearest neighbors with metadata and squared-L2 distances.

        Searches both the main index and the write-ahead segment, then merges.
        """
        q = np.ascontiguousarray(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} does not match index dim {self.dim}")

        with self._lock:
            index = self._index
            metadata = self._metadata
            wal_embs = list(self._wal_embeddings)
            wal_meta = list(self._wal_metadata)

        scored: list[tuple[float, dict]] = []

        if index is not None and index.ntotal > 0:
            top = min(k, index.ntotal)
            distances, ids = index.search(q, top)
            for dist, idx in zip(distances[0], ids[0]):
                if idx < 0:
                    continue
                row = dict(metadata[idx])
                row["distance"] = float(dist)
                scored.append((float(dist), row))

        if wal_embs:
            wal_arr = np.asarray(wal_embs, dtype=np.float32)
            diffs = wal_arr - q
            wal_dists = (diffs * diffs).sum(axis=1)
            for dist, md in zip(wal_dists, wal_meta):
                row = dict(md)
                row["distance"] = float(dist)
                scored.append((float(dist), row))

        scored.sort(key=lambda pair: pair[0])
        return [row for _, row in scored[:k]]

    # --- hot reload -----------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the main index and a metadata sidecar at ``path`` and ``path + '.meta'``."""
        with self._lock:
            if self._index is None:
                raise RuntimeError("no index built; call build_index() first")
            faiss.write_index(self._index, path)
            with open(path + ".meta", "wb") as f:
                pickle.dump(self._metadata, f)

    def hot_reload(self, new_index_path: str) -> None:
        """Load a new index version without downtime. Swap atomically."""
        new_index = faiss.read_index(new_index_path)
        if new_index.d != self.dim:
            raise ValueError(f"index dim {new_index.d} does not match {self.dim}")

        meta_path = new_index_path + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                new_metadata = pickle.load(f)
        else:
            new_metadata = []

        with self._lock:
            self._index = new_index
            self._metadata = list(new_metadata)

    # --- write-ahead segment -------------------------------------------
    def append_to_wal(self, embedding: np.ndarray, metadata: dict) -> None:
        """Add to write-ahead segment (searched alongside the main index)."""
        emb = np.ascontiguousarray(np.asarray(embedding, dtype=np.float32).reshape(-1))
        if emb.shape[0] != self.dim:
            raise ValueError(f"embedding dim {emb.shape[0]} does not match {self.dim}")
        with self._lock:
            self._wal_embeddings.append(emb)
            self._wal_metadata.append(dict(metadata))

    def merge_wal(self) -> None:
        """Merge the write-ahead segment into the main index, then clear it."""
        with self._lock:
            if not self._wal_embeddings:
                return
            wal_arr = np.asarray(self._wal_embeddings, dtype=np.float32)
            wal_meta = list(self._wal_metadata)
            self._wal_embeddings.clear()
            self._wal_metadata.clear()

            if self._index is None:
                # No main index yet — initialise from WAL.
                self.build_index(wal_arr, wal_meta)
                return

            self._index.add(wal_arr)
            self._metadata.extend(wal_meta)

    # --- stats ----------------------------------------------------------
    def get_index_stats(self) -> dict:
        with self._lock:
            main_total = self._index.ntotal if self._index is not None else 0
            wal_size = len(self._wal_embeddings)
            return {
                "total_vectors": main_total + wal_size,
                "wal_size": wal_size,
                "index_type": self.index_type,
                "memory_bytes": (main_total + wal_size) * self.dim * 4,
            }


class CachedRetriever:
    """LRU cache on LSH-bucketed query embeddings.

    During trending events, cache hit rate can exceed 80%.
    """

    def __init__(
        self,
        index_manager: FAISSIndexManager,
        cache_size: int = 10000,
        lsh_bits: int = 16,
        seed: int = 42,
    ) -> None:
        self.index = index_manager
        self.cache_size = cache_size
        self.lsh_bits = lsh_bits

        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((lsh_bits, index_manager.dim)).astype(np.float32)

        self._cache: OrderedDict[tuple[int, bytes], list[dict]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _bucket_key(self, query: np.ndarray, k: int) -> tuple[int, bytes]:
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        bits = (self._proj @ q) > 0
        return (k, np.packbits(bits.astype(np.uint8)).tobytes())

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        key = self._bucket_key(query_embedding, k)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self._hits += 1
            return cached

        self._misses += 1
        result = self.index.search(query_embedding, k)
        self._cache[key] = result
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def cache_stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total > 0 else 0.0,
        }
