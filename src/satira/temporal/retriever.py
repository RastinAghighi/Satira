from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import torch

from satira.temporal.index_manager import CachedRetriever, FAISSIndexManager


class TemporalContextRetriever:
    """Async wrapper around FAISS retrieval with timeout and fallback.

    This is called in PARALLEL with vision and text encoding.
    If retrieval takes longer than timeout_ms, falls back to a
    learned default embedding. The model is trained with 15% temporal
    dropout to handle this gracefully.
    """

    def __init__(
        self,
        index_manager: FAISSIndexManager,
        default_embedding: torch.Tensor | None = None,
        timeout_ms: float = 80.0,
        top_k: int = 5,
        text_encoder: Any = None,
    ) -> None:
        self.index_manager = index_manager
        self.cached_retriever = CachedRetriever(index_manager)
        self.timeout_ms = timeout_ms
        self.top_k = top_k
        self.text_encoder = text_encoder

        if default_embedding is None:
            default_embedding = torch.zeros(index_manager.dim, dtype=torch.float32)
        elif default_embedding.dim() != 1 or default_embedding.shape[0] != index_manager.dim:
            raise ValueError(
                f"default_embedding must be 1-D of size {index_manager.dim}, "
                f"got shape {tuple(default_embedding.shape)}"
            )
        self.default_embedding = default_embedding.detach().clone()

    async def retrieve(
        self,
        query_text: str,
        text_encoder: Any = None,
    ) -> torch.Tensor:
        """Encode the query, search the index, and return a mean-pooled context vector.

        On timeout returns ``default_embedding`` instead.
        """
        encoder = text_encoder if text_encoder is not None else self.text_encoder
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._sync_retrieve, query_text, encoder),
                timeout=self.timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            return self.default_embedding.clone()

    async def retrieve_with_timeout(
        self,
        query_text: str,
    ) -> tuple[torch.Tensor, bool]:
        """Like ``retrieve`` but also reports whether the LRU cache served the request."""
        hits_before = self.cached_retriever.cache_stats()["hits"]
        embedding = await self.retrieve(query_text)
        hits_after = self.cached_retriever.cache_stats()["hits"]
        return embedding, hits_after > hits_before

    # --- internals ------------------------------------------------------
    def _sync_retrieve(self, query_text: str, encoder: Any) -> torch.Tensor:
        if encoder is None:
            raise ValueError("a text_encoder is required (pass to __init__ or retrieve())")

        query_np = self._encode_query(query_text, encoder)
        results = self.cached_retriever.retrieve(query_np, k=self.top_k)
        return self._pool(results)

    def _encode_query(self, query_text: str, encoder: Any) -> np.ndarray:
        emb = encoder.encode(query_text)
        if isinstance(emb, torch.Tensor):
            arr = emb.detach().cpu().numpy()
        else:
            arr = np.asarray(emb)
        arr = np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)
        if arr.shape[0] != self.index_manager.dim:
            raise ValueError(
                f"encoded query dim {arr.shape[0]} does not match "
                f"index dim {self.index_manager.dim}"
            )
        return arr

    def _pool(self, results: list[dict]) -> torch.Tensor:
        embeddings = [r["embedding"] for r in results if "embedding" in r]
        if not embeddings:
            return self.default_embedding.clone()
        arr = np.stack(
            [np.asarray(e, dtype=np.float32).reshape(-1) for e in embeddings],
            axis=0,
        )
        pooled = arr.mean(axis=0)
        return torch.from_numpy(np.ascontiguousarray(pooled))
