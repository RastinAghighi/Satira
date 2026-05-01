from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone

import torch


class GraphEmbeddingCache:
    """Stores precomputed GNN embeddings for all graph nodes.

    Supports versioned snapshots for compatibility tracking. At inference
    time the model does not run the GNN — it looks up precomputed
    embeddings here in sub-millisecond time.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        self.embedding_dim = embedding_dim
        self._embeddings: dict[str, torch.Tensor] = {}

    # --- read / write ---------------------------------------------------
    def set(self, node_id: str, embedding: torch.Tensor) -> None:
        if embedding.dim() != 1 or embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"embedding must be 1-D of size {self.embedding_dim}, got shape {tuple(embedding.shape)}"
            )
        self._embeddings[node_id] = embedding.detach().clone()

    def get(self, node_id: str) -> torch.Tensor | None:
        return self._embeddings.get(node_id)

    def mget(self, node_ids: list[str]) -> list[torch.Tensor | None]:
        return [self._embeddings.get(nid) for nid in node_ids]

    def __len__(self) -> int:
        return len(self._embeddings)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._embeddings

    # --- pooling --------------------------------------------------------
    def attention_pool(
        self,
        embeddings: list[torch.Tensor],
        weights: list[float],
    ) -> torch.Tensor:
        """Weighted attention pool over retrieved embeddings.

        Weights typically come from ``EntityResolutionResult.graph_weight``.
        Returns the normalized weighted sum: ``sum(w_i / sum(w) * e_i)``.
        Entries with zero weight contribute nothing; if every weight is
        zero (or no embeddings are provided), a zero vector is returned.
        """
        if len(embeddings) != len(weights):
            raise ValueError(
                f"embeddings and weights length mismatch: {len(embeddings)} vs {len(weights)}"
            )
        if not embeddings:
            return torch.zeros(self.embedding_dim)

        stacked = torch.stack([e.detach() for e in embeddings], dim=0)
        w = torch.tensor(weights, dtype=stacked.dtype)
        total = w.sum()
        if total <= 0:
            return torch.zeros(self.embedding_dim, dtype=stacked.dtype)
        normalized = w / total
        return (stacked * normalized.unsqueeze(-1)).sum(dim=0)

    # --- snapshots ------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "embeddings": {nid: emb.detach().clone() for nid, emb in self._embeddings.items()},
            "stats": self.compute_distribution_stats(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def load_snapshot(self, snapshot: dict) -> None:
        dim = snapshot.get("embedding_dim", self.embedding_dim)
        self.embedding_dim = dim
        self._embeddings = {
            nid: emb.detach().clone() for nid, emb in snapshot["embeddings"].items()
        }

    def compute_distribution_stats(self) -> dict:
        if not self._embeddings:
            return {
                "count": 0,
                "mean": torch.zeros(self.embedding_dim),
                "std": torch.zeros(self.embedding_dim),
                "norm_mean": 0.0,
                "norm_std": 0.0,
                "top_singular_values": torch.zeros(0),
            }

        stacked = torch.stack(list(self._embeddings.values()), dim=0)
        norms = stacked.norm(dim=1)
        if stacked.shape[0] >= 2:
            std = stacked.std(dim=0, unbiased=False)
            norm_std = float(norms.std(unbiased=False).item())
        else:
            std = torch.zeros(self.embedding_dim, dtype=stacked.dtype)
            norm_std = 0.0

        try:
            sv = torch.linalg.svdvals(stacked)
        except RuntimeError:
            sv = torch.zeros(0)
        top_k = sv[: min(10, sv.shape[0])]

        return {
            "count": stacked.shape[0],
            "mean": stacked.mean(dim=0),
            "std": std,
            "norm_mean": float(norms.mean().item()),
            "norm_std": norm_std,
            "top_singular_values": top_k,
        }


class GraphEmbeddingVersionStore:
    """Maintains rolling snapshots for temporally augmented training.

    Stores recent snapshots so the trainer can sample older graph states
    and build robustness to GNN drift.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, dict] = {}

    def save_snapshot(self, embeddings: dict, gnn_version: str) -> str:
        version_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        self._snapshots[version_id] = {
            "version_id": version_id,
            "gnn_version": gnn_version,
            "saved_at": datetime.now(timezone.utc),
            "embeddings": {
                nid: emb.detach().clone() if isinstance(emb, torch.Tensor) else emb
                for nid, emb in embeddings.items()
            },
        }
        return version_id

    def load_snapshot(self, version_id: str) -> dict:
        if version_id not in self._snapshots:
            raise KeyError(f"unknown snapshot version {version_id!r}")
        return self._snapshots[version_id]

    def list_versions(self, last_n_days: int = 21) -> list[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=last_n_days)
        recent = [
            (snap["saved_at"], vid)
            for vid, snap in self._snapshots.items()
            if snap["saved_at"] >= cutoff
        ]
        recent.sort(key=lambda pair: pair[0])
        return [vid for _, vid in recent]
