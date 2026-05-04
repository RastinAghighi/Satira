"""Post-surge correction pipeline.

After a surge subsides, moderators work through the deferred review
backlog. Confirmed decisions restore full confidence; incorrect merges
trigger entity splits and targeted re-classification of content where
the graph context meaningfully shaped the original prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from satira.graph.embedding_cache import GraphEmbeddingCache
from satira.graph.schema import EdgeType, EntityNode
from satira.graph.store import GraphStore
from satira.moderator.review_queue import ReviewItem


@dataclass
class _ClassificationRecord:
    content_id: str
    mentioned_entities: list[str]
    graph_contribution: float


class CorrectionPipeline:
    """Apply moderator corrections and surgically re-queue affected content.

    Re-classification is gated on two signals: the graph context must
    have contributed more than ``GRAPH_CONTRIB_THRESHOLD`` to the
    original prediction, and the embedding the content effectively saw
    must have moved by more than ``EMBEDDING_DELTA_THRESHOLD``. Content
    where the graph was a passenger isn't worth re-scoring.
    """

    GRAPH_CONTRIB_THRESHOLD = 0.20
    EMBEDDING_DELTA_THRESHOLD = 0.1

    VALID_ACTIONS = ("merge", "split", "confirm")

    def __init__(
        self,
        graph_store: GraphStore,
        embedding_cache: GraphEmbeddingCache,
    ) -> None:
        self.graph_store = graph_store
        self.embedding_cache = embedding_cache
        self._deferred_backlog: dict[str, ReviewItem] = {}
        self._reclassification_queue: list[str] = []
        self._reclass_seen: set[str] = set()
        self._classifications: dict[str, _ClassificationRecord] = {}

    # --- registration ---------------------------------------------------
    def register_deferred(self, item: ReviewItem) -> None:
        """Track an auto-resolved item awaiting human confirmation."""
        self._deferred_backlog[item.id] = item

    def record_classification(
        self,
        content_id: str,
        mentioned_entities: list[str],
        graph_contribution: float,
    ) -> None:
        """Record per-prediction graph attribution for later re-queuing."""
        self._classifications[content_id] = _ClassificationRecord(
            content_id=content_id,
            mentioned_entities=list(mentioned_entities),
            graph_contribution=graph_contribution,
        )

    # --- main entrypoint ------------------------------------------------
    def apply_correction(self, correction: dict) -> dict:
        action = correction.get("action")
        if action not in self.VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {self.VALID_ACTIONS}, got {action!r}"
            )

        if action == "merge":
            result = self._apply_merge(correction)
        elif action == "split":
            result = self._apply_split(correction)
        else:
            result = self._apply_confirm(correction)

        deferred_id = correction.get("deferred_item_id")
        if deferred_id is not None:
            self._deferred_backlog.pop(deferred_id, None)

        return result

    # --- queue accessors ------------------------------------------------
    def get_deferred_backlog(self) -> list[ReviewItem]:
        return list(self._deferred_backlog.values())

    def get_reclassification_queue(self) -> list[str]:
        return list(self._reclassification_queue)

    # --- per-action handlers --------------------------------------------
    def _apply_merge(self, correction: dict) -> dict:
        source = correction["source_entity"]
        target = correction["target_entity"]
        new_embeddings: dict[str, torch.Tensor] = correction.get("new_embeddings", {})

        prev_source = self._snapshot(source)
        prev_target = self._snapshot(target)
        source_content = self._content_mentioning([source])
        target_content = self._content_mentioning([target])

        affected_nodes = self.graph_store.merge_entities(source, target)

        # Source entity is gone — drop its cache entry too.
        self.embedding_cache._embeddings.pop(source, None)
        if target in new_embeddings:
            self.embedding_cache.set(target, new_embeddings[target])
        new_target = self.embedding_cache.get(target)

        queued = 0
        # Content that mentioned source now resolves to target — their
        # effective embedding shift is target_new vs source_old.
        queued += self._enqueue_if_shifted(source_content, source, prev_source, new_target)
        # Content that mentioned target sees target_new vs target_old.
        queued += self._enqueue_if_shifted(target_content, target, prev_target, new_target)

        return {
            "affected_nodes": len(affected_nodes),
            "reclassifications_queued": queued,
        }

    def _apply_split(self, correction: dict) -> dict:
        from_entity = correction["from_entity"]
        new_entity: EntityNode = correction["new_entity"]
        content_to_reassign = list(correction.get("content_to_reassign", []))
        new_embeddings: dict[str, torch.Tensor] = correction.get("new_embeddings", {})

        prev_from = self._snapshot(from_entity)
        affected: set[str] = {from_entity, new_entity.id}

        self.graph_store.add_entity(new_entity)
        for cid in content_to_reassign:
            edges = self.graph_store._out.get(cid, set())
            if (from_entity, EdgeType.MENTIONS) not in edges:
                continue
            self.graph_store._out[cid].discard((from_entity, EdgeType.MENTIONS))
            self.graph_store._in[from_entity].discard((cid, EdgeType.MENTIONS))
            self.graph_store.add_edge(cid, new_entity.id, EdgeType.MENTIONS)
            affected.add(cid)

        for eid, emb in new_embeddings.items():
            self.embedding_cache.set(eid, emb)

        new_from = self.embedding_cache.get(from_entity)
        new_split = self.embedding_cache.get(new_entity.id)

        # Reassigned content now resolves to the new split-off entity.
        reassigned = set(content_to_reassign)
        # Content that stayed on from_entity sees from_new vs from_old.
        remaining = self._content_mentioning([from_entity]) - reassigned

        queued = 0
        queued += self._enqueue_if_shifted(reassigned, from_entity, prev_from, new_split)
        queued += self._enqueue_if_shifted(remaining, from_entity, prev_from, new_from)

        return {
            "affected_nodes": len(affected),
            "reclassifications_queued": queued,
        }

    def _apply_confirm(self, _correction: dict) -> dict:
        # Confirmation restores full confidence — no graph mutation,
        # nothing to re-score.
        return {"affected_nodes": 0, "reclassifications_queued": 0}

    # --- internals ------------------------------------------------------
    def _snapshot(self, entity_id: str) -> torch.Tensor | None:
        emb = self.embedding_cache.get(entity_id)
        return emb.clone() if emb is not None else None

    def _content_mentioning(self, entity_ids: list[str]) -> set[str]:
        out: set[str] = set()
        for eid in entity_ids:
            for src_id, et in self.graph_store._in.get(eid, set()):
                if et == EdgeType.MENTIONS:
                    out.add(src_id)
        return out

    def _enqueue_if_shifted(
        self,
        content_ids: set[str],
        original_entity: str,
        prev_emb: torch.Tensor | None,
        new_emb: torch.Tensor | None,
    ) -> int:
        if not self._delta_significant(prev_emb, new_emb):
            return 0
        queued = 0
        for cid in content_ids:
            rec = self._classifications.get(cid)
            if rec is None or rec.graph_contribution <= self.GRAPH_CONTRIB_THRESHOLD:
                continue
            if original_entity not in rec.mentioned_entities:
                continue
            if cid in self._reclass_seen:
                continue
            self._reclass_seen.add(cid)
            self._reclassification_queue.append(cid)
            queued += 1
        return queued

    def _delta_significant(
        self,
        prev: torch.Tensor | None,
        new: torch.Tensor | None,
    ) -> bool:
        # Without before/after we can't measure — be conservative and
        # treat the change as significant rather than silently skipping
        # potentially-affected content.
        if prev is None or new is None:
            return True
        return torch.linalg.norm(new - prev).item() >= self.EMBEDDING_DELTA_THRESHOLD
