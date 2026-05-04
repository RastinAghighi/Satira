from datetime import datetime

import pytest
import torch

from satira.graph.embedding_cache import GraphEmbeddingCache
from satira.graph.schema import ContentNode, EdgeType, EntityNode
from satira.graph.store import GraphStore
from satira.moderator.corrections import CorrectionPipeline
from satira.moderator.review_queue import ReviewItem


EMBED_DIM = 4


def _entity(eid: str, name: str = "X") -> EntityNode:
    return EntityNode(
        id=eid,
        canonical_name=name,
        entity_type="person",
        aliases=[],
        created_at=datetime(2026, 1, 1),
    )


def _content(cid: str) -> ContentNode:
    return ContentNode(
        id=cid,
        image_hash=f"h-{cid}",
        extracted_text="t",
        timestamp=datetime(2026, 1, 1),
        source_id=None,
    )


def _build_pipeline() -> tuple[CorrectionPipeline, GraphStore, GraphEmbeddingCache]:
    store = GraphStore()
    cache = GraphEmbeddingCache(embedding_dim=EMBED_DIM)
    return CorrectionPipeline(store, cache), store, cache


# --- correction triggers graph update --------------------------------
def test_merge_correction_updates_graph_and_drops_source_embedding() -> None:
    pipeline, store, cache = _build_pipeline()
    store.add_entity(_entity("e1", "alice s"))
    store.add_entity(_entity("e2", "alice smith"))
    c = _content("c1")
    store.add_content(c)
    store.add_edge("c1", "e1", EdgeType.MENTIONS)

    cache.set("e1", torch.ones(EMBED_DIM))
    cache.set("e2", torch.ones(EMBED_DIM) * 2)

    result = pipeline.apply_correction(
        {
            "action": "merge",
            "source_entity": "e1",
            "target_entity": "e2",
        }
    )

    assert store.get_entity("e1") is None
    assert store.get_entity("e2") is not None
    # mention edge was rewired to the target
    assert "e2" in store.get_neighbors("c1", EdgeType.MENTIONS)
    # source embedding is purged
    assert cache.get("e1") is None
    assert result["affected_nodes"] >= 1


def test_split_correction_creates_entity_and_rewires_mentions() -> None:
    pipeline, store, cache = _build_pipeline()
    store.add_entity(_entity("e_merged", "ambiguous"))
    for cid in ("c1", "c2"):
        store.add_content(_content(cid))
        store.add_edge(cid, "e_merged", EdgeType.MENTIONS)

    new_entity = _entity("e_split", "disambiguated")
    result = pipeline.apply_correction(
        {
            "action": "split",
            "from_entity": "e_merged",
            "new_entity": new_entity,
            "content_to_reassign": ["c2"],
        }
    )

    assert store.get_entity("e_split") is not None
    assert store.get_neighbors("c1", EdgeType.MENTIONS) == ["e_merged"]
    assert store.get_neighbors("c2", EdgeType.MENTIONS) == ["e_split"]
    assert result["affected_nodes"] >= 2


# --- only high-graph-contribution content gets re-queued -------------
def test_reclassification_queue_excludes_low_graph_contribution() -> None:
    pipeline, store, cache = _build_pipeline()
    store.add_entity(_entity("e1"))
    store.add_entity(_entity("e2"))
    for cid in ("c_high", "c_low"):
        store.add_content(_content(cid))
        store.add_edge(cid, "e1", EdgeType.MENTIONS)

    cache.set("e1", torch.zeros(EMBED_DIM))
    cache.set("e2", torch.zeros(EMBED_DIM))

    # c_high relied heavily on the graph; c_low barely used it.
    pipeline.record_classification("c_high", ["e1"], graph_contribution=0.50)
    pipeline.record_classification("c_low", ["e1"], graph_contribution=0.05)

    # Provide an updated target embedding far from the prior one so the
    # delta clears the threshold.
    new_target = torch.ones(EMBED_DIM) * 10.0
    result = pipeline.apply_correction(
        {
            "action": "merge",
            "source_entity": "e1",
            "target_entity": "e2",
            "new_embeddings": {"e2": new_target},
        }
    )

    queue = pipeline.get_reclassification_queue()
    assert queue == ["c_high"]
    assert result["reclassifications_queued"] == 1


def test_no_reclassification_when_embedding_delta_below_threshold() -> None:
    pipeline, store, cache = _build_pipeline()
    store.add_entity(_entity("e1"))
    store.add_entity(_entity("e2"))
    store.add_content(_content("c1"))
    store.add_edge("c1", "e1", EdgeType.MENTIONS)

    base = torch.zeros(EMBED_DIM)
    cache.set("e1", base.clone())
    cache.set("e2", base.clone())

    pipeline.record_classification("c1", ["e1"], graph_contribution=0.80)

    # New embedding sits within EMBEDDING_DELTA_THRESHOLD of the prior one.
    nudged = base.clone()
    nudged[0] = pipeline.EMBEDDING_DELTA_THRESHOLD / 2.0
    pipeline.apply_correction(
        {
            "action": "merge",
            "source_entity": "e1",
            "target_entity": "e2",
            "new_embeddings": {"e2": nudged},
        }
    )

    assert pipeline.get_reclassification_queue() == []


def test_confirm_does_not_reclassify_or_mutate_graph() -> None:
    pipeline, store, cache = _build_pipeline()
    store.add_entity(_entity("e1"))
    store.add_content(_content("c1"))
    store.add_edge("c1", "e1", EdgeType.MENTIONS)
    pipeline.record_classification("c1", ["e1"], graph_contribution=0.9)

    result = pipeline.apply_correction({"action": "confirm"})

    assert result == {"affected_nodes": 0, "reclassifications_queued": 0}
    assert store.get_entity("e1") is not None
    assert pipeline.get_reclassification_queue() == []


# --- deferred backlog tracking ---------------------------------------
def test_deferred_backlog_is_returned_and_cleared_on_correction() -> None:
    pipeline, store, _cache = _build_pipeline()
    store.add_entity(_entity("e1"))
    store.add_entity(_entity("e2"))

    item_a = ReviewItem(id="r_a", mention_text="alice", candidate_entities=[("e1", 0.7)])
    item_b = ReviewItem(id="r_b", mention_text="bob", candidate_entities=[("e2", 0.7)])
    pipeline.register_deferred(item_a)
    pipeline.register_deferred(item_b)

    backlog = pipeline.get_deferred_backlog()
    assert {it.id for it in backlog} == {"r_a", "r_b"}

    pipeline.apply_correction(
        {
            "action": "merge",
            "source_entity": "e1",
            "target_entity": "e2",
            "deferred_item_id": "r_a",
        }
    )

    remaining = pipeline.get_deferred_backlog()
    assert [it.id for it in remaining] == ["r_b"]


def test_apply_correction_rejects_unknown_action() -> None:
    pipeline, _store, _cache = _build_pipeline()
    with pytest.raises(ValueError):
        pipeline.apply_correction({"action": "banana"})
