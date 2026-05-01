import json
from datetime import datetime

import pytest

from satira.graph.schema import (
    ContentNode,
    EdgeType,
    EntityNode,
    EventNode,
    SourceNode,
    TemplateNode,
)
from satira.graph.store import GraphStore


def _entity(eid: str, name: str = "", aliases: list[str] | None = None) -> EntityNode:
    return EntityNode(
        id=eid,
        canonical_name=name or eid,
        entity_type="person",
        aliases=list(aliases or []),
        created_at=datetime(2026, 1, 1, 12, 0, 0),
    )


def _content(cid: str, source_id: str | None = None) -> ContentNode:
    return ContentNode(
        id=cid,
        image_hash=f"hash-{cid}",
        extracted_text=f"text {cid}",
        timestamp=datetime(2026, 2, 3, 4, 5, 6),
        source_id=source_id,
    )


def _source(sid: str) -> SourceNode:
    return SourceNode(id=sid, domain=f"{sid}.com", account_id=None, credibility_label="news")


def _event(eid: str) -> EventNode:
    return EventNode(
        id=eid,
        topic_cluster="cluster-A",
        date_range=(datetime(2026, 1, 1), datetime(2026, 1, 31)),
    )


def _template(tid: str) -> TemplateNode:
    return TemplateNode(id=tid, perceptual_hash=f"phash-{tid}", layout_features=[0.1, 0.2, 0.3])


# --- add / get ----------------------------------------------------------
def test_add_and_get_entity_round_trips() -> None:
    store = GraphStore()
    e = _entity("e1", "Alice", ["A."])

    store.add_entity(e)

    fetched = store.get_entity("e1")
    assert fetched is e
    assert fetched.canonical_name == "Alice"
    assert fetched.aliases == ["A."]


def test_get_entity_returns_none_for_missing_id() -> None:
    store = GraphStore()
    assert store.get_entity("nope") is None


def test_get_entity_returns_none_for_non_entity_node() -> None:
    store = GraphStore()
    store.add_source(_source("s1"))
    assert store.get_entity("s1") is None


def test_add_entity_rejects_duplicate_id() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1"))
    with pytest.raises(ValueError):
        store.add_entity(_entity("e1"))


# --- edges / neighbors --------------------------------------------------
def test_add_edge_and_get_neighbors_returns_targets() -> None:
    store = GraphStore()
    store.add_content(_content("c1"))
    store.add_entity(_entity("e1"))
    store.add_entity(_entity("e2"))

    store.add_edge("c1", "e1", EdgeType.MENTIONS)
    store.add_edge("c1", "e2", EdgeType.MENTIONS)

    assert sorted(store.get_neighbors("c1")) == ["e1", "e2"]


def test_get_neighbors_filters_by_edge_type() -> None:
    store = GraphStore()
    store.add_content(_content("c1"))
    store.add_entity(_entity("e1"))
    store.add_source(_source("s1"))

    store.add_edge("c1", "e1", EdgeType.MENTIONS)
    store.add_edge("c1", "s1", EdgeType.POSTED_BY)

    assert store.get_neighbors("c1", EdgeType.MENTIONS) == ["e1"]
    assert store.get_neighbors("c1", EdgeType.POSTED_BY) == ["s1"]


def test_get_neighbors_unknown_node_returns_empty() -> None:
    store = GraphStore()
    assert store.get_neighbors("ghost") == []


def test_add_edge_rejects_unknown_endpoints() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1"))
    with pytest.raises(KeyError):
        store.add_edge("e1", "missing", EdgeType.INVOLVED_IN)
    with pytest.raises(KeyError):
        store.add_edge("missing", "e1", EdgeType.INVOLVED_IN)


# --- merge --------------------------------------------------------------
def test_merge_redirects_inbound_edges_to_target() -> None:
    store = GraphStore()
    store.add_content(_content("c1"))
    store.add_entity(_entity("e_dup", "Alice"))
    store.add_entity(_entity("e_keep", "Alice Smith"))
    store.add_edge("c1", "e_dup", EdgeType.MENTIONS)

    affected = store.merge_entities("e_dup", "e_keep")

    assert store.get_entity("e_dup") is None
    assert store.get_neighbors("c1", EdgeType.MENTIONS) == ["e_keep"]
    assert "c1" in affected
    assert "e_keep" in affected


def test_merge_redirects_outbound_edges_to_target() -> None:
    store = GraphStore()
    store.add_entity(_entity("e_dup"))
    store.add_entity(_entity("e_keep"))
    store.add_event(_event("ev1"))
    store.add_edge("e_dup", "ev1", EdgeType.INVOLVED_IN)

    store.merge_entities("e_dup", "e_keep")

    assert store.get_neighbors("e_keep", EdgeType.INVOLVED_IN) == ["ev1"]
    assert store.get_entity("e_dup") is None


def test_merge_absorbs_aliases_and_canonical_name() -> None:
    store = GraphStore()
    store.add_entity(_entity("e_dup", "Bob R.", ["Bobby"]))
    store.add_entity(_entity("e_keep", "Robert", ["Rob"]))

    store.merge_entities("e_dup", "e_keep")

    keep = store.get_entity("e_keep")
    assert keep is not None
    assert "Rob" in keep.aliases
    assert "Bob R." in keep.aliases
    assert "Bobby" in keep.aliases
    assert keep.canonical_name == "Robert"


def test_merge_dedupes_parallel_edges_and_avoids_self_loops() -> None:
    store = GraphStore()
    store.add_content(_content("c1"))
    store.add_entity(_entity("e_dup"))
    store.add_entity(_entity("e_keep"))
    store.add_edge("c1", "e_dup", EdgeType.MENTIONS)
    store.add_edge("c1", "e_keep", EdgeType.MENTIONS)
    # an edge from dup -> keep would become a self-loop on keep after merge
    store.add_edge("e_dup", "e_keep", EdgeType.INVOLVED_IN)

    store.merge_entities("e_dup", "e_keep")

    assert store.get_neighbors("c1", EdgeType.MENTIONS) == ["e_keep"]
    assert "e_keep" not in store.get_neighbors("e_keep")


def test_merge_unknown_entity_raises() -> None:
    store = GraphStore()
    store.add_entity(_entity("e_keep"))
    with pytest.raises(KeyError):
        store.merge_entities("missing", "e_keep")


# --- snapshot / restore --------------------------------------------------
def test_snapshot_restore_round_trip_preserves_nodes_and_edges() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1", "Alice", ["A."]))
    store.add_source(_source("s1"))
    store.add_event(_event("ev1"))
    store.add_template(_template("t1"))
    store.add_content(_content("c1", source_id="s1"))
    store.add_edge("c1", "e1", EdgeType.MENTIONS)
    store.add_edge("c1", "s1", EdgeType.POSTED_BY)
    store.add_edge("c1", "ev1", EdgeType.REFERENCES)
    store.add_edge("c1", "t1", EdgeType.USES)
    store.add_edge("e1", "ev1", EdgeType.INVOLVED_IN)

    snap = store.snapshot()

    restored = GraphStore()
    restored.restore(snap)

    e1 = restored.get_entity("e1")
    assert e1 is not None
    assert e1.canonical_name == "Alice"
    assert e1.aliases == ["A."]
    assert e1.created_at == datetime(2026, 1, 1, 12, 0, 0)

    assert sorted(restored.get_neighbors("c1")) == sorted(["e1", "s1", "ev1", "t1"])
    assert restored.get_neighbors("c1", EdgeType.POSTED_BY) == ["s1"]
    assert restored.get_neighbors("e1", EdgeType.INVOLVED_IN) == ["ev1"]


def test_snapshot_is_json_serializable() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1", "Alice"))
    store.add_event(_event("ev1"))
    store.add_edge("e1", "ev1", EdgeType.INVOLVED_IN)

    snap = store.snapshot()

    encoded = json.dumps(snap)
    decoded = json.loads(encoded)
    assert decoded == snap


def test_restore_replaces_existing_state() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1"))
    snap = store.snapshot()

    store.add_entity(_entity("e2"))
    assert store.get_entity("e2") is not None

    store.restore(snap)
    assert store.get_entity("e2") is None
    assert store.get_entity("e1") is not None
