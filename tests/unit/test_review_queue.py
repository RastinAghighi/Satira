from datetime import datetime, timedelta

from satira.graph.schema import ContentNode, EdgeType, EntityNode
from satira.graph.store import GraphStore
from satira.moderator.review_queue import (
    ReviewItem,
    ReviewQueueManager,
)


def _entity(eid: str, name: str = "X") -> EntityNode:
    return EntityNode(
        id=eid,
        canonical_name=name,
        entity_type="person",
        aliases=[],
        created_at=datetime(2026, 1, 1),
    )


def _make_item(
    iid: str,
    text: str = "alice smith",
    candidates: list[tuple[str, float]] | None = None,
    affected: int = 0,
    embedding_impact: float = 0.0,
    created_at: datetime | None = None,
    auto_resolve_at: datetime | None = None,
) -> ReviewItem:
    cands = candidates if candidates is not None else [("e1", 0.75)]
    return ReviewItem(
        id=iid,
        mention_text=text,
        candidate_entities=cands,
        similarity_score=cands[0][1] if cands else 0.0,
        created_at=created_at or datetime.utcnow(),
        auto_resolve_at=auto_resolve_at,
        affected_content_count=affected,
        embedding_impact=embedding_impact,
    )


# --- push + get_next_cluster ------------------------------------------
def test_push_and_get_next_cluster_returns_highest_priority() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1"))
    store.add_entity(_entity("e2"))
    mgr = ReviewQueueManager(store)

    low = _make_item("low", text="bob", candidates=[("e2", 0.75)], affected=1)
    high = _make_item("high", text="alice", candidates=[("e1", 0.75)], affected=200, embedding_impact=0.9)
    mgr.push(low)
    mgr.push(high)

    cluster = mgr.get_next_cluster()
    assert cluster is not None
    assert cluster.representative.id == "high"


def test_get_next_cluster_returns_none_when_empty() -> None:
    mgr = ReviewQueueManager(GraphStore())
    assert mgr.get_next_cluster() is None


# --- resolve_cluster --------------------------------------------------
def test_resolve_cluster_applies_to_all_members() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1"))
    mgr = ReviewQueueManager(store)

    for i in range(3):
        mgr.push(_make_item(f"m{i}", text=f"alice smith {i}", candidates=[("e1", 0.75)]))

    cluster = mgr.get_next_cluster()
    assert cluster is not None
    assert len(cluster.members) == 3

    resolved = mgr.resolve_cluster(cluster.cluster_id, action="merge", target_entity="e1")
    assert resolved == 3
    assert mgr.stats()["queue_depth"] == 0
    actions = [r["action"] for r in mgr.resolutions]
    targets = [r["target_entity"] for r in mgr.resolutions]
    assert actions == ["merge", "merge", "merge"]
    assert all(t == "e1" for t in targets)


def test_resolve_cluster_rejects_invalid_action() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1"))
    mgr = ReviewQueueManager(store)
    mgr.push(_make_item("m0"))
    cluster = mgr.get_next_cluster()
    assert cluster is not None

    try:
        mgr.resolve_cluster(cluster.cluster_id, action="banana")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid action")


# --- process_stale_items ----------------------------------------------
def test_process_stale_items_auto_resolves_old_items() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1"))
    mgr = ReviewQueueManager(store)

    now = datetime.utcnow()
    fresh = _make_item("fresh", created_at=now, auto_resolve_at=now + timedelta(minutes=30))
    stale = _make_item(
        "stale",
        created_at=now - timedelta(minutes=45),
        auto_resolve_at=now - timedelta(minutes=15),
    )
    mgr.push(fresh)
    mgr.push(stale)

    n = mgr.process_stale_items()
    assert n == 1
    assert mgr.stats()["queue_depth"] == 1
    assert len(mgr.deferred_review) == 1
    assert mgr.deferred_review[0]["item_id"] == "stale"


def test_process_stale_items_returns_zero_when_nothing_expired() -> None:
    mgr = ReviewQueueManager(GraphStore())
    mgr.push(_make_item("x", candidates=[]))
    assert mgr.process_stale_items() == 0


# --- clustering -------------------------------------------------------
def test_clustering_groups_similar_mentions_together() -> None:
    store = GraphStore()
    store.add_entity(_entity("e_alice"))
    store.add_entity(_entity("e_bob"))
    mgr = ReviewQueueManager(store)

    for i in range(5):
        mgr.push(_make_item(f"a{i}", text=f"alice s. {i}", candidates=[("e_alice", 0.7)]))
    for i in range(2):
        mgr.push(_make_item(f"b{i}", text=f"bob jones {i}", candidates=[("e_bob", 0.7)]))

    clusters = mgr.cluster_pending_items()
    assert len(clusters) == 2
    sizes = sorted(len(c.members) for c in clusters)
    assert sizes == [2, 5]


def test_clustering_separates_unrelated_mentions() -> None:
    store = GraphStore()
    for eid in ("e1", "e2", "e3"):
        store.add_entity(_entity(eid))
    mgr = ReviewQueueManager(store)

    mgr.push(_make_item("m1", text="alpha", candidates=[("e1", 0.7)]))
    mgr.push(_make_item("m2", text="beta", candidates=[("e2", 0.7)]))
    mgr.push(_make_item("m3", text="gamma", candidates=[("e3", 0.7)]))

    clusters = mgr.cluster_pending_items()
    assert len(clusters) == 3


# --- priority scoring -------------------------------------------------
def test_high_impact_items_score_higher_than_low_impact() -> None:
    store = GraphStore()
    store.add_entity(_entity("e_high"))
    store.add_entity(_entity("e_low"))
    mgr = ReviewQueueManager(store)

    high = _make_item(
        "high",
        candidates=[("e_high", 0.75), ("e_low", 0.74)],  # very ambiguous
        affected=200,
        embedding_impact=0.9,
    )
    low = _make_item(
        "low",
        candidates=[("e_low", 0.75)],
        affected=1,
        embedding_impact=0.0,
    )
    mgr.push(high)
    mgr.push(low)

    assert high.priority_score > low.priority_score


def test_priority_uses_graph_affected_content_when_count_unset() -> None:
    store = GraphStore()
    store.add_entity(_entity("e_busy"))
    store.add_entity(_entity("e_quiet"))
    # Wire 30 content nodes mentioning e_busy
    for i in range(30):
        c = ContentNode(
            id=f"c{i}",
            image_hash=f"h{i}",
            extracted_text="t",
            timestamp=datetime(2026, 1, 1),
            source_id=None,
        )
        store.add_content(c)
        store.add_edge(c.id, "e_busy", EdgeType.MENTIONS)

    mgr = ReviewQueueManager(store)
    busy = _make_item("busy", candidates=[("e_busy", 0.75)])
    quiet = _make_item("quiet", candidates=[("e_quiet", 0.75)])
    mgr.push(busy)
    mgr.push(quiet)

    assert busy.affected_content_count == 30
    assert quiet.affected_content_count == 0
    assert busy.priority_score > quiet.priority_score


def test_stats_reports_queue_depth_and_auto_resolve_rate() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1"))
    mgr = ReviewQueueManager(store)

    now = datetime.utcnow()
    mgr.push(_make_item("a", auto_resolve_at=now - timedelta(minutes=1)))
    mgr.push(_make_item("b"))
    mgr.process_stale_items()

    cluster = mgr.get_next_cluster()
    assert cluster is not None
    mgr.resolve_cluster(cluster.cluster_id, action="merge", target_entity="e1")

    s = mgr.stats()
    assert s["queue_depth"] == 0
    assert 0.0 < s["auto_resolve_rate"] < 1.0
    assert s["avg_resolution_time"] >= 0.0
    assert s["clusters_pending"] == 0
