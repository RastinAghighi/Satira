from datetime import datetime

from satira.graph.batch_resolver import BatchResolver, MergeTracker
from satira.graph.schema import EntityNode
from satira.graph.store import GraphStore


def _entity(eid: str, name: str, etype: str = "person", aliases: list[str] | None = None) -> EntityNode:
    return EntityNode(
        id=eid,
        canonical_name=name,
        entity_type=etype,
        aliases=list(aliases or []),
        created_at=datetime(2026, 1, 1),
    )


# --- merge / review / create thresholds ---------------------------------
def test_high_similarity_mention_gets_merge_action() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1", "Alice Smith"))
    resolver = BatchResolver(
        store, entity_embeddings={"e1": [1.0, 0.0, 0.0]}
    )

    decisions = resolver.resolve_batch([
        {
            "text": "Alice Smith",
            "embedding": [1.0, 0.0, 0.0],
            "entity_type": "person",
            "cooccurring_entity_ids": ["e1"],
        }
    ])

    assert len(decisions) == 1
    d = decisions[0]
    assert d["action"] == "merge"
    assert d["target_entity"] == "e1"
    assert d["score"] >= 0.85


def test_low_similarity_mention_gets_create_action() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1", "Alice", etype="person"))
    resolver = BatchResolver(store)  # no entity embeddings

    decisions = resolver.resolve_batch([
        {
            "text": "Zorblax Quux",
            "embedding": [0.0, 0.0, 1.0],
            "entity_type": "location",
            "cooccurring_entity_ids": [],
        }
    ])

    d = decisions[0]
    assert d["action"] == "create"
    assert d["target_entity"] is None
    assert d["score"] < 0.65


def test_mid_similarity_mention_gets_review_action() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1", "Alice Smith", etype="person"))
    resolver = BatchResolver(
        store, entity_embeddings={"e1": [1.0, 0.0, 0.0]}
    )

    # alice smyth: edit distance 1 from "alice smith" → string ~0.91
    # embedding [0.8,0.6,0] · [1,0,0] = 0.8
    # type matches → 1.0; no cooccurrence → 0
    # score = 0.4*0.8 + 0.25*0.91 + 0.2*0 + 0.15*1 ≈ 0.69
    decisions = resolver.resolve_batch([
        {
            "text": "Alice Smyth",
            "embedding": [0.8, 0.6, 0.0],
            "entity_type": "person",
        }
    ])

    d = decisions[0]
    assert d["action"] == "review"
    assert d["target_entity"] == "e1"
    assert 0.65 <= d["score"] < 0.85


# --- blocking reduces comparisons ---------------------------------------
def test_blocking_reduces_comparison_count_vs_naive() -> None:
    store = GraphStore()
    distinct_first_tokens = [
        "alpha", "beta", "gamma", "delta", "epsilon",
        "zeta", "eta", "theta", "iota", "kappa",
    ]
    for i, tok in enumerate(distinct_first_tokens):
        store.add_entity(_entity(f"e{i}", f"{tok.capitalize()} Surname"))

    resolver = BatchResolver(store)

    mentions = [
        {"text": "alpha foo"},
        {"text": "beta bar"},
        {"text": "gamma baz"},
    ]
    resolver.resolve_batch(mentions)

    naive = resolver.last_naive_comparison_count
    blocked = resolver.last_comparison_count
    assert naive == 3 * 10
    assert blocked < naive
    # Each mention should hit just its first-token bucket → 1 candidate each
    assert blocked == 3


def test_blocking_unions_strategies() -> None:
    store = GraphStore()
    store.add_entity(_entity("e_ft", "Alpha Person"))
    store.add_entity(_entity("e_cooc", "Beta Person"))
    resolver = BatchResolver(store)

    mentions = [{
        "text": "alpha individual",  # FirstToken hits e_ft
        "cooccurring_entity_ids": ["e_cooc"],  # cooc hits e_cooc
    }]
    resolver.resolve_batch(mentions)

    assert resolver.last_comparison_count == 2  # both e_ft and e_cooc considered


# --- MergeTracker --------------------------------------------------------
def test_merge_tracker_records_source_target_and_edges() -> None:
    tracker = MergeTracker()
    tracker.record_merge(
        source_id="e_dup",
        target_id="e_keep",
        affected_edges=[("c1", "e_dup", "mentions"), ("c2", "e_dup", "mentions")],
    )

    affected = tracker.get_priority_recompute_set()
    assert "e_dup" in affected
    assert "e_keep" in affected
    assert "c1" in affected
    assert "c2" in affected


def test_merge_tracker_accepts_plain_string_node_ids() -> None:
    tracker = MergeTracker()
    tracker.record_merge("a", "b", ["c", "d"])
    assert tracker.get_priority_recompute_set() == {"a", "b", "c", "d"}


def test_merge_tracker_accumulates_across_multiple_merges() -> None:
    tracker = MergeTracker()
    tracker.record_merge("s1", "t1", ["n1"])
    tracker.record_merge("s2", "t2", ["n2"])
    assert tracker.get_priority_recompute_set() == {"s1", "t1", "n1", "s2", "t2", "n2"}


def test_merge_tracker_clear_resets_state() -> None:
    tracker = MergeTracker()
    tracker.record_merge("s", "t", ["x"])
    tracker.clear()
    assert tracker.get_priority_recompute_set() == set()


# --- empty input ---------------------------------------------------------
def test_resolve_empty_batch_returns_empty_list() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1", "Alice"))
    resolver = BatchResolver(store)
    assert resolver.resolve_batch([]) == []
    assert resolver.last_comparison_count == 0
