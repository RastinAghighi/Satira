import time
from datetime import datetime

import pytest

from satira.graph.entity_resolution import EntityResolutionResult, MentionNormalizer
from satira.graph.schema import EntityNode
from satira.graph.store import GraphStore


def _entity(eid: str, name: str, aliases: list[str] | None = None) -> EntityNode:
    return EntityNode(
        id=eid,
        canonical_name=name,
        entity_type="person",
        aliases=list(aliases or []),
        created_at=datetime(2026, 1, 1),
    )


def _normalizer_with(*entities: EntityNode) -> MentionNormalizer:
    store = GraphStore()
    for e in entities:
        store.add_entity(e)
    norm = MentionNormalizer()
    norm.load_from_graph(store)
    return norm


# --- exact lookup --------------------------------------------------------
def test_exact_alias_resolution_returns_full_confidence() -> None:
    norm = _normalizer_with(_entity("e1", "Alice Smith", ["A. Smith", "Allie"]))

    eid, score = norm.normalize("Alice Smith")
    assert eid == "e1"
    assert score == 1.0

    eid, score = norm.normalize("allie")
    assert eid == "e1"
    assert score == 1.0


def test_exact_lookup_is_case_and_whitespace_insensitive() -> None:
    norm = _normalizer_with(_entity("e1", "Bob Jones"))
    eid, score = norm.normalize("  bob jones  ")
    assert eid == "e1"
    assert score == 1.0


# --- fuzzy lookup --------------------------------------------------------
def test_fuzzy_match_resolves_single_typo() -> None:
    norm = _normalizer_with(_entity("e1", "Catherine Pemberton"))
    eid, score = norm.normalize("catherine pembrton")  # missing one char
    assert eid == "e1"
    assert score >= 0.92
    assert score < 1.0


def test_fuzzy_match_below_threshold_returns_none() -> None:
    norm = _normalizer_with(_entity("e1", "Bob"))
    eid, score = norm.normalize("Xander")
    assert eid is None
    assert score == 0.0


# --- unresolved ----------------------------------------------------------
def test_unresolved_mention_returns_none() -> None:
    norm = _normalizer_with(_entity("e1", "Alice"))
    eid, score = norm.normalize("zzz unrelated")
    assert eid is None
    assert score == 0.0


def test_empty_mention_returns_none() -> None:
    norm = _normalizer_with(_entity("e1", "Alice"))
    eid, score = norm.normalize("   ")
    assert eid is None
    assert score == 0.0


# --- register_alias ------------------------------------------------------
def test_register_alias_makes_subsequent_lookup_succeed() -> None:
    norm = _normalizer_with(_entity("e1", "Alice"))
    assert norm.normalize("Ally A.") == (None, 0.0)

    norm.register_alias("Ally A.", "e1")

    eid, score = norm.normalize("Ally A.")
    assert eid == "e1"
    assert score == 1.0


def test_register_alias_updates_stats() -> None:
    norm = _normalizer_with(_entity("e1", "Alice"))
    before = norm.stats()
    norm.register_alias("New Nickname", "e1")
    after = norm.stats()
    assert after["total_aliases"] == before["total_aliases"] + 1


# --- graph_weight --------------------------------------------------------
def test_graph_weight_mapping_per_resolution_type() -> None:
    assert EntityResolutionResult("e1", 1.0, "exact_alias").graph_weight == 1.0
    assert EntityResolutionResult("e1", 0.95, "high_confidence").graph_weight == 0.85
    assert EntityResolutionResult("e1", 0.7, "provisional").graph_weight == 0.5
    assert EntityResolutionResult(None, 0.0, "unresolved").graph_weight == 0.0


# --- performance ---------------------------------------------------------
def test_ten_thousand_lookups_complete_under_one_hundred_ms() -> None:
    entities = [
        _entity(f"e{i}", f"Person Number {i}", [f"alias-{i}", f"nick{i}"])
        for i in range(500)
    ]
    norm = _normalizer_with(*entities)

    queries = [f"Person Number {i % 500}" for i in range(10_000)]

    start = time.perf_counter()
    for q in queries:
        norm.normalize(q)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 100, f"10k lookups took {elapsed_ms:.1f}ms, budget is 100ms"


# --- load_from_graph -----------------------------------------------------
def test_load_from_graph_picks_up_canonical_and_aliases() -> None:
    store = GraphStore()
    store.add_entity(_entity("e1", "Alice", ["A.", "Ally"]))
    store.add_entity(_entity("e2", "Bob"))
    norm = MentionNormalizer()
    norm.load_from_graph(store)

    assert norm.normalize("Alice") == ("e1", 1.0)
    assert norm.normalize("A.") == ("e1", 1.0)
    assert norm.normalize("Ally") == ("e1", 1.0)
    assert norm.normalize("Bob") == ("e2", 1.0)
    stats = norm.stats()
    assert stats["total_entities"] == 2
    assert stats["total_aliases"] >= 4
