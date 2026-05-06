"""Unit tests for the spaCy NER ingest stage.

A real spaCy model would be slow to load and require a network
download in CI, so the tests inject a tiny fake ``nlp`` object that
emits whatever entity spans the test author hands it. That keeps the
suite hermetic while still exercising every code path the production
extractor walks.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import pytest

from satira.graph.entity_resolution import MentionNormalizer
from satira.graph.schema import ContentNode, EdgeType, EntityNode, SourceNode
from satira.graph.store import GraphStore
from satira.ingest.entity_extraction import (
    EntityExtractor,
    ExtractedEntity,
)
from satira.ingest.image_pipeline import ProcessedItem


# --- fake spaCy plumbing ----------------------------------------------------
@dataclass
class _FakeSpan:
    text: str
    label_: str
    start_char: int
    end_char: int


@dataclass
class _FakeDoc:
    ents: list[_FakeSpan]


class _FakeNLP:
    """Minimal stand-in for a spaCy ``Language`` object.

    The mapping is keyed by document text, so each test can declare
    exactly which entities the ``nlp`` should "find" in a given input.
    """

    def __init__(self, mapping: dict[str, list[tuple[str, str, int, int]]]) -> None:
        self._mapping = mapping
        self.calls: list[str] = []
        self.pipe_calls: list[list[str]] = []

    def __call__(self, text: str) -> _FakeDoc:
        self.calls.append(text)
        return _FakeDoc([_FakeSpan(*tup) for tup in self._mapping.get(text, [])])

    def pipe(self, texts: Iterable[str]) -> Iterable[_FakeDoc]:
        materialised = list(texts)
        self.pipe_calls.append(materialised)
        for t in materialised:
            yield _FakeDoc([_FakeSpan(*tup) for tup in self._mapping.get(t, [])])


_UNSET: Any = object()


def _extractor(
    mapping: dict[str, list[tuple[str, str, int, int]]] | None = None,
    allowed_labels: Iterable[str] | None = _UNSET,
) -> tuple[EntityExtractor, _FakeNLP]:
    nlp = _FakeNLP(mapping or {})
    if allowed_labels is _UNSET:
        ex = EntityExtractor(nlp=nlp)
    else:
        ex = EntityExtractor(nlp=nlp, allowed_labels=allowed_labels)
    return ex, nlp


def _processed(
    source_url: str = "https://example.com/a",
    text: str = "",
    domain: str = "example.com",
    label: str = "satire",
    metadata: dict[str, Any] | None = None,
) -> ProcessedItem:
    md: dict[str, Any] = {"label": label}
    if metadata:
        md.update(metadata)
    return ProcessedItem(
        source_url=source_url,
        image_url=f"{source_url}.png",
        title="t",
        text=text,
        timestamp=datetime(2026, 1, 1),
        source_domain=domain,
        metadata=md,
        image_path="/tmp/x.png",
        image_dimensions=(300, 300),
        perceptual_hash="0" * 16,
        file_size_bytes=100,
    )


# --- extract ---------------------------------------------------------------
def test_extract_returns_entities_with_offsets() -> None:
    ex, _ = _extractor({
        "Alice met Bob.": [
            ("Alice", "PERSON", 0, 5),
            ("Bob", "PERSON", 10, 13),
        ],
    })
    out = ex.extract("Alice met Bob.")
    assert out == [
        ExtractedEntity(text="Alice", label="PERSON", start=0, end=5),
        ExtractedEntity(text="Bob", label="PERSON", start=10, end=13),
    ]


def test_extract_returns_empty_for_empty_text() -> None:
    ex, nlp = _extractor({})
    assert ex.extract("") == []
    # nlp should not be invoked at all when there's nothing to parse.
    assert nlp.calls == []


def test_extract_filters_by_allowed_labels() -> None:
    # Default allowed_labels excludes DATE / CARDINAL — those should be dropped.
    ex, _ = _extractor({
        "stuff": [
            ("Alice", "PERSON", 0, 5),
            ("Tuesday", "DATE", 6, 13),
            ("42", "CARDINAL", 14, 16),
        ],
    })
    labels = [e.label for e in ex.extract("stuff")]
    assert labels == ["PERSON"]


def test_allowed_labels_none_lets_everything_through() -> None:
    ex, _ = _extractor(
        {"x": [("Tuesday", "DATE", 0, 7), ("42", "CARDINAL", 8, 10)]},
        allowed_labels=None,
    )
    assert {e.label for e in ex.extract("x")} == {"DATE", "CARDINAL"}


# --- extract_batch ---------------------------------------------------------
def test_extract_batch_preserves_input_order() -> None:
    ex, _ = _extractor({
        "doc-a": [("Alice", "PERSON", 0, 5)],
        "doc-b": [("Bob", "PERSON", 0, 3)],
    })
    out = ex.extract_batch(["doc-a", "doc-b"])
    assert [e[0].text for e in out] == ["Alice", "Bob"]


def test_extract_batch_handles_empty_strings_in_position() -> None:
    ex, nlp = _extractor({
        "doc-a": [("Alice", "PERSON", 0, 5)],
        "doc-c": [("Carol", "PERSON", 0, 5)],
    })
    out = ex.extract_batch(["doc-a", "", "doc-c"])
    assert len(out) == 3
    assert [e.text for e in out[0]] == ["Alice"]
    assert out[1] == []
    assert [e.text for e in out[2]] == ["Carol"]
    # Empty texts are not handed to spaCy.
    assert nlp.pipe_calls == [["doc-a", "doc-c"]]


def test_extract_batch_handles_all_empty_input() -> None:
    ex, nlp = _extractor({})
    assert ex.extract_batch([]) == []
    assert ex.extract_batch(["", "", ""]) == [[], [], []]
    assert nlp.pipe_calls == []


# --- populate_graph: structural edges ---------------------------------------
def test_populate_graph_creates_content_and_source_nodes() -> None:
    store = GraphStore()
    norm = MentionNormalizer()
    ex, _ = _extractor({"text": []})

    item = _processed(text="text")
    stats = ex.populate_graph([item], store, norm)

    # Exactly one content + one source node added.
    assert stats["content_added"] == 1
    kinds = sorted(store._node_kind.values())
    assert kinds == ["content", "source"]


def test_populate_graph_adds_posted_by_edge() -> None:
    store = GraphStore()
    norm = MentionNormalizer()
    ex, _ = _extractor({"text": []})

    ex.populate_graph([_processed(text="text")], store, norm)

    content_id = next(
        nid for nid, k in store._node_kind.items() if k == "content"
    )
    source_id = next(
        nid for nid, k in store._node_kind.items() if k == "source"
    )
    neighbors = store.get_neighbors(content_id, EdgeType.POSTED_BY)
    assert neighbors == [source_id]


def test_populate_graph_propagates_credibility_label_from_metadata() -> None:
    store = GraphStore()
    norm = MentionNormalizer()
    ex, _ = _extractor({"text": []})

    ex.populate_graph(
        [_processed(text="text", domain="onion.com", label="satire")],
        store,
        norm,
    )
    source = next(
        n for nid, n in store._nodes.items() if store._node_kind[nid] == "source"
    )
    assert isinstance(source, SourceNode)
    assert source.domain == "onion.com"
    assert source.credibility_label == "satire"


def test_populate_graph_falls_back_to_unknown_credibility() -> None:
    store = GraphStore()
    norm = MentionNormalizer()
    ex, _ = _extractor({"text": []})

    item = _processed(text="text")
    item.metadata.pop("label")
    ex.populate_graph([item], store, norm)

    source = next(
        n for nid, n in store._nodes.items() if store._node_kind[nid] == "source"
    )
    assert source.credibility_label == "unknown"


# --- populate_graph: resolution -------------------------------------------
def test_populate_graph_resolves_known_entities_into_mentions_edges() -> None:
    store = GraphStore()
    alice = EntityNode(
        id="ent:alice",
        canonical_name="Alice Smith",
        entity_type="person",
        aliases=["Allie"],
        created_at=datetime(2026, 1, 1),
    )
    store.add_entity(alice)
    norm = MentionNormalizer()
    norm.load_from_graph(store)

    ex, _ = _extractor({
        "Alice met someone": [
            ("Alice Smith", "PERSON", 0, 11),
            ("Mystery Person", "PERSON", 16, 30),
        ],
    })

    stats = ex.populate_graph(
        [_processed(text="Alice met someone")], store, norm
    )

    assert stats["entities_resolved"] == 1
    assert stats["entities_pending"] == 1

    content_id = next(
        nid for nid, k in store._node_kind.items() if k == "content"
    )
    mentions = store.get_neighbors(content_id, EdgeType.MENTIONS)
    assert mentions == ["ent:alice"]


def test_populate_graph_queues_unresolved_with_cooccurrence() -> None:
    store = GraphStore()
    alice = EntityNode(
        id="ent:alice",
        canonical_name="Alice",
        entity_type="person",
        aliases=[],
        created_at=datetime(2026, 1, 1),
    )
    store.add_entity(alice)
    norm = MentionNormalizer()
    norm.load_from_graph(store)

    ex, _ = _extractor({
        "doc": [
            ("Alice", "PERSON", 0, 5),
            ("Acme Corp", "ORG", 6, 15),
            ("Wakanda", "GPE", 16, 23),
        ],
    })

    stats = ex.populate_graph([_processed(text="doc")], store, norm)

    pending = stats["pending_mentions"]
    assert {m["text"] for m in pending} == {"Acme Corp", "Wakanda"}

    by_text = {m["text"]: m for m in pending}
    assert by_text["Acme Corp"]["entity_type"] == "organization"
    assert by_text["Wakanda"]["entity_type"] == "location"
    # The resolved Alice entity should ride along as cooccurring context
    # so Tier 2 can use it as a blocking key.
    for m in pending:
        assert m["cooccurring_entity_ids"] == ["ent:alice"]
        assert m["content_id"].startswith("content:")


def test_populate_graph_dedupes_repeated_resolutions() -> None:
    """If spaCy emits the same surface form twice, we still only resolve it
    once into the cooccurrence list — but each mention edge is added (and
    dedupped by GraphStore's set-backed adjacency)."""
    store = GraphStore()
    store.add_entity(EntityNode(
        id="ent:alice",
        canonical_name="Alice",
        entity_type="person",
        aliases=[],
        created_at=datetime(2026, 1, 1),
    ))
    norm = MentionNormalizer()
    norm.load_from_graph(store)

    ex, _ = _extractor({
        "doc": [
            ("Alice", "PERSON", 0, 5),
            ("Alice", "PERSON", 10, 15),
        ],
    })

    stats = ex.populate_graph([_processed(text="doc")], store, norm)

    content_id = next(
        nid for nid, k in store._node_kind.items() if k == "content"
    )
    # Two raw mentions, but the graph stores one edge thanks to set semantics.
    assert stats["entities_resolved"] == 2
    assert store.get_neighbors(content_id, EdgeType.MENTIONS) == ["ent:alice"]


# --- populate_graph: idempotence ------------------------------------------
def test_populate_graph_skips_duplicate_content() -> None:
    store = GraphStore()
    norm = MentionNormalizer()
    ex, _ = _extractor({"text": []})

    item = _processed(text="text")
    first = ex.populate_graph([item], store, norm)
    second = ex.populate_graph([item], store, norm)

    assert first["content_added"] == 1
    assert second["content_added"] == 0
    # Same source URL → same content node, not a duplicate.
    content_ids = [
        nid for nid, k in store._node_kind.items() if k == "content"
    ]
    assert len(content_ids) == 1


def test_populate_graph_reuses_source_across_items() -> None:
    store = GraphStore()
    norm = MentionNormalizer()
    ex, _ = _extractor({"a": [], "b": []})

    items = [
        _processed(source_url="https://example.com/a", text="a"),
        _processed(source_url="https://example.com/b", text="b"),
    ]
    ex.populate_graph(items, store, norm)

    source_ids = [
        nid for nid, k in store._node_kind.items() if k == "source"
    ]
    assert len(source_ids) == 1


def test_populate_graph_separates_sources_by_account_id() -> None:
    store = GraphStore()
    norm = MentionNormalizer()
    ex, _ = _extractor({"a": [], "b": []})

    items = [
        _processed(
            source_url="https://twitter.com/post/1",
            text="a",
            domain="twitter.com",
            metadata={"account_id": "alice_handle"},
        ),
        _processed(
            source_url="https://twitter.com/post/2",
            text="b",
            domain="twitter.com",
            metadata={"account_id": "bob_handle"},
        ),
    ]
    ex.populate_graph(items, store, norm)

    source_ids = sorted(
        nid for nid, k in store._node_kind.items() if k == "source"
    )
    assert len(source_ids) == 2


def test_populate_graph_handles_empty_items_without_calling_pipe() -> None:
    store = GraphStore()
    norm = MentionNormalizer()
    ex, nlp = _extractor({})

    stats = ex.populate_graph([], store, norm)

    assert stats == {
        "entities_resolved": 0,
        "entities_pending": 0,
        "content_added": 0,
        "pending_mentions": [],
    }
    assert nlp.pipe_calls == []


# --- constructor surface ---------------------------------------------------
def test_constructor_surfaces_missing_model_with_helpful_message() -> None:
    # Either spaCy is missing (RuntimeError about install) or it's
    # installed and the model isn't (RuntimeError about spacy download).
    # Both should surface as RuntimeError, not the raw OSError/ImportError.
    with pytest.raises(RuntimeError):
        EntityExtractor(model_name="nonexistent_model_definitely_not_installed")
