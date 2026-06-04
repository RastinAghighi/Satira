"""Unit tests for the HuggingFace loader adapters.

The adapters are pure row -> ScrapedItem functions, so these tests feed
synthetic rows directly — no network, no `datasets` import. The focus is
the LIAR2 5-class mapping (getting the veracity order backwards would
silently corrupt every label) and the licensing policy guard.
"""
from __future__ import annotations

import pytest

from satira.ingest import KNOWN_FACTCHECK_DATASETS
from satira.ingest.huggingface_loader import (
    _FIVE_CLASS,
    _LIAR2_LABEL_NAMES,
    _LIAR2_TO_SATIRA,
    _adapt_liar2,
)


def _row(**over):
    base = {
        "statement": "The unemployment rate fell to a record low last quarter.",
        "label": 5,
        "context": "a campaign speech",
        "speaker": "jane doe",
        "justification": "PolitiFact rated this because official BLS data shows...",
    }
    base.update(over)
    return base


# --- mapping integrity ------------------------------------------------------
def test_liar2_label_names_cover_six_way_scale() -> None:
    assert set(_LIAR2_LABEL_NAMES) == {0, 1, 2, 3, 4, 5}
    assert _LIAR2_LABEL_NAMES[0] == "pants-fire"  # most false
    assert _LIAR2_LABEL_NAMES[5] == "true"  # most true


def test_every_liar2_category_maps_into_the_five_class_taxonomy() -> None:
    for name in _LIAR2_LABEL_NAMES.values():
        assert name in _LIAR2_TO_SATIRA, f"unmapped LIAR2 category {name!r}"
        assert _LIAR2_TO_SATIRA[name] in _FIVE_CLASS


@pytest.mark.parametrize(
    ("label_int", "expected"),
    [
        (0, "fabricated"),          # pants-fire: egregious falsehood, NOT satire
        (1, "fabricated"),          # false
        (2, "misleading_context"),  # barely-true
        (3, "misleading_context"),  # half-true
        (4, "authentic"),           # mostly-true: substantially accurate
        (5, "authentic"),           # true
    ],
)
def test_liar2_adapter_maps_each_label(label_int: int, expected: str) -> None:
    item = _adapt_liar2(_row(label=label_int))
    assert item is not None
    assert item.metadata["label"] == expected
    assert item.metadata["original_label"] == _LIAR2_LABEL_NAMES[label_int]


def test_liar2_contributes_nothing_to_the_satire_class() -> None:
    # LIAR2 is political fact-checking data with no actual satire. Political
    # falsehood ("pants-fire") is fabrication, not intentional humour, so no
    # category may route into the satire class. This guards the corrected
    # mapping against regressing to the old pants-fire -> satire error.
    assert "satire" not in set(_LIAR2_TO_SATIRA.values())
    for label_int in _LIAR2_LABEL_NAMES:
        item = _adapt_liar2(_row(label=label_int))
        assert item is not None
        assert item.metadata["label"] != "satire"


# --- item shape -------------------------------------------------------------
def test_liar2_item_is_text_only_with_statement_as_title() -> None:
    item = _adapt_liar2(_row(statement="Taxes will rise 90 percent.", context="a tweet"))
    assert item is not None
    assert item.image_url is None
    assert item.title == "Taxes will rise 90 percent."
    assert item.text == "a tweet"
    assert item.source_domain == "politifact.com"


def test_liar2_item_carries_license_and_citation() -> None:
    item = _adapt_liar2(_row())
    assert item is not None
    md = item.metadata
    assert md["source_type"] == "huggingface"
    assert md["hf_dataset"] == "chengxuphd/liar2"
    assert md["license"] == "apache-2.0"
    assert "IEEE Access" in md["citation"]
    assert md["speaker"] == "jane doe"


def test_liar2_adapter_does_not_leak_justification_into_input() -> None:
    # The fact-checker's justification reveals the label; it must not end
    # up in any field the model reads (title/text).
    just = "PolitiFact rated this because official BLS data shows..."
    item = _adapt_liar2(_row(justification=just))
    assert item is not None
    assert just not in item.title
    assert just not in item.text


# --- skips ------------------------------------------------------------------
@pytest.mark.parametrize("bad", [{"statement": ""}, {"statement": "   "}])
def test_liar2_adapter_skips_empty_statement(bad: dict) -> None:
    assert _adapt_liar2(_row(**bad)) is None


@pytest.mark.parametrize("label", [None, 9, -1, "true"])
def test_liar2_adapter_skips_unknown_label(label: object) -> None:
    assert _adapt_liar2(_row(label=label)) is None


def test_liar2_adapter_skips_missing_label_key() -> None:
    row = _row()
    del row["label"]
    assert _adapt_liar2(row) is None


# --- licensing policy guard -------------------------------------------------
def test_factcheck_specs_are_all_license_pinned_and_cited() -> None:
    # The whole point of KNOWN_FACTCHECK_DATASETS is that only
    # research-licensed, attributable datasets get in. Enforce it.
    assert KNOWN_FACTCHECK_DATASETS, "expected at least one fact-checking dataset"
    for spec in KNOWN_FACTCHECK_DATASETS:
        assert spec.license, f"{spec.dataset_id} has no pinned license"
        assert spec.citation, f"{spec.dataset_id} has no citation string"


def test_liar2_is_registered() -> None:
    ids = {s.dataset_id for s in KNOWN_FACTCHECK_DATASETS}
    assert "chengxuphd/liar2" in ids
