"""Unit tests for the source credibility classifier.

Covers each tier (curated allowlist → domain heuristics → content
history) and tier ordering, plus input-normalisation edge cases that
real-world callers tend to hit (URLs passed instead of domains, leading
``www.``, ports, mixed case).
"""
from __future__ import annotations

import pytest

from satira.ingest.source_credibility import (
    CATEGORIES,
    KNOWN_NEWS,
    KNOWN_SATIRE,
    MIXED,
    NEWS,
    SATIRE,
    UNKNOWN,
    SourceClassification,
    SourceCredibilityClassifier,
)


# --- SourceClassification dataclass -----------------------------------------
class TestSourceClassification:
    def test_round_trip(self):
        c = SourceClassification(category=SATIRE, confidence=0.8, reasoning="ok")
        assert c.category == SATIRE
        assert c.confidence == 0.8
        assert c.reasoning == "ok"

    def test_rejects_unknown_category(self):
        with pytest.raises(ValueError):
            SourceClassification(category="GARBAGE", confidence=0.5, reasoning="x")

    @pytest.mark.parametrize("conf", [-0.1, 1.5, 2.0])
    def test_rejects_out_of_range_confidence(self, conf):
        with pytest.raises(ValueError):
            SourceClassification(category=SATIRE, confidence=conf, reasoning="x")


# --- Tier 1: curated allowlists --------------------------------------------
class TestCuratedTier:
    @pytest.fixture
    def clf(self):
        return SourceCredibilityClassifier()

    @pytest.mark.parametrize(
        "domain",
        ["theonion.com", "babylonbee.com", "reductress.com", "clickhole.com"],
    )
    def test_known_satire(self, clf, domain):
        verdict = clf.classify(domain)
        assert verdict.category == SATIRE
        assert verdict.confidence == 1.0
        assert "allowlist" in verdict.reasoning

    @pytest.mark.parametrize(
        "domain", ["reuters.com", "apnews.com", "bbc.co.uk", "npr.org"]
    )
    def test_known_news(self, clf, domain):
        verdict = clf.classify(domain)
        assert verdict.category == NEWS
        assert verdict.confidence == 1.0
        assert "allowlist" in verdict.reasoning

    def test_known_satire_and_news_are_disjoint(self):
        assert not (KNOWN_SATIRE & KNOWN_NEWS)

    def test_categories_constant_is_complete(self):
        assert set(CATEGORIES) == {SATIRE, NEWS, MIXED, UNKNOWN}


# --- input normalisation ----------------------------------------------------
class TestNormalisation:
    @pytest.fixture
    def clf(self):
        return SourceCredibilityClassifier()

    @pytest.mark.parametrize(
        "raw",
        [
            "theonion.com",
            "TheOnion.com",
            "WWW.theonion.com",
            "  theonion.com  ",
            "https://www.theonion.com/articles/foo",
            "http://theonion.com:8080/path",
        ],
    )
    def test_curated_match_survives_messy_input(self, clf, raw):
        assert clf.classify(raw).category == SATIRE

    def test_empty_domain_is_unknown(self, clf):
        verdict = clf.classify("")
        assert verdict.category == UNKNOWN
        assert verdict.confidence == 0.0


# --- Tier 2: heuristics -----------------------------------------------------
class TestHeuristicTier:
    @pytest.fixture
    def clf(self):
        return SourceCredibilityClassifier()

    @pytest.mark.parametrize(
        "domain",
        [
            "satireblog.com",
            "globalsatirenetwork.org",
            "comedyhour.io",
            "spoofdaily.net",
            "parodypress.co",
            "fake-news-times.com",
        ],
    )
    def test_satire_keyword_in_domain(self, clf, domain):
        verdict = clf.classify(domain)
        assert verdict.category == SATIRE
        # Heuristic confidence sits in the 0.6–0.8 band per the spec.
        assert 0.6 <= verdict.confidence <= 0.8

    def test_unknown_when_no_signals(self, clf):
        verdict = clf.classify("randomblog42.org")
        assert verdict.category == UNKNOWN
        assert verdict.confidence == 0.0

    def test_suspicious_tld_alone_is_not_enough(self, clf):
        # No domain-age lookup configured → suspicious TLD alone falls
        # through to UNKNOWN. The combined signal needs both.
        verdict = clf.classify("randomthing.xyz")
        assert verdict.category == UNKNOWN

    def test_suspicious_tld_plus_young_domain_is_mixed(self):
        clf = SourceCredibilityClassifier(domain_age_lookup=lambda _d: 30.0)
        verdict = clf.classify("brandnew.xyz")
        assert verdict.category == MIXED
        assert 0.6 <= verdict.confidence <= 0.8
        assert "low-cost TLD" in verdict.reasoning

    def test_old_suspicious_tld_is_not_mixed(self):
        clf = SourceCredibilityClassifier(domain_age_lookup=lambda _d: 5_000.0)
        # A long-established .xyz domain is no longer "fresh suspicion".
        assert clf.classify("longlived.xyz").category == UNKNOWN

    def test_age_lookup_failure_is_swallowed(self):
        def boom(_d: str) -> float | None:
            raise RuntimeError("WHOIS server down")

        clf = SourceCredibilityClassifier(domain_age_lookup=boom)
        # Failure → treat as unknown age, fall through to UNKNOWN, no raise.
        assert clf.classify("randomthing.xyz").category == UNKNOWN


# --- Tier 3: content history ------------------------------------------------
class TestContentHistoryTier:
    @pytest.fixture
    def clf(self):
        # Lower the sample threshold so individual tests stay readable.
        return SourceCredibilityClassifier(min_history_samples=5)

    def test_satire_dominance(self, clf):
        clf.update_from_content_history(
            "smallunknown.example", ["satire"] * 9 + ["news"] * 1
        )
        verdict = clf.classify("smallunknown.example")
        assert verdict.category == SATIRE
        assert verdict.confidence >= 0.7

    def test_news_dominance(self, clf):
        clf.update_from_content_history(
            "regional.example", ["news"] * 9 + ["satire"] * 1
        )
        verdict = clf.classify("regional.example")
        assert verdict.category == NEWS
        assert verdict.confidence >= 0.7

    def test_mixed_when_neither_dominates(self, clf):
        clf.update_from_content_history(
            "messy.example", ["satire"] * 5 + ["news"] * 5
        )
        verdict = clf.classify("messy.example")
        assert verdict.category == MIXED

    def test_below_sample_threshold_is_unknown(self, clf):
        # 3 samples < min_history_samples (5) → no verdict from history.
        clf.update_from_content_history("tiny.example", ["satire"] * 3)
        assert clf.classify("tiny.example").category == UNKNOWN

    def test_authentic_alias_counts_as_news(self, clf):
        clf.update_from_content_history(
            "wire.example", ["authentic"] * 9 + ["satire"] * 1
        )
        assert clf.classify("wire.example").category == NEWS

    def test_history_accumulates_across_calls(self, clf):
        clf.update_from_content_history("incremental.example", ["satire"] * 4)
        # First call alone is below threshold and only one category present.
        assert clf.classify("incremental.example").category == UNKNOWN
        clf.update_from_content_history("incremental.example", ["satire"] * 5)
        assert clf.classify("incremental.example").category == SATIRE

    def test_history_for_returns_counts(self, clf):
        clf.update_from_content_history(
            "counts.example",
            ["satire", "news", "news", "garbage", "satire"],
        )
        assert clf.history_for("counts.example") == {
            "satire": 2,
            "news": 2,
            "other": 1,
        }

    def test_history_for_unknown_domain_is_empty(self, clf):
        assert clf.history_for("never-seen.example") == {}

    def test_empty_domain_update_is_noop(self, clf):
        clf.update_from_content_history("", ["satire"] * 100)
        assert clf.history_for("") == {}


# --- tier ordering ----------------------------------------------------------
class TestTierOrdering:
    def test_curated_overrides_heuristic(self):
        # Pretend a curated outlet has "satire" in its name (Reductress
        # actually does). Curated verdict (NEWS) must win regardless.
        clf = SourceCredibilityClassifier(
            known_news=["satirepatrol.example"],
            known_satire=[],  # explicitly drop the default satire list
        )
        verdict = clf.classify("satirepatrol.example")
        assert verdict.category == NEWS
        assert verdict.confidence == 1.0

    def test_curated_overrides_history(self):
        clf = SourceCredibilityClassifier(min_history_samples=3)
        # Even a strong satire history can't override a curated NEWS entry.
        clf.update_from_content_history("reuters.com", ["satire"] * 50)
        verdict = clf.classify("reuters.com")
        assert verdict.category == NEWS
        assert verdict.confidence == 1.0

    def test_heuristic_overrides_history(self):
        clf = SourceCredibilityClassifier(min_history_samples=3)
        # Strong news-history signal but the domain itself screams satire.
        clf.update_from_content_history(
            "satirezone.example", ["news"] * 100
        )
        # Heuristic catches "satire" substring → SATIRE wins over history.
        assert clf.classify("satirezone.example").category == SATIRE

    def test_history_used_when_other_tiers_silent(self):
        clf = SourceCredibilityClassifier(min_history_samples=3)
        clf.update_from_content_history(
            "neutralname.example", ["satire"] * 10
        )
        verdict = clf.classify("neutralname.example")
        assert verdict.category == SATIRE
        assert verdict.confidence < 1.0  # not curated


# --- constructor validation -------------------------------------------------
class TestConstructor:
    def test_rejects_nonpositive_min_samples(self):
        with pytest.raises(ValueError):
            SourceCredibilityClassifier(min_history_samples=0)
        with pytest.raises(ValueError):
            SourceCredibilityClassifier(min_history_samples=-3)

    def test_custom_known_lists_are_normalised(self):
        clf = SourceCredibilityClassifier(
            known_satire=["WWW.MyJoke.com", "https://other.test/"],
            known_news=[],
        )
        assert clf.classify("myjoke.com").category == SATIRE
        assert clf.classify("https://www.other.test/path").category == SATIRE

    def test_empty_lists_disable_curated_tier(self):
        clf = SourceCredibilityClassifier(known_satire=[], known_news=[])
        # theonion.com would normally hit the curated tier; without it
        # we should fall through to the heuristic tier on the substring.
        verdict = clf.classify("theonion.com")
        assert verdict.category == SATIRE
        assert verdict.confidence < 1.0  # heuristic, not curated
