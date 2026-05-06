"""Source credibility classifier for the ingest pipeline.

Classifies content sources (publishers, accounts, domains) into a small
set of credibility categories so downstream stages — training-data
weighting, retrieval ranking, the moderator review queue — can treat a
record from The Onion differently from one from Reuters without each of
those stages having to re-derive the distinction.

Three tiers, in order of confidence:

1. **Hardcoded curation** — manually vetted allowlists of known-satire
   and known-news outlets. The default lists are intentionally short:
   only outlets whose nature is well established and unlikely to flip.
2. **Domain heuristics** — cheap structural signals that don't need
   network calls: satire keywords baked into the registrable domain,
   suspicious low-cost TLDs, etc. Domain-age (WHOIS) checks are *opt-in*
   via the ``domain_age_lookup`` constructor hook so this module doesn't
   pull in a WHOIS dependency and tests stay hermetic.
3. **Content history** — once we've classified enough content from a
   source, the per-source satire/news ratio becomes the strongest
   signal. :meth:`update_from_content_history` feeds those
   classifications back in, and subsequent ``classify`` calls consult
   them when neither tier 1 nor tier 2 produced a verdict.

Confidences are picked so a caller can threshold meaningfully: curated
verdicts are 1.0, heuristic verdicts 0.6–0.85, and ``UNKNOWN`` is
always 0.0 — so a check like ``c.confidence >= 0.7`` cleanly excludes
"we don't know" without picking up borderline heuristics.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass


logger = logging.getLogger(__name__)


# --- categories -------------------------------------------------------------
SATIRE = "SATIRE"
NEWS = "NEWS"
MIXED = "MIXED"
UNKNOWN = "UNKNOWN"

CATEGORIES: tuple[str, ...] = (SATIRE, NEWS, MIXED, UNKNOWN)


# --- thresholds -------------------------------------------------------------
# Below this many observed content classifications, history is too
# noisy to rely on — even an 80% satire ratio over 3 items shouldn't
# overrule "we have no other signal".
_MIN_HISTORY_SAMPLES = 10

# A source is treated as a category when its content-history share of
# that category clears the dominance threshold; if both satire and news
# clear the minority threshold but neither dominates, the source is
# MIXED.
_DOMINANCE_THRESHOLD = 0.8
_MINORITY_THRESHOLD = 0.2

# Heuristic-tier confidences. Kept inside the spec'd 0.6–0.8 band, with
# content-history dominance allowed to push to 0.85 since it's grounded
# in actual observations rather than structural guesswork.
_HEURISTIC_HIGH = 0.8
_HEURISTIC_MED = 0.7
_HEURISTIC_LOW = 0.6
_HISTORY_DOMINANT = 0.85

# TLDs frequently used for low-cost throwaway / spam / parody domains.
# Hitting one isn't damning on its own — combined with an obviously new
# registration it's enough to flag a source as MIXED (low credibility).
_SUSPICIOUS_TLDS: frozenset[str] = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "info",
})

# Substrings (not whole-word boundaries) because publishers routinely
# concatenate brand tokens, e.g. ``satireblog.com``. False positives at
# the heuristic tier are tolerable: confidence is 0.8, not 1.0.
_SATIRE_SUBSTRINGS: tuple[str, ...] = (
    "satire", "satirical", "satirically",
    "spoof", "parody", "comedy", "humor", "humour",
    "fakenews", "fake-news", "fake_news",
    "theonion", "babylonbee", "reductress",
)


# --- public types -----------------------------------------------------------
@dataclass(frozen=True)
class SourceClassification:
    """Verdict for a single source.

    ``confidence`` is on [0, 1]: 1.0 for curated entries, 0.6–0.85 for
    heuristic / history-based verdicts, 0.0 for ``UNKNOWN``.
    ``reasoning`` is a short human-readable explanation suitable for
    logging, telemetry, or surfacing in a moderator UI.
    """

    category: str
    confidence: float
    reasoning: str

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(
                f"category must be one of {CATEGORIES}, got {self.category!r}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence!r}"
            )


# --- known-source allowlists ------------------------------------------------
# These are intentionally curated and short. Adding to either set is a
# product decision: we want maintainers to understand each entry, not
# paste a 500-row CSV that will rot and start producing wrong verdicts.
KNOWN_SATIRE: frozenset[str] = frozenset({
    "theonion.com",
    "babylonbee.com",
    "reductress.com",
    "clickhole.com",
    "thebeaverton.com",
    "thehardtimes.net",
    "newsthump.com",
    "thedailymash.co.uk",
    "thespoof.com",
    "waterfordwhispersnews.com",
    "private-eye.co.uk",
    "thedailycurrant.com",
})

KNOWN_NEWS: frozenset[str] = frozenset({
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "npr.org",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "economist.com",
    "ft.com",
    "bloomberg.com",
    "abcnews.go.com",
    "cbsnews.com",
    "nbcnews.com",
    "cnn.com",
    "aljazeera.com",
    "pbs.org",
})


# Hook signature: takes a normalised domain, returns its age in days,
# or ``None`` when the lookup couldn't determine an age. Pluggable so
# this module never has to depend on a WHOIS client and tests can
# inject deterministic ages.
DomainAgeLookup = Callable[[str], "float | None"]


# --- classifier -------------------------------------------------------------
class SourceCredibilityClassifier:
    """Tiered source-credibility classifier.

    Tier order is curated lists → domain heuristics → content history,
    and the first tier to return a non-``UNKNOWN`` verdict wins. The
    one exception is the content-history tier, which can override an
    ``UNKNOWN`` heuristic verdict once it has accumulated at least
    ``min_history_samples`` observations.
    """

    def __init__(
        self,
        known_satire: Iterable[str] | None = None,
        known_news: Iterable[str] | None = None,
        *,
        domain_age_lookup: DomainAgeLookup | None = None,
        min_history_samples: int = _MIN_HISTORY_SAMPLES,
    ) -> None:
        if min_history_samples <= 0:
            raise ValueError(
                f"min_history_samples must be positive, got {min_history_samples}"
            )
        self._known_satire = frozenset(
            self._normalize(d)
            for d in (known_satire if known_satire is not None else KNOWN_SATIRE)
            if d
        )
        self._known_news = frozenset(
            self._normalize(d)
            for d in (known_news if known_news is not None else KNOWN_NEWS)
            if d
        )
        self._domain_age_lookup = domain_age_lookup
        self._min_history_samples = min_history_samples
        self._history: dict[str, dict[str, int]] = defaultdict(
            lambda: {"satire": 0, "news": 0, "other": 0}
        )

    # --- public API -----------------------------------------------------
    def classify(self, domain: str) -> SourceClassification:
        norm = self._normalize(domain)
        if not norm:
            return SourceClassification(
                category=UNKNOWN,
                confidence=0.0,
                reasoning="empty or invalid domain",
            )

        if norm in self._known_satire:
            return SourceClassification(
                category=SATIRE,
                confidence=1.0,
                reasoning=f"{norm!r} is in the curated satire allowlist",
            )
        if norm in self._known_news:
            return SourceClassification(
                category=NEWS,
                confidence=1.0,
                reasoning=f"{norm!r} is in the curated news allowlist",
            )

        heuristic = self._classify_heuristic(norm)
        if heuristic.category != UNKNOWN:
            return heuristic

        history = self._classify_from_history(norm)
        if history is not None:
            return history

        return SourceClassification(
            category=UNKNOWN,
            confidence=0.0,
            reasoning=f"no curated, heuristic, or history-based signal for {norm!r}",
        )

    def update_from_content_history(
        self,
        domain: str,
        content_classifications: Iterable[str],
    ) -> None:
        """Record content-classification observations for ``domain``.

        Each entry should be ``"satire"``, ``"news"`` (``"authentic"``
        and ``"real"`` are aliased), or any other string (counted as
        ``"other"``). Counts accumulate across calls so a caller can
        stream classifications in batches without rebuilding state.
        """
        norm = self._normalize(domain)
        if not norm:
            return
        bucket = self._history[norm]
        for c in content_classifications:
            key = (c or "").strip().lower()
            if key == "satire":
                bucket["satire"] += 1
            elif key in {"news", "authentic", "real"}:
                bucket["news"] += 1
            else:
                bucket["other"] += 1

    def history_for(self, domain: str) -> dict[str, int]:
        """Return a copy of the per-domain content-history counts.

        Useful for tests and telemetry. Returns an empty dict if no
        history has been recorded for ``domain``.
        """
        return dict(self._history.get(self._normalize(domain), {}))

    # --- internals ------------------------------------------------------
    @staticmethod
    def _normalize(domain: str) -> str:
        if not domain:
            return ""
        d = domain.strip().lower()
        # Strip a leading scheme if a caller accidentally hands us a URL.
        for scheme in ("http://", "https://"):
            if d.startswith(scheme):
                d = d[len(scheme):]
                break
        # Drop any path/query the URL might have carried.
        d = d.split("/", 1)[0]
        # Strip a port, if present.
        d = d.split(":", 1)[0]
        if d.startswith("www."):
            d = d[4:]
        return d

    def _classify_heuristic(self, domain: str) -> SourceClassification:
        for needle in _SATIRE_SUBSTRINGS:
            if needle in domain:
                return SourceClassification(
                    category=SATIRE,
                    confidence=_HEURISTIC_HIGH,
                    reasoning=(
                        f"{domain!r} contains satire-related substring "
                        f"{needle!r}"
                    ),
                )

        tld = domain.rsplit(".", 1)[-1] if "." in domain else ""
        suspicious_tld = tld in _SUSPICIOUS_TLDS

        age_days: float | None = None
        if self._domain_age_lookup is not None:
            try:
                age_days = self._domain_age_lookup(domain)
            except Exception as exc:  # noqa: BLE001 — lookups can fail many ways
                logger.debug("domain age lookup for %r failed: %s", domain, exc)
                age_days = None

        # Either signal alone is too weak to act on, but a brand-new
        # domain on a low-cost TLD together is a reasonable "treat as
        # low credibility" cue.
        if suspicious_tld and age_days is not None and age_days < 180:
            return SourceClassification(
                category=MIXED,
                confidence=_HEURISTIC_MED,
                reasoning=(
                    f"{domain!r} uses low-cost TLD .{tld} and is only "
                    f"{int(age_days)} days old — treat as low-credibility"
                ),
            )

        return SourceClassification(
            category=UNKNOWN,
            confidence=0.0,
            reasoning="no heuristic match",
        )

    def _classify_from_history(self, domain: str) -> SourceClassification | None:
        bucket = self._history.get(domain)
        if not bucket:
            return None
        total = bucket["satire"] + bucket["news"] + bucket["other"]
        if total < self._min_history_samples:
            return None

        satire_ratio = bucket["satire"] / total
        news_ratio = bucket["news"] / total

        if satire_ratio >= _DOMINANCE_THRESHOLD:
            return SourceClassification(
                category=SATIRE,
                confidence=_HISTORY_DOMINANT,
                reasoning=(
                    f"{satire_ratio:.0%} of {total} observed items from "
                    f"{domain!r} were satire"
                ),
            )
        if news_ratio >= _DOMINANCE_THRESHOLD:
            return SourceClassification(
                category=NEWS,
                confidence=_HISTORY_DOMINANT,
                reasoning=(
                    f"{news_ratio:.0%} of {total} observed items from "
                    f"{domain!r} were news"
                ),
            )
        if (
            satire_ratio >= _MINORITY_THRESHOLD
            and news_ratio >= _MINORITY_THRESHOLD
        ):
            return SourceClassification(
                category=MIXED,
                confidence=_HEURISTIC_LOW,
                reasoning=(
                    f"{domain!r} content history is mixed: "
                    f"{satire_ratio:.0%} satire / {news_ratio:.0%} news "
                    f"over {total} items"
                ),
            )
        return None


# Quick sanity check at import time — keeps the curated lists from
# silently overlapping if someone edits one and forgets the other.
_overlap = KNOWN_SATIRE & KNOWN_NEWS
if _overlap:  # pragma: no cover — defensive
    raise RuntimeError(
        f"KNOWN_SATIRE and KNOWN_NEWS must be disjoint; overlap: {sorted(_overlap)}"
    )
