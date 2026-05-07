"""Domain normalization for the ingest pipeline.

Scrapers stamp ``source_domain`` from a few different inputs (RSS
``<link>``, GDELT's ``domain`` field, HuggingFace ``article_link``).
Each path was extracting its own way, which produced two failure
modes in the Tier 1 build:

* Subdomains drifted apart from their parent. A run that pulled
  ``theonion.com``, ``local.theonion.com``, ``politics.theonion.com``,
  and ``entertainment.theonion.com`` ended up with four "sources" the
  source-balance cap treated independently — letting the Onion
  collectively dominate even though no single subdomain crossed the
  25% line.
* Embedded-protocol URLs slipped through. The ``raquiba/Sarcasm_News_Headline``
  HuggingFace corpus contained ``article_link`` values like
  ``https://www.huffingtonpost.comhttp://...``. ``urlparse`` on those
  returns a netloc of ``www.huffingtonpost.comhttp:`` (the parser stops
  at the next ``/``), which surfaced in the dataset as the source
  ``huffingtonpost.comhttp:``.

:func:`normalize_domain` is the single chokepoint every scraper now
funnels through. It strips scheme/userinfo/port/path, lowercases,
drops ``www.``, rejects malformed inputs, and folds known subdomains
into their canonical parent so source counts and the credibility
allowlists agree on what an outlet is called.
"""
from __future__ import annotations

from urllib.parse import urlparse


# Outlets we routinely see under multiple subdomains. Anything that
# matches one of these as a strict suffix is folded down to the parent;
# the cap on source dominance can then act on the outlet as a whole
# rather than on one of its subsections.
#
# The list is intentionally conservative: only outlets that *actually*
# appeared in a Tier 1 run with split subdomains, or whose parent
# matches an entry in the KNOWN_SATIRE / KNOWN_NEWS allowlists in
# ``source_credibility``. Adding a new entry is a product decision —
# pasting a long list here would silently change credibility verdicts.
_PARENT_DOMAINS: tuple[str, ...] = (
    "theonion.com",
    "babylonbee.com",
    "reductress.com",
    "huffingtonpost.com",
    "huffpost.com",
    "bbc.co.uk",
    "bbc.com",
    "nytimes.com",
    "theguardian.com",
    "npr.org",
    "aljazeera.com",
    "reuters.com",
    "washingtonpost.com",
    "wsj.com",
    "ft.com",
    "bloomberg.com",
    "cnn.com",
    "nbcnews.com",
    "cbsnews.com",
    "abcnews.go.com",
)


def _consolidate_subdomain(host: str) -> str:
    """Fold ``host`` to its parent if it matches one of :data:`_PARENT_DOMAINS`."""
    for parent in _PARENT_DOMAINS:
        if host == parent or host.endswith("." + parent):
            return parent
    return host


def normalize_domain(value: str | None) -> str:
    """Return the canonical normalized domain for a URL or bare hostname.

    Accepts either a full URL (``https://www.example.com/path``) or a
    bare host (``www.example.com``). Returns the empty string for
    inputs that don't look like a parseable hostname — callers should
    treat that as "unknown source" rather than retry-with-fallback.

    The return value is lowercase, with the leading ``www.`` and any
    port removed, and with known multi-subdomain outlets consolidated
    under their parent (see :data:`_PARENT_DOMAINS`).
    """
    if not value:
        return ""
    s = value.strip().lower()
    if not s:
        return ""

    # Embedded-protocol URLs (e.g. "https://www.foo.comhttp://...") are
    # the failure mode that produced "huffingtonpost.comhttp:" entries
    # in past runs. ``urlparse`` happily returns the truncated netloc
    # ``www.foo.comhttp:`` on these because it stops at the next ``/``.
    # Reject the input outright; the source field is more useful blank
    # than corrupted.
    if s.count("://") > 1:
        return ""

    if "://" in s:
        try:
            parsed = urlparse(s)
        except ValueError:
            return ""
        host = parsed.netloc or ""
        # urlparse keeps userinfo on the netloc (e.g. "user:pw@host").
        if "@" in host:
            host = host.rsplit("@", 1)[1]
    else:
        # Bare host (possibly with a path appended). Take everything
        # before the first slash so callers can pass "example.com/x"
        # without us treating "/x" as part of the hostname.
        host = s.split("/", 1)[0]

    # Strip port.
    host = host.split(":", 1)[0]
    # Strip a leading ``www.`` (and any stray dots that survive).
    if host.startswith("www."):
        host = host[4:]
    host = host.strip(".")

    # A real hostname has at least one dot, no spaces, and no remaining
    # path/scheme separators. Everything else is junk we don't want
    # masquerading as a source.
    if not host or "." not in host or " " in host or "/" in host or ":" in host:
        return ""

    return _consolidate_subdomain(host)
