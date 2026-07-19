"""Canonical label taxonomy for Satira.

Single source of truth for the class set, the string<->int mapping, and the
subset helpers. Anything that needs to turn a label string into an integer
index (or back) imports from here instead of hand-rolling its own dict.

The canonical taxonomy has FOUR classes::

    0  authentic
    1  satire
    2  misleading_context
    3  fabricated

``parody`` is not a separate class. It is merged into ``satire`` and accepted
as an input alias only (see :data:`ALIASES`), so legacy sources that still emit
``"parody"`` map onto ``satire`` rather than being rejected.
"""
from __future__ import annotations

from enum import IntEnum


class Label(IntEnum):
    """Canonical class indices. The integer values are the label ids."""

    AUTHENTIC = 0
    SATIRE = 1
    MISLEADING_CONTEXT = 2
    FABRICATED = 3


#: Canonical class names in index order: ("authentic", "satire", ...).
CANONICAL_NAMES: tuple[str, ...] = tuple(member.name.lower() for member in Label)

#: name -> index for the canonical classes.
STR_TO_INT: dict[str, int] = {name: int(member) for name, member in zip(CANONICAL_NAMES, Label)}

#: index -> name for the canonical classes.
INT_TO_STR: dict[int, str] = {int(member): name for name, member in zip(CANONICAL_NAMES, Label)}

#: Input aliases folded onto a canonical class. ``parody`` was its own class in
#: the previous 5-class taxonomy; it is now merged into ``satire``.
ALIASES: dict[str, int] = {"parody": int(Label.SATIRE)}

#: Binary subset used by the tier-1 build and the first real training run:
#: authentic vs satire, sharing the canonical indices 0 and 1.
BINARY_NAMES: tuple[str, str] = ("authentic", "satire")

#: name -> index restricted to the binary subset (a drop-in for a bespoke
#: ``{"authentic": 0, "satire": 1}`` map; ``.get()`` returns ``None`` for any
#: out-of-subset label so callers can drop it).
BINARY_STR_TO_INT: dict[str, int] = {name: STR_TO_INT[name] for name in BINARY_NAMES}


def normalize(name: str) -> str:
    """Canonicalize a raw label string for lookup (strip + lowercase)."""
    return name.strip().lower()


def str_to_int(name: str, *, allow_aliases: bool = True) -> int:
    """Map a label string to its canonical integer index.

    Raises :class:`ValueError` on any string that is neither a canonical class
    name nor (when ``allow_aliases``) a known alias. It never silently maps an
    unknown string to a default class.
    """
    if not isinstance(name, str):
        raise TypeError(f"label must be a str, got {type(name).__name__}")
    key = normalize(name)
    if key in STR_TO_INT:
        return STR_TO_INT[key]
    if allow_aliases and key in ALIASES:
        return ALIASES[key]
    expected = sorted(STR_TO_INT)
    if allow_aliases:
        expected = expected + [f"{a!r}(alias)" for a in sorted(ALIASES)]
    raise ValueError(f"unknown label {name!r}; expected one of {expected}")


def int_to_str(index: int) -> str:
    """Map a canonical integer index to its class name.

    Raises :class:`ValueError` on an out-of-range index.
    """
    try:
        return INT_TO_STR[int(index)]
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(
            f"unknown label index {index!r}; expected 0..{len(CANONICAL_NAMES) - 1}"
        ) from exc


def class_names(num_classes: int | None = None) -> list[str]:
    """Canonical class names, or the first ``num_classes`` of them.

    ``num_classes`` selects a prefix subset — e.g. ``2`` yields
    ``["authentic", "satire"]`` for the binary run. Raises if it falls outside
    ``1..len(CANONICAL_NAMES)``.
    """
    if num_classes is None:
        return list(CANONICAL_NAMES)
    if not 1 <= num_classes <= len(CANONICAL_NAMES):
        raise ValueError(
            f"num_classes must be in 1..{len(CANONICAL_NAMES)}, got {num_classes}"
        )
    return list(CANONICAL_NAMES[:num_classes])
