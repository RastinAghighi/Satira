"""Runnable entry point for the offline graph resolution job.

Invoked as ``python -m satira.graph.offline_pipeline`` — this is the target of
``docker/Dockerfile.offline`` (``CMD ["python", "-m",
"satira.graph.offline_pipeline"]``) and the ``offline`` service in
``docker-compose.yml``.

Like :mod:`satira.inference.worker`, this is a *thin wrapper* around existing
classes, not new resolution logic. It loads a persisted
:class:`~satira.graph.store.GraphStore` snapshot, builds the Tier-2
:class:`~satira.graph.batch_resolver.BatchResolver` over it, pulls a batch of
pending mentions, runs one ``resolve_batch`` pass, and reports the decision
summary. The merge/review/create *decisions* come straight from the resolver;
this module does not auto-apply merges — applying a merge is a moderation-policy
step that belongs to the review flow, not to a batch runner.

Data-source seam (deliberate, documented): the "warm path" is meant to pull
unresolved mentions off a queue every few minutes. That queue/DB integration is
not wired in this codebase yet, so pending mentions are read from a JSON file
(``--pending``) — an explicit, inspectable stand-in for the eventual source.
With no ``--pending`` given, a pass resolves an empty batch (a no-op) and exits
cleanly, which is what keeps the module runnable and testable today.

Usage::

    # single pass over a snapshot + pending file, then exit
    python -m satira.graph.offline_pipeline --graph-snapshot graph.json --pending pending.json

    # warm-path loop: re-read pending and resolve every 300s
    python -m satira.graph.offline_pipeline --graph-snapshot graph.json --pending pending.json --interval 300

    # build the resolver and exit 0 (readiness/CI check)
    python -m satira.graph.offline_pipeline --check
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

from satira.config import Settings
from satira.graph.batch_resolver import BatchResolver
from satira.graph.store import GraphStore


logger = logging.getLogger("satira.graph.offline_pipeline")


def _entity_count(store: GraphStore) -> int:
    return sum(1 for kind in store._node_kind.values() if kind == "entity")


def build_resolver(
    config: Settings | None = None,
    *,
    graph_snapshot: str | None = None,
) -> BatchResolver:
    """Build a :class:`BatchResolver` over a :class:`GraphStore`.

    If ``graph_snapshot`` points at an existing JSON snapshot (as produced by
    :meth:`GraphStore.snapshot`), it is restored into the store first; otherwise
    the resolver starts over an empty store.
    """
    config = config or Settings()  # reserved for future thresholds/knobs
    store = GraphStore()

    if graph_snapshot:
        path = Path(graph_snapshot)
        if path.exists():
            store.restore(json.loads(path.read_text(encoding="utf-8")))
            logger.info(
                "restored graph snapshot from %s (%d entities)",
                path,
                _entity_count(store),
            )
        else:
            logger.warning(
                "graph snapshot %s not found; starting from an empty store", path
            )

    return BatchResolver(store)


def load_pending(pending_path: str | None) -> list[dict]:
    """Read pending mentions (a JSON list of mention dicts) from ``pending_path``."""
    if not pending_path:
        return []
    path = Path(pending_path)
    if not path.exists():
        logger.warning("pending-mentions file %s not found; nothing to resolve", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"pending-mentions file {path} must contain a JSON list, got {type(data).__name__}"
        )
    return data


def run_once(resolver: BatchResolver, pending: list[dict]) -> list[dict]:
    """Run one resolution pass and log a decision summary."""
    decisions = resolver.resolve_batch(pending)
    summary = Counter(d["action"] for d in decisions)
    logger.info(
        "resolved %d mention(s): %s | blocking narrowed comparisons %d -> %d",
        len(decisions),
        dict(summary),
        resolver.last_naive_comparison_count,
        resolver.last_comparison_count,
    )
    return decisions


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Satira offline graph resolution job.")
    parser.add_argument(
        "--graph-snapshot",
        default=None,
        help="Path to a GraphStore JSON snapshot to restore before resolving.",
    )
    parser.add_argument(
        "--pending",
        default=None,
        help="Path to a JSON list of pending mention dicts to resolve.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="If > 0, loop forever, re-reading --pending and resolving every "
        "INTERVAL seconds (the warm-path cadence). Default: single pass then exit.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Build the resolver, then exit 0 without resolving (readiness/CI check).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    resolver = build_resolver(graph_snapshot=args.graph_snapshot)

    if args.check:
        logger.info(
            "self-check OK: resolver built over %d entities",
            _entity_count(resolver.graph_store),
        )
        return 0

    if args.interval and args.interval > 0:
        logger.info("offline resolver loop started; interval=%ss", args.interval)
        while True:
            run_once(resolver, load_pending(args.pending))
            time.sleep(args.interval)

    run_once(resolver, load_pending(args.pending))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
