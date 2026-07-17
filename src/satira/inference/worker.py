"""Runnable entry point for the inference worker (``python -m satira.inference.worker``).

This is a *thin process wrapper*, not new inference logic. It assembles the
components that already exist — :class:`~satira.inference.context_resolver.ContextResolver`,
:class:`~satira.models.engine.SatireDetectionEngine`, and the
:class:`~satira.inference.batcher.DynamicBatcher` owned by
:class:`~satira.inference.pipeline.InferencePipeline` — into one long-lived
process, starts the dynamic-batching loop via ``pipeline.initialize()``, and
blocks until it receives SIGINT/SIGTERM, at which point it drains and shuts the
batcher down cleanly. Every decision about preprocessing, batching, and the
forward pass stays in the classes wired together here.

``docker/Dockerfile.inference`` runs this module (``CMD ["python", "-m",
"satira.inference.worker"]``); the container needs a valid, importable,
long-running target and this provides one.

Encoder gap (deliberate, documented): this codebase ships *mock* vision/text/OCR
encoders for tests but no production encoder yet, so :func:`build_pipeline`
leaves those unset. The worker therefore boots the batching machinery and awaits
that integration rather than inventing an encoder. When real encoders land, pass
them into :func:`build_pipeline` — that is the only change needed here.

Usage::

    python -m satira.inference.worker            # run until SIGINT/SIGTERM
    python -m satira.inference.worker --check     # boot + shut down, for CI/HEALTHCHECK
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal

import torch

from satira.config import Settings
from satira.graph.embedding_cache import GraphEmbeddingCache
from satira.graph.entity_resolution import MentionNormalizer
from satira.inference.context_resolver import ContextResolver
from satira.inference.pipeline import InferencePipeline
from satira.models.engine import SatireDetectionEngine
from satira.temporal.index_manager import FAISSIndexManager
from satira.temporal.retriever import TemporalContextRetriever


logger = logging.getLogger("satira.inference.worker")


def _select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_pipeline(
    config: Settings | None = None,
    *,
    device: str | None = None,
) -> InferencePipeline:
    """Wire the existing inference components into an :class:`InferencePipeline`.

    Sub-component dimensions come from ``config`` (``graph_dim`` for the graph
    embedding cache, ``temporal_dim`` for the FAISS index). The vision/text/OCR
    encoders are intentionally left ``None`` — see the module docstring.
    """
    config = config or Settings()
    device = device or _select_device()

    normalizer = MentionNormalizer()
    graph_cache = GraphEmbeddingCache(embedding_dim=config.graph_dim)
    temporal = TemporalContextRetriever(
        FAISSIndexManager(dim=config.temporal_dim),
        timeout_ms=config.batch_timeout_ms,
    )
    resolver = ContextResolver(
        mention_normalizer=normalizer,
        graph_cache=graph_cache,
        temporal_retriever=temporal,
    )

    model = SatireDetectionEngine(config)
    model.eval()

    return InferencePipeline(
        config=config,
        context_resolver=resolver,
        model=model,
        device=device,
    )


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop: asyncio.Event) -> None:
    """Set ``stop`` on SIGINT/SIGTERM, portably.

    ``loop.add_signal_handler`` is the clean path on POSIX (where the container
    runs). Windows' ProactorEventLoop doesn't implement it, so fall back to
    ``signal.signal`` there so a developer can still Ctrl-C the process.
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError, OSError):
            try:
                signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
            except (ValueError, OSError):  # signal not settable on this platform
                pass


async def _serve(pipeline: InferencePipeline, *, check_only: bool = False) -> None:
    await pipeline.initialize()
    logger.info("inference worker initialized (device=%s)", pipeline._device)

    if check_only:
        await pipeline.shutdown()
        logger.info("self-check OK: batcher started and stopped cleanly")
        return

    stop = asyncio.Event()
    _install_signal_handlers(asyncio.get_running_loop(), stop)
    logger.info("inference worker ready; awaiting shutdown signal (SIGINT/SIGTERM)")
    try:
        await stop.wait()
    finally:
        logger.info("shutdown signal received; draining batcher")
        await pipeline.shutdown()
        logger.info("inference worker stopped")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Satira inference worker.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Build and initialize the pipeline, then shut down and exit 0 "
        "(readiness/CI check; does not serve).",
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
    pipeline = build_pipeline()
    asyncio.run(_serve(pipeline, check_only=args.check))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
