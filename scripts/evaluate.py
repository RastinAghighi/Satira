"""Evaluate a trained Satira checkpoint on a dataset.

Loads a checkpoint produced by ``scripts/train.py``, runs
``ModelEvaluator`` over a dataset, prints the ``EvalReport`` summary,
and optionally writes the report as JSON.

The data pipeline is still on placeholder mock datasets, so the eval
batch iterator synthesises engine-input tensors (v_patches, t_tokens,
temporal_ctx, graph_ctx) of the right shape per the config. When real
encoder outputs land, swap ``_iter_eval_batches`` for the real loader.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Iterator

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satira.config import Settings  # noqa: E402
from satira.data.datasets import SatireDataset, create_mock_datasets  # noqa: E402
from satira.models.engine import SatireDetectionEngine  # noqa: E402
from satira.training.evaluation import ModelEvaluator  # noqa: E402


logger = logging.getLogger("satira.evaluate")


DATASET_CHOICES = ("tier1", "tier2", "tier3", "all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Satira checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="all",
        help="Which mock tier to evaluate against, or 'all' to concatenate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override eval batch size (defaults to config.batch_size).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file with Settings overrides. Used as a fallback "
            "when the checkpoint did not embed its config under metadata."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device to run evaluation on.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the EvalReport as JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthesised inputs (default: 42).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning("CUDA not available — falling back to CPU.")
    return torch.device("cpu")


def load_config_for_checkpoint(
    checkpoint: dict, fallback_path: Path | None
) -> Settings:
    """Prefer the config embedded in checkpoint metadata; fall back to a
    JSON file or environment-driven defaults.
    """
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    embedded = metadata.get("config") if isinstance(metadata, dict) else None
    if isinstance(embedded, dict) and embedded:
        return Settings(**embedded)

    if fallback_path is not None:
        if not fallback_path.is_file():
            raise FileNotFoundError(f"--config file not found: {fallback_path}")
        with fallback_path.open("r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        if not isinstance(overrides, dict):
            raise ValueError(
                f"--config must contain a JSON object, got {type(overrides).__name__}"
            )
        return Settings(**overrides)

    return Settings()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def select_dataset(name: str) -> SatireDataset:
    tier1, tier2, tier3 = create_mock_datasets()
    if name == "tier1":
        return tier1
    if name == "tier2":
        return tier2
    if name == "tier3":
        return tier3
    if name == "all":
        manifest = (
            tier1.data_manifest + tier2.data_manifest + tier3.data_manifest
        )
        return SatireDataset(manifest)
    raise ValueError(f"unknown dataset {name!r}")


def _iter_eval_batches(
    dataset: SatireDataset,
    config: Settings,
    batch_size: int,
    device: torch.device,
) -> Iterator[dict]:
    """Yield evaluation batches with synthesised engine inputs.

    Real encoder outputs are not wired up yet; this function fills the
    contract ModelEvaluator expects (v_patches, t_tokens, temporal_ctx,
    graph_ctx, label) so the rest of the eval path is exercised.
    """
    n = len(dataset)
    if n == 0:
        return

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs = end - start

        labels = []
        for i in range(start, end):
            raw = dataset[i]["label"]
            if isinstance(raw, torch.Tensor):
                labels.append(int(raw.item()))
            elif isinstance(raw, (int, bool)):
                labels.append(int(raw))
            else:
                labels.append(0)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

        v = torch.randn(bs, 10, config.vision_dim, device=device)
        t = torch.randn(bs, 12, config.text_dim, device=device)
        temp = torch.randn(bs, config.temporal_dim, device=device)
        graph = torch.randn(bs, config.graph_dim, device=device)

        yield {
            "v_patches": v,
            "t_tokens": t,
            "temporal_ctx": temp,
            "graph_ctx": graph,
            "label": label_tensor,
        }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        torch.manual_seed(args.seed)

        device = pick_device(args.device)
        logger.info("device=%s", device)

        ckpt = load_checkpoint(args.checkpoint, device)
        logger.info("loaded checkpoint: %s", args.checkpoint)

        config = load_config_for_checkpoint(ckpt, args.config)
        batch_size = args.batch_size or config.batch_size
        logger.info(
            "config: d_model=%d num_heads=%d num_classes=%d batch_size=%d",
            config.d_model,
            config.num_heads,
            config.num_classes,
            batch_size,
        )

        model = SatireDetectionEngine(config)
        if "model_state" not in ckpt:
            raise KeyError("checkpoint is missing 'model_state'")
        model.load_state_dict(ckpt["model_state"])

        dataset = select_dataset(args.dataset)
        logger.info("dataset=%s n=%d", args.dataset, len(dataset))

        evaluator = ModelEvaluator(model=model, config=config, device=str(device))
        report = evaluator.evaluate(
            _iter_eval_batches(dataset, config, batch_size, device)
        )

        print("\n" + report.summary())

        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("w", encoding="utf-8") as fh:
                json.dump(report.to_dict(), fh, indent=2)
            logger.info("wrote report: %s", args.report)

        return 0

    except Exception as exc:  # noqa: BLE001
        logger.error("evaluation failed: %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
