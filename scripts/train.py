"""Main training entry point for the satire-detection engine.

Drives ``SatireTrainer`` through the 3-phase curriculum (or a single
forced phase), supports resuming from a checkpoint, and writes a final
checkpoint with metadata (training stats, graph snapshot window, GNN
version) plus a console evaluation summary.

Run ``python scripts/train.py --help`` for CLI options.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satira import __version__ as SATIRA_VERSION  # noqa: E402
from satira.config import Settings  # noqa: E402
from satira.data.datasets import SatireDataset, create_mock_datasets  # noqa: E402
from satira.models.engine import SatireDetectionEngine  # noqa: E402
from satira.training.evaluation import ModelEvaluator  # noqa: E402
from satira.training.trainer import SatireTrainer  # noqa: E402


logger = logging.getLogger("satira.train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Satira satire-detection engine.",
    )
    parser.add_argument(
        "--phase",
        choices=("1", "2", "3", "all"),
        default="all",
        help=(
            "Run only a specific curriculum phase, or 'all' for the full "
            "auto-advancing 3-phase schedule."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Maximum number of training epochs (default: 25).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from the config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume from.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file with Settings overrides. "
            "Environment variables prefixed with SATIRA_ also override."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Directory to write checkpoints into (default: ./checkpoints).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device to train on. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic mock data + init (default: 42).",
    )
    parser.add_argument(
        "--gnn-version",
        type=str,
        default="v0",
        help=(
            "GNN architecture version this run was trained against; "
            "stored in checkpoint metadata for compatibility checks."
        ),
    )
    parser.add_argument(
        "--graph-snapshot-start",
        type=str,
        default=None,
        help=(
            "ISO timestamp marking the start of the graph snapshot window "
            "this run was trained against."
        ),
    )
    parser.add_argument(
        "--graph-snapshot-end",
        type=str,
        default=None,
        help=(
            "ISO timestamp marking the end of the graph snapshot window."
        ),
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
    logger.warning("CUDA not available — falling back to CPU; training will be slower.")
    return torch.device("cpu")


def load_config(config_path: Path | None, batch_size_override: int | None) -> Settings:
    overrides: dict[str, Any] = {}
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(f"--config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            overrides = json.load(fh)
        if not isinstance(overrides, dict):
            raise ValueError(
                f"--config must contain a JSON object, got {type(overrides).__name__}"
            )
    if batch_size_override is not None:
        overrides["batch_size"] = batch_size_override
    return Settings(**overrides)


def build_datasets() -> tuple[
    tuple[SatireDataset, SatireDataset, SatireDataset], SatireDataset
]:
    """Return (train_tiers, val_dataset).

    The data pipeline is still on placeholder mock datasets. When real
    datasets land, this is the function to swap.
    """
    tier1, tier2, tier3 = create_mock_datasets()
    val = create_mock_datasets()[0]
    return (tier1, tier2, tier3), val


def force_phase(trainer: SatireTrainer, phase: int) -> None:
    """Pin the trainer to a specific phase and disable auto-advancement."""
    trainer.phase_controller._phase = phase  # type: ignore[attr-defined]
    trainer.model.freeze_for_phase(phase)
    trainer.optimizer = trainer._build_optimizer()


def run_training(
    trainer: SatireTrainer,
    epochs: int,
    forced_phase: int | None,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Run the training loop and return a history + summary dict."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    phase_transitions: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        phase_before = trainer.phase_controller.phase
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate()

        advanced = False
        if forced_phase is None:
            advanced = trainer._handle_phase_transition(epoch, train_metrics)
            if advanced:
                phase_transitions.append(
                    {
                        "epoch": epoch,
                        "from": phase_before,
                        "to": trainer.phase_controller.phase,
                    }
                )

        if val_metrics["loss"] < trainer.best_val_loss - 1e-6:
            trainer.best_val_loss = val_metrics["loss"]
            trainer._epochs_since_val_improvement = 0
            trainer.best_checkpoint_path = trainer.save_checkpoint(
                checkpoint_dir / "best.pt"
            )
        else:
            trainer._epochs_since_val_improvement += 1

        logger.info(
            "epoch %d phase %d train_loss=%.4f val_loss=%.4f "
            "train_acc=%.4f val_acc=%.4f%s",
            epoch,
            train_metrics["phase"],
            train_metrics["loss"],
            val_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["accuracy"],
            f" (phase advanced {phase_before}->{trainer.phase_controller.phase})"
            if advanced
            else "",
        )

        history.append(
            {
                "epoch": epoch,
                "phase": train_metrics["phase"],
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "gate_activation_variance": train_metrics["gate_activation_variance"],
                "grad_norm": train_metrics["grad_norm"],
                "projection_grad_norm": train_metrics["projection_grad_norm"],
            }
        )

        if (
            forced_phase is None
            and trainer._epochs_since_val_improvement >= trainer.EARLY_STOP_PATIENCE
        ):
            logger.info("early stopping at epoch %d", epoch)
            break

    return {
        "history": history,
        "phase_transitions": phase_transitions,
        "best_val_loss": trainer.best_val_loss,
        "best_checkpoint": str(trainer.best_checkpoint_path)
        if trainer.best_checkpoint_path
        else None,
        "final_phase": trainer.phase_controller.phase,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
    }


def save_final_checkpoint(
    trainer: SatireTrainer,
    config: Settings,
    args: argparse.Namespace,
    summary: dict[str, Any],
    out_path: Path,
) -> None:
    """Write the final checkpoint with rich metadata for deployment."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_window = (
        args.graph_snapshot_start or _now_iso(),
        args.graph_snapshot_end or _now_iso(),
    )
    payload = {
        "model_state": trainer.model.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "phase": trainer.phase_controller.phase,
        "best_val_loss": trainer.best_val_loss,
        "metadata": {
            "satira_version": SATIRA_VERSION,
            "gnn_version": args.gnn_version,
            "graph_snapshot_window": list(snapshot_window),
            "training_stats": {
                "final_phase": summary["final_phase"],
                "final_train_loss": summary["final_train_loss"],
                "final_val_loss": summary["final_val_loss"],
                "best_val_loss": summary["best_val_loss"],
                "epochs_run": len(summary["history"]),
                "phase_transitions": summary["phase_transitions"],
            },
            "config": config.model_dump(),
            "trained_at": _now_iso(),
        },
    }
    torch.save(payload, out_path)
    logger.info("wrote final checkpoint: %s", out_path)


def evaluation_summary(trainer: SatireTrainer, config: Settings) -> str:
    """Compact text summary using ``trainer.validate`` so we don't need a
    real DataLoader for the quick post-training read-out.
    """
    metrics = trainer.validate()
    lines = [
        "=== final evaluation ===",
        f"  val_loss:           {metrics['loss']:.4f}",
        f"  val_accuracy:       {metrics['accuracy']:.4f}",
        f"  calibration_error:  {metrics['calibration_error']:.4f}",
    ]
    f1s = metrics.get("f1_per_class") or []
    names = list(getattr(config, "CLASS_NAMES", []) or [])
    for i, f1 in enumerate(f1s):
        name = names[i] if i < len(names) else f"class_{i}"
        lines.append(f"  f1[{name:>22s}]: {f1:.4f}")
    return "\n".join(lines)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _phase_arg_to_int(phase: str) -> int | None:
    return None if phase == "all" else int(phase)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    try:
        torch.manual_seed(args.seed)

        device = pick_device(args.device)
        logger.info("device=%s", device)

        config = load_config(args.config, args.batch_size)
        logger.info(
            "config: d_model=%d num_heads=%d num_classes=%d batch_size=%d "
            "learning_rate=%.2e",
            config.d_model,
            config.num_heads,
            config.num_classes,
            config.batch_size,
            config.learning_rate,
        )

        train_tiers, val_dataset = build_datasets()
        logger.info(
            "datasets: tier1=%d tier2=%d tier3=%d val=%d",
            len(train_tiers[0]),
            len(train_tiers[1]),
            len(train_tiers[2]),
            len(val_dataset),
        )

        model = SatireDetectionEngine(config)
        param_counts = model.count_parameters()
        logger.info("total parameters: %s", f"{param_counts['total']:,}")

        trainer = SatireTrainer(
            model=model,
            config=config,
            train_datasets=train_tiers,
            val_dataset=val_dataset,
            device=str(device),
        )

        if args.checkpoint is not None:
            if not args.checkpoint.is_file():
                raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
            logger.info("resuming from checkpoint: %s", args.checkpoint)
            trainer.load_checkpoint(args.checkpoint)

        forced_phase = _phase_arg_to_int(args.phase)
        if forced_phase is not None:
            logger.info("forcing phase %d (auto-advancement disabled)", forced_phase)
            force_phase(trainer, forced_phase)

        summary = run_training(
            trainer=trainer,
            epochs=args.epochs,
            forced_phase=forced_phase,
            checkpoint_dir=args.checkpoint_dir,
        )

        final_path = args.checkpoint_dir / "final.pt"
        save_final_checkpoint(trainer, config, args, summary, final_path)

        print("\n" + evaluation_summary(trainer, config))
        print(
            f"\nbest_val_loss={summary['best_val_loss']:.4f} "
            f"final_phase={summary['final_phase']} "
            f"epochs_run={len(summary['history'])}"
        )
        if summary["best_checkpoint"]:
            print(f"best checkpoint: {summary['best_checkpoint']}")
        print(f"final checkpoint: {final_path}")

        return 0

    except Exception as exc:  # noqa: BLE001 — surface every failure at the entry point
        logger.error("training failed: %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
