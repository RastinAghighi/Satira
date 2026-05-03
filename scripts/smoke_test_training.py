"""End-to-end smoke test for the satire-detection training pipeline.

Runs a small 5-epoch curriculum-training loop over mock datasets to confirm
the engine, phased loss, scheduler, and trainer cooperate without errors and
loss/grad behaviour looks sane. Not intended to produce a useful model.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

# Make src/ importable when run as a script from the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satira.config import Settings  # noqa: E402
from satira.data.datasets import SatireDataset, create_mock_datasets  # noqa: E402
from satira.models.engine import SatireDetectionEngine  # noqa: E402
from satira.training.trainer import SatireTrainer  # noqa: E402


CHECKPOINT_DIR = Path("./checkpoints/smoke_test")
REPORT_PATH = CHECKPOINT_DIR / "report.json"
NUM_EPOCHS = 5
SAMPLES_PER_TIER = 50
VAL_SIZE = 20
SEED = 42


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print(
        "[warn] CUDA not available — falling back to CPU. "
        "Training will be noticeably slower than on GPU."
    )
    return torch.device("cpu")


def truncate_dataset(ds: SatireDataset, n: int) -> SatireDataset:
    return SatireDataset(ds.data_manifest[:n], transform=ds.transform)


def build_mixed_validation_dataset(
    tier1: SatireDataset, tier2: SatireDataset, tier3: SatireDataset, total: int
) -> SatireDataset:
    """Pull samples from each tier in roughly equal proportion, taking from
    the *tail* of each manifest so they don't overlap with the truncated
    train tiers."""
    per_tier = total // 3
    remainder = total - per_tier * 3
    counts = [per_tier + (1 if i < remainder else 0) for i in range(3)]
    manifest: list[dict] = []
    for ds, count in zip((tier1, tier2, tier3), counts):
        manifest.extend(ds.data_manifest[-count:])
    return SatireDataset(manifest)


def per_projection_grad_norms(model: SatireDetectionEngine) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in ("v_proj", "t_proj", "temp_proj", "graph_proj"):
        module = getattr(model, name)
        total_sq = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().pow(2).sum().item()
        out[name] = math.sqrt(total_sq)
    return out


def grads_have_nan_or_inf(model: SatireDetectionEngine) -> tuple[bool, str | None]:
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if torch.isnan(g).any().item():
            return True, f"NaN in grad of {name}"
        if torch.isinf(g).any().item():
            return True, f"Inf in grad of {name}"
    return False, None


def format_summary_table(history: list[dict[str, Any]]) -> str:
    headers = [
        "Ep",
        "Phase",
        "TrainLoss",
        "ValLoss",
        "TrainAcc",
        "GateVar",
        "PrjGrad",
        "T1",
        "T2",
        "T3",
    ]
    rows = [headers]
    for h in history:
        rows.append(
            [
                str(h["epoch"]),
                str(h["phase"]),
                f"{h['train_loss']:.4f}",
                f"{h['val_loss']:.4f}",
                f"{h['train_accuracy']:.4f}",
                f"{h['gate_variance']:.4f}",
                f"{h['projection_grad_norm']:.4f}",
                f"{h['tier_weights']['tier1_easy']:.2f}",
                f"{h['tier_weights']['tier2_contradiction']:.2f}",
                f"{h['tier_weights']['tier3_hard_negatives']:.2f}",
            ]
        )
    widths = [max(len(r[c]) for r in rows) for c in range(len(headers))]
    sep = "+".join("-" * (w + 2) for w in widths)
    sep = f"+{sep}+"
    lines = [sep]
    for i, row in enumerate(rows):
        lines.append(
            "| " + " | ".join(cell.ljust(widths[c]) for c, cell in enumerate(row)) + " |"
        )
        if i == 0:
            lines.append(sep)
    lines.append(sep)
    return "\n".join(lines)


def run_smoke_test() -> int:
    torch.manual_seed(SEED)

    device = pick_device()
    print(f"[setup] device = {device}")

    config = Settings(d_model=128, num_heads=4)
    print(
        f"[setup] config: d_model={config.d_model} num_heads={config.num_heads} "
        f"num_classes={config.num_classes} batch_size={config.batch_size}"
    )

    tier1, tier2, tier3 = create_mock_datasets()
    tier1 = truncate_dataset(tier1, SAMPLES_PER_TIER)
    tier2 = truncate_dataset(tier2, SAMPLES_PER_TIER)
    tier3 = truncate_dataset(tier3, SAMPLES_PER_TIER)
    val_dataset = build_mixed_validation_dataset(*create_mock_datasets(), total=VAL_SIZE)
    print(
        f"[data] tier1={len(tier1)} tier2={len(tier2)} tier3={len(tier3)} "
        f"val={len(val_dataset)}"
    )

    model = SatireDetectionEngine(config)
    param_counts = model.count_parameters()
    print(f"[model] parameter counts: {param_counts}")
    print(f"[model] total parameters: {param_counts['total']:,}")

    trainer = SatireTrainer(
        model=model,
        config=config,
        train_datasets=(tier1, tier2, tier3),
        val_dataset=val_dataset,
        device=str(device),
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    nonzero_streams = {k: False for k in ("v_proj", "t_proj", "temp_proj", "graph_proj")}
    grad_finite_all_epochs = True
    grad_failure_reason: str | None = None
    phase_sequence: list[int] = []
    phase_transitions: list[dict[str, Any]] = []

    for epoch in range(1, NUM_EPOCHS + 1):
        phase_before = trainer.phase_controller.phase
        train_metrics = trainer.train_epoch(epoch)

        # Inspect grads while they still exist (zero_grad is at start of next epoch).
        has_bad, bad_reason = grads_have_nan_or_inf(model)
        if has_bad:
            grad_finite_all_epochs = False
            grad_failure_reason = bad_reason

        proj_norms = per_projection_grad_norms(model)
        for k, v in proj_norms.items():
            if v > 0.0:
                nonzero_streams[k] = True

        val_metrics = trainer.validate()

        advanced = trainer._handle_phase_transition(epoch, train_metrics)
        phase_after = trainer.phase_controller.phase
        if advanced:
            phase_transitions.append(
                {"epoch": epoch, "from": phase_before, "to": phase_after}
            )

        ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch}.pt"
        trainer.save_checkpoint(ckpt_path)

        tier_weights = trainer.scheduler.get_tier_weights(epoch)

        record = {
            "epoch": epoch,
            "phase": train_metrics["phase"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "gate_variance": train_metrics["gate_activation_variance"],
            "grad_norm": train_metrics["grad_norm"],
            "projection_grad_norm": train_metrics["projection_grad_norm"],
            "per_projection_grad_norm": proj_norms,
            "tier_weights": tier_weights,
            "checkpoint": str(ckpt_path),
            "phase_advanced": advanced,
        }
        history.append(record)
        phase_sequence.append(phase_after)

        print(f"\n--- epoch {epoch} ---")
        print(f"  phase           : {record['phase']}")
        print(f"  train loss      : {record['train_loss']:.6f}")
        print(f"  val loss        : {record['val_loss']:.6f}")
        print(f"  train accuracy  : {record['train_accuracy']:.4f}")
        if record["phase"] >= 2:
            print(f"  gate variance   : {record['gate_variance']:.6f}")
        else:
            print(
                f"  gate variance   : (n/a in phase 1; observed {record['gate_variance']:.6f})"
            )
        print(f"  proj grad norms : {{{', '.join(f'{k}={v:.4f}' for k, v in proj_norms.items())}}}")
        print(f"  total grad norm : {record['grad_norm']:.6f}")
        print(
            f"  tier weights    : "
            f"t1={tier_weights['tier1_easy']:.3f} "
            f"t2={tier_weights['tier2_contradiction']:.3f} "
            f"t3={tier_weights['tier3_hard_negatives']:.3f}"
        )
        if advanced:
            print(f"  phase advanced  : {phase_before} -> {phase_after}")
        print(f"  checkpoint      : {ckpt_path}")

    # ---- assertions ----
    failures: list[str] = []

    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]
    if not (final_loss < initial_loss):
        failures.append(
            f"loss did not decrease over the run (initial={initial_loss:.4f}, "
            f"final={final_loss:.4f})"
        )

    if not grad_finite_all_epochs:
        failures.append(f"non-finite gradient detected: {grad_failure_reason}")

    if any(p < 1 or p > 3 for p in phase_sequence):
        failures.append(f"phase out of range over run: {phase_sequence}")
    for prev, nxt in zip(phase_sequence, phase_sequence[1:]):
        if nxt < prev:
            failures.append(f"phase regressed: {phase_sequence}")
            break

    missing_streams = [k for k, v in nonzero_streams.items() if not v]
    if missing_streams:
        failures.append(
            f"input streams with zero gradient across all epochs: {missing_streams}"
        )

    summary_table = format_summary_table(history)
    print("\n" + summary_table)

    passed = len(failures) == 0
    status_line = "SMOKE TEST PASSED" if passed else f"SMOKE TEST FAILED: {'; '.join(failures)}"
    print("\n" + status_line)

    report = {
        "passed": passed,
        "failures": failures,
        "device": str(device),
        "seed": SEED,
        "config_overrides": {"d_model": config.d_model, "num_heads": config.num_heads},
        "parameter_counts": param_counts,
        "history": history,
        "phase_sequence": phase_sequence,
        "phase_transitions": phase_transitions,
        "nonzero_streams": nonzero_streams,
        "grad_finite_all_epochs": grad_finite_all_epochs,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(f"[report] wrote {REPORT_PATH}")

    return 0 if passed else 1


def main() -> int:
    try:
        return run_smoke_test()
    except Exception as exc:  # noqa: BLE001 — surface every failure with a clear trace
        print(f"\nSMOKE TEST FAILED: unhandled exception: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
