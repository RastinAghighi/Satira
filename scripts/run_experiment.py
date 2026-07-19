"""Execute the first real-data training run and emit its diagnostic artifacts.

Binary run (authentic=0, satire=1) over the image-bearing corpus, all context
streams cold-started to the learned fallback. Trains the three-phase curriculum,
then evaluates on the held-out test split and writes a run directory with: loss
curves per phase, the confusion matrix, per-class precision/recall/F1 + macro-F1,
ECE with reliability-diagram data, the learned temperature, the mean
contradiction-gate activation per class on val (the satire-vs-authentic gate
separation check), and two reference baselines.

Read this before reading any number it produces: in this corpus the source
outlet perfectly determines the label, and both the CLIP/RoBERTa features and the
text baseline are downstream of the source. Every metric below is therefore
inflated by construction. This run exists to validate that the data path, the
padding masks, the deterministic cold-start substitution, and the
contradiction-gate mechanism are wired correctly end to end — not to characterize
how well the model separates satire from authentic reporting.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import matplotlib

matplotlib.use("Agg")  # headless; write PNGs without a display
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from satira import labels  # noqa: E402
from satira.config import Settings  # noqa: E402
from satira.data.corpus_dataset import (  # noqa: E402
    DEFAULT_EMBEDDING_DIR,
    collate_embeddings,
)
from satira.models.engine import SatireDetectionEngine  # noqa: E402
from satira.training.losses import per_sample_gate_activation  # noqa: E402
from satira.training.trainer import SatireTrainer  # noqa: E402

# Reuse the dataset builder and training loop from the training entry point.
import train as train_entry  # noqa: E402

logger = logging.getLogger("satira.run_experiment")

NUM_ECE_BINS = 15


# --- config -----------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run #1: real-data wiring/mechanism validation.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=REPO_ROOT / "runs" / "run1_binary")
    parser.add_argument(
        "--class-weight-scheme",
        choices=("uniform", "inverse_frequency"),
        default="uniform",
        help="Loss class weighting; uniform is this run's default.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args(argv)


def build_binary_config(args: argparse.Namespace) -> Settings:
    """2-class head over the canonical authentic/satire subset."""
    return Settings(
        num_classes=2,
        CLASS_GATE_TARGETS={0: 0.1, 1: 0.9},
        batch_size=args.batch_size,
        class_weight_scheme=args.class_weight_scheme,
    )


def pick_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- forward pass over a split ---------------------------------------------
@torch.no_grad()
def forward_split(
    model: SatireDetectionEngine,
    dataset,
    config: Settings,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the model over a whole cached-embedding split.

    Returns ``(logits, targets, gate_activation)`` on CPU, where gate_activation
    is the mask-aware per-sample mean contradiction-gate value.
    """
    model.eval()
    logits_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    gate_parts: list[torch.Tensor] = []
    batch_size = config.batch_size
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = collate_embeddings(samples)
        n = len(samples)
        v = batch["vision"].to(device)
        t = batch["text"].to(device)
        temp = torch.zeros(n, config.temporal_dim, device=device)
        graph = torch.zeros(n, config.graph_dim, device=device)
        mask = batch["text_key_padding_mask"].to(device)
        logits, _t2v, _v2t, t_gate, v_gate = model(
            v,
            t,
            temp,
            graph,
            text_key_padding_mask=mask,
            temporal_present=batch["temporal_present"].to(device),
            graph_present=batch["graph_present"].to(device),
        )
        logits_parts.append(logits.cpu())
        target_parts.append(batch["label"])
        gate_parts.append(per_sample_gate_activation(t_gate, v_gate, mask).cpu())
    return (
        torch.cat(logits_parts),
        torch.cat(target_parts),
        torch.cat(gate_parts),
    )


# --- metrics ----------------------------------------------------------------
def classification_metrics(preds: np.ndarray, targets: np.ndarray, class_names: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "per_class": per_class,
    }


def reliability_and_ece(probs: torch.Tensor, targets: torch.Tensor, num_bins: int = NUM_ECE_BINS) -> tuple[list[dict], float]:
    """Reliability-diagram bins and the Expected Calibration Error over them."""
    confidence, preds = probs.max(dim=-1)
    correct = (preds == targets).float()
    n = confidence.numel()
    edges = torch.linspace(0.0, 1.0, num_bins + 1)
    bins: list[dict] = []
    ece = 0.0
    for i in range(num_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        if i == num_bins - 1:
            in_bin = (confidence >= lo) & (confidence <= hi)
        else:
            in_bin = (confidence >= lo) & (confidence < hi)
        count = int(in_bin.sum().item())
        avg_conf = float(confidence[in_bin].mean().item()) if count else None
        avg_acc = float(correct[in_bin].mean().item()) if count else None
        bins.append(
            {"lo": lo, "hi": hi, "count": count, "avg_confidence": avg_conf, "avg_accuracy": avg_acc}
        )
        if count:
            ece += (count / n) * abs(avg_conf - avg_acc)
    return bins, float(ece)


def gate_activation_per_class(
    gate: torch.Tensor, targets: torch.Tensor, class_names: list[str]
) -> dict:
    """Mean/variance of the per-sample gate activation, per class."""
    out: dict = {}
    for c, name in enumerate(class_names):
        vals = gate[targets == c]
        if vals.numel() == 0:
            out[name] = {"mean": None, "variance": None, "count": 0}
        else:
            out[name] = {
                "mean": float(vals.mean().item()),
                "variance": float(vals.var(unbiased=False).item()),
                "count": int(vals.numel()),
            }
    return out


# --- baselines --------------------------------------------------------------
def _mean_pooled_text(items, cache_dir: Path) -> np.ndarray:
    """Mean-pool each item's cached RoBERTa token states over the real tokens."""
    rows = []
    for item in items:
        cached = torch.load(cache_dir / f"{item.item_id}.pt", map_location="cpu", weights_only=True)
        text = cached["text"].float()  # (text_len, 768) already trimmed to real length
        rows.append(text.mean(dim=0).numpy())
    return np.vstack(rows)


def majority_class_baseline(train_labels: list[int], test_labels: list[int], class_names: list[str]) -> dict:
    majority = int(np.bincount(train_labels).argmax())
    preds = np.full(len(test_labels), majority, dtype=int)
    targets = np.asarray(test_labels)
    return {
        "predicts_class": class_names[majority],
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def logreg_text_baseline(train_items, test_items, cache_dir: Path, class_names: list[str]) -> dict:
    x_train = _mean_pooled_text(train_items, cache_dir)
    x_test = _mean_pooled_text(test_items, cache_dir)
    y_train = np.asarray([it.label for it in train_items])
    y_test = np.asarray([it.label for it in test_items])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return {
        "features": "mean-pooled RoBERTa-base (text only)",
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
    }


# --- plots ------------------------------------------------------------------
def plot_loss_curves(history: list[dict], phase_transitions: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train loss")
    ax.plot(epochs, [h["val_loss"] for h in history], marker="s", label="val loss")
    for transition in phase_transitions:
        ax.axvline(transition["epoch"], color="gray", linestyle="--", alpha=0.6)
        ax.text(
            transition["epoch"], ax.get_ylim()[1],
            f" p{transition['from']}->{transition['to']}", va="top", fontsize=8, color="gray",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training / validation loss (phase boundaries dashed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_reliability(bins: list[dict], path: Path) -> None:
    centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
    accs = [b["avg_accuracy"] if b["avg_accuracy"] is not None else np.nan for b in bins]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(centers, accs, marker="o", label="observed")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- driver -----------------------------------------------------------------
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    config = build_binary_config(args)
    class_names = labels.class_names(config.num_classes)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("device=%s run_dir=%s", device, run_dir)

    # Datasets (prints the per-class split summary).
    bundle = train_entry.build_datasets(mock=False, config=config)
    train_tiers = bundle["train_tiers"]
    val_ds, test_ds = bundle["val"], bundle["test"]

    model = SatireDetectionEngine(config)
    trainer = SatireTrainer(
        model=model,
        config=config,
        train_datasets=train_tiers,
        val_dataset=val_ds,
        device=str(device),
        class_weights=bundle["class_weights"],
        collate_fn=collate_embeddings,
    )
    # Run the full three-phase schedule; disable early stopping so every phase is
    # exercised across the epoch budget.
    trainer.EARLY_STOP_PATIENCE = args.epochs + 1

    start = time.perf_counter()
    summary = train_entry.run_training(
        trainer=trainer,
        epochs=args.epochs,
        forced_phase=None,
        checkpoint_dir=run_dir / "checkpoints",
    )
    wall = time.perf_counter() - start
    logger.info("training done in %.1fs; phase transitions: %s", wall, summary["phase_transitions"])

    # Evaluate the trained model on the held-out test split, and val for gates.
    test_logits, test_targets, _test_gate = forward_split(model, test_ds, config, device)
    val_logits, val_targets, val_gate = forward_split(model, val_ds, config, device)

    test_probs = F.softmax(test_logits, dim=-1)
    test_preds = test_probs.argmax(dim=-1)

    cls_metrics = classification_metrics(
        test_preds.numpy(), test_targets.numpy(), class_names
    )
    cm = sk_confusion_matrix(test_targets.numpy(), test_preds.numpy(), labels=list(range(len(class_names))))
    reliability, ece = reliability_and_ece(test_probs, test_targets)
    gates = gate_activation_per_class(val_gate, val_targets, class_names)
    temperature = float(model.classifier.temperature.detach().item())

    # Reference baselines for context.
    train_items = list(train_tiers[0].items)
    baselines = {
        "majority_class": majority_class_baseline(
            [it.label for it in train_items], test_targets.tolist(), class_names
        ),
        "logreg_text": logreg_text_baseline(
            train_items, list(test_ds.items), DEFAULT_EMBEDDING_DIR, class_names
        ),
    }

    # Gate separation (satire minus authentic) is the mechanism check.
    sat_mean = gates.get("satire", {}).get("mean")
    auth_mean = gates.get("authentic", {}).get("mean")
    gate_separation = (
        float(sat_mean - auth_mean) if sat_mean is not None and auth_mean is not None else None
    )

    caveat = (
        "Source outlet perfectly determines label in this corpus; all metrics "
        "are inflated by construction. This run validates wiring and the "
        "contradiction-gate mechanism only, not model quality."
    )
    results = {
        "caveat": caveat,
        "git_commit": _git_commit(),
        "seed": args.seed,
        "device": str(device),
        "wall_seconds": round(wall, 1),
        "config": config.model_dump(),
        "class_names": class_names,
        "split_sizes": {
            "train": len(train_tiers[0]),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "training": {
            "epochs_run": len(summary["history"]),
            "final_phase": summary["final_phase"],
            "phase_transitions": summary["phase_transitions"],
            "best_val_loss": summary["best_val_loss"],
        },
        "test_metrics": {
            **cls_metrics,
            "ece": ece,
            "confusion_matrix": cm.tolist(),
            "learned_temperature": temperature,
        },
        "val_gate_activation": {
            "per_class": gates,
            "satire_minus_authentic_separation": gate_separation,
        },
        "reliability_bins": reliability,
        "baselines": baselines,
    }

    # Write artifacts.
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(summary["history"], indent=2), encoding="utf-8")
    plot_loss_curves(summary["history"], summary["phase_transitions"], run_dir / "loss_curves.png")
    plot_confusion(cm, class_names, run_dir / "confusion_matrix.png")
    plot_reliability(reliability, run_dir / "reliability_diagram.png")

    _write_summary_txt(run_dir / "summary.txt", results)

    print("\n" + (run_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"\nartifacts written to {run_dir}")
    return 0


def _write_summary_txt(path: Path, results: dict) -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("RUN #1 - real-data wiring/mechanism validation (binary authentic/satire)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("CAVEAT: " + results["caveat"])
    lines.append("")
    lines.append(f"git commit : {results['git_commit']}")
    lines.append(f"seed       : {results['seed']}   device: {results['device']}")
    lines.append(f"splits     : {results['split_sizes']}")
    t = results["training"]
    lines.append(
        f"training   : {t['epochs_run']} epochs, final phase {t['final_phase']}, "
        f"transitions {t['phase_transitions']}"
    )
    lines.append("")
    tm = results["test_metrics"]
    lines.append("-- test split (see caveat) --")
    lines.append(f"  accuracy   : {tm['accuracy']:.4f}")
    lines.append(f"  macro F1   : {tm['macro_f1']:.4f}")
    lines.append(f"  ECE        : {tm['ece']:.4f}")
    lines.append(f"  temperature: {tm['learned_temperature']:.4f}")
    for name, m in tm["per_class"].items():
        lines.append(
            f"    {name:>18s}: p={m['precision']:.3f} r={m['recall']:.3f} "
            f"f1={m['f1']:.3f} n={m['support']}"
        )
    lines.append(f"  confusion (rows=true, cols=pred): {tm['confusion_matrix']}")
    lines.append("")
    lines.append("-- contradiction-gate activation on val (mechanism check) --")
    for name, g in results["val_gate_activation"]["per_class"].items():
        if g["mean"] is None:
            lines.append(f"    {name:>18s}: (no samples)")
        else:
            lines.append(f"    {name:>18s}: mean={g['mean']:.3f} var={g['variance']:.4f} n={g['count']}")
    sep = results["val_gate_activation"]["satire_minus_authentic_separation"]
    lines.append(f"  satire - authentic separation: {sep:.3f}" if sep is not None else "  separation: n/a")
    lines.append("")
    lines.append("-- reference baselines (see caveat) --")
    b = results["baselines"]
    lines.append(
        f"    majority-class ({b['majority_class']['predicts_class']}): "
        f"acc={b['majority_class']['accuracy']:.4f} macroF1={b['majority_class']['macro_f1']:.4f}"
    )
    lines.append(
        f"    logreg text-only            : "
        f"acc={b['logreg_text']['accuracy']:.4f} macroF1={b['logreg_text']['macro_f1']:.4f}"
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
