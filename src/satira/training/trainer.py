import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from satira.config import Settings
from satira.data.datasets import CurriculumDataLoader, SatireDataset
from satira.models.engine import SatireDetectionEngine
from satira.training.curriculum import CurriculumScheduler, PhaseTransitionController
from satira.training.losses import PhasedLossFunction


logger = logging.getLogger(__name__)


class SatireTrainer:
    """Drives the 3-phase curriculum training loop end-to-end.

    The trainer owns the engine, phased loss, curriculum data loader, and
    phase transition controller. Each epoch it samples a curriculum-mixed
    batch, runs forward + phased loss + backward with gradient clipping,
    and reports loss/accuracy/gate-variance/grad-norm. After each epoch it
    asks the phase controller whether to advance, and on advance refreshes
    freezing and the optimizer to match the new phase.
    """

    GRAD_CLIP_NORM = 1.0
    LOSS_PLATEAU_PATIENCE = 3
    EARLY_STOP_PATIENCE = 5

    def __init__(
        self,
        model: SatireDetectionEngine,
        config: Settings,
        train_datasets: tuple[SatireDataset, SatireDataset, SatireDataset],
        val_dataset: SatireDataset,
        device: str = "cuda",
    ) -> None:
        if len(train_datasets) != 3:
            raise ValueError(
                f"train_datasets must be (tier1, tier2, tier3); got {len(train_datasets)}"
            )

        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = model.to(self.device)

        self.scheduler = CurriculumScheduler(total_epochs=25)
        self.phase_controller = PhaseTransitionController(patience=self.LOSS_PLATEAU_PATIENCE)

        self.data_loader = CurriculumDataLoader(
            tier1=train_datasets[0],
            tier2=train_datasets[1],
            tier3=train_datasets[2],
            scheduler=self.scheduler,
            batch_size=config.batch_size,
        )
        self.val_dataset = val_dataset

        class_weights = torch.ones(config.num_classes)
        self.loss_fn = PhasedLossFunction(
            class_gate_targets=config.CLASS_GATE_TARGETS,
            class_weights=class_weights,
            gamma=config.focal_gamma,
            lambda_consistency=config.consistency_loss_weight,
            gate_loss_weight=config.gate_loss_weight,
        ).to(self.device)

        self.model.freeze_for_phase(self.phase_controller.phase)
        self.optimizer = self._build_optimizer()

        self.best_val_loss = float("inf")
        self.best_checkpoint_path: Optional[Path] = None
        self._epochs_since_val_improvement = 0

    def _build_optimizer(self) -> torch.optim.AdamW:
        if self.phase_controller.phase < 3:
            cfg = self.phase_controller.get_optimizer_config()
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            return torch.optim.AdamW(trainable, lr=cfg["learning_rate"])

        return torch.optim.AdamW(self.model.get_parameter_groups())

    def _move_batch_to_device(self, batch: dict) -> dict:
        moved: dict = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def _synthesize_engine_inputs(self, batch_size: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Build engine inputs of the right shape from the batch.

        The dataset is still placeholder (random images, no real text encoder
        yet), so we synthesize the four input streams here. When the data
        pipeline lands, this method routes real encoder outputs to the engine.
        """
        cfg = self.config
        v = torch.randn(batch_size, 10, cfg.vision_dim, device=self.device)
        t = torch.randn(batch_size, 12, cfg.text_dim, device=self.device)
        temp = torch.randn(batch_size, cfg.temporal_dim, device=self.device)
        graph = torch.randn(batch_size, cfg.graph_dim, device=self.device)
        return v, t, temp, graph

    def _coerce_label_tensor(self, batch: dict) -> torch.Tensor:
        labels = batch["label"]
        if isinstance(labels, torch.Tensor):
            return labels.to(self.device, dtype=torch.long)
        coerced = []
        for lbl in labels:
            if isinstance(lbl, torch.Tensor):
                coerced.append(int(lbl))
            elif isinstance(lbl, (int, bool)):
                coerced.append(int(lbl))
            elif isinstance(lbl, str):
                coerced.append(0)
            else:
                coerced.append(int(lbl))
        return torch.tensor(coerced, dtype=torch.long, device=self.device)

    @staticmethod
    def _gradient_norm(parameters) -> float:
        total = 0.0
        for p in parameters:
            if p.grad is not None:
                total += p.grad.detach().pow(2).sum().item()
        return total**0.5

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        phase = self.phase_controller.phase

        batch = self.data_loader.get_batch(epoch=epoch)
        batch = self._move_batch_to_device(batch)
        targets = self._coerce_label_tensor(batch)
        batch_size = targets.size(0)

        v, t, temp, graph = self._synthesize_engine_inputs(batch_size)

        logits, _t2v, _v2t, t_gate, v_gate = self.model(v, t, temp, graph)

        logits_alt: Optional[torch.Tensor] = None
        if phase == 3:
            v2, t2, temp2, graph2 = self._synthesize_engine_inputs(batch_size)
            logits_alt, *_ = self.model(v2, t2, temp2, graph2)

        loss = self.loss_fn.compute(
            phase=phase,
            logits=logits,
            targets=targets,
            t_gate=t_gate,
            v_gate=v_gate,
            logits_alt_snapshot=logits_alt,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm_pre_clip = self._gradient_norm(trainable)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=self.GRAD_CLIP_NORM)

        projection_grad_norm = self._gradient_norm(
            list(self.model.v_proj.parameters())
            + list(self.model.t_proj.parameters())
            + list(self.model.temp_proj.parameters())
            + list(self.model.graph_proj.parameters())
        )

        self.optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == targets).float().mean().item()

            gate_mean = 0.5 * (
                t_gate.reshape(batch_size, -1).mean(dim=-1)
                + v_gate.reshape(batch_size, -1).mean(dim=-1)
            )
            gate_variance = gate_mean.var(unbiased=False).item()

        return {
            "loss": float(loss.detach().item()),
            "accuracy": accuracy,
            "gate_activation_variance": gate_variance,
            "grad_norm": grad_norm_pre_clip,
            "projection_grad_norm": projection_grad_norm,
            "phase": phase,
            "epoch": epoch,
        }

    def validate(self) -> dict:
        self.model.eval()
        ds = self.val_dataset
        if len(ds) == 0:
            return {"loss": float("inf"), "accuracy": 0.0, "calibration_error": 0.0}

        batch_size = min(self.config.batch_size, len(ds))
        indices = list(range(batch_size))
        targets = self._coerce_label_tensor(
            {"label": [ds[i]["label"] for i in indices]}
        )

        v, t, temp, graph = self._synthesize_engine_inputs(batch_size)

        with torch.no_grad():
            logits, *_ = self.model(v, t, temp, graph)
            loss = F.cross_entropy(logits, targets)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == targets).float().mean().item()

            probs = F.softmax(logits, dim=-1)
            confidence, _ = probs.max(dim=-1)
            correct = (preds == targets).float()
            calibration_error = (confidence - correct).abs().mean().item()

            num_classes = self.config.num_classes
            f1_per_class = []
            for c in range(num_classes):
                tp = ((preds == c) & (targets == c)).sum().item()
                fp = ((preds == c) & (targets != c)).sum().item()
                fn = ((preds != c) & (targets == c)).sum().item()
                denom = 2 * tp + fp + fn
                f1_per_class.append((2 * tp / denom) if denom > 0 else 0.0)

        return {
            "loss": float(loss.item()),
            "accuracy": accuracy,
            "calibration_error": calibration_error,
            "f1_per_class": f1_per_class,
        }

    def _handle_phase_transition(self, epoch: int, train_metrics: dict) -> bool:
        prev_phase = self.phase_controller.phase
        advanced = self.phase_controller.should_advance_phase(
            epoch=epoch, metrics=train_metrics
        )
        if advanced and self.phase_controller.phase != prev_phase:
            self.model.freeze_for_phase(self.phase_controller.phase)
            self.optimizer = self._build_optimizer()
            logger.info(
                "phase transition %d -> %d at epoch %d",
                prev_phase,
                self.phase_controller.phase,
                epoch,
            )
        return advanced

    def save_checkpoint(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "phase": self.phase_controller.phase,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        target_phase = int(ckpt.get("phase", 1))
        self.phase_controller._phase = target_phase  # type: ignore[attr-defined]
        self.model.freeze_for_phase(target_phase)
        self.optimizer = self._build_optimizer()
        if "optimizer_state" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except (ValueError, KeyError):
                logger.warning("optimizer state incompatible with rebuilt optimizer; skipping")
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

    def run(
        self,
        max_epochs: int = 25,
        checkpoint_dir: str | Path = "./checkpoints",
    ) -> dict:
        checkpoint_dir = Path(checkpoint_dir)
        history: list[dict] = []

        for epoch in range(1, max_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            self._handle_phase_transition(epoch, train_metrics)

            logger.info(
                "epoch %d phase %d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                train_metrics["phase"],
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["accuracy"],
            )

            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

            if val_metrics["loss"] < self.best_val_loss - 1e-6:
                self.best_val_loss = val_metrics["loss"]
                self._epochs_since_val_improvement = 0
                self.best_checkpoint_path = self.save_checkpoint(
                    checkpoint_dir / "best.pt"
                )
            else:
                self._epochs_since_val_improvement += 1
                if self._epochs_since_val_improvement >= self.EARLY_STOP_PATIENCE:
                    logger.info("early stopping at epoch %d", epoch)
                    break

        return {
            "history": history,
            "best_val_loss": self.best_val_loss,
            "best_checkpoint": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "final_phase": self.phase_controller.phase,
        }
