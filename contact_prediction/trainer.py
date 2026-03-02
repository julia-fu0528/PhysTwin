"""Trainer for Contact Prediction using HuggingFace Accelerate.

Supports:
- Mixed precision (fp16/bf16)
- Gradient accumulation
- Checkpointing (best F1)
- Wandb logging
- Focal loss for class imbalance
- Comprehensive metrics (accuracy, precision, recall, F1, AUROC, AUPRC)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the positive class.
        gamma: Focusing parameter (gamma >= 0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (B, 1) raw logits.
            targets: (B, 1) binary targets.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class ContactPredictionTrainer:
    """Trainer for Contact Prediction model.

    Args:
        config: ContactPredictionConfig instance.
        model: Optional pre-created model.
    """

    def __init__(self, config, model=None):
        self.config = config

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="wandb" if config.use_wandb else None,
            project_dir=config.output_dir,
        )

        # Create model if not provided
        if model is None:
            from .models import ContactTransformer
            model = ContactTransformer(config)
        self.model = model

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log(f"Total parameters: {total_params:,}")
        self.log(f"Trainable parameters: {trainable_params:,}")

        # Loss function
        if config.use_focal_loss:
            self.criterion = FocalLoss(
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
            )
            self.log(f"Using Focal Loss (alpha={config.focal_alpha}, gamma={config.focal_gamma})")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            self.log("Using BCEWithLogitsLoss")

        # Optimizer: only trainable parameters
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler: cosine with warmup
        self.warmup_steps = 0  # Will be set in train()

        # Initialize wandb
        if config.use_wandb and self.accelerator.is_main_process:
            init_kwargs = {}
            if hasattr(config, "cam_name") and config.cam_name:
                init_kwargs["wandb"] = {"name": f"contact_pred_{config.cam_name}"}
                
            self.accelerator.init_trackers(
                project_name=config.wandb_project,
                config=config.__dict__,
                init_kwargs=init_kwargs,
            )

        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Save config
        if self.accelerator.is_main_process:
            config.save(os.path.join(config.output_dir, "config.json"))

        # Best metric tracking
        self.best_f1 = 0.0
        self.best_epoch = 0

    def _get_lr(self, step: int, total_steps: int) -> float:
        """Compute learning rate with cosine schedule and linear warmup."""
        if step < self.warmup_steps:
            return self.config.learning_rate * step / max(self.warmup_steps, 1)
        else:
            progress = (step - self.warmup_steps) / max(
                total_steps - self.warmup_steps, 1
            )
            return self.config.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))

    def _set_lr(self, lr: float):
        """Set learning rate for optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        pos_weight: Optional[float] = None,
    ):
        """Run training loop.

        Args:
            train_dataloader: Training data loader.
            eval_dataloader: Optional evaluation data loader.
            pos_weight: Optional positive class weight for BCEWithLogitsLoss.
        """
        # Optionally update loss with dataset-computed pos_weight
        if pos_weight is not None and not self.config.use_focal_loss:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
            self.log(f"Updated BCEWithLogitsLoss with pos_weight={pos_weight:.3f}")

        # Prepare with accelerator
        (
            self.model,
            self.optimizer,
            train_dataloader,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader
        )

        if eval_dataloader is not None:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)

        # Calculate warmup steps
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * self.config.num_epochs
        self.warmup_steps = steps_per_epoch * self.config.warmup_epochs
        self.log(f"Total steps: {total_steps}, Warmup steps: {self.warmup_steps}")

        # Resume from checkpoint
        start_epoch = 0
        global_step = 0
        if self.config.resume_from is not None:
            start_epoch, global_step = self._load_checkpoint(self.config.resume_from)

        # Training loop
        for epoch in range(start_epoch, self.config.num_epochs):
            self.model.train()
            # Keep visual encoder in eval if frozen
            if self.config.freeze_visual_encoder:
                unwrapped = self.accelerator.unwrap_model(self.model)
                unwrapped.visual_encoder.eval()

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            epoch_start = time.time()

            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.model):
                    logits = self.model(
                        batch["rgb"],
                        batch["gripper_action"],
                    )

                    loss = self.criterion(logits, batch["target"])

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update LR
                lr = self._get_lr(global_step, total_steps)
                self._set_lr(lr)

                # Track metrics
                with torch.no_grad():
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct = (preds == batch["target"]).sum().item()
                    total = batch["target"].numel()

                epoch_loss += loss.item()
                epoch_correct += correct
                epoch_total += total

                if (step + 1) % self.config.log_interval == 0:
                    avg_loss = epoch_loss / (step + 1)
                    acc = epoch_correct / max(epoch_total, 1)
                    self.log(
                        f"Epoch {epoch+1}/{self.config.num_epochs} "
                        f"Step {step+1}/{steps_per_epoch} "
                        f"Loss: {avg_loss:.4f} Acc: {acc:.4f} LR: {lr:.6f}"
                    )

                    if self.config.use_wandb:
                        self.accelerator.log(
                            {
                                "train/loss": avg_loss,
                                "train/accuracy": acc,
                                "train/lr": lr,
                                "train/step": global_step,
                            },
                            step=global_step,
                        )

                global_step += 1

            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(steps_per_epoch, 1)
            avg_acc = epoch_correct / max(epoch_total, 1)
            self.log(
                f"Epoch {epoch+1} done in {epoch_time:.1f}s. "
                f"Train Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}"
            )

            # Evaluation
            if (
                eval_dataloader is not None
                and (epoch + 1) % self.config.eval_interval == 0
            ):
                eval_metrics = self._eval_epoch(eval_dataloader)
                self.log(
                    f"Eval - Acc: {eval_metrics['accuracy']:.4f} "
                    f"(Rand: {eval_metrics['acc_random_guess']:.4f}, Maj: {eval_metrics['acc_majority']:.4f}) "
                    f"Prec: {eval_metrics['precision']:.4f} "
                    f"Rec: {eval_metrics['recall']:.4f} "
                    f"F1: {eval_metrics['f1']:.4f} "
                    f"AUROC: {eval_metrics['auroc']:.4f} "
                    f"AUPRC: {eval_metrics['auprc']:.4f}"
                )

                if self.config.use_wandb:
                    log_dict = {f"eval/{k}": v for k, v in eval_metrics.items()
                                if k != "confusion_matrix"}
                    log_dict["eval/epoch"] = epoch + 1
                    self.accelerator.log(log_dict, step=global_step)

                # Save best model by F1
                if eval_metrics["f1"] > self.best_f1:
                    self.best_f1 = eval_metrics["f1"]
                    self.best_epoch = epoch + 1
                    self._save_checkpoint("best", epoch, global_step)
                    self.log(f"New best F1: {self.best_f1:.4f} at epoch {self.best_epoch}")

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(f"epoch_{epoch+1}", epoch, global_step)

        # Save final checkpoint
        self._save_checkpoint("final", self.config.num_epochs - 1, global_step)
        self.log(
            f"Training complete. Best F1: {self.best_f1:.4f} at epoch {self.best_epoch}"
        )

        if self.config.use_wandb:
            self.accelerator.end_training()

    @torch.no_grad()
    def _eval_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate for one epoch with comprehensive metrics."""
        self.model.eval()

        all_logits = []
        all_targets = []

        for batch in dataloader:
            logits = self.model(
                batch["rgb"],
                batch["gripper_action"],
            )
            # Gather from all processes
            logits_gathered = self.accelerator.gather(logits)
            targets_gathered = self.accelerator.gather(batch["target"])

            all_logits.append(logits_gathered.cpu())
            all_targets.append(targets_gathered.cpu())

        self.model.train()
        if self.config.freeze_visual_encoder:
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.visual_encoder.eval()

        # Only compute metrics on main process
        if not self.accelerator.is_main_process:
            return {}

        all_logits = torch.cat(all_logits, dim=0).numpy()    # (N, 1)
        all_targets = torch.cat(all_targets, dim=0).numpy()  # (N, 1)

        probs = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
        preds = (probs > 0.5).astype(np.float32)

        targets_flat = all_targets.flatten()
        preds_flat = preds.flatten()
        probs_flat = probs.flatten()

        # Core metrics
        metrics = {
            "accuracy": accuracy_score(targets_flat, preds_flat),
            "precision": precision_score(targets_flat, preds_flat, zero_division=0),
            "recall": recall_score(targets_flat, preds_flat, zero_division=0),
            "f1": f1_score(targets_flat, preds_flat, zero_division=0),
        }

        # Random guessing baseline probabilities
        pos_ratio = float(targets_flat.mean()) if len(targets_flat) > 0 else 0.0
        neg_ratio = 1.0 - pos_ratio
        metrics["acc_random_guess"] = pos_ratio**2 + neg_ratio**2
        metrics["acc_majority"] = max(pos_ratio, neg_ratio)

        # AUROC and AUPRC: need at least both classes present
        if len(np.unique(targets_flat)) > 1:
            metrics["auroc"] = roc_auc_score(targets_flat, probs_flat)
            metrics["auprc"] = average_precision_score(targets_flat, probs_flat)
        else:
            metrics["auroc"] = 0.0
            metrics["auprc"] = 0.0

        # Confusion matrix
        cm = confusion_matrix(targets_flat, preds_flat, labels=[0, 1])
        metrics["confusion_matrix"] = cm.tolist()

        # Per-class accuracy
        if cm.sum() > 0:
            metrics["acc_no_contact"] = cm[0, 0] / max(cm[0].sum(), 1)
            metrics["acc_contact"] = cm[1, 1] / max(cm[1].sum(), 1)

        return metrics

    def _save_checkpoint(self, name: str, epoch: int, global_step: int):
        """Save a checkpoint."""
        if not self.accelerator.is_main_process:
            return

        ckpt_dir = os.path.join(self.config.output_dir, f"checkpoint_{name}")
        os.makedirs(ckpt_dir, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        ckpt = {
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_f1": self.best_f1,
            "config": self.config.__dict__,
        }

        torch.save(ckpt, os.path.join(ckpt_dir, "checkpoint.pt"))
        self.log(f"Saved checkpoint: {ckpt_dir}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint. Returns (epoch, global_step)."""
        ckpt_file = os.path.join(checkpoint_path, "checkpoint.pt")
        if not os.path.exists(ckpt_file):
            ckpt_file = checkpoint_path  # Try direct path

        self.log(f"Loading checkpoint from {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu")

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_f1 = ckpt.get("best_f1", 0.0)

        epoch = ckpt.get("epoch", 0) + 1  # Resume from next epoch
        global_step = ckpt.get("global_step", 0)

        self.log(f"Resumed from epoch {epoch}, step {global_step}, best F1: {self.best_f1:.4f}")
        return epoch, global_step

    def log(self, message: str):
        """Log a message (only on main process)."""
        if self.accelerator.is_main_process:
            print(message, flush=True)
