"""Trainer for ParticleFormer using HuggingFace Accelerate.

Supports:
- Multi-GPU training
- Mixed precision (fp16/bf16)
- Gradient accumulation
- Checkpointing
- Logging to wandb (optional)
"""

import os
import json
import time
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed

from .config import ParticleFormerConfig
from .models import ParticleFormer
from .losses import HybridLoss
from .losses import HybridLoss
from .data import create_dataloader
from .metrics import ChamferMetric, TrackMetric, RenderMetric


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


class ParticleFormerTrainer:
    """Trainer for ParticleFormer model.
    
    Uses HuggingFace Accelerate for distributed training support.
    
    Args:
        config: ParticleFormerConfig instance.
        model: ParticleFormer model (optional, created from config if not provided).
    """
    
    def __init__(
        self,
        config: ParticleFormerConfig,
        model: Optional[ParticleFormer] = None,
    ):
        self.config = config
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision if config.mixed_precision != "no" else None,
            log_with="wandb" if config.use_wandb else None,
            project_dir=config.output_dir,
        )
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Create model
        if model is None:
            self.model = ParticleFormer.from_config(config)
        else:
            self.model = model
        
        # Create loss function
        self.loss_fn = HybridLoss(
            alpha=config.loss_alpha,
            hd_temperature=0.1,
            object_only=True,
        )
        
        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = None  # Created after dataloader
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        if self.accelerator.is_main_process:
            config.save(self.output_dir / "config.json")
        
        # Initialize wandb
        if config.use_wandb and self.accelerator.is_main_process:
            run_name = f"{config.object_name}_ep_{config.ep_idx}"
            self.accelerator.init_trackers(
                project_name="deformable_dynamics",
                config={
                    "method": "ParticleFormer",
                    "object_name": config.object_name,
                    "ep_idx": config.ep_idx,
                    **config.__dict__
                },
                init_kwargs={
                    "wandb": {
                        "name": run_name,
                        "resume": "allow"
                    }
                }
            )

        # Initialize metrics
        self.chamfer_metric = ChamferMetric(self.output_dir)
        self.track_metric = TrackMetric(self.output_dir)
        # Only initialize render metric if strictly needed or just keep it ready
        self.render_metric = RenderMetric(self.output_dir, skip_render=False)
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        """Run training loop.
        
        Args:
            train_dataloader: Training data loader.
            eval_dataloader: Optional evaluation data loader.
        """
        config = self.config
        
        # Create scheduler
        num_training_steps = len(train_dataloader) * config.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=config.learning_rate * 0.01,
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, train_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, self.scheduler
        )
        
        if eval_dataloader is not None:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        # Resume from checkpoint if specified
        if config.resume_from is not None:
            self._load_checkpoint(config.resume_from)
        
        # Training loop
        self.log(f"Starting training for {config.num_epochs} epochs")
        self.log(f"Total training steps: {num_training_steps}")
        
        for epoch in range(self.epoch, config.num_epochs):
            self.epoch = epoch
            train_metrics = self._train_epoch(train_dataloader)
            
            # Log epoch metrics
            self.log(f"Epoch {epoch}: {train_metrics}")
            
            # Evaluation
            if eval_dataloader is not None and (epoch + 1) % config.eval_interval == 0:
                eval_metrics = self._eval_epoch(eval_dataloader)
                self.log(f"Eval: {eval_metrics}")
                
                # Save best model
                if eval_metrics.get("loss", float('inf')) < self.best_loss:
                    self.best_loss = eval_metrics["loss"]
                    self._save_checkpoint("best")
            
            # Save periodic checkpoint
            if (epoch + 1) % config.save_interval == 0:
                self._save_checkpoint(f"epoch_{epoch}")
        
        # Save final model
        self._save_checkpoint("final")
        
        # FINAL LOGGING TO WANDB
        if eval_dataloader is not None:
             self.log("Performing final evaluation for WandB...")
             final_metrics = self._eval_full_rollout(eval_dataloader)
             self.log(f"Final Metrics: {final_metrics}")
             if config.use_wandb:
                 self.accelerator.log(final_metrics)

        if config.use_wandb:
            self.accelerator.end_training()
        
        self.log("Training complete!")
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_cd = 0.0
        total_hd = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            with self.accelerator.accumulate(self.model):
                loss, metrics = self._train_step(batch)
                
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += metrics["loss"]
            total_cd += metrics["chamfer_distance"]
            total_hd += metrics["hausdorff_distance"]
            num_batches += 1
            
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.log(
                    f"Step {self.global_step}: loss={metrics['loss']:.6f}, "
                    f"cd={metrics['chamfer_distance']:.6f}, "
                    f"hd={metrics['hausdorff_distance']:.6f}, "
                    f"lr={lr:.2e}"
                )
        
        return {
            "loss": total_loss / num_batches,
            "chamfer_distance": total_cd / num_batches,
            "hausdorff_distance": total_hd / num_batches,
        }
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Single training step."""
        positions = batch["positions"]           # (batch, num_particles, 3)
        actions = batch["actions"]               # (batch, num_steps, num_particles, 3)
        targets = batch["targets"]               # (batch, num_steps, num_particles, 3)
        is_controller = batch["is_controller"]   # (batch, num_particles)
        object_ids = batch["object_ids"]         # (batch,)
        padding_mask = batch["padding_mask"]     # (batch, num_particles)
        
        num_steps = actions.shape[1]
    
        # Sub-sampling logic: Exactly 32 object particles (using FPS) and 2 controller particles
        N_OBJ_TARGET = 32
        N_CTRL_TARGET = 2
        batch_size, n_total, _ = positions.shape
        
        effective_mask = torch.zeros_like(padding_mask)
        for i in range(batch_size):
            ctrl_idx = torch.where(is_controller[i] & padding_mask[i])[0]
            obj_idx = torch.where((~is_controller[i]) & padding_mask[i])[0]
            
            # Select first 2 controller particles
            if len(ctrl_idx) > 0:
                ctrl_keep = ctrl_idx[:N_CTRL_TARGET]
                effective_mask[i, ctrl_keep] = True
            
            # Use FPS for object particles
            if len(obj_idx) > N_OBJ_TARGET:
                obj_xyz = positions[i, obj_idx].unsqueeze(0) # (1, N_obj, 3)
                fps_indices = farthest_point_sample(obj_xyz, N_OBJ_TARGET) # (1, 32)
                obj_keep = obj_idx[fps_indices[0]]
                effective_mask[i, obj_keep] = True
            elif len(obj_idx) > 0:
                effective_mask[i, obj_idx] = True
        
        # Override padding_mask for this step
        padding_mask = effective_mask
            
        # Debug: Check if we have any object particles left to train on
        if self.global_step % 10 == 0:
            obj_counts = (~is_controller & effective_mask).sum(dim=1).float().mean().item()
            ctrl_counts = (is_controller & effective_mask).sum(dim=1).float().mean().item()
            print(f"  [Step {self.global_step}] Particles: Obj={obj_counts:.1f}, Ctrl={ctrl_counts:.1f}")
        
        # Update padding_mask for both model and loss
        padding_mask = effective_mask
        
        # Autoregressive rollout
        pred_positions_list = []
        current_positions = positions
            
        for t in range(num_steps):
            current_actions = actions[:, t]  # (batch, num_particles, 3)
            
            # Forward pass
            next_positions, _ = self.model(
                positions=current_positions,
                actions=current_actions,
                object_ids=object_ids,
                is_controller=is_controller,
                attention_mask=padding_mask,
            )
            
            # For controller particles, use ground truth positions
            controller_mask = is_controller.unsqueeze(-1).expand_as(next_positions)
            gt_controller = targets[:, t]  # Use target as ground truth for controllers
            next_positions = torch.where(controller_mask, gt_controller, next_positions)
            
            pred_positions_list.append(next_positions)
            current_positions = next_positions
            
        pred_positions = torch.stack(pred_positions_list, dim=1)  # (batch, num_steps, num_particles, 3)
        
        # Compute loss
        loss, metrics = self.loss_fn(
            pred_positions=pred_positions,
            target_positions=targets,
            is_controller=is_controller,
            padding_mask=padding_mask,
        )
        
        return loss, metrics
    
    @torch.no_grad()
    def _eval_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_cd = 0.0
        total_hd = 0.0
        num_batches = 0
        
        for batch in dataloader:
            positions = batch["positions"]
            actions = batch["actions"]
            targets = batch["targets"]
            is_controller = batch["is_controller"]
            object_ids = batch["object_ids"]
            padding_mask = batch["padding_mask"]
            
            num_steps = actions.shape[1]
            
            # Autoregressive rollout
            pred_positions_list = []
            current_positions = positions
            
            for t in range(num_steps):
                current_actions = actions[:, t]
                next_positions, _ = self.model(
                    positions=current_positions,
                    actions=current_actions,
                    object_ids=object_ids,
                    is_controller=is_controller,
                    attention_mask=padding_mask,
                )
                
                controller_mask = is_controller.unsqueeze(-1).expand_as(next_positions)
                gt_controller = targets[:, t]
                next_positions = torch.where(controller_mask, gt_controller, next_positions)
                
                pred_positions_list.append(next_positions)
                current_positions = next_positions
            
            pred_positions = torch.stack(pred_positions_list, dim=1)
            
            loss, metrics = self.loss_fn(
                pred_positions=pred_positions,
                target_positions=targets,
                is_controller=is_controller,
                padding_mask=padding_mask,
            )
            
            total_loss += metrics["loss"]
            total_cd += metrics["chamfer_distance"]
            total_hd += metrics["hausdorff_distance"]
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "chamfer_distance": total_cd / num_batches,
            "hausdorff_distance": total_hd / num_batches,
        }
    
    def _save_checkpoint(self, name: str):
        """Save a checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint_dir = self.output_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Unwrap model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save model state
        torch.save(
            unwrapped_model.state_dict(),
            checkpoint_dir / "model.pt"
        )
        
        # Save optimizer and scheduler state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }, checkpoint_dir / "training_state.pt")
        
        self.log(f"Saved checkpoint: {checkpoint_dir}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model state
        model_state = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(model_state)
        
        # Load training state
        training_state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu")
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self.scheduler and training_state["scheduler"]:
            self.scheduler.load_state_dict(training_state["scheduler"])
        self.epoch = training_state["epoch"]
        self.global_step = training_state["global_step"]
        self.best_loss = training_state["best_loss"]
        
        self.log(f"Loaded checkpoint from {checkpoint_dir}")
    
    def log(self, message: str):
        """Log a message (only on main process)."""
        if self.accelerator.is_main_process:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

    @torch.no_grad()
    def _eval_full_rollout(self, dataloader: DataLoader) -> Dict[str, float]:
        """Perform full rollout evaluation and compute physics metrics."""
        self.model.eval()
        
        counts = {"chamfer": 0, "track": 0, "render": 0}
        sums = {
            "train/chamfer_error": 0.0, "test/chamfer_error": 0.0,
            "train/chamfer_frame_num": 0.0, "test/chamfer_frame_num": 0.0,
            "train/track_error": 0.0, "test/track_error": 0.0,
            "train/psnr": 0.0, "train/ssim": 0.0, "train/lpips": 0.0,
            "test/psnr": 0.0, "test/ssim": 0.0, "test/lpips": 0.0
        }
        
        # To log video to wandb once per eval
        first_video_path = None
        
        
        eval_episodes = 0
        max_eval_episodes = 5 
        
        for batch in dataloader:
            if eval_episodes >= max_eval_episodes:
                break
            
            episode_paths = batch["episode_paths"]
            
            # Process unique episodes in this batch
            unique_paths = list(set(episode_paths))
            
            for ep_path in unique_paths:
                if eval_episodes >= max_eval_episodes:
                    break
                
                # Load initial state for this episode
                gt_path = Path(ep_path) / "final_data.pkl"
                if not gt_path.exists():
                    continue
                    
                with open(gt_path, "rb") as f:
                    data = pickle.load(f)
                
                # Setup proper rollout
                object_points = data["object_points"] # (T, N, 3)
                controller_points = data["controller_points"] # (T, M, 3)
                
                # Check split to know where "train" ends and "test" begins
                # Usually we rollout from start of test split.
                split_path = Path(ep_path) / "split.json"
                if split_path.exists():
                     with open(split_path, "r") as f:
                        split_data = json.load(f)
                     test_range = split_data.get("test", [0, 0])
                     # We want to rollout the test sequence. Particle Former predictions usually match GT length.
                     # If we start at T=0, we predict everything.
                     # The user says: "start from initial frame of training split"?? 
                     # Wait, user said: "start from initial frame of training split."
                     # Usually training split starts at 0? 
                     # If training split is [0, 100], do they mean frame 0?
                     # Let's assume start_idx = train_range[0].
                     train_range = split_data.get("train", [0,0])
                     start_idx = 0 # Force start from frame 0 for full rollout
                else:
                     start_idx = 0
                
                # Slicing data from start_idx
                # We want to predict from start_idx to end.
                # Adjust object_points and controller_points
                
                # Careful: The data loader might have already sliced it? 
                # No, we loaded `final_data.pkl` which is raw full episode.
                
                object_points = object_points[start_idx:]
                controller_points = controller_points[start_idx:]
                
                if len(object_points) == 0:
                     continue
                
                # Initial state
                # Combine object and controller
                curr_pos_obj = object_points[0]
                curr_pos_ctrl = controller_points[0]
                
                # Create initial tensors
                # Need to match model input shape: (1, N+M, 3)
                num_obj_total = len(curr_pos_obj)
                num_ctrl_total = len(curr_pos_ctrl)
                
                # Sub-sample for Transformer stability (matches training)
                N_OBJ_TARGET = 32
                N_CTRL_TARGET = 2
                
                if num_obj_total > N_OBJ_TARGET:
                    # Use FPS for object particles in evaluation too
                    obj_xyz = torch.from_numpy(curr_pos_obj).float().unsqueeze(0).to(self.accelerator.device)
                    fps_indices = farthest_point_sample(obj_xyz, N_OBJ_TARGET)
                    fps_indices = fps_indices[0].cpu().numpy()
                    curr_pos_obj = curr_pos_obj[fps_indices]
                    object_points = object_points[:, fps_indices]
                
                if num_ctrl_total > N_CTRL_TARGET:
                    curr_pos_ctrl = curr_pos_ctrl[:N_CTRL_TARGET]
                    controller_points = controller_points[:, :N_CTRL_TARGET] # (T, 2, 3)
                
                curr_pos = np.concatenate([curr_pos_obj, curr_pos_ctrl], axis=0)
                curr_pos = torch.from_numpy(curr_pos).float().unsqueeze(0).to(self.accelerator.device)
                
                num_obj = len(curr_pos_obj)
                num_ctrl = len(curr_pos_ctrl)
                total_particles = num_obj + num_ctrl
                
                is_controller = torch.zeros(total_particles, dtype=torch.bool, device=self.accelerator.device)
                is_controller[num_obj:] = True
                is_controller = is_controller.unsqueeze(0) # (1, N+M)
                
                # Object ID (assuming 0 for now or getting from batch if we could map it back)
                # We'll just use 0 (rope/default)
                object_ids = torch.zeros(1, dtype=torch.long, device=self.accelerator.device)
                
                # Rollout loop
                num_frames = len(object_points)
                # Limit rollout length if needed? No, full length.
                
                pred_positions_list = [curr_pos]
                
                # We need actions (controller velocities) for the whole sequence
                # Calculate actions from GT controller points
                # action[t] moves state from t to t+1
                actions_full = np.zeros((num_frames, total_particles, 3), dtype=np.float32)
                # Controller velocity
                actions_full[0:num_frames-1, num_obj:] = controller_points[1:] - controller_points[:-1]
                actions_full = torch.from_numpy(actions_full).float().to(self.accelerator.device)
                
                current_positions = curr_pos
                
                print(f"\nStarting rollout for {os.path.basename(ep_path)}: {num_frames} frames")
                for t in range(num_frames - 1):
                    current_actions = actions_full[t].unsqueeze(0) # (1, P, 3)
                    
                    # Forward
                    next_positions, displacement = self.model(
                        positions=current_positions,
                        actions=current_actions,
                        object_ids=object_ids,
                        is_controller=is_controller,
                        attention_mask=None # No padding in full rollout single batch
                    )
                    
                    # Force controller positions to GT
                    gt_ctrl_next = torch.from_numpy(controller_points[t+1]).float().to(self.accelerator.device)
                    next_positions[0, num_obj:] = gt_ctrl_next
                    
                    if t % 50 == 0 or t == num_frames - 2:
                        disp_abs = displacement.abs().mean().item()
                        pos_std = next_positions[0, :num_obj].std().item()
                        print(f"  Step {t:3d}: AvgDisp={disp_abs:.6f}, ObjStd={pos_std:.4f}")
                    
                    pred_positions_list.append(next_positions)
                    current_positions = next_positions

                    
                # Stack predictions: (T, N+M, 3) - squeeze batch dim
                pred_positions = torch.cat(pred_positions_list, dim=0)
                
                # Eval Metrics
                # Chamfer
                c_res = self.chamfer_metric.evaluate(ep_path, pred_positions[:, :num_obj], self.epoch)
                for k in ["train/chamfer_error", "test/chamfer_error", "train/chamfer_frame_num", "test/chamfer_frame_num"]:
                    if k in c_res:
                        sums[k] += c_res[k]
                counts["chamfer"] += 1
                
                # Track
                t_res = self.track_metric.evaluate(ep_path, pred_positions[:, :num_obj], self.epoch)
                for k in ["train/track_error", "test/track_error"]:
                    if k in t_res:
                        sums[k] += t_res[k]
                counts["track"] += 1
                
                # Render
                r_res = self.render_metric.evaluate(ep_path, pred_positions, self.epoch)
                for k in ["train/psnr", "train/ssim", "train/lpips", "test/psnr", "test/ssim", "test/lpips"]:
                    if k in r_res:
                        sums[k] += r_res[k]
                
                if "test/comparison_video" in r_res and first_video_path is None:
                    first_video_path = r_res["test/comparison_video"]
                
                counts["render"] += 1
                eval_episodes += 1
        
        # Average metrics
        metrics = {}
        if counts["chamfer"] > 0:
            for k in ["train/chamfer_error", "test/chamfer_error", "train/chamfer_frame_num", "test/chamfer_frame_num"]:
                metrics[k] = sums[k] / counts["chamfer"]
        if counts["track"] > 0:
            for k in ["train/track_error", "test/track_error"]:
                metrics[k] = sums[k] / counts["track"]
        if counts["render"] > 0:
            for k in ["train/psnr", "train/ssim", "train/lpips", "test/psnr", "test/ssim", "test/lpips"]:
                metrics[k] = sums[k] / counts["render"]
            
        if first_video_path and self.config.use_wandb and self.accelerator.is_main_process:
            metrics["test/comparison_video"] = wandb.Video(first_video_path, fps=30, format="mp4")
            
        return metrics


def train_from_config(config_path: str):
    """Train ParticleFormer from a config file.
    
    Args:
        config_path: Path to config JSON file.
    """
    config = ParticleFormerConfig.load(config_path)
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        data_root=config.data_root,
        split_path=config.split_json,
        batch_size=config.batch_size,
        rollout_steps=config.rollout_steps,
        num_workers=4,
        shuffle=True,
        split="train",
    )
    
    val_dataloader = create_dataloader(
        data_root=config.data_root,
        split_path=config.split_json,
        batch_size=config.batch_size,
        rollout_steps=config.rollout_steps,
        num_workers=4,
        shuffle=False,
        split="test",
    )
    
    # Create trainer and train
    trainer = ParticleFormerTrainer(config)
    trainer.train(train_dataloader, val_dataloader)


