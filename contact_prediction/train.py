#!/usr/bin/env python3
"""Training script for Contact Prediction.

Usage:
    # Train on all objects, episodes 0-3 train, episode 4 test
    pixi run python -m contact_prediction.train

    # Debug with a single object
    pixi run python -m contact_prediction.train --objects 001-rope --num_epochs 2

    # With wandb logging
    pixi run python -m contact_prediction.train --use_wandb
"""

import argparse
import json
import os
from pathlib import Path

from .config import ContactPredictionConfig, OBJ_NAMES
from .trainer import ContactPredictionTrainer
from .data import create_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Contact Prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config JSON file (overrides all other args)",
    )

    # Data arguments
    parser.add_argument(
        "--data_root", type=str,
        default="/oscar/data/gdk/hli230/projects/vitac-particle/processed",
        help="Root directory containing object data folders",
    )
    parser.add_argument(
        "--objects", type=str, nargs="+", default=None,
        help="Object names to train on. Defaults to all 55 OBJ_NAMES. "
             "Pass a subset for debugging.",
    )
    parser.add_argument(
        "--train_episodes", type=int, nargs="+", default=[0, 1, 2, 3],
        help="Episode IDs for training",
    )
    parser.add_argument(
        "--test_episodes", type=int, nargs="+", default=[4],
        help="Episode IDs for testing",
    )
    parser.add_argument(
        "--cam_name", type=str, default="brics-odroid-022_cam1",
        help="Camera name for RGB frames",
    )
    parser.add_argument(
        "--img_size", type=int, default=224,
        help="Image size for DINOv2 input",
    )

    # Model arguments
    parser.add_argument(
        "--visual_encoder", type=str, default="dinov2_vitb14",
        help="DINOv2 model name",
    )
    parser.add_argument(
        "--freeze_visual_encoder", action="store_true", default=True,
        help="Freeze DINOv2 visual encoder",
    )
    parser.add_argument(
        "--no_freeze_visual_encoder", action="store_true",
        help="Unfreeze DINOv2 visual encoder (fine-tune)",
    )
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Loss arguments
    parser.add_argument(
        "--use_focal_loss", action="store_true", default=True,
        help="Use focal loss for class imbalance",
    )
    parser.add_argument("--no_focal_loss", action="store_true",
                        help="Use standard BCE loss instead of focal loss")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    # Logging arguments
    parser.add_argument("--output_dir", type=str, default="outputs/contact_prediction")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="contact_prediction")

    # Other arguments
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Determine object list
    object_names = args.objects if args.objects is not None else OBJ_NAMES

    # Handle boolean flag pairs
    freeze_visual = args.freeze_visual_encoder and not args.no_freeze_visual_encoder
    use_focal = args.use_focal_loss and not args.no_focal_loss

    # Create config
    if args.config is not None:
        print(f"Loading config from {args.config}")
        config = ContactPredictionConfig.load(args.config)
    else:
        config = ContactPredictionConfig(
            data_root=args.data_root,
            cam_name=args.cam_name,
            img_size=args.img_size,
            train_episodes=args.train_episodes,
            test_episodes=args.test_episodes,
            visual_encoder=args.visual_encoder,
            freeze_visual_encoder=freeze_visual,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            warmup_epochs=args.warmup_epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_focal_loss=use_focal,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            output_dir=args.output_dir,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            mixed_precision=args.mixed_precision,
            seed=args.seed,
            resume_from=args.resume_from,
        )

    print("=" * 60)
    print("Contact Prediction Training")
    print("=" * 60)
    print(f"Objects: {len(object_names)} ({object_names[:3]}...)")
    print(f"Train episodes: {config.train_episodes}")
    print(f"Test episodes: {config.test_episodes}")
    print(f"Config: {json.dumps(config.__dict__, indent=2, default=str)}")
    print("=" * 60)

    # Create dataloaders
    print("Creating training dataloader...")
    train_dataloader = create_dataloader(
        data_root=config.data_root,
        object_names=object_names,
        episode_ids=config.train_episodes,
        cam_name=config.cam_name,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    print("Creating evaluation dataloader...")
    eval_dataloader = create_dataloader(
        data_root=config.data_root,
        object_names=object_names,
        episode_ids=config.test_episodes,
        cam_name=config.cam_name,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Evaluation samples: {len(eval_dataloader.dataset)}")

    # Get pos_weight from training dataset
    pos_weight = train_dataloader.dataset.pos_weight

    # Create trainer
    print("Creating trainer...")
    trainer = ContactPredictionTrainer(config)

    # Start training
    print("Starting training...")
    trainer.train(train_dataloader, eval_dataloader, pos_weight=pos_weight)


if __name__ == "__main__":
    main()
