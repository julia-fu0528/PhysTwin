#!/usr/bin/env python3
"""Training script for ParticleFormer.

Usage:
    # Basic training with default config
    python -m particleformer.train
    
    # Training with custom config
    python -m particleformer.train --config path/to/config.json
    
    # Training with command line overrides
    python -m particleformer.train --batch_size 8 --learning_rate 1e-4
    
    # Multi-GPU training with accelerate
    accelerate launch -m particleformer.train --config config.json
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
import json
from pathlib import Path

from .config import ParticleFormerConfig
from .trainer import ParticleFormerTrainer
from .data import create_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ParticleFormer model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file",
    )
    
    # Data arguments
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/data/ParticleData/processed",
        help="Root directory containing object data folders",
    )
    parser.add_argument(
        "--object",
        type=str,
        default="001-rope",
        help="Object name to train on (if no global split.json)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode ID to train on (if no global split.json)",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default="split.json",
        help="Path to split.json file",
    )
    
    # Model arguments
    parser.add_argument(
        "--d_model",
        type=int,
        default=256,
        help="Transformer model dimension",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=512,
        help="Feedforward dimension in transformer",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=32,
        help="Object embedding dimension",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=5,
        help="Number of rollout steps for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    
    # Loss arguments
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=0.5,
        help="Weight for Chamfer Distance in hybrid loss (0-1)",
    )
    
    # Logging arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/particleformer",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="particleformer",
        help="Wandb project name",
    )
    
    # Other arguments
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Load or create config
    if args.config is not None:
        print(f"Loading config from {args.config}")
        config = ParticleFormerConfig.load(args.config)
        # Override with command line args if provided
        for key, value in vars(args).items():
            if key != "config" and value is not None:
                if hasattr(config, key):
                    # Only override if explicitly provided (not default)
                    setattr(config, key, value)
    else:
        # Create config from command line args
        config = ParticleFormerConfig(
            data_root=args.data_root,
            split_json=args.split_json,
            object_name=args.object,
            ep_idx=args.episode,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            rollout_steps=args.rollout_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            loss_alpha=args.loss_alpha,
            output_dir=args.output_dir,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            mixed_precision=args.mixed_precision,
            seed=args.seed,
            resume_from=args.resume_from,
        )
    
    print("=" * 60)
    print("ParticleFormer Training")
    print("=" * 60)
    print(f"Config: {json.dumps(config.__dict__, indent=2, default=str)}")
    print("=" * 60)
    
    # Create dataloaders
    print("Creating dataloaders...")
    
    # Make split_json path relative to workspace if not absolute
    split_path = Path(args.split_json)
    if not split_path.is_absolute():
        # Try relative to current directory first
        if not split_path.exists():
            # Try relative to PhysTwin directory
            phystwin_dir = Path(__file__).parent.parent
            split_path = phystwin_dir / args.split_json
    
    train_dataloader = create_dataloader(
        data_root=config.data_root,
        split_path=str(split_path),
        batch_size=config.batch_size,
        rollout_steps=config.rollout_steps,
        num_workers=args.num_workers,
        shuffle=True,
        split="train",
        object_name=args.object,
        episode_ids=[args.episode],
    )
    
    val_dataloader = create_dataloader(
        data_root=config.data_root,
        split_path=str(split_path),
        batch_size=config.batch_size,
        rollout_steps=config.rollout_steps,
        num_workers=args.num_workers,
        shuffle=False,
        split="test",
        object_name=args.object,
        episode_ids=[args.episode],
    )
    
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    print(f"Batches per epoch: {len(train_dataloader)}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = ParticleFormerTrainer(config)
    
    # Start training
    print("Starting training...")
    trainer.train(train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
