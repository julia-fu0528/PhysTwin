"""Configuration dataclass for ParticleFormer."""

from dataclasses import dataclass, field
from typing import Optional, List
import json
from pathlib import Path


@dataclass
class ParticleFormerConfig:
    """Configuration for ParticleFormer model and training."""
    
    # Model architecture
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    
    # Embedding dimensions
    embedding_dim: int = 32
    num_objects: int = 10  # Max number of object types
    
    # Input dimensions
    position_dim: int = 3
    action_dim: int = 3
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 8
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    
    # Rollout parameters
    rollout_steps: int = 5  # k in paper
    
    # Loss parameters
    loss_alpha: float = 0.5  # Balance between CD and HD
    
    # Data parameters
    data_root: str = "/mnt/data/ParticleData/processed"
    split_json: str = "split.json"
    object_name: str = "001-rope"
    ep_idx: int = 0
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50
    eval_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "particleformer"
    
    # Checkpointing
    output_dir: str = "outputs/particleformer"
    resume_from: Optional[str] = None
    
    # Device
    device: str = "cuda"
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    
    # Seed
    seed: int = 42
    
    @property
    def input_dim(self) -> int:
        """Total input dimension for state projector."""
        return self.position_dim + self.embedding_dim + self.action_dim
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ParticleFormerConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model % self.nhead == 0, \
            f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
        assert 0.0 <= self.loss_alpha <= 1.0, \
            f"loss_alpha must be in [0, 1], got {self.loss_alpha}"
        assert self.rollout_steps >= 1, \
            f"rollout_steps must be >= 1, got {self.rollout_steps}"
