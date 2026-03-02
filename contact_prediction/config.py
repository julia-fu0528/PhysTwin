"""Configuration dataclass for Contact Prediction."""

from dataclasses import dataclass, field
from typing import Optional, List
import json
from pathlib import Path


# The 55 object names from submit_pgnd_multi_episode.sh
OBJ_NAMES = [
    "002-rope-silk", "003-cable", "004-rubber-band", "001-rope",
    "006-fur", "008-pink-cloth", "010-orange-cloth", "011-green-cloth",
    "012-hat-cloth", "013-glove-cloth", "016-shirt-cloth",
    "015-airbag-cloth", "017-chessboard-cloth", "018-trashbag-cloth",
    "019-trashbag-plastic-cloth", "021-bag-cloth", "022-handkerchief",
    "024-glass-cleaner-cloth", "023-cleaning-cloth",
    "025-bag-small-cloth", "027-umbrella-bag-cloth", "026-sock-cloth",
    "030-foam-flat-cloth", "029-foam-cloth", "038-mat-cloth",
    "040-paper-cloth", "043-dog", "045-cat", "046-sponge",
    "048-butter-sponge", "059-shoe", "062-banana", "063-flower",
    "068-nylon-rope", "082-curtain-cloth", "088-snake", "090-sloth",
    "092-squirrel", "096-octopus", "100-puppet", "095-watermelon",
    "103-ice-pack-cloth", "109-pouch-cloth", "110-shower-cap-cloth",
    "113-collar", "115-cotton-gauze-cloth", "120-bread-plush",
    "118-envelope-cloth", "117-bubble-wrap-cloth",
    "121-croissant-plush", "125-rabbit", "135-makeup-sponge",
    "147-baking-mold", "148-crepe-paper-cloth",
    "156-mesh-produce-bag-cloth", "150-shredded-packing-paper-cloth",
    "157-sack-cloth", "159-purse", "163-bear", "164-sheep",
]


@dataclass
class ContactPredictionConfig:
    """Configuration for Contact Prediction model and training."""

    # Data parameters
    data_root: str = "/oscar/data/gdk/hli230/projects/vitac-particle/processed"
    cam_name: str = "brics-odroid-022_cam1"
    img_size: int = 224
    train_episodes: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    test_episodes: List[int] = field(default_factory=lambda: [4])

    # Model architecture
    visual_encoder: str = "dinov2_vitb14"  # DINOv2 model name
    freeze_visual_encoder: bool = True
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Gripper action dimension (mean 3D velocity of controller particles)
    action_dim: int = 3

    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 50
    warmup_epochs: int = 5
    gradient_accumulation_steps: int = 1

    # Loss: handle class imbalance
    use_focal_loss: bool = True
    focal_alpha: float = 0.25  # weighting for positive (contact) class
    focal_gamma: float = 2.0   # focusing parameter

    # Logging
    log_interval: int = 10
    save_interval: int = 5
    eval_interval: int = 1
    use_wandb: bool = False
    wandb_project: str = "contact_prediction"

    # Checkpointing
    output_dir: str = "outputs/contact_prediction"
    resume_from: Optional[str] = None

    # Device
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"

    # Seed
    seed: int = 42

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ContactPredictionConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model % self.nhead == 0, \
            f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
        assert self.img_size > 0, f"img_size must be positive, got {self.img_size}"
