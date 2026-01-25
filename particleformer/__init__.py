"""
ParticleFormer: A Transformer-based 3D Point Cloud World Model

This package implements ParticleFormer for multi-object, multi-material
robotic manipulation dynamics prediction.
"""

from .config import ParticleFormerConfig
from .models import ParticleFormer, ObjectEmbedding
from .losses import hybrid_loss, chamfer_distance, hausdorff_distance

__version__ = "0.1.0"
__all__ = [
    "ParticleFormerConfig",
    "ParticleFormer",
    "ObjectEmbedding",
    "hybrid_loss",
    "chamfer_distance",
    "hausdorff_distance",
]
