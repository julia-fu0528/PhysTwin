"""
Contact Prediction: Transformer-based next-frame contact state prediction.

Uses DINOv2 visual features, current contact state, and gripper action
to predict whether contact will occur at the next timestep.
"""

from .config import ContactPredictionConfig

__version__ = "0.1.0"
__all__ = ["ContactPredictionConfig"]
