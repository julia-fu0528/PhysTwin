import numpy
numpy.bool = bool  # to avoid deprecation error

from . import utils

# Optional imports - may fail if dependencies are not installed
try:
    from . import data
except ImportError as e:
    print(f"Warning: Could not import pgnd.data: {e}")
    data = None

try:
    from . import models
except ImportError as e:
    print(f"Warning: Could not import pgnd.models: {e}")
    models = None

try:
    from . import sim
except ImportError as e:
    print(f"Warning: Could not import pgnd.sim: {e}")
    sim = None
