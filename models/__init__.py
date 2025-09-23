"""
Deepfake detection models package.
"""

# Import simplified models that work with basic dependencies
from .simple_cnn_detector import (
    SimpleCNNDeepfakeDetector,
    SimpleMultiScaleCNN,
    SimpleAttentionCNN,
    create_simple_cnn_model,
    SIMPLE_MODEL_CONFIGS
)

# For backward compatibility, create aliases
CNNDeepfakeDetector = SimpleCNNDeepfakeDetector
MultiScaleCNN = SimpleMultiScaleCNN
AttentionCNN = SimpleAttentionCNN
create_cnn_model = create_simple_cnn_model
MODEL_CONFIGS = SIMPLE_MODEL_CONFIGS

# Placeholder functions for modules that require additional dependencies
def create_transformer_model(*args, **kwargs):
    """Placeholder for transformer models (requires additional dependencies)."""
    raise NotImplementedError("Transformer models require additional dependencies. Use CNN models instead.")

def create_ensemble_model(*args, **kwargs):
    """Placeholder for ensemble models (requires additional dependencies)."""
    raise NotImplementedError("Ensemble models require additional dependencies. Use CNN models instead.")

# Placeholder classes for modules that require additional dependencies
class VisionTransformerDetector:
    """Placeholder for transformer models (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Transformer models require additional dependencies. Use CNN models instead.")

class TemporalTransformer:
    """Placeholder for transformer models (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Transformer models require additional dependencies. Use CNN models instead.")

class MultiModalTransformer:
    """Placeholder for transformer models (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Transformer models require additional dependencies. Use CNN models instead.")

class SwinTransformerDetector:
    """Placeholder for transformer models (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Transformer models require additional dependencies. Use CNN models instead.")

class EnsembleDetector:
    """Placeholder for ensemble models (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Ensemble models require additional dependencies. Use CNN models instead.")

class HeterogeneousEnsemble:
    """Placeholder for ensemble models (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Ensemble models require additional dependencies. Use CNN models instead.")

class StackingEnsemble:
    """Placeholder for ensemble models (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Ensemble models require additional dependencies. Use CNN models instead.")

class DynamicEnsemble:
    """Placeholder for ensemble models (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Ensemble models require additional dependencies. Use CNN models instead.")

# Placeholder configurations
TRANSFORMER_CONFIGS = {}
ENSEMBLE_CONFIGS = {}

__all__ = [
    # CNN Models (working)
    "CNNDeepfakeDetector",
    "MultiScaleCNN", 
    "AttentionCNN",
    "create_cnn_model",
    "MODEL_CONFIGS",
    
    # Simplified CNN Models
    "SimpleCNNDeepfakeDetector",
    "SimpleMultiScaleCNN",
    "SimpleAttentionCNN",
    "create_simple_cnn_model",
    "SIMPLE_MODEL_CONFIGS",
    
    # Placeholder Models (not implemented)
    "VisionTransformerDetector",
    "TemporalTransformer",
    "MultiModalTransformer", 
    "SwinTransformerDetector",
    "create_transformer_model",
    "TRANSFORMER_CONFIGS",
    
    # Placeholder Ensemble Models (not implemented)
    "EnsembleDetector",
    "HeterogeneousEnsemble",
    "StackingEnsemble",
    "DynamicEnsemble",
    "create_ensemble_model",
    "ENSEMBLE_CONFIGS"
]