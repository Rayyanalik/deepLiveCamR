"""
Utility package for deepfake detection research platform.
"""

# Import simplified modules that work with basic dependencies
from .simple_preprocessing import (
    SimpleFaceDetector,
    SimpleImagePreprocessor,
    SimpleVideoPreprocessor,
    create_simple_preprocessing_pipeline
)

# For backward compatibility, create aliases
FaceDetector = SimpleFaceDetector
ImagePreprocessor = SimpleImagePreprocessor
VideoPreprocessor = SimpleVideoPreprocessor
create_preprocessing_pipeline = create_simple_preprocessing_pipeline

# Placeholder classes for modules that require additional dependencies
class Config:
    """Placeholder for configuration (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Full configuration requires additional dependencies. Use simplified preprocessing instead.")

class ModelConfig:
    """Placeholder for model configuration (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Full configuration requires additional dependencies. Use simplified preprocessing instead.")

class DetectionConfig:
    """Placeholder for detection configuration (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Full configuration requires additional dependencies. Use simplified preprocessing instead.")

class DataConfig:
    """Placeholder for data configuration (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Full configuration requires additional dependencies. Use simplified preprocessing instead.")

class EvaluationConfig:
    """Placeholder for evaluation configuration (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Full configuration requires additional dependencies. Use simplified preprocessing instead.")

class WebConfig:
    """Placeholder for web configuration (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Full configuration requires additional dependencies. Use simplified preprocessing instead.")

class DataAugmentation:
    """Placeholder for data augmentation (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Data augmentation requires additional dependencies. Use simplified preprocessing instead.")

__all__ = [
    # Simplified modules (working)
    "SimpleFaceDetector",
    "SimpleImagePreprocessor",
    "SimpleVideoPreprocessor",
    "create_simple_preprocessing_pipeline",
    
    # Backward compatibility aliases
    "FaceDetector",
    "ImagePreprocessor",
    "VideoPreprocessor",
    "create_preprocessing_pipeline",
    
    # Placeholder modules (not implemented)
    "Config",
    "ModelConfig", 
    "DetectionConfig",
    "DataConfig",
    "EvaluationConfig",
    "WebConfig",
    "DataAugmentation"
]