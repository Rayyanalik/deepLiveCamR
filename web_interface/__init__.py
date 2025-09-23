"""
Web interface package for deepfake detection research platform.
"""

# Import simplified modules that work with basic dependencies
from .simple_streamlit_app import SimpleDeepfakeDetectionApp
from .simple_gradio_app import SimpleGradioApp
from .real_time_video_detector import RealTimeVideoDetector

# For backward compatibility, create aliases
DeepfakeDetectionApp = SimpleDeepfakeDetectionApp
GradioDeepfakeApp = SimpleGradioApp
RealTimeDetector = RealTimeVideoDetector

# Placeholder functions for modules that require additional dependencies
def create_interface(*args, **kwargs):
    """Placeholder for create_interface (requires additional dependencies)."""
    raise NotImplementedError("Full interface requires additional dependencies. Use simplified interfaces instead.")

__all__ = [
    # Simplified modules (working)
    "SimpleDeepfakeDetectionApp",
    "SimpleGradioApp", 
    "RealTimeVideoDetector",
    
    # Backward compatibility aliases
    "DeepfakeDetectionApp",
    "RealTimeDetector",
    "GradioDeepfakeApp",
    "create_interface"
]