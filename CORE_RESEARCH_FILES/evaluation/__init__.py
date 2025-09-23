"""
Evaluation package for deepfake detection models.
"""

# Import simplified modules that work with basic dependencies
from .simple_metrics import (
    SimpleDetectionMetrics,
    SimpleRealTimeMetrics,
    create_simple_evaluation_suite
)

# For backward compatibility, create aliases
DetectionMetrics = SimpleDetectionMetrics
RealTimeMetrics = SimpleRealTimeMetrics
create_evaluation_suite = create_simple_evaluation_suite

# Placeholder classes for modules that require additional dependencies
class BenchmarkSuite:
    """Placeholder for benchmark suite (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Benchmark suite requires additional dependencies. Use simplified metrics instead.")

class ModelAnalyzer:
    """Placeholder for model analyzer (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Model analyzer requires additional dependencies. Use simplified metrics instead.")

class BenchmarkConfig:
    """Placeholder for benchmark configuration (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Benchmark configuration requires additional dependencies. Use simplified metrics instead.")

class ModelBenchmarker:
    """Placeholder for model benchmarker (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Model benchmarker requires additional dependencies. Use simplified metrics instead.")

class RealTimeBenchmarker:
    """Placeholder for real-time benchmarker (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Real-time benchmarker requires additional dependencies. Use simplified metrics instead.")

class ScalabilityAnalyzer:
    """Placeholder for scalability analyzer (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Scalability analyzer requires additional dependencies. Use simplified metrics instead.")

class DetectionVisualizer:
    """Placeholder for detection visualizer (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Detection visualizer requires additional dependencies. Use simplified metrics instead.")

class InteractiveVisualizer:
    """Placeholder for interactive visualizer (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Interactive visualizer requires additional dependencies. Use simplified metrics instead.")

class RealTimeVisualizer:
    """Placeholder for real-time visualizer (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Real-time visualizer requires additional dependencies. Use simplified metrics instead.")

class FeatureVisualizer:
    """Placeholder for feature visualizer (requires additional dependencies)."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Feature visualizer requires additional dependencies. Use simplified metrics instead.")

# Placeholder functions
def run_comprehensive_benchmark(*args, **kwargs):
    """Placeholder for comprehensive benchmark (requires additional dependencies)."""
    raise NotImplementedError("Comprehensive benchmark requires additional dependencies. Use simplified metrics instead.")

def create_visualization_suite(*args, **kwargs):
    """Placeholder for visualization suite (requires additional dependencies)."""
    raise NotImplementedError("Visualization suite requires additional dependencies. Use simplified metrics instead.")

# Placeholder configurations
EXAMPLE_MODEL_CONFIGS = {}

__all__ = [
    # Simplified modules (working)
    "SimpleDetectionMetrics",
    "SimpleRealTimeMetrics",
    "create_simple_evaluation_suite",
    
    # Backward compatibility aliases
    "DetectionMetrics",
    "RealTimeMetrics",
    "create_evaluation_suite",
    
    # Placeholder modules (not implemented)
    "BenchmarkSuite",
    "ModelAnalyzer",
    "BenchmarkConfig",
    "ModelBenchmarker",
    "RealTimeBenchmarker",
    "ScalabilityAnalyzer",
    "run_comprehensive_benchmark",
    "EXAMPLE_MODEL_CONFIGS",
    
    # Placeholder visualization modules (not implemented)
    "DetectionVisualizer",
    "InteractiveVisualizer",
    "RealTimeVisualizer", 
    "FeatureVisualizer",
    "create_visualization_suite"
]