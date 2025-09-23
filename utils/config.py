"""
Configuration management for the Deepfake Detection Research Platform.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml

@dataclass
class ModelConfig:
    """Configuration for detection models."""
    model_type: str = "cnn"  # cnn, transformer, ensemble
    input_size: tuple = (224, 224)
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    dropout_rate: float = 0.5
    pretrained: bool = True
    
@dataclass
class DetectionConfig:
    """Configuration for detection settings."""
    confidence_threshold: float = 0.5
    real_time_enabled: bool = True
    max_fps: int = 30
    detection_interval: int = 1  # frames
    face_detection_model: str = "mediapipe"
    
@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_dir: str = "data/"
    synthetic_dir: str = "data/synthetic/"
    real_dir: str = "data/real/"
    augmentation_enabled: bool = True
    normalization: str = "imagenet"  # imagenet, custom, none
    
@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    metrics: List[str] = None
    cross_validation_folds: int = 5
    test_split: float = 0.2
    validation_split: float = 0.2
    save_predictions: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

@dataclass
class WebConfig:
    """Configuration for web interface."""
    host: str = "localhost"
    port: int = 8501
    debug: bool = False
    theme: str = "light"
    max_file_size: int = 100  # MB

class Config:
    """Main configuration class."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.model = ModelConfig()
        self.detection = DetectionConfig()
        self.data = DataConfig()
        self.evaluation = EvaluationConfig()
        self.web = WebConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'detection': self.detection.__dict__,
            'data': self.data.__dict__,
            'evaluation': self.evaluation.__dict__,
            'web': self.web.__dict__
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def get_model_path(self, model_name: str) -> str:
        """Get path for model file."""
        return os.path.join("models", f"{model_name}.pth")
    
    def get_data_path(self, subdir: str = "") -> str:
        """Get path for data directory."""
        if subdir:
            return os.path.join(self.data.data_dir, subdir)
        return self.data.data_dir
    
    def create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data.data_dir,
            self.data.synthetic_dir,
            self.data.real_dir,
            "models",
            "logs",
            "results"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Global configuration instance
config = Config()

# Default configuration values
DEFAULT_CONFIG = {
    'model': {
        'model_type': 'cnn',
        'input_size': [224, 224],
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'dropout_rate': 0.5,
        'pretrained': True
    },
    'detection': {
        'confidence_threshold': 0.5,
        'real_time_enabled': True,
        'max_fps': 30,
        'detection_interval': 1,
        'face_detection_model': 'mediapipe'
    },
    'data': {
        'data_dir': 'data/',
        'synthetic_dir': 'data/synthetic/',
        'real_dir': 'data/real/',
        'augmentation_enabled': True,
        'normalization': 'imagenet'
    },
    'evaluation': {
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        'cross_validation_folds': 5,
        'test_split': 0.2,
        'validation_split': 0.2,
        'save_predictions': True
    },
    'web': {
        'host': 'localhost',
        'port': 8501,
        'debug': False,
        'theme': 'light',
        'max_file_size': 100
    }
}
