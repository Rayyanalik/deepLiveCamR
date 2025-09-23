"""
Simplified CNN-based deepfake detection model implementation.
Works with basic PyTorch dependencies only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional

class SimpleCNNDeepfakeDetector(nn.Module):
    """Simplified CNN-based deepfake detection model."""
    
    def __init__(self, 
                 model_name: str = "resnet18",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 input_size: Tuple[int, int] = (224, 224)):
        super(SimpleCNNDeepfakeDetector, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Load backbone model (using torchvision models only)
        if model_name == "resnet18":
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final classification layer
        elif model_name == "resnet34":
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)

class SimpleMultiScaleCNN(nn.Module):
    """Simplified multi-scale CNN for detecting artifacts at different scales."""
    
    def __init__(self, 
                 num_classes: int = 2,
                 dropout_rate: float = 0.5):
        super(SimpleMultiScaleCNN, self).__init__()
        
        # Different scale branches
        self.scale_1 = self._create_branch(224, 64)  # Fine details
        self.scale_2 = self._create_branch(112, 128)  # Medium details
        self.scale_3 = self._create_branch(56, 256)   # Coarse details
        
        # Fusion layer
        total_features = 64 + 128 + 256
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def _create_branch(self, input_size: int, out_channels: int) -> nn.Module:
        """Create a CNN branch for specific scale."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale processing."""
        # Resize input for different scales
        x1 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x2 = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        x3 = F.interpolate(x, size=(56, 56), mode='bilinear', align_corners=False)
        
        # Extract features from each scale
        f1 = self.scale_1(x1)
        f2 = self.scale_2(x2)
        f3 = self.scale_3(x3)
        
        # Concatenate features
        combined_features = torch.cat([f1, f2, f3], dim=1)
        
        # Final classification
        output = self.fusion(combined_features)
        return output

class SimpleAttentionCNN(nn.Module):
    """Simplified CNN with attention mechanism for focusing on important regions."""
    
    def __init__(self, 
                 backbone_name: str = "resnet18",
                 num_classes: int = 2,
                 attention_dim: int = 256,
                 dropout_rate: float = 0.5):
        super(SimpleAttentionCNN, self).__init__()
        
        # Backbone
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Classification
        output = self.classifier(attended_features)
        return output

def create_simple_cnn_model(model_type: str = "resnet18",
                          num_classes: int = 2,
                          pretrained: bool = True,
                          **kwargs) -> nn.Module:
    """Factory function to create simplified CNN models."""
    
    if model_type in ["resnet18", "resnet34", "resnet50", "mobilenet_v2"]:
        return SimpleCNNDeepfakeDetector(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == "multiscale":
        return SimpleMultiScaleCNN(num_classes=num_classes, **kwargs)
    elif model_type == "attention":
        return SimpleAttentionCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Simplified model configurations
SIMPLE_MODEL_CONFIGS = {
    "resnet18": {
        "model_type": "resnet18",
        "input_size": (224, 224),
        "pretrained": True,
        "dropout_rate": 0.5
    },
    "resnet34": {
        "model_type": "resnet34",
        "input_size": (224, 224),
        "pretrained": True,
        "dropout_rate": 0.5
    },
    "resnet50": {
        "model_type": "resnet50",
        "input_size": (224, 224),
        "pretrained": True,
        "dropout_rate": 0.5
    },
    "mobilenet_v2": {
        "model_type": "mobilenet_v2",
        "input_size": (224, 224),
        "pretrained": True,
        "dropout_rate": 0.3
    },
    "multiscale": {
        "model_type": "multiscale",
        "input_size": (224, 224),
        "dropout_rate": 0.5
    },
    "attention": {
        "model_type": "attention",
        "attention_dim": 256,
        "dropout_rate": 0.5
    }
}
