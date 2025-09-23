"""
CNN-based deepfake detection model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional
import timm

class CNNDeepfakeDetector(nn.Module):
    """CNN-based deepfake detection model."""
    
    def __init__(self, 
                 model_name: str = "resnet50",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 input_size: Tuple[int, int] = (224, 224)):
        super(CNNDeepfakeDetector, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Load backbone model
        if model_name in ["resnet50", "resnet101", "resnet152"]:
            self.backbone = getattr(models, model_name)(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final classification layer
        elif model_name in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]:
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            self.feature_dim = self.backbone.classifier.in_features
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

class MultiScaleCNN(nn.Module):
    """Multi-scale CNN for detecting artifacts at different scales."""
    
    def __init__(self, 
                 num_classes: int = 2,
                 dropout_rate: float = 0.5):
        super(MultiScaleCNN, self).__init__()
        
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

class TemporalCNN(nn.Module):
    """CNN with temporal modeling for video sequences."""
    
    def __init__(self, 
                 backbone_name: str = "resnet50",
                 sequence_length: int = 16,
                 num_classes: int = 2,
                 dropout_rate: float = 0.5):
        super(TemporalCNN, self).__init__()
        
        self.sequence_length = sequence_length
        
        # Spatial feature extractor
        if backbone_name == "resnet50":
            self.spatial_extractor = models.resnet50(pretrained=True)
            self.feature_dim = self.spatial_extractor.fc.in_features
            self.spatial_extractor.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Temporal modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.feature_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for video sequence."""
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for spatial feature extraction
        x_reshaped = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract spatial features
        spatial_features = self.spatial_extractor(x_reshaped)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        # Transpose for temporal convolution
        temporal_input = spatial_features.transpose(1, 2)  # (batch, features, seq_len)
        
        # Temporal modeling
        temporal_features = self.temporal_conv(temporal_input)
        temporal_features = temporal_features.squeeze(-1)  # (batch, features)
        
        # Classification
        output = self.classifier(temporal_features)
        return output

class AttentionCNN(nn.Module):
    """CNN with attention mechanism for focusing on important regions."""
    
    def __init__(self, 
                 backbone_name: str = "resnet50",
                 num_classes: int = 2,
                 attention_dim: int = 256,
                 dropout_rate: float = 0.5):
        super(AttentionCNN, self).__init__()
        
        # Backbone
        if backbone_name == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
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

def create_cnn_model(model_type: str = "resnet50",
                    num_classes: int = 2,
                    pretrained: bool = True,
                    **kwargs) -> nn.Module:
    """Factory function to create CNN models."""
    
    if model_type in ["resnet50", "resnet101", "resnet152", 
                      "efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]:
        return CNNDeepfakeDetector(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == "multiscale":
        return MultiScaleCNN(num_classes=num_classes, **kwargs)
    elif model_type == "temporal":
        return TemporalCNN(num_classes=num_classes, **kwargs)
    elif model_type == "attention":
        return AttentionCNN(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Model configurations
MODEL_CONFIGS = {
    "resnet50": {
        "model_type": "resnet50",
        "input_size": (224, 224),
        "pretrained": True,
        "dropout_rate": 0.5
    },
    "efficientnet_b0": {
        "model_type": "efficientnet_b0",
        "input_size": (224, 224),
        "pretrained": True,
        "dropout_rate": 0.3
    },
    "multiscale": {
        "model_type": "multiscale",
        "input_size": (224, 224),
        "dropout_rate": 0.5
    },
    "temporal": {
        "model_type": "temporal",
        "sequence_length": 16,
        "dropout_rate": 0.5
    },
    "attention": {
        "model_type": "attention",
        "attention_dim": 256,
        "dropout_rate": 0.5
    }
}
