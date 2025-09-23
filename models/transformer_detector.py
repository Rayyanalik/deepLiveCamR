"""
Transformer-based deepfake detection model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig, AutoModel, AutoConfig
from typing import Tuple, Optional, List
import math

class VisionTransformerDetector(nn.Module):
    """Vision Transformer for deepfake detection."""
    
    def __init__(self,
                 model_name: str = "google/vit-base-patch16-224",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.1,
                 freeze_backbone: bool = False):
        super(VisionTransformerDetector, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained ViT model
        if pretrained:
            self.backbone = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig.from_pretrained(model_name)
            self.backbone = ViTModel(config)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        self.feature_dim = self.backbone.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classification head
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Get ViT outputs
        outputs = self.backbone(pixel_values=x)
        
        # Use CLS token representation
        cls_representation = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Classification
        output = self.classifier(cls_representation)
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        outputs = self.backbone(pixel_values=x)
        return outputs.last_hidden_state[:, 0]  # CLS token

class TemporalTransformer(nn.Module):
    """Transformer for temporal sequence modeling in videos."""
    
    def __init__(self,
                 spatial_model_name: str = "google/vit-base-patch16-224",
                 sequence_length: int = 16,
                 num_classes: int = 2,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout_rate: float = 0.1):
        super(TemporalTransformer, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Spatial feature extractor (ViT)
        self.spatial_extractor = ViTModel.from_pretrained(spatial_model_name)
        spatial_dim = self.spatial_extractor.config.hidden_size
        
        # Project spatial features to temporal dimension
        self.spatial_projection = nn.Linear(spatial_dim, hidden_dim)
        
        # Temporal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for video sequence."""
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for spatial feature extraction
        x_reshaped = x.view(batch_size * seq_len, channels, height, width)
        
        # Extract spatial features
        spatial_outputs = self.spatial_extractor(pixel_values=x_reshaped)
        spatial_features = spatial_outputs.last_hidden_state[:, 0]  # CLS tokens
        
        # Reshape back to sequence
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        # Project to temporal dimension
        temporal_features = self.spatial_projection(spatial_features)
        
        # Add positional encoding
        temporal_features = self.pos_encoding(temporal_features)
        
        # Temporal modeling
        temporal_output = self.temporal_transformer(temporal_features)
        
        # Global average pooling
        pooled_output = temporal_output.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled_output)
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class MultiModalTransformer(nn.Module):
    """Multi-modal transformer for combining different input modalities."""
    
    def __init__(self,
                 visual_model_name: str = "google/vit-base-patch16-224",
                 num_classes: int = 2,
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout_rate: float = 0.1):
        super(MultiModalTransformer, self).__init__()
        
        # Visual encoder
        self.visual_encoder = ViTModel.from_pretrained(visual_model_name)
        visual_dim = self.visual_encoder.config.hidden_size
        
        # Audio encoder (placeholder - would need actual audio features)
        self.audio_dim = 128  # Placeholder dimension
        self.audio_projection = nn.Linear(self.audio_dim, hidden_dim)
        
        # Project visual features
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        
        # Cross-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.cross_modal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Modality embeddings
        self.modality_embeddings = nn.Embedding(2, hidden_dim)  # visual, audio
    
    def forward(self, visual_input: torch.Tensor, 
                audio_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with multi-modal input."""
        batch_size = visual_input.size(0)
        
        # Visual features
        visual_outputs = self.visual_encoder(pixel_values=visual_input)
        visual_features = visual_outputs.last_hidden_state[:, 0]  # CLS token
        visual_features = self.visual_projection(visual_features)
        
        # Add modality embedding
        visual_features = visual_features + self.modality_embeddings(
            torch.zeros(batch_size, dtype=torch.long, device=visual_input.device)
        )
        
        # Prepare multi-modal sequence
        if audio_input is not None:
            # Audio features (placeholder)
            audio_features = self.audio_projection(audio_input)
            audio_features = audio_features + self.modality_embeddings(
                torch.ones(batch_size, dtype=torch.long, device=visual_input.device)
            )
            
            # Concatenate modalities
            multi_modal_features = torch.stack([visual_features, audio_features], dim=1)
        else:
            multi_modal_features = visual_features.unsqueeze(1)
        
        # Cross-modal modeling
        cross_modal_output = self.cross_modal_transformer(multi_modal_features)
        
        # Global pooling
        pooled_output = cross_modal_output.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled_output)
        return output

class SwinTransformerDetector(nn.Module):
    """Swin Transformer for deepfake detection."""
    
    def __init__(self,
                 model_name: str = "microsoft/swin-base-patch4-window7-224",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.1):
        super(SwinTransformerDetector, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained Swin Transformer
        if pretrained:
            self.backbone = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel(config)
        
        # Get feature dimension
        self.feature_dim = self.backbone.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Get Swin Transformer outputs
        outputs = self.backbone(pixel_values=x)
        
        # Use pooled output
        pooled_output = outputs.pooler_output
        
        # Classification
        output = self.classifier(pooled_output)
        return output

def create_transformer_model(model_type: str = "vit_base",
                           num_classes: int = 2,
                           pretrained: bool = True,
                           **kwargs) -> nn.Module:
    """Factory function to create transformer models."""
    
    if model_type == "vit_base":
        return VisionTransformerDetector(
            model_name="google/vit-base-patch16-224",
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == "vit_large":
        return VisionTransformerDetector(
            model_name="google/vit-large-patch16-224",
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == "swin_base":
        return SwinTransformerDetector(
            model_name="microsoft/swin-base-patch4-window7-224",
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == "temporal":
        return TemporalTransformer(num_classes=num_classes, **kwargs)
    elif model_type == "multimodal":
        return MultiModalTransformer(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown transformer model type: {model_type}")

# Transformer model configurations
TRANSFORMER_CONFIGS = {
    "vit_base": {
        "model_type": "vit_base",
        "input_size": (224, 224),
        "pretrained": True,
        "dropout_rate": 0.1
    },
    "vit_large": {
        "model_type": "vit_large",
        "input_size": (224, 224),
        "pretrained": True,
        "dropout_rate": 0.1
    },
    "swin_base": {
        "model_type": "swin_base",
        "input_size": (224, 224),
        "pretrained": True,
        "dropout_rate": 0.1
    },
    "temporal": {
        "model_type": "temporal",
        "sequence_length": 16,
        "hidden_dim": 256,
        "num_heads": 8,
        "num_layers": 4,
        "dropout_rate": 0.1
    },
    "multimodal": {
        "model_type": "multimodal",
        "hidden_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "dropout_rate": 0.1
    }
}
