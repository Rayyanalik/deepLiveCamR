"""
Ensemble deepfake detection model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from .cnn_detector import CNNDeepfakeDetector, MultiScaleCNN, TemporalCNN
from .transformer_detector import VisionTransformerDetector, TemporalTransformer

class EnsembleDetector(nn.Module):
    """Ensemble of multiple deepfake detection models."""
    
    def __init__(self,
                 models: List[nn.Module],
                 fusion_method: str = "weighted_average",
                 weights: Optional[List[float]] = None,
                 num_classes: int = 2):
        super(EnsembleDetector, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        
        # Initialize weights
        if weights is None:
            self.weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        else:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
        
        # Fusion layer for learned combination
        if fusion_method == "learned_fusion":
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.num_models * num_classes, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(predictions, dim=1)  # (batch, num_models, num_classes)
        
        if self.fusion_method == "weighted_average":
            # Weighted average of predictions
            weights = F.softmax(self.weights, dim=0)
            weighted_preds = torch.sum(stacked_preds * weights.view(1, -1, 1), dim=1)
            return weighted_preds
        
        elif self.fusion_method == "learned_fusion":
            # Learned fusion of predictions
            flattened_preds = stacked_preds.view(stacked_preds.size(0), -1)
            return self.fusion_layer(flattened_preds)
        
        elif self.fusion_method == "max_voting":
            # Maximum voting
            return torch.max(stacked_preds, dim=1)[0]
        
        elif self.fusion_method == "average":
            # Simple average
            return torch.mean(stacked_preds, dim=1)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get predictions from individual models."""
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        return predictions

class HeterogeneousEnsemble(nn.Module):
    """Ensemble combining different types of models (CNN, Transformer, etc.)."""
    
    def __init__(self,
                 cnn_models: List[nn.Module],
                 transformer_models: List[nn.Module],
                 fusion_method: str = "attention",
                 num_classes: int = 2):
        super(HeterogeneousEnsemble, self).__init__()
        
        self.cnn_models = nn.ModuleList(cnn_models)
        self.transformer_models = nn.ModuleList(transformer_models)
        self.num_classes = num_classes
        
        # Attention-based fusion
        if fusion_method == "attention":
            self.attention_weights = nn.Sequential(
                nn.Linear(num_classes * (len(cnn_models) + len(transformer_models)), 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, len(cnn_models) + len(transformer_models)),
                nn.Softmax(dim=1)
            )
        
        # Cross-modal fusion
        elif fusion_method == "cross_modal":
            self.cross_modal_fusion = nn.Sequential(
                nn.Linear(num_classes * 2, 128),  # CNN + Transformer features
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through heterogeneous ensemble."""
        # CNN predictions
        cnn_preds = []
        for model in self.cnn_models:
            pred = model(x)
            cnn_preds.append(pred)
        
        # Transformer predictions
        transformer_preds = []
        for model in self.transformer_models:
            pred = model(x)
            transformer_preds.append(pred)
        
        # Combine all predictions
        all_preds = cnn_preds + transformer_preds
        stacked_preds = torch.stack(all_preds, dim=1)
        
        # Apply fusion method
        if hasattr(self, 'attention_weights'):
            # Attention-based fusion
            flattened_preds = stacked_preds.view(stacked_preds.size(0), -1)
            attention_weights = self.attention_weights(flattened_preds)
            weighted_preds = torch.sum(stacked_preds * attention_weights.unsqueeze(-1), dim=1)
            return weighted_preds
        
        elif hasattr(self, 'cross_modal_fusion'):
            # Cross-modal fusion
            cnn_avg = torch.mean(torch.stack(cnn_preds), dim=0)
            transformer_avg = torch.mean(torch.stack(transformer_preds), dim=0)
            combined_features = torch.cat([cnn_avg, transformer_avg], dim=1)
            return self.cross_modal_fusion(combined_features)
        
        else:
            # Simple average
            return torch.mean(stacked_preds, dim=1)

class StackingEnsemble(nn.Module):
    """Stacking ensemble with meta-learner."""
    
    def __init__(self,
                 base_models: List[nn.Module],
                 meta_learner_type: str = "neural_network",
                 num_classes: int = 2):
        super(StackingEnsemble, self).__init__()
        
        self.base_models = nn.ModuleList(base_models)
        self.num_base_models = len(base_models)
        self.meta_learner_type = meta_learner_type
        self.num_classes = num_classes
        
        # Neural network meta-learner
        if meta_learner_type == "neural_network":
            self.meta_learner = nn.Sequential(
                nn.Linear(self.num_base_models * num_classes, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )
        
        # Traditional ML meta-learners (for CPU inference)
        self.traditional_meta_learners = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "svm": SVC(probability=True, random_state=42),
            "logistic_regression": LogisticRegression(random_state=42)
        }
        
        self.current_meta_learner = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stacking ensemble."""
        # Get base model predictions
        base_predictions = []
        for model in self.base_models:
            pred = model(x)
            base_predictions.append(pred)
        
        # Stack predictions
        stacked_preds = torch.stack(base_predictions, dim=1)
        
        if self.meta_learner_type == "neural_network":
            # Neural network meta-learner
            flattened_preds = stacked_preds.view(stacked_preds.size(0), -1)
            return self.meta_learner(flattened_preds)
        
        else:
            # Traditional ML meta-learner (requires CPU inference)
            return self._traditional_meta_learning(stacked_preds)
    
    def _traditional_meta_learning(self, stacked_preds: torch.Tensor) -> torch.Tensor:
        """Traditional ML meta-learning."""
        if self.current_meta_learner is None:
            # Use simple average if no meta-learner trained
            return torch.mean(stacked_preds, dim=1)
        
        # Convert to numpy for traditional ML
        preds_np = stacked_preds.detach().cpu().numpy()
        preds_reshaped = preds_np.reshape(preds_np.shape[0], -1)
        
        # Get meta-learner predictions
        meta_preds = self.current_meta_learner.predict_proba(preds_reshaped)
        
        return torch.tensor(meta_preds, dtype=torch.float32, device=stacked_preds.device)
    
    def train_meta_learner(self, X_train: torch.Tensor, y_train: torch.Tensor,
                          meta_learner_name: str = "random_forest"):
        """Train traditional ML meta-learner."""
        # Get base model predictions on training data
        base_predictions = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                pred = model(X_train)
                base_predictions.append(pred.cpu().numpy())
        
        # Stack predictions
        stacked_preds = np.stack(base_predictions, axis=1)
        stacked_preds = stacked_preds.reshape(stacked_preds.shape[0], -1)
        
        # Train meta-learner
        self.current_meta_learner = self.traditional_meta_learners[meta_learner_name]
        self.current_meta_learner.fit(stacked_preds, y_train.cpu().numpy())

class DynamicEnsemble(nn.Module):
    """Dynamic ensemble that adapts weights based on input characteristics."""
    
    def __init__(self,
                 models: List[nn.Module],
                 adaptation_network: nn.Module,
                 num_classes: int = 2):
        super(DynamicEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.adaptation_network = adaptation_network
        self.num_classes = num_classes
        
        # Weight generation network
        self.weight_generator = nn.Sequential(
            nn.Linear(adaptation_network.output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(models)),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic weight adaptation."""
        # Get input characteristics
        input_features = self.adaptation_network(x)
        
        # Generate dynamic weights
        dynamic_weights = self.weight_generator(input_features)
        
        # Get model predictions
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Apply dynamic weights
        stacked_preds = torch.stack(predictions, dim=1)
        weighted_preds = torch.sum(stacked_preds * dynamic_weights.unsqueeze(-1), dim=1)
        
        return weighted_preds

class InputAdaptationNetwork(nn.Module):
    """Network to analyze input characteristics for dynamic ensemble."""
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224)):
        super(InputAdaptationNetwork, self).__init__()
        
        self.input_size = input_size
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Characteristic analysis
        self.characteristic_analyzer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        
        self.output_dim = 32
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract input characteristics."""
        features = self.feature_extractor(x)
        characteristics = self.characteristic_analyzer(features)
        return characteristics

def create_ensemble_model(ensemble_type: str = "weighted_average",
                         model_configs: List[Dict] = None,
                         num_classes: int = 2,
                         **kwargs) -> nn.Module:
    """Factory function to create ensemble models."""
    
    if model_configs is None:
        # Default model configurations
        model_configs = [
            {"type": "cnn", "model_name": "resnet50"},
            {"type": "cnn", "model_name": "efficientnet_b0"},
            {"type": "transformer", "model_name": "vit_base"},
            {"type": "cnn", "model_name": "multiscale"}
        ]
    
    # Create individual models
    models = []
    for config in model_configs:
        if config["type"] == "cnn":
            model = CNNDeepfakeDetector(
                model_name=config["model_name"],
                num_classes=num_classes,
                **kwargs
            )
        elif config["type"] == "transformer":
            model = VisionTransformerDetector(
                model_name=f"google/vit-{config['model_name']}-patch16-224",
                num_classes=num_classes,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        models.append(model)
    
    # Create ensemble
    if ensemble_type == "weighted_average":
        return EnsembleDetector(models, fusion_method="weighted_average", **kwargs)
    elif ensemble_type == "learned_fusion":
        return EnsembleDetector(models, fusion_method="learned_fusion", **kwargs)
    elif ensemble_type == "stacking":
        return StackingEnsemble(models, **kwargs)
    elif ensemble_type == "dynamic":
        adaptation_net = InputAdaptationNetwork()
        return DynamicEnsemble(models, adaptation_net, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

# Ensemble configurations
ENSEMBLE_CONFIGS = {
    "simple_ensemble": {
        "ensemble_type": "weighted_average",
        "model_configs": [
            {"type": "cnn", "model_name": "resnet50"},
            {"type": "cnn", "model_name": "efficientnet_b0"}
        ]
    },
    "advanced_ensemble": {
        "ensemble_type": "learned_fusion",
        "model_configs": [
            {"type": "cnn", "model_name": "resnet50"},
            {"type": "cnn", "model_name": "efficientnet_b0"},
            {"type": "transformer", "model_name": "vit_base"},
            {"type": "cnn", "model_name": "multiscale"}
        ]
    },
    "stacking_ensemble": {
        "ensemble_type": "stacking",
        "meta_learner_type": "neural_network",
        "model_configs": [
            {"type": "cnn", "model_name": "resnet50"},
            {"type": "transformer", "model_name": "vit_base"}
        ]
    },
    "dynamic_ensemble": {
        "ensemble_type": "dynamic",
        "model_configs": [
            {"type": "cnn", "model_name": "resnet50"},
            {"type": "cnn", "model_name": "efficientnet_b0"},
            {"type": "transformer", "model_name": "vit_base"}
        ]
    }
}
