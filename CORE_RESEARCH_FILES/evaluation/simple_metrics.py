"""
Simplified evaluation metrics for deepfake detection models.
Works with basic dependencies only.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

class SimpleDetectionMetrics:
    """Simplified metrics for deepfake detection evaluation."""
    
    def __init__(self, num_classes: int = 2, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or ["Real", "Fake"]
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.inference_times = []
    
    def update(self, 
               predictions: Union[torch.Tensor, np.ndarray],
               targets: Union[torch.Tensor, np.ndarray],
               probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None,
               inference_time: Optional[float] = None):
        """Update metrics with new batch of predictions."""
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
        
        # Store predictions and targets
        self.predictions.extend(predictions.flatten())
        self.targets.extend(targets.flatten())
        
        if probabilities is not None:
            self.probabilities.extend(probabilities)
        
        if inference_time is not None:
            self.inference_times.append(inference_time)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        if not self.predictions:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Basic classification metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Per-class metrics
        precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name.lower()}_precision'] = precision_per_class[i]
            metrics[f'{class_name.lower()}_recall'] = recall_per_class[i]
            metrics[f'{class_name.lower()}_f1'] = f1_per_class[i]
        
        # ROC-AUC if probabilities available
        if self.probabilities:
            probabilities = np.array(self.probabilities)
            if probabilities.ndim > 1 and probabilities.shape[1] > 1:
                # Multi-class probabilities
                roc_auc = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
                metrics['roc_auc'] = roc_auc
            else:
                # Binary probabilities
                roc_auc = roc_auc_score(targets, probabilities)
                metrics['roc_auc'] = roc_auc
        
        # Performance metrics
        if self.inference_times:
            avg_inference_time = np.mean(self.inference_times)
            std_inference_time = np.std(self.inference_times)
            metrics['avg_inference_time_ms'] = avg_inference_time * 1000
            metrics['std_inference_time_ms'] = std_inference_time * 1000
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if not self.predictions:
            return np.array([])
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        return confusion_matrix(targets, predictions)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if not self.predictions:
            return "No predictions available"
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        return classification_report(targets, predictions, target_names=self.class_names)

class SimpleRealTimeMetrics:
    """Simplified metrics for real-time detection evaluation."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.frame_times = []
        self.detection_times = []
        self.predictions = []
        self.targets = []
    
    def update_frame(self, 
                    frame_time: float,
                    detection_time: float,
                    prediction: int,
                    target: int):
        """Update metrics for a single frame."""
        self.frame_times.append(frame_time)
        self.detection_times.append(detection_time)
        self.predictions.append(prediction)
        self.targets.append(target)
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if len(self.frame_times) < 2:
            return 0.0
        
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff == 0:
            return 0.0
        
        return (len(self.frame_times) - 1) / time_diff
    
    def get_avg_detection_time(self) -> float:
        """Get average detection time per frame."""
        if not self.detection_times:
            return 0.0
        return np.mean(self.detection_times)
    
    def get_accuracy(self) -> float:
        """Get current accuracy."""
        if not self.predictions:
            return 0.0
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        return accuracy_score(targets, predictions)

def create_simple_evaluation_suite(num_classes: int = 2, 
                                 class_names: List[str] = None) -> Dict[str, object]:
    """Create simplified evaluation suite."""
    return {
        'detection_metrics': SimpleDetectionMetrics(num_classes, class_names),
        'realtime_metrics': SimpleRealTimeMetrics()
    }
