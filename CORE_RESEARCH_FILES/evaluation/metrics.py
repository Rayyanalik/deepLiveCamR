"""
Evaluation metrics for deepfake detection models.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

class DetectionMetrics:
    """Comprehensive metrics for deepfake detection evaluation."""
    
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
        self.batch_times = []
    
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

class RealTimeMetrics:
    """Metrics specifically for real-time detection evaluation."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.frame_times = []
        self.detection_times = []
        self.predictions = []
        self.targets = []
        self.frame_counts = []
    
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
        self.frame_counts.append(len(self.frame_times))
    
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
    
    def get_windowed_metrics(self) -> Dict[str, List[float]]:
        """Get metrics computed over sliding windows."""
        if len(self.predictions) < self.window_size:
            return {}
        
        windowed_accuracy = []
        windowed_fps = []
        
        for i in range(self.window_size, len(self.predictions)):
            # Accuracy over window
            window_preds = self.predictions[i-self.window_size:i]
            window_targets = self.targets[i-self.window_size:i]
            acc = accuracy_score(window_targets, window_preds)
            windowed_accuracy.append(acc)
            
            # FPS over window
            window_times = self.frame_times[i-self.window_size:i]
            if len(window_times) > 1:
                time_diff = window_times[-1] - window_times[0]
                if time_diff > 0:
                    fps = (len(window_times) - 1) / time_diff
                    windowed_fps.append(fps)
                else:
                    windowed_fps.append(0.0)
            else:
                windowed_fps.append(0.0)
        
        return {
            'windowed_accuracy': windowed_accuracy,
            'windowed_fps': windowed_fps
        }

class BenchmarkSuite:
    """Comprehensive benchmarking suite for deepfake detection models."""
    
    def __init__(self):
        self.results = defaultdict(list)
    
    def benchmark_model(self, 
                      model: torch.nn.Module,
                      dataloader: torch.utils.data.DataLoader,
                      device: torch.device,
                      model_name: str = "model") -> Dict[str, float]:
        """Benchmark a single model."""
        model.eval()
        
        metrics = DetectionMetrics()
        total_time = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                # Measure inference time
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                
                # Get predictions
                if output.dim() > 1:
                    predictions = torch.argmax(output, dim=1)
                    probabilities = torch.softmax(output, dim=1)
                else:
                    predictions = (output > 0.5).long()
                    probabilities = torch.sigmoid(output)
                
                # Update metrics
                metrics.update(predictions, target, probabilities, inference_time)
                
                total_time += inference_time
                num_batches += 1
        
        # Compute final metrics
        final_metrics = metrics.compute_metrics()
        final_metrics['total_time'] = total_time
        final_metrics['avg_batch_time'] = total_time / num_batches if num_batches > 0 else 0
        final_metrics['throughput_fps'] = len(metrics.predictions) / total_time if total_time > 0 else 0
        
        # Store results
        for key, value in final_metrics.items():
            self.results[f"{model_name}_{key}"].append(value)
        
        return final_metrics
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compare multiple models."""
        comparison = {}
        
        for model_name, results in model_results.items():
            comparison[model_name] = {
                'accuracy': results.get('accuracy', 0),
                'f1_score': results.get('f1_score', 0),
                'roc_auc': results.get('roc_auc', 0),
                'avg_inference_time_ms': results.get('avg_inference_time_ms', 0),
                'throughput_fps': results.get('throughput_fps', 0)
            }
        
        return comparison
    
    def get_best_model(self, model_results: Dict[str, Dict[str, float]], 
                      metric: str = 'f1_score') -> Tuple[str, float]:
        """Get the best performing model for a given metric."""
        best_model = None
        best_score = -1
        
        for model_name, results in model_results.items():
            score = results.get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model, best_score

class ModelAnalyzer:
    """Advanced model analysis tools."""
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_prediction_confidence(self, 
                                    model: torch.nn.Module,
                                    dataloader: torch.utils.data.DataLoader,
                                    device: torch.device) -> Dict[str, np.ndarray]:
        """Analyze prediction confidence distribution."""
        model.eval()
        
        all_confidences = []
        correct_confidences = []
        incorrect_confidences = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if output.dim() > 1:
                    probabilities = torch.softmax(output, dim=1)
                    confidences = torch.max(probabilities, dim=1)[0]
                    predictions = torch.argmax(output, dim=1)
                else:
                    probabilities = torch.sigmoid(output)
                    confidences = torch.max(torch.stack([probabilities, 1-probabilities]), dim=0)[0]
                    predictions = (output > 0.5).long()
                
                # Store confidences
                confidences_np = confidences.cpu().numpy()
                predictions_np = predictions.cpu().numpy()
                targets_np = target.cpu().numpy()
                
                all_confidences.extend(confidences_np)
                
                # Separate correct and incorrect predictions
                correct_mask = predictions_np == targets_np
                correct_confidences.extend(confidences_np[correct_mask])
                incorrect_confidences.extend(confidences_np[~correct_mask])
        
        return {
            'all_confidences': np.array(all_confidences),
            'correct_confidences': np.array(correct_confidences),
            'incorrect_confidences': np.array(incorrect_confidences)
        }
    
    def analyze_feature_importance(self, 
                                 model: torch.nn.Module,
                                 sample_input: torch.Tensor,
                                 device: torch.device) -> Dict[str, torch.Tensor]:
        """Analyze feature importance using gradient-based methods."""
        model.eval()
        sample_input = sample_input.to(device)
        sample_input.requires_grad_(True)
        
        # Forward pass
        output = model(sample_input)
        
        # Compute gradients
        if output.dim() > 1:
            target_class = torch.argmax(output, dim=1)
            loss = torch.nn.functional.cross_entropy(output, target_class)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(output, torch.sigmoid(output))
        
        loss.backward()
        
        # Get gradients
        gradients = sample_input.grad
        
        return {
            'gradients': gradients,
            'gradient_magnitude': torch.abs(gradients),
            'gradient_mean': torch.mean(torch.abs(gradients), dim=0)
        }
    
    def analyze_model_complexity(self, model: torch.nn.Module) -> Dict[str, int]:
        """Analyze model complexity."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count layers
        num_layers = len(list(model.modules()))
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'num_layers': num_layers
        }

def create_evaluation_suite(num_classes: int = 2, 
                           class_names: List[str] = None) -> Dict[str, object]:
    """Create a complete evaluation suite."""
    return {
        'detection_metrics': DetectionMetrics(num_classes, class_names),
        'realtime_metrics': RealTimeMetrics(),
        'benchmark_suite': BenchmarkSuite(),
        'model_analyzer': ModelAnalyzer()
    }
