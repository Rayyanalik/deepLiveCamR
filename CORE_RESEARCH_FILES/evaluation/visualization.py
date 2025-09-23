"""
Visualization tools for deepfake detection evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
from pathlib import Path

class DetectionVisualizer:
    """Comprehensive visualization tools for deepfake detection results."""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_confusion_matrix(self, 
                            y_true: Union[np.ndarray, List],
                            y_pred: Union[np.ndarray, List],
                            class_names: List[str] = None,
                            normalize: bool = True,
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix."""
        if class_names is None:
            class_names = ["Real", "Fake"]
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, 
                      y_true: Union[np.ndarray, List],
                      y_scores: Union[np.ndarray, List],
                      title: str = "ROC Curve",
                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, 
                                  y_true: Union[np.ndarray, List],
                                  y_scores: Union[np.ndarray, List],
                                  title: str = "Precision-Recall Curve",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Plot precision-recall curve."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(recall, precision, color='darkorange', lw=2,
               label=f'AP = {avg_precision:.3f}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_comparison(self, 
                              metrics_data: Dict[str, Dict[str, float]],
                              metrics: List[str] = None,
                              title: str = "Model Performance Comparison",
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot comparison of multiple metrics across models."""
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Prepare data
        model_names = list(metrics_data.keys())
        metric_values = {metric: [metrics_data[model].get(metric, 0) for model in model_names] 
                        for metric in metrics}
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = metric_values[metric]
            
            bars = ax.bar(model_names, values, color=self.colors[:len(model_names)])
            ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            if len(max(model_names, key=len)) > 8:
                ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confidence_distribution(self, 
                                    confidences: Dict[str, np.ndarray],
                                    title: str = "Prediction Confidence Distribution",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot distribution of prediction confidences."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (label, conf) in enumerate(confidences.items()):
            ax.hist(conf, bins=30, alpha=0.7, label=label, 
                   color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, 
                               importance_scores: Union[np.ndarray, torch.Tensor],
                               feature_names: List[str] = None,
                               title: str = "Feature Importance",
                               top_k: int = 20,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot feature importance scores."""
        if isinstance(importance_scores, torch.Tensor):
            importance_scores = importance_scores.detach().cpu().numpy()
        
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1][:top_k]
        sorted_scores = importance_scores[sorted_indices]
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importance_scores))]
        
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(sorted_scores)), sorted_scores, 
                      color=self.colors[0], alpha=0.8)
        
        ax.set_yticks(range(len(sorted_scores)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
            ax.text(score + 0.01, i, f'{score:.3f}', 
                   va='center', ha='left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

class InteractiveVisualizer:
    """Interactive visualizations using Plotly."""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def create_interactive_confusion_matrix(self, 
                                          y_true: Union[np.ndarray, List],
                                          y_pred: Union[np.ndarray, List],
                                          class_names: List[str] = None) -> go.Figure:
        """Create interactive confusion matrix."""
        if class_names is None:
            class_names = ["Real", "Fake"]
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Interactive Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=600,
            height=500
        )
        
        return fig
    
    def create_interactive_roc_curve(self, 
                                    y_true: Union[np.ndarray, List],
                                    y_scores: Union[np.ndarray, List]) -> go.Figure:
        """Create interactive ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='darkorange', width=3)
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Interactive ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=600,
            height=500
        )
        
        return fig
    
    def create_model_comparison_dashboard(self, 
                                        model_results: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create interactive model comparison dashboard."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[metric.replace('_', ' ').title() for metric in metrics],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]]
        )
        
        model_names = list(model_results.keys())
        
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            values = [model_results[model].get(metric, 0) for model in model_names]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=metric,
                    showlegend=False,
                    marker_color=self.colors[i % len(self.colors)]
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Interactive Model Performance Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig

class RealTimeVisualizer:
    """Real-time visualization for live detection monitoring."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_buffer = {
            'timestamps': [],
            'predictions': [],
            'confidences': [],
            'fps': [],
            'latency': []
        }
    
    def update_buffer(self, 
                     prediction: int,
                     confidence: float,
                     fps: float,
                     latency: float):
        """Update real-time data buffer."""
        import time
        timestamp = time.time()
        
        self.data_buffer['timestamps'].append(timestamp)
        self.data_buffer['predictions'].append(prediction)
        self.data_buffer['confidences'].append(confidence)
        self.data_buffer['fps'].append(fps)
        self.data_buffer['latency'].append(latency)
        
        # Keep only recent data
        for key in self.data_buffer:
            if len(self.data_buffer[key]) > self.window_size:
                self.data_buffer[key] = self.data_buffer[key][-self.window_size:]
    
    def create_realtime_plot(self) -> go.Figure:
        """Create real-time monitoring plot."""
        if not self.data_buffer['timestamps']:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Predictions', 'Confidence', 'FPS', 'Latency'],
            vertical_spacing=0.1
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=self.data_buffer['timestamps'],
                y=self.data_buffer['predictions'],
                mode='lines+markers',
                name='Predictions',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Confidence
        fig.add_trace(
            go.Scatter(
                x=self.data_buffer['timestamps'],
                y=self.data_buffer['confidences'],
                mode='lines+markers',
                name='Confidence',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        # FPS
        fig.add_trace(
            go.Scatter(
                x=self.data_buffer['timestamps'],
                y=self.data_buffer['fps'],
                mode='lines+markers',
                name='FPS',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        # Latency
        fig.add_trace(
            go.Scatter(
                x=self.data_buffer['timestamps'],
                y=self.data_buffer['latency'],
                mode='lines+markers',
                name='Latency (ms)',
                line=dict(color='orange')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Real-time Detection Monitoring",
            height=600,
            showlegend=False
        )
        
        return fig

class FeatureVisualizer:
    """Visualization tools for feature analysis."""
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    def plot_feature_embedding(self, 
                              features: Union[np.ndarray, torch.Tensor],
                              labels: Union[np.ndarray, List],
                              method: str = "tsne",
                              title: str = "Feature Embedding Visualization",
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot 2D embedding of features."""
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Reduce dimensionality
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        features_2d = reducer.fit_transform(features)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                      c=self.colors[i % len(self.colors)], 
                      label=f'Class {label}', alpha=0.7, s=50)
        
        ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attention_heatmap(self, 
                             attention_weights: Union[np.ndarray, torch.Tensor],
                             input_image: Union[np.ndarray, torch.Tensor],
                             title: str = "Attention Heatmap",
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot attention heatmap overlay on input image."""
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        if isinstance(input_image, torch.Tensor):
            input_image = input_image.detach().cpu().numpy()
        
        # Normalize attention weights
        attention_weights = (attention_weights - attention_weights.min()) / \
                          (attention_weights.max() - attention_weights.min())
        
        # Resize attention to match image
        if attention_weights.shape != input_image.shape[:2]:
            attention_weights = cv2.resize(attention_weights, 
                                         (input_image.shape[1], input_image.shape[0]))
        
        # Create heatmap
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(input_image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Attention heatmap
        im = ax2.imshow(attention_weights, cmap='hot', interpolation='nearest')
        ax2.set_title('Attention Heatmap')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2)
        
        # Overlay
        ax3.imshow(input_image)
        ax3.imshow(attention_weights, cmap='hot', alpha=0.5, interpolation='nearest')
        ax3.set_title('Attention Overlay')
        ax3.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def create_visualization_suite() -> Dict[str, object]:
    """Create complete visualization suite."""
    return {
        'detection_visualizer': DetectionVisualizer(),
        'interactive_visualizer': InteractiveVisualizer(),
        'realtime_visualizer': RealTimeVisualizer(),
        'feature_visualizer': FeatureVisualizer()
    }
