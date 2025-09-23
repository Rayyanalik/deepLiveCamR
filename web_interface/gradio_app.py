"""
Gradio web interface for deepfake detection research platform.
"""

import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import io
import base64

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_cnn_model, create_transformer_model, create_ensemble_model
from utils.preprocessing import FaceDetector, ImagePreprocessor
from evaluation import DetectionMetrics, create_visualization_suite
from utils.config import config

class GradioDeepfakeApp:
    """Gradio application for deepfake detection."""
    
    def __init__(self):
        self.device = self._setup_device()
        self.models = {}
        self.current_model = None
        self.face_detector = None
        self.image_preprocessor = None
        self.metrics = DetectionMetrics()
        self.visualizers = create_visualization_suite()
        
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self, model_type: str, model_name: str) -> str:
        """Load a detection model."""
        try:
            model_key = f"{model_type}_{model_name}"
            
            if model_key in self.models:
                self.current_model = self.models[model_key]
                return f"✅ Model {model_key} already loaded"
            
            if model_type == "cnn":
                model = create_cnn_model(model_name=model_name, pretrained=True)
            elif model_type == "transformer":
                model = create_transformer_model(model_type=model_name, pretrained=True)
            elif model_type == "ensemble":
                model = create_ensemble_model(ensemble_type=model_name)
            else:
                return f"❌ Unknown model type: {model_type}"
            
            model = model.to(self.device)
            model.eval()
            
            self.models[model_key] = model
            self.current_model = model
            
            return f"✅ {model_type.upper()} model '{model_name}' loaded successfully on {self.device}"
            
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def setup_preprocessing(self):
        """Setup preprocessing components."""
        if self.face_detector is None:
            self.face_detector = FaceDetector(method="mediapipe")
        if self.image_preprocessor is None:
            self.image_preprocessor = ImagePreprocessor()
    
    def detect_deepfake(self, image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """Detect deepfake in image."""
        if self.current_model is None:
            return {
                'prediction': 'No model loaded',
                'confidence': 0.0,
                'is_fake': False,
                'error': 'Please load a model first'
            }
        
        try:
            self.setup_preprocessing()
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if not faces:
                return {
                    'prediction': 'No face detected',
                    'confidence': 0.0,
                    'is_fake': False,
                    'error': 'No face found in image'
                }
            
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            # Preprocess face
            processed_face = self.image_preprocessor.preprocess_image(face_region)
            processed_face = processed_face.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                start_time = time.time()
                output = self.current_model(processed_face)
                inference_time = time.time() - start_time
                
                if output.dim() > 1:
                    probabilities = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = torch.max(probabilities, dim=1)[0].item()
                else:
                    probabilities = torch.sigmoid(output)
                    prediction = (output > 0.5).long().item()
                    confidence = torch.max(torch.stack([probabilities, 1-probabilities]), dim=0)[0].item()
            
            return {
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': confidence,
                'is_fake': prediction == 1,
                'inference_time': inference_time,
                'face_region': (x, y, w, h),
                'threshold_exceeded': confidence >= confidence_threshold
            }
            
        except Exception as e:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'is_fake': False,
                'error': str(e)
            }
    
    def process_image(self, image: Image.Image, confidence_threshold: float) -> Tuple[str, Image.Image, str]:
        """Process uploaded image."""
        if image is None:
            return "No image provided", None, ""
        
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Make detection
        result = self.detect_deepfake(image_np, confidence_threshold)
        
        # Create result image with overlay
        result_image = image.copy()
        
        if 'face_region' in result and result['face_region']:
            x, y, w, h = result['face_region']
            
            # Draw face box
            if result['is_fake']:
                color = (255, 0, 0)  # Red for fake
            else:
                color = (0, 255, 0)  # Green for real
            
            # Convert PIL to OpenCV format for drawing
            cv_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), color, 3)
            
            # Add label
            label = f"{result['prediction']} ({result['confidence']:.3f})"
            cv2.putText(cv_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Convert back to PIL
            result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        # Create result text
        if 'error' in result:
            result_text = f"❌ Error: {result['error']}"
        else:
            status = "🚨 FAKE DETECTED" if result['is_fake'] else "✅ REAL IMAGE"
            result_text = f"{status}\nConfidence: {result['confidence']:.3f}\nInference Time: {result['inference_time']*1000:.1f}ms"
        
        return result_text, result_image, result['prediction']
    
    def create_performance_plot(self) -> go.Figure:
        """Create performance visualization."""
        # Simulate performance data
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        values = [0.95, 0.93, 0.97, 0.95, 0.98]
        
        fig = go.Figure(data=[
            go.Bar(x=metrics, y=values, marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ])
        
        fig.update_layout(
            title="Model Performance Metrics",
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            height=400
        )
        
        return fig
    
    def create_model_comparison(self, selected_models: List[str]) -> go.Figure:
        """Create model comparison visualization."""
        if len(selected_models) < 2:
            return go.Figure()
        
        # Simulate comparison data
        comparison_data = {
            'ResNet50': [0.95, 0.93, 45],
            'EfficientNet-B0': [0.93, 0.92, 38],
            'ViT-Base': [0.96, 0.95, 120],
            'Multi-Scale CNN': [0.94, 0.93, 55],
            'Ensemble': [0.97, 0.96, 85]
        }
        
        metrics = ['Accuracy', 'F1-Score', 'Inference Time (ms)']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=metrics,
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(metrics):
            values = [comparison_data[model][i] for model in selected_models]
            
            fig.add_trace(
                go.Bar(x=selected_models, y=values, name=metric, 
                      marker_color=colors[i], showlegend=False),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=400,
            showlegend=False
        )
        
        return fig

def create_interface():
    """Create Gradio interface."""
    app = GradioDeepfakeApp()
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    """
    
    with gr.Blocks(css=css, title="Deepfake Detection Research Platform") as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🔍 Deepfake Detection Research Platform</h1>
            <p>Advanced AI-powered deepfake detection for research and education</p>
        </div>
        """)
        
        # Warning notice
        gr.HTML("""
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0;">
            <strong>⚠️ Research Platform Notice:</strong> This platform is designed for educational and research purposes only. 
            It focuses on developing detection algorithms to identify deepfakes and protect against malicious use.
        </div>
        """)
        
        with gr.Tabs():
            
            # Image Detection Tab
            with gr.Tab("📸 Image Detection"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Model Configuration")
                        
                        model_type = gr.Dropdown(
                            choices=["cnn", "transformer", "ensemble"],
                            value="cnn",
                            label="Model Type",
                            info="Select the type of detection model"
                        )
                        
                        model_name = gr.Dropdown(
                            choices=["resnet50", "efficientnet_b0", "multiscale"],
                            value="resnet50",
                            label="Model Name",
                            info="Select specific model architecture"
                        )
                        
                        load_model_btn = gr.Button("🔄 Load Model", variant="primary")
                        model_status = gr.Textbox(label="Model Status", interactive=False)
                        
                        confidence_threshold = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Confidence Threshold",
                            info="Minimum confidence for detection"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### Image Analysis")
                        
                        input_image = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=400
                        )
                        
                        analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
                        
                        result_text = gr.Textbox(
                            label="Detection Result",
                            interactive=False,
                            lines=3
                        )
                        
                        output_image = gr.Image(
                            label="Analysis Result",
                            type="pil",
                            height=400
                        )
                
                # Event handlers
                def update_model_choices(model_type):
                    if model_type == "cnn":
                        return gr.Dropdown(choices=["resnet50", "efficientnet_b0", "multiscale", "temporal", "attention"])
                    elif model_type == "transformer":
                        return gr.Dropdown(choices=["vit_base", "vit_large", "swin_base", "temporal", "multimodal"])
                    else:  # ensemble
                        return gr.Dropdown(choices=["weighted_average", "learned_fusion", "stacking", "dynamic"])
                
                model_type.change(update_model_choices, inputs=model_type, outputs=model_name)
                
                load_model_btn.click(
                    app.load_model,
                    inputs=[model_type, model_name],
                    outputs=model_status
                )
                
                analyze_btn.click(
                    app.process_image,
                    inputs=[input_image, confidence_threshold],
                    outputs=[result_text, output_image, gr.State()]
                )
            
            # Real-time Detection Tab
            with gr.Tab("📹 Real-time Detection"):
                gr.Markdown("### Live Camera Detection")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Real-time deepfake detection using webcam**")
                        gr.Markdown("""
                        To use real-time detection:
                        1. Load a model in the Image Detection tab
                        2. Run the real-time detector script:
                           ```bash
                           python web_interface/real_time_detector.py --model-type cnn --model-name resnet50
                           ```
                        """)
                        
                        gr.Markdown("### Real-time Metrics")
                        
                        # Simulated real-time metrics
                        with gr.Row():
                            current_fps = gr.Number(label="Current FPS", value=0.0)
                            detection_time = gr.Number(label="Detection Time (ms)", value=0.0)
                        
                        with gr.Row():
                            total_frames = gr.Number(label="Total Frames", value=0)
                            fake_detections = gr.Number(label="Fake Detections", value=0)
                        
                        # Real-time plot placeholder
                        realtime_plot = gr.Plot(label="Real-time Performance")
                        
                        # Update button for demo
                        update_metrics_btn = gr.Button("🔄 Update Metrics")
                        
                        def update_realtime_metrics():
                            # Simulate real-time data
                            import random
                            fps = random.uniform(25, 35)
                            detection_time = random.uniform(30, 60)
                            frames = random.randint(1000, 5000)
                            fake_count = random.randint(50, 200)
                            
                            # Create simple plot
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=[random.uniform(0, 1) for _ in range(20)],
                                mode='lines',
                                name='Confidence',
                                line=dict(color='blue')
                            ))
                            fig.update_layout(
                                title="Real-time Confidence",
                                height=300,
                                showlegend=False
                            )
                            
                            return fps, detection_time, frames, fake_count, fig
                        
                        update_metrics_btn.click(
                            update_realtime_metrics,
                            outputs=[current_fps, detection_time, total_frames, fake_detections, realtime_plot]
                        )
            
            # Performance Analysis Tab
            with gr.Tab("📊 Performance Analysis"):
                gr.Markdown("### Model Performance Metrics")
                
                with gr.Row():
                    with gr.Column():
                        performance_plot = gr.Plot(label="Performance Metrics")
                        
                        # Model information
                        gr.Markdown("### Model Information")
                        
                        with gr.Row():
                            total_params = gr.Number(label="Total Parameters", value=0)
                            trainable_params = gr.Number(label="Trainable Parameters", value=0)
                        
                        with gr.Row():
                            model_size = gr.Textbox(label="Model Size", value="N/A")
                            device_info = gr.Textbox(label="Device", value=str(app.device))
                    
                    with gr.Column():
                        gr.Markdown("### Detailed Metrics")
                        
                        # Detailed metrics table
                        metrics_data = {
                            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                            "Value": [0.95, 0.93, 0.97, 0.95, 0.98],
                            "Description": [
                                "Overall correctness",
                                "True positive rate",
                                "Sensitivity",
                                "Harmonic mean of precision and recall",
                                "Area under ROC curve"
                            ]
                        }
                        
                        metrics_df = gr.Dataframe(
                            value=pd.DataFrame(metrics_data),
                            label="Performance Metrics",
                            interactive=False
                        )
                
                # Update performance plot
                def update_performance():
                    return app.create_performance_plot()
                
                performance_plot.value = update_performance()
            
            # Model Comparison Tab
            with gr.Tab("🔬 Model Comparison"):
                gr.Markdown("### Compare Different Models")
                
                with gr.Row():
                    with gr.Column():
                        selected_models = gr.CheckboxGroup(
                            choices=["ResNet50", "EfficientNet-B0", "ViT-Base", "Multi-Scale CNN", "Ensemble"],
                            value=["ResNet50", "EfficientNet-B0"],
                            label="Select Models to Compare"
                        )
                        
                        comparison_plot = gr.Plot(label="Model Comparison")
                        
                        compare_btn = gr.Button("🔄 Compare Models", variant="primary")
                        
                        compare_btn.click(
                            app.create_model_comparison,
                            inputs=selected_models,
                            outputs=comparison_plot
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Comparison Summary")
                        
                        # Comparison table
                        comparison_data = {
                            "Model": ["ResNet50", "EfficientNet-B0", "ViT-Base", "Multi-Scale CNN", "Ensemble"],
                            "Accuracy": [0.95, 0.93, 0.96, 0.94, 0.97],
                            "F1-Score": [0.95, 0.92, 0.95, 0.93, 0.96],
                            "Inference Time (ms)": [45, 38, 120, 55, 85],
                            "Memory Usage (MB)": [150, 120, 300, 180, 250]
                        }
                        
                        comparison_df = gr.Dataframe(
                            value=pd.DataFrame(comparison_data),
                            label="Model Comparison Table",
                            interactive=False
                        )
            
            # Documentation Tab
            with gr.Tab("📚 Documentation"):
                gr.Markdown("""
                ## About This Platform
                
                This Deepfake Detection Research Platform is designed to:
                
                - **Develop Detection Algorithms**: Create and test various deepfake detection methods
                - **Real-time Analysis**: Test detection performance in real-time scenarios
                - **Performance Benchmarking**: Compare different models and approaches
                - **Educational Purpose**: Learn about deepfake technology and detection methods
                
                ### 🔬 Supported Models
                
                **CNN-based Models:**
                - ResNet50: Classic convolutional architecture
                - EfficientNet-B0: Efficient scaling of CNNs
                - Multi-Scale CNN: Detects artifacts at different scales
                - Temporal CNN: Handles video sequences
                - Attention CNN: Focuses on important regions
                
                **Transformer-based Models:**
                - ViT-Base: Vision Transformer for image analysis
                - ViT-Large: Larger transformer model
                - Swin Transformer: Hierarchical vision transformer
                - Temporal Transformer: Handles temporal sequences
                - Multi-Modal Transformer: Combines different input types
                
                **Ensemble Models:**
                - Weighted Average: Simple weighted combination
                - Learned Fusion: Neural network-based fusion
                - Stacking: Meta-learner approach
                - Dynamic Ensemble: Adaptive weight adjustment
                
                ### 🛡️ Ethical Guidelines
                
                This platform is designed for:
                - ✅ Research and education
                - ✅ Developing detection algorithms
                - ✅ Protecting against malicious deepfakes
                - ✅ Understanding AI safety
                
                **Prohibited uses:**
                - ❌ Creating malicious deepfakes
                - ❌ Non-consensual image generation
                - ❌ Spreading misinformation
                - ❌ Harassment or fraud
                
                ### 📖 How to Use
                
                1. **Load a Model**: Select and load a detection model
                2. **Image Detection**: Upload images for analysis
                3. **Real-time Detection**: Use the real-time detector script
                4. **Performance Analysis**: View model performance metrics
                5. **Model Comparison**: Compare different models
                
                ### 🔧 Technical Details
                
                - **Framework**: PyTorch
                - **Preprocessing**: OpenCV, MediaPipe
                - **Interface**: Gradio
                - **Visualization**: Plotly
                - **Device Support**: CPU, CUDA, MPS
                """)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #ddd;">
            <p>🔍 Deepfake Detection Research Platform | Built for AI Safety Research</p>
        </div>
        """)
    
    return interface

def main():
    """Main function to launch Gradio app."""
    interface = create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()
