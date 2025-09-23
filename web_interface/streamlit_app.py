"""
Streamlit web interface for deepfake detection research platform.
"""

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional
import io
import base64

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_cnn_model, create_transformer_model, create_ensemble_model
from utils.preprocessing import FaceDetector, ImagePreprocessor
from evaluation import DetectionMetrics, RealTimeMetrics, create_visualization_suite
from utils.config import config

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection Research Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DeepfakeDetectionApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        self.device = self._setup_device()
        self.models = {}
        self.current_model = None
        self.face_detector = None
        self.image_preprocessor = None
        self.metrics = DetectionMetrics()
        self.realtime_metrics = RealTimeMetrics()
        self.visualizers = create_visualization_suite()
        
    def _setup_device(self):
        """Setup compute device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self, model_type: str, model_name: str, **kwargs):
        """Load a detection model."""
        try:
            if model_type == "cnn":
                model = create_cnn_model(model_name=model_name, **kwargs)
            elif model_type == "transformer":
                model = create_transformer_model(model_type=model_name, **kwargs)
            elif model_type == "ensemble":
                model = create_ensemble_model(ensemble_type=model_name, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model = model.to(self.device)
            model.eval()
            
            self.models[f"{model_type}_{model_name}"] = model
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    def setup_preprocessing(self):
        """Setup preprocessing components."""
        if self.face_detector is None:
            self.face_detector = FaceDetector(method="mediapipe")
        if self.image_preprocessor is None:
            self.image_preprocessor = ImagePreprocessor()
    
    def detect_deepfake(self, image: np.ndarray, model: torch.nn.Module) -> Dict:
        """Detect deepfake in image."""
        try:
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
                output = model(processed_face)
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
                'face_region': (x, y, w, h)
            }
            
        except Exception as e:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'is_fake': False,
                'error': str(e)
            }

def main():
    """Main application function."""
    app = DeepfakeDetectionApp()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Deepfake Detection Research Platform</h1>', 
                unsafe_allow_html=True)
    
    # Warning box
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Research Platform Notice:</strong> This platform is designed for educational and research purposes only. 
        It focuses on developing detection algorithms to identify deepfakes and protect against malicious use.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model selection
        st.subheader("Model Selection")
        model_type = st.selectbox(
            "Model Type",
            ["cnn", "transformer", "ensemble"],
            help="Select the type of detection model to use"
        )
        
        if model_type == "cnn":
            model_name = st.selectbox(
                "CNN Model",
                ["resnet50", "efficientnet_b0", "multiscale", "temporal", "attention"],
                help="Select CNN architecture"
            )
        elif model_type == "transformer":
            model_name = st.selectbox(
                "Transformer Model",
                ["vit_base", "vit_large", "swin_base", "temporal", "multimodal"],
                help="Select transformer architecture"
            )
        else:  # ensemble
            model_name = st.selectbox(
                "Ensemble Type",
                ["weighted_average", "learned_fusion", "stacking", "dynamic"],
                help="Select ensemble method"
            )
        
        # Load model button
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                model = app.load_model(model_type, model_name)
                if model is not None:
                    app.current_model = model
                    st.success(f"✅ {model_type.upper()} model '{model_name}' loaded successfully!")
                else:
                    st.error("❌ Failed to load model")
        
        # Detection settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence for detection"
        )
        
        show_face_boxes = st.checkbox(
            "Show Face Detection Boxes",
            value=True,
            help="Display bounding boxes around detected faces"
        )
        
        # Device info
        st.subheader("System Info")
        st.info(f"🖥️ Device: {app.device}")
        st.info(f"🐍 Python: {sys.version.split()[0]}")
        st.info(f"🔥 PyTorch: {torch.__version__}")
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📸 Image Detection", 
        "📹 Real-time Detection", 
        "📊 Performance Analysis", 
        "🔬 Model Comparison",
        "📚 Documentation"
    ])
    
    with tab1:
        st.header("📸 Image Deepfake Detection")
        
        if app.current_model is None:
            st.warning("⚠️ Please load a model first using the sidebar.")
        else:
            app.setup_preprocessing()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Upload Image")
                
                # File upload
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['png', 'jpg', 'jpeg'],
                    help="Upload an image to analyze for deepfake detection"
                )
                
                # Camera input
                st.subheader("Or Use Camera")
                if st.button("📷 Take Photo"):
                    camera_input = st.camera_input("Take a photo")
                    if camera_input:
                        uploaded_file = camera_input
            
            with col2:
                st.subheader("Detection Results")
                
                if uploaded_file is not None:
                    # Load and display image
                    image = Image.open(uploaded_file)
                    image_np = np.array(image)
                    
                    # Make detection
                    result = app.detect_deepfake(image_np, app.current_model)
                    
                    # Display results
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        # Prediction result
                        if result['is_fake']:
                            st.error(f"🚨 **FAKE DETECTED**")
                            st.error(f"Confidence: {result['confidence']:.3f}")
                        else:
                            st.success(f"✅ **REAL IMAGE**")
                            st.success(f"Confidence: {result['confidence']:.3f}")
                        
                        # Additional info
                        st.info(f"⏱️ Inference Time: {result['inference_time']*1000:.1f} ms")
                        
                        # Draw face box if requested
                        if show_face_boxes and 'face_region' in result:
                            x, y, w, h = result['face_region']
                            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Display image
                        st.image(image_np, caption="Analyzed Image", use_column_width=True)
    
    with tab2:
        st.header("📹 Real-time Detection")
        
        if app.current_model is None:
            st.warning("⚠️ Please load a model first using the sidebar.")
        else:
            app.setup_preprocessing()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Live Camera Feed")
                
                # Camera input for real-time
                camera_input = st.camera_input("Real-time Detection", key="realtime_camera")
                
                if camera_input:
                    # Process frame
                    image = Image.open(camera_input)
                    image_np = np.array(image)
                    
                    # Make detection
                    result = app.detect_deepfake(image_np, app.current_model)
                    
                    # Update real-time metrics
                    if 'inference_time' in result:
                        app.realtime_metrics.update_frame(
                            frame_time=time.time(),
                            detection_time=result['inference_time'],
                            prediction=1 if result['is_fake'] else 0,
                            target=0  # Assuming real for demo
                        )
            
            with col2:
                st.subheader("Real-time Metrics")
                
                # Display current metrics
                current_fps = app.realtime_metrics.get_fps()
                current_accuracy = app.realtime_metrics.get_accuracy()
                avg_detection_time = app.realtime_metrics.get_avg_detection_time()
                
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric("Current FPS", f"{current_fps:.1f}")
                    st.metric("Detection Time", f"{avg_detection_time*1000:.1f} ms")
                
                with col2_2:
                    st.metric("Accuracy", f"{current_accuracy:.3f}")
                    st.metric("Status", "🟢 Active" if current_fps > 10 else "🟡 Slow")
                
                # Real-time plot
                if len(app.realtime_metrics.frame_times) > 10:
                    windowed_metrics = app.realtime_metrics.get_windowed_metrics()
                    
                    if 'windowed_fps' in windowed_metrics:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=windowed_metrics['windowed_fps'],
                            mode='lines',
                            name='FPS',
                            line=dict(color='blue')
                        ))
                        fig.update_layout(
                            title="Real-time FPS",
                            xaxis_title="Frame",
                            yaxis_title="FPS",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📊 Performance Analysis")
        
        if app.current_model is None:
            st.warning("⚠️ Please load a model first using the sidebar.")
        else:
            st.subheader("Model Performance Metrics")
            
            # Simulate some performance data for demo
            performance_data = {
                'Accuracy': 0.95,
                'Precision': 0.93,
                'Recall': 0.97,
                'F1-Score': 0.95,
                'ROC-AUC': 0.98
            }
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            for i, (metric, value) in enumerate(performance_data.items()):
                with [col1, col2, col3][i % 3]:
                    st.metric(metric, f"{value:.3f}")
            
            # Performance visualization
            st.subheader("Performance Visualization")
            
            # Create performance chart
            fig = go.Figure(data=[
                go.Bar(x=list(performance_data.keys()), 
                      y=list(performance_data.values()),
                      marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ])
            
            fig.update_layout(
                title="Model Performance Metrics",
                xaxis_title="Metrics",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model complexity info
            st.subheader("Model Information")
            
            total_params = sum(p.numel() for p in app.current_model.parameters())
            trainable_params = sum(p.numel() for p in app.current_model.parameters() if p.requires_grad)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Parameters", f"{total_params:,}")
            with col2:
                st.metric("Trainable Parameters", f"{trainable_params:,}")
    
    with tab4:
        st.header("🔬 Model Comparison")
        
        st.subheader("Compare Different Models")
        
        # Model comparison interface
        comparison_models = st.multiselect(
            "Select models to compare",
            ["ResNet50", "EfficientNet-B0", "ViT-Base", "Multi-Scale CNN", "Ensemble"],
            default=["ResNet50", "EfficientNet-B0"]
        )
        
        if len(comparison_models) >= 2:
            # Simulate comparison data
            comparison_data = {
                'Model': comparison_models,
                'Accuracy': [0.95, 0.93, 0.96, 0.94, 0.97],
                'F1-Score': [0.95, 0.92, 0.95, 0.93, 0.96],
                'Inference Time (ms)': [45, 38, 120, 55, 85],
                'Memory Usage (MB)': [150, 120, 300, 180, 250]
            }
            
            # Create comparison chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Accuracy', 'F1-Score', 'Inference Time', 'Memory Usage'],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            metrics = ['Accuracy', 'F1-Score', 'Inference Time (ms)', 'Memory Usage (MB)']
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, metric in enumerate(metrics):
                row = i // 2 + 1
                col = i % 2 + 1
                
                fig.add_trace(
                    go.Bar(x=comparison_data['Model'], 
                          y=comparison_data[metric],
                          name=metric,
                          marker_color=colors[i],
                          showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="Model Performance Comparison",
                height=600,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary table
            st.subheader("Comparison Summary")
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Please select at least 2 models to compare.")
    
    with tab5:
        st.header("📚 Documentation")
        
        st.subheader("About This Platform")
        
        st.markdown("""
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
        
        1. **Load a Model**: Select and load a detection model from the sidebar
        2. **Image Detection**: Upload images or use camera for detection
        3. **Real-time Detection**: Monitor live camera feed
        4. **Performance Analysis**: View model performance metrics
        5. **Model Comparison**: Compare different models
        
        ### 🔧 Technical Details
        
        - **Framework**: PyTorch
        - **Preprocessing**: OpenCV, MediaPipe
        - **Interface**: Streamlit
        - **Visualization**: Plotly
        - **Device Support**: CPU, CUDA, MPS
        """)
        
        # Contact information
        st.subheader("📞 Contact & Support")
        
        st.markdown("""
        For questions, issues, or contributions:
        
        - 📧 Email: research@deepfake-detection.org
        - 🐛 Issues: GitHub Issues
        - 📖 Documentation: Full documentation available
        - 🤝 Contributing: See contribution guidelines
        """)

if __name__ == "__main__":
    main()
