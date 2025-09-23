"""
Simplified Gradio interface for deepfake detection research platform.
Works with basic dependencies only.
"""

import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import time
from typing import Dict, Tuple, Optional

# Import our simplified modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simple_cnn_detector import create_simple_cnn_model
from utils.simple_preprocessing import SimpleFaceDetector, SimpleImagePreprocessor

class SimpleGradioApp:
    """Simplified Gradio application class."""
    
    def __init__(self):
        self.device = self._setup_device()
        self.model = None
        self.face_detector = None
        self.image_preprocessor = None
        self.model_loaded = False
        
    def _setup_device(self):
        """Setup compute device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self, model_name: str) -> str:
        """Load a detection model."""
        try:
            self.model = create_simple_cnn_model(model_name, pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.face_detector = SimpleFaceDetector(method="opencv")
            self.image_preprocessor = SimpleImagePreprocessor()
            self.model_loaded = True
            
            return f"✅ {model_name.upper()} model loaded successfully on {self.device}"
            
        except Exception as e:
            self.model_loaded = False
            return f"❌ Error loading model: {str(e)}"
    
    def detect_deepfake(self, image: Image.Image, confidence_threshold: float) -> Tuple[str, Image.Image]:
        """Detect deepfake in image."""
        if not self.model_loaded:
            return "❌ Please load a model first", image
        
        try:
            # Convert PIL to numpy
            image_np = np.array(image)
            
            # Detect faces
            faces = self.face_detector.detect_faces(image_np)
            
            if not faces:
                return "❌ No face detected in image", image
            
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_region = image_np[y:y+h, x:x+w]
            
            # Preprocess face
            processed_face = self.image_preprocessor.preprocess_image(face_region)
            processed_face = processed_face.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                start_time = time.time()
                output = self.model(processed_face)
                inference_time = time.time() - start_time
                
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(output, dim=1).item()
                confidence = torch.max(probabilities, dim=1)[0].item()
            
            # Create result image with overlay
            result_image = image.copy()
            
            # Draw face box
            if prediction == 1:  # Fake
                color = (255, 0, 0)  # Red
                status = "🚨 FAKE DETECTED"
            else:  # Real
                color = (0, 255, 0)  # Green
                status = "✅ REAL IMAGE"
            
            # Convert PIL to OpenCV format for drawing
            cv_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), color, 3)
            
            # Add label
            label = f"{'Fake' if prediction == 1 else 'Real'} ({confidence:.3f})"
            cv2.putText(cv_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Convert back to PIL
            result_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # Create result text
            result_text = f"{status}\nConfidence: {confidence:.3f}\nInference Time: {inference_time*1000:.1f}ms"
            
            return result_text, result_image
            
        except Exception as e:
            return f"❌ Error during detection: {str(e)}", image

def create_gradio_interface():
    """Create Gradio interface."""
    app = SimpleGradioApp()
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(css=css, title="Deepfake Detection Research Platform") as interface:
        
        # Header
        gr.Markdown("""
        # 🔍 Deepfake Detection Research Platform
        
        <div class="warning-box">
        <strong>⚠️ Research Platform Notice:</strong> This platform is designed for educational and research purposes only. 
        It focuses on developing detection algorithms to identify deepfakes and protect against malicious use.
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                gr.Markdown("### 🔧 Configuration")
                
                model_name = gr.Dropdown(
                    choices=["resnet18", "resnet34", "resnet50", "mobilenet_v2", "multiscale", "attention"],
                    value="resnet18",
                    label="CNN Model",
                    info="Select CNN architecture"
                )
                
                load_button = gr.Button("🔄 Load Model", variant="primary")
                
                model_status = gr.Textbox(
                    label="Model Status",
                    value="No model loaded",
                    interactive=False
                )
                
                confidence_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Minimum confidence for detection"
                )
                
                # System info
                gr.Markdown("### 🖥️ System Info")
                device_info = gr.Textbox(
                    label="Device",
                    value=f"Device: {app.device}",
                    interactive=False
                )
                
            with gr.Column(scale=2):
                # Image detection
                gr.Markdown("### 📸 Image Detection")
                
                with gr.Row():
                    input_image = gr.Image(
                        label="Upload Image",
                        type="pil",
                        height=300
                    )
                    
                    output_image = gr.Image(
                        label="Detection Result",
                        height=300
                    )
                
                detect_button = gr.Button("🔍 Detect Deepfake", variant="primary")
                
                result_text = gr.Textbox(
                    label="Detection Result",
                    lines=3,
                    interactive=False
                )
        
        # Event handlers
        def load_model_handler(model_name):
            return app.load_model(model_name)
        
        def detect_handler(image, threshold):
            if image is None:
                return "❌ Please upload an image", None
            return app.detect_deepfake(image, threshold)
        
        # Connect events
        load_button.click(
            fn=load_model_handler,
            inputs=[model_name],
            outputs=[model_status]
        )
        
        detect_button.click(
            fn=detect_handler,
            inputs=[input_image, confidence_threshold],
            outputs=[result_text, output_image]
        )
        
        # Examples
        gr.Markdown("### 📖 Examples")
        
        # Add some example images if available
        examples = []
        
        gr.Examples(
            examples=examples,
            inputs=[input_image],
            outputs=[output_image],
            fn=lambda x: app.detect_deepfake(x, 0.5)[1] if app.model_loaded else None,
            cache_examples=False
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 📚 About This Platform
        
        This Deepfake Detection Research Platform is designed for:
        
        - **🔬 Research**: Develop and test detection algorithms
        - **📚 Education**: Learn about deepfake technology
        - **🛡️ Protection**: Identify and prevent malicious deepfakes
        - **⚡ Real-time**: Test detection performance
        
        ### 🛡️ Ethical Guidelines
        
        **✅ Allowed Uses:**
        - Research and education
        - Developing detection algorithms
        - Protecting against malicious deepfakes
        - Understanding AI safety
        
        **❌ Prohibited Uses:**
        - Creating malicious deepfakes
        - Non-consensual image generation
        - Spreading misinformation
        - Harassment or fraud
        
        ### 🔧 Technical Details
        
        - **Framework**: PyTorch
        - **Preprocessing**: OpenCV
        - **Interface**: Gradio
        - **Device Support**: CPU, CUDA, MPS
        
        ---
        <div style="text-align: center;">
        <p>🔍 Deepfake Detection Research Platform | Built for AI Safety Research</p>
        </div>
        """)
    
    return interface

def main():
    """Main function to launch Gradio interface."""
    print("🚀 Launching Deepfake Detection Research Platform...")
    print("🔍 Gradio Interface")
    print("🛡️ Research Platform - Educational Use Only")
    
    interface = create_gradio_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
