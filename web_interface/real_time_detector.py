"""
Real-time deepfake detection using webcam.
"""

import cv2
import torch
import numpy as np
import time
import argparse
from typing import Dict, Optional, Tuple
import threading
import queue
from collections import deque

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_cnn_model, create_transformer_model, create_ensemble_model
from utils.preprocessing import FaceDetector, ImagePreprocessor
from evaluation import RealTimeMetrics
from utils.config import config

class RealTimeDetector:
    """Real-time deepfake detection system."""
    
    def __init__(self, 
                 model_type: str = "cnn",
                 model_name: str = "resnet50",
                 confidence_threshold: float = 0.5,
                 target_fps: int = 30,
                 display_width: int = 640,
                 display_height: int = 480):
        
        self.model_type = model_type
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.target_fps = target_fps
        self.display_width = display_width
        self.display_height = display_height
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize components
        self.model = None
        self.face_detector = None
        self.image_preprocessor = None
        self.metrics = RealTimeMetrics()
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        self.predictions = deque(maxlen=100)
        
        # Threading
        self.detection_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        self.running = False
        
        # Statistics
        self.total_frames = 0
        self.fake_detections = 0
        self.real_detections = 0
        
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_model(self) -> bool:
        """Load the detection model."""
        try:
            print(f"Loading {self.model_type} model: {self.model_name}")
            
            if self.model_type == "cnn":
                self.model = create_cnn_model(model_name=self.model_name, pretrained=True)
            elif self.model_type == "transformer":
                self.model = create_transformer_model(model_type=self.model_name, pretrained=True)
            elif self.model_type == "ensemble":
                self.model = create_ensemble_model(ensemble_type=self.model_name)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def setup_preprocessing(self):
        """Setup preprocessing components."""
        self.face_detector = FaceDetector(method="mediapipe")
        self.image_preprocessor = ImagePreprocessor()
        print("✅ Preprocessing components initialized")
    
    def detect_face_in_frame(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Detect and extract face from frame."""
        try:
            faces = self.face_detector.detect_faces(frame)
            
            if not faces:
                return None
            
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            return face_region, (x, y, w, h)
            
        except Exception as e:
            print(f"Face detection error: {str(e)}")
            return None
    
    def predict_deepfake(self, face_region: np.ndarray) -> Dict:
        """Predict if face region is a deepfake."""
        try:
            # Preprocess face
            processed_face = self.image_preprocessor.preprocess_image(face_region)
            processed_face = processed_face.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                start_time = time.time()
                output = self.model(processed_face)
                detection_time = time.time() - start_time
                
                if output.dim() > 1:
                    probabilities = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output, dim=1).item()
                    confidence = torch.max(probabilities, dim=1)[0].item()
                else:
                    probabilities = torch.sigmoid(output)
                    prediction = (output > 0.5).long().item()
                    confidence = torch.max(torch.stack([probabilities, 1-probabilities]), dim=0)[0].item()
            
            return {
                'is_fake': prediction == 1,
                'confidence': confidence,
                'detection_time': detection_time,
                'prediction': prediction
            }
            
        except Exception as e:
            return {
                'is_fake': False,
                'confidence': 0.0,
                'detection_time': 0.0,
                'error': str(e)
            }
    
    def detection_worker(self):
        """Worker thread for detection processing."""
        while self.running:
            try:
                # Get frame from queue
                frame_data = self.detection_queue.get(timeout=1.0)
                frame, face_region, face_box = frame_data
                
                # Make prediction
                result = self.predict_deepfake(face_region)
                result['face_box'] = face_box
                result['frame'] = frame
                
                # Put result in queue
                self.result_queue.put(result, timeout=1.0)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Detection worker error: {str(e)}")
    
    def draw_overlay(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw detection overlay on frame."""
        overlay_frame = frame.copy()
        
        if 'face_box' in result:
            x, y, w, h = result['face_box']
            
            # Choose color based on prediction
            if result['is_fake']:
                color = (0, 0, 255)  # Red for fake
                label = "FAKE"
            else:
                color = (0, 255, 0)  # Green for real
                label = "REAL"
            
            # Draw face bounding box
            cv2.rectangle(overlay_frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(overlay_frame, (x, y-30), (x+label_size[0]+10, y), color, -1)
            cv2.putText(overlay_frame, label, (x+5, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw confidence
            confidence_text = f"Conf: {result['confidence']:.3f}"
            cv2.putText(overlay_frame, confidence_text, (x, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return overlay_frame
    
    def draw_stats(self, frame: np.ndarray) -> np.ndarray:
        """Draw performance statistics on frame."""
        stats_frame = frame.copy()
        
        # Calculate current FPS
        if len(self.frame_times) > 1:
            fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        else:
            fps = 0
        
        # Calculate average detection time
        avg_detection_time = np.mean(self.detection_times) if self.detection_times else 0
        
        # Draw stats
        stats_text = [
            f"FPS: {fps:.1f}",
            f"Detection Time: {avg_detection_time*1000:.1f}ms",
            f"Total Frames: {self.total_frames}",
            f"Fake Detections: {self.fake_detections}",
            f"Real Detections: {self.real_detections}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(stats_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(stats_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25
        
        return stats_frame
    
    def run(self, camera_index: int = 0):
        """Run real-time detection."""
        print("🚀 Starting real-time deepfake detection...")
        
        # Load model
        if not self.load_model():
            return
        
        # Setup preprocessing
        self.setup_preprocessing()
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print(f"📹 Camera initialized: {self.display_width}x{self.display_height} @ {self.target_fps}fps")
        
        # Start detection worker thread
        self.running = True
        detection_thread = threading.Thread(target=self.detection_worker)
        detection_thread.daemon = True
        detection_thread.start()
        
        print("🎯 Detection started. Press 'q' to quit, 's' to save frame")
        
        try:
            while True:
                frame_start_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Resize frame for display
                display_frame = cv2.resize(frame, (self.display_width, self.display_height))
                
                # Detect face
                face_result = self.detect_face_in_frame(frame)
                
                if face_result is not None:
                    face_region, face_box = face_result
                    
                    # Try to add to detection queue (non-blocking)
                    try:
                        self.detection_queue.put_nowait((frame, face_region, face_box))
                    except queue.Full:
                        pass  # Skip this frame if queue is full
                
                # Try to get detection result (non-blocking)
                try:
                    result = self.result_queue.get_nowait()
                    
                    # Update statistics
                    if result['is_fake']:
                        self.fake_detections += 1
                    else:
                        self.real_detections += 1
                    
                    self.detection_times.append(result['detection_time'])
                    self.predictions.append(result['prediction'])
                    
                    # Draw overlay
                    display_frame = self.draw_overlay(display_frame, result)
                    
                except queue.Empty:
                    pass  # No new result available
                
                # Draw statistics
                display_frame = self.draw_stats(display_frame)
                
                # Display frame
                cv2.imshow('Real-time Deepfake Detection', display_frame)
                
                # Update frame timing
                frame_end_time = time.time()
                self.frame_times.append(frame_end_time)
                self.total_frames += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"detection_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"💾 Frame saved as {filename}")
                
                # Control frame rate
                elapsed = frame_end_time - frame_start_time
                target_frame_time = 1.0 / self.target_fps
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
        
        except KeyboardInterrupt:
            print("\n⏹️ Detection stopped by user")
        
        finally:
            # Cleanup
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self.print_final_stats()
    
    def print_final_stats(self):
        """Print final performance statistics."""
        print("\n" + "="*50)
        print("📊 FINAL PERFORMANCE STATISTICS")
        print("="*50)
        
        if self.frame_times:
            total_time = self.frame_times[-1] - self.frame_times[0]
            avg_fps = len(self.frame_times) / total_time if total_time > 0 else 0
            print(f"Average FPS: {avg_fps:.2f}")
        
        if self.detection_times:
            avg_detection_time = np.mean(self.detection_times)
            std_detection_time = np.std(self.detection_times)
            print(f"Average Detection Time: {avg_detection_time*1000:.2f}ms ± {std_detection_time*1000:.2f}ms")
        
        print(f"Total Frames Processed: {self.total_frames}")
        print(f"Fake Detections: {self.fake_detections}")
        print(f"Real Detections: {self.real_detections}")
        
        if self.total_frames > 0:
            fake_rate = self.fake_detections / self.total_frames * 100
            print(f"Fake Detection Rate: {fake_rate:.2f}%")
        
        print("="*50)

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Real-time Deepfake Detection")
    
    parser.add_argument("--model-type", type=str, default="cnn",
                       choices=["cnn", "transformer", "ensemble"],
                       help="Type of detection model")
    
    parser.add_argument("--model-name", type=str, default="resnet50",
                       help="Specific model name")
    
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Confidence threshold for detection")
    
    parser.add_argument("--target-fps", type=int, default=30,
                       help="Target FPS for processing")
    
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index")
    
    parser.add_argument("--width", type=int, default=640,
                       help="Display width")
    
    parser.add_argument("--height", type=int, default=480,
                       help="Display height")
    
    args = parser.parse_args()
    
    # Create detector
    detector = RealTimeDetector(
        model_type=args.model_type,
        model_name=args.model_name,
        confidence_threshold=args.confidence_threshold,
        target_fps=args.target_fps,
        display_width=args.width,
        display_height=args.height
    )
    
    # Run detection
    detector.run(camera_index=args.camera)

if __name__ == "__main__":
    main()
