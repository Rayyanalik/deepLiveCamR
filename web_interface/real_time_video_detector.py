"""
Real-time video deepfake detection for webcam and video files.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import time
from typing import List, Tuple, Optional, Dict, Callable
import threading
from collections import deque

from models.simple_cnn_detector import create_simple_cnn_model
from utils.simple_preprocessing import SimpleFaceDetector, SimpleImagePreprocessor, SimpleVideoPreprocessor

class RealTimeVideoDetector:
    """Real-time video deepfake detection system."""
    
    def __init__(self, 
                 model_name: str = "resnet18",
                 confidence_threshold: float = 0.5,
                 temporal_window: int = 5,
                 device: str = "auto"):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.temporal_window = temporal_window
        self.device = self._setup_device(device)
        
        # Initialize components
        self.model = None
        self.face_detector = SimpleFaceDetector(method="opencv")
        self.image_preprocessor = SimpleImagePreprocessor()
        self.video_preprocessor = SimpleVideoPreprocessor()
        
        # Temporal analysis
        self.prediction_history = deque(maxlen=temporal_window)
        self.confidence_history = deque(maxlen=temporal_window)
        
        # State management
        self.is_running = False
        self.current_frame = None
        self.detection_results = {}
        
        # Callbacks
        self.on_frame_processed = None
        self.on_detection_result = None
    
    def _setup_device(self, device: str):
        """Setup compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def load_model(self) -> bool:
        """Load the detection model."""
        try:
            self.model = create_simple_cnn_model(self.model_name, pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect_single_frame(self, frame: np.ndarray) -> Dict:
        """Detect deepfake in a single frame."""
        if self.model is None:
            return {
                'prediction': 'No model loaded',
                'confidence': 0.0,
                'is_fake': False,
                'error': 'Please load a model first'
            }
        
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            
            if not faces:
                return {
                    'prediction': 'No face detected',
                    'confidence': 0.0,
                    'is_fake': False,
                    'error': 'No face found in frame'
                }
            
            # Use the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
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
            
            result = {
                'prediction': 'Fake' if prediction == 1 else 'Real',
                'confidence': confidence,
                'is_fake': prediction == 1,
                'inference_time': inference_time,
                'face_region': (x, y, w, h),
                'threshold_exceeded': confidence >= self.confidence_threshold
            }
            
            # Update temporal history
            self.prediction_history.append(prediction)
            self.confidence_history.append(confidence)
            
            return result
            
        except Exception as e:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'is_fake': False,
                'error': str(e)
            }
    
    def get_temporal_analysis(self) -> Dict:
        """Get temporal analysis of recent predictions."""
        if len(self.prediction_history) < 2:
            return {'temporal_consistency': 1.0, 'avg_confidence': 0.0}
        
        # Calculate temporal consistency
        predictions = list(self.prediction_history)
        consistency = 1.0 - (len(set(predictions)) - 1) / len(predictions)
        
        # Calculate average confidence
        avg_confidence = np.mean(list(self.confidence_history))
        
        return {
            'temporal_consistency': consistency,
            'avg_confidence': avg_confidence,
            'prediction_trend': 'stable' if consistency > 0.8 else 'unstable'
        }
    
    def process_video_file(self, video_path: str, 
                          output_callback: Optional[Callable] = None) -> Dict:
        """Process a video file for deepfake detection."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': 'Could not open video file'}
        
        results = {
            'total_frames': 0,
            'processed_frames': 0,
            'fake_frames': 0,
            'real_frames': 0,
            'no_face_frames': 0,
            'error_frames': 0,
            'frame_results': [],
            'temporal_analysis': {}
        }
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            results['total_frames'] += 1
            
            # Process frame
            result = self.detect_single_frame(frame)
            result['frame_number'] = frame_count
            results['frame_results'].append(result)
            
            # Update statistics
            if 'error' in result:
                results['error_frames'] += 1
            elif result['prediction'] == 'No face detected':
                results['no_face_frames'] += 1
            elif result['is_fake']:
                results['fake_frames'] += 1
            else:
                results['real_frames'] += 1
            
            results['processed_frames'] += 1
            
            # Call output callback if provided
            if output_callback:
                output_callback(frame, result, frame_count)
        
        cap.release()
        
        # Final temporal analysis
        results['temporal_analysis'] = self.get_temporal_analysis()
        
        return results
    
    def start_webcam_detection(self, camera_index: int = 0,
                              output_callback: Optional[Callable] = None) -> bool:
        """Start real-time webcam detection."""
        if not self.load_model():
            return False
        
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                # Try different camera indices
                for i in range(5):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        camera_index = i
                        break
                else:
                    return False
            
            self.is_running = True
            self.camera_cap = cap
            
            def detection_loop():
                while self.is_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    self.current_frame = frame.copy()
                    
                    # Process frame
                    result = self.detect_single_frame(frame)
                    self.detection_results = result
                    
                    # Call output callback if provided
                    if output_callback:
                        output_callback(frame, result)
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.1)  # ~10 FPS for better stability
            
            # Start detection in separate thread
            self.detection_thread = threading.Thread(target=detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting webcam detection: {e}")
            return False
    
    def stop_webcam_detection(self):
        """Stop real-time webcam detection."""
        self.is_running = False
        if hasattr(self, 'detection_thread'):
            self.detection_thread.join()
        if hasattr(self, 'camera_cap'):
            self.camera_cap.release()
    
    def get_current_result(self) -> Dict:
        """Get the current detection result."""
        result = self.detection_results.copy()
        result['temporal_analysis'] = self.get_temporal_analysis()
        return result
    
    def draw_detection_overlay(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw detection results on frame."""
        overlay_frame = frame.copy()
        
        if 'face_region' in result and result['face_region']:
            x, y, w, h = result['face_region']
            
            # Choose color based on prediction
            if result['is_fake']:
                color = (0, 0, 255)  # Red for fake
                label = "FAKE"
            else:
                color = (0, 255, 0)  # Green for real
                label = "REAL"
            
            # Draw face rectangle
            cv2.rectangle(overlay_frame, (x, y), (x+w, y+h), color, 3)
            
            # Draw label
            label_text = f"{label} ({result['confidence']:.3f})"
            cv2.putText(overlay_frame, label_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw temporal analysis
        temporal = self.get_temporal_analysis()
        temporal_text = f"Consistency: {temporal['temporal_consistency']:.2f}"
        cv2.putText(overlay_frame, temporal_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return overlay_frame

def create_video_detector(model_name: str = "resnet18",
                         confidence_threshold: float = 0.5,
                         temporal_window: int = 5) -> RealTimeVideoDetector:
    """Factory function to create video detector."""
    return RealTimeVideoDetector(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        temporal_window=temporal_window
    )
