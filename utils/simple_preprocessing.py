"""
Simplified image and video preprocessing utilities for deepfake detection.
Works with basic dependencies only.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, List, Optional, Union

class SimpleFaceDetector:
    """Simplified face detection utility using OpenCV."""
    
    def __init__(self, method: str = "opencv"):
        self.method = method
        if method == "opencv":
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image and return bounding boxes."""
        if self.method == "opencv":
            return self._detect_opencv(image)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _detect_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(x, y, w, h) for x, y, w, h in faces]
    
    def get_largest_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get the largest face in the image."""
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Find face with largest area
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        return largest_face

class SimpleImagePreprocessor:
    """Simplified image preprocessing for deepfake detection models."""
    
    def __init__(self, input_size: Tuple[int, int] = (224, 224), 
                 normalization: str = "imagenet"):
        self.input_size = input_size
        self.normalization = normalization
        
        # Define normalization values
        if normalization == "imagenet":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        elif normalization == "custom":
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            self.mean = [0.0, 0.0, 0.0]
            self.std = [1.0, 1.0, 1.0]
        
        # Create transforms
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image], 
                        augment: bool = False) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        return self.transform(image)
    
    def preprocess_batch(self, images: List[np.ndarray], 
                        augment: bool = False) -> torch.Tensor:
        """Preprocess batch of images."""
        processed_images = []
        for image in images:
            processed_images.append(self.preprocess_image(image, augment))
        return torch.stack(processed_images)

class SimpleVideoPreprocessor:
    """Simplified video preprocessing utilities."""
    
    def __init__(self, target_fps: int = 30, max_frames: int = 100):
        self.target_fps = target_fps
        self.max_frames = max_frames
    
    def extract_frames(self, video_path: str, 
                      frame_indices: Optional[List[int]] = None) -> List[np.ndarray]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if frame_indices is None:
            # Extract frames at target FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / self.target_fps))
            
            frame_count = 0
            while len(frames) < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                frame_count += 1
        else:
            # Extract specific frames
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
        
        cap.release()
        return frames
    
    def extract_frames_from_camera(self, cap: cv2.VideoCapture, 
                                  num_frames: int = 10) -> List[np.ndarray]:
        """Extract frames from camera feed."""
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        return frames
    
    def preprocess_video(self, video_path: str, 
                        face_detector: SimpleFaceDetector,
                        image_preprocessor: SimpleImagePreprocessor) -> List[torch.Tensor]:
        """Preprocess video for detection."""
        frames = self.extract_frames(video_path)
        processed_frames = []
        
        for frame in frames:
            # Detect faces
            faces = face_detector.detect_faces(frame)
            
            if faces:
                # Use the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                
                # Preprocess face
                processed_face = image_preprocessor.preprocess_image(face_region)
                processed_frames.append(processed_face)
        
        return processed_frames
    
    def preprocess_camera_frames(self, frames: List[np.ndarray],
                               face_detector: SimpleFaceDetector,
                               image_preprocessor: SimpleImagePreprocessor) -> List[torch.Tensor]:
        """Preprocess frames from camera for real-time detection."""
        processed_frames = []
        
        for frame in frames:
            # Detect faces
            faces = face_detector.detect_faces(frame)
            
            if faces:
                # Use the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                
                # Preprocess face
                processed_face = image_preprocessor.preprocess_image(face_region)
                processed_frames.append(processed_face)
        
        return processed_frames

def create_simple_preprocessing_pipeline(input_size: Tuple[int, int] = (224, 224),
                                       normalization: str = "imagenet",
                                       face_detection_method: str = "opencv") -> dict:
    """Create simplified preprocessing pipeline."""
    return {
        'face_detector': SimpleFaceDetector(method=face_detection_method),
        'image_preprocessor': SimpleImagePreprocessor(input_size, normalization),
        'video_preprocessor': SimpleVideoPreprocessor()
    }
