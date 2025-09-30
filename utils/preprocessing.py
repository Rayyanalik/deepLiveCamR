"""
Image and video preprocessing utilities for deepfake detection.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, List, Optional, Union
# Try to import mediapipe; fall back gracefully if unavailable (e.g., Python 3.13 wheels)
try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None
from face_recognition import face_locations, face_encodings
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FaceDetector:
    """Face detection utility using multiple methods."""
    
    def __init__(self, method: str = "mediapipe"):
        self.method = method
        if method == "mediapipe":
            if mp is None:
                # Fallback to OpenCV if mediapipe isn't available
                self.method = "opencv"
            
        if self.method == "mediapipe":
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
        elif self.method == "opencv":
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image and return bounding boxes."""
        if self.method == "mediapipe":
            return self._detect_mediapipe(image)
        elif self.method == "opencv":
            return self._detect_opencv(image)
        elif self.method == "face_recognition":
            return self._detect_face_recognition(image)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def _detect_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                faces.append((x, y, width, height))
        
        return faces
    
    def _detect_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(x, y, w, h) for x, y, w, h in faces]
    
    def _detect_face_recognition(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using face_recognition library."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locs = face_locations(rgb_image)
        return [(top, right, bottom, left) for top, right, bottom, left in face_locs]

class ImagePreprocessor:
    """Image preprocessing for deepfake detection models."""
    
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
        
        # Augmentation transforms
        self.augmentation_transform = A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
            A.GaussNoise(p=0.2),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image], 
                        augment: bool = False) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if augment:
            image_np = np.array(image)
            augmented = self.augmentation_transform(image=image_np)
            return augmented['image']
        else:
            return self.transform(image)
    
    def preprocess_batch(self, images: List[np.ndarray], 
                        augment: bool = False) -> torch.Tensor:
        """Preprocess batch of images."""
        processed_images = []
        for image in images:
            processed_images.append(self.preprocess_image(image, augment))
        return torch.stack(processed_images)

class VideoPreprocessor:
    """Video preprocessing utilities."""
    
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
    
    def preprocess_video(self, video_path: str, 
                        face_detector: FaceDetector,
                        image_preprocessor: ImagePreprocessor) -> List[torch.Tensor]:
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

class DataAugmentation:
    """Data augmentation for training data."""
    
    def __init__(self):
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)
        ])
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image."""
        augmented = self.augmentation_pipeline(image=image)
        return augmented['image']
    
    def augment_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Apply augmentation to batch of images."""
        return [self.augment_image(img) for img in images]

def create_preprocessing_pipeline(input_size: Tuple[int, int] = (224, 224),
                                normalization: str = "imagenet",
                                face_detection_method: str = "mediapipe") -> dict:
    """Create complete preprocessing pipeline."""
    return {
        'face_detector': FaceDetector(method=face_detection_method),
        'image_preprocessor': ImagePreprocessor(input_size, normalization),
        'video_preprocessor': VideoPreprocessor(),
        'augmentation': DataAugmentation()
    }
