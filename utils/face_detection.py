"""
Face detection utilities for deepfake detection.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Union
import dlib
from face_recognition import face_locations, face_encodings, face_landmarks
import torch
import torchvision.transforms as transforms
from PIL import Image

class FaceDetector:
    """Comprehensive face detection utility supporting multiple methods."""
    
    def __init__(self, method: str = "mediapipe", confidence_threshold: float = 0.5):
        self.method = method
        self.confidence_threshold = confidence_threshold
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the selected face detection method."""
        if self.method == "mediapipe":
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=self.confidence_threshold
            )
        elif self.method == "opencv":
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        elif self.method == "dlib":
            # Initialize dlib's face detector
            self.dlib_detector = dlib.get_frontal_face_detector()
        elif self.method == "face_recognition":
            # No initialization needed for face_recognition
            pass
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image and return bounding boxes."""
        if self.method == "mediapipe":
            return self._detect_mediapipe(image)
        elif self.method == "opencv":
            return self._detect_opencv(image)
        elif self.method == "dlib":
            return self._detect_dlib(image)
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
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                faces.append((x, y, width, height))
        
        return faces
    
    def _detect_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar cascades."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return [(x, y, w, h) for x, y, w, h in faces]
    
    def _detect_dlib(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using dlib."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.dlib_detector(gray)
        
        faces = []
        for detection in detections:
            x = detection.left()
            y = detection.top()
            w = detection.width()
            h = detection.height()
            faces.append((x, y, w, h))
        
        return faces
    
    def _detect_face_recognition(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using face_recognition library."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locs = face_locations(rgb_image)
        return [(top, right, bottom, left) for top, right, bottom, left in face_locs]
    
    def get_largest_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get the largest face in the image."""
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Find face with largest area
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        return largest_face
    
    def extract_face_region(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract face region from image."""
        x, y, w, h = face_box
        return image[y:y+h, x:x+w]
    
    def draw_face_boxes(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]], 
                       color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Draw bounding boxes around detected faces."""
        result_image = image.copy()
        for x, y, w, h in faces:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), color, thickness)
        return result_image

class FaceLandmarkDetector:
    """Detect facial landmarks for more detailed analysis."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Detect facial landmarks."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = []
        h, w, _ = image.shape
        
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        return landmarks
    
    def draw_landmarks(self, image: np.ndarray, landmarks: List[Tuple[int, int]], 
                      color: Tuple[int, int, int] = (0, 0, 255), radius: int = 2) -> np.ndarray:
        """Draw facial landmarks on image."""
        result_image = image.copy()
        for x, y in landmarks:
            cv2.circle(result_image, (x, y), radius, color, -1)
        return result_image

class FaceQualityAssessor:
    """Assess the quality of detected faces."""
    
    def __init__(self):
        self.face_detector = FaceDetector()
    
    def assess_face_quality(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Assess various quality metrics of a face."""
        x, y, w, h = face_box
        face_region = image[y:y+h, x:x+w]
        
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate quality metrics
        metrics = {}
        
        # Size quality
        metrics['size_score'] = min(1.0, (w * h) / (100 * 100))  # Normalize to 100x100
        
        # Brightness quality
        brightness = np.mean(gray_face)
        metrics['brightness_score'] = 1.0 - abs(brightness - 127) / 127
        
        # Contrast quality
        contrast = np.std(gray_face)
        metrics['contrast_score'] = min(1.0, contrast / 50)  # Normalize to 50
        
        # Sharpness quality (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        metrics['sharpness_score'] = min(1.0, laplacian_var / 1000)  # Normalize to 1000
        
        # Overall quality score
        metrics['overall_score'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def is_face_suitable_for_detection(self, image: np.ndarray, face_box: Tuple[int, int, int, int], 
                                     min_score: float = 0.5) -> bool:
        """Check if face is suitable for deepfake detection."""
        quality_metrics = self.assess_face_quality(image, face_box)
        return quality_metrics['overall_score'] >= min_score

class FacePreprocessor:
    """Preprocess faces for deepfake detection."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess face image for model input."""
        # Convert BGR to RGB
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_face = Image.fromarray(rgb_face)
        
        # Apply transforms
        tensor_face = self.transform(pil_face)
        
        return tensor_face
    
    def preprocess_faces_batch(self, face_images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess multiple face images."""
        processed_faces = []
        for face_image in face_images:
            processed_face = self.preprocess_face(face_image)
            processed_faces.append(processed_face)
        
        return torch.stack(processed_faces)

class FaceTracker:
    """Track faces across video frames."""
    
    def __init__(self, max_disappeared: int = 30):
        self.max_disappeared = max_disappeared
        self.face_detector = FaceDetector()
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
    
    def update(self, image: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
        """Update face tracking with new frame."""
        # Detect faces in current frame
        face_boxes = self.face_detector.detect_faces(image)
        
        # If no faces detected, mark all as disappeared
        if len(face_boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return self.objects
        
        # If no existing objects, register all faces
        if len(self.objects) == 0:
            for face_box in face_boxes:
                self._register(face_box)
        else:
            # Match existing objects with new detections
            object_centroids = np.array([self._get_centroid(box) for box in self.objects.values()])
            face_centroids = np.array([self._get_centroid(box) for box in face_boxes])
            
            # Compute distance matrix
            D = self._compute_distance_matrix(object_centroids, face_centroids)
            
            # Find minimum values and sort
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            # Track used row and column indices
            used_row_indices = set()
            used_col_indices = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                
                object_id = list(self.objects.keys())[row]
                self.objects[object_id] = face_boxes[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_rows = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_cols = set(range(0, D.shape[1])).difference(used_col_indices)
            
            # If more objects than detections, mark as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = list(self.objects.keys())[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self._deregister(object_id)
            else:
                # Register new objects
                for col in unused_cols:
                    self._register(face_boxes[col])
        
        return self.objects
    
    def _register(self, face_box: Tuple[int, int, int, int]):
        """Register a new face object."""
        self.objects[self.next_object_id] = face_box
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def _deregister(self, object_id: int):
        """Deregister a face object."""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def _get_centroid(self, face_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get centroid of face bounding box."""
        x, y, w, h = face_box
        return (x + w // 2, y + h // 2)
    
    def _compute_distance_matrix(self, centroids1: np.ndarray, centroids2: np.ndarray) -> np.ndarray:
        """Compute distance matrix between two sets of centroids."""
        D = np.linalg.norm(centroids1[:, np.newaxis] - centroids2, axis=2)
        return D

def create_face_detection_pipeline(method: str = "mediapipe", 
                                 confidence_threshold: float = 0.5,
                                 target_size: Tuple[int, int] = (224, 224)) -> Dict[str, object]:
    """Create a complete face detection pipeline."""
    return {
        'detector': FaceDetector(method, confidence_threshold),
        'landmark_detector': FaceLandmarkDetector(),
        'quality_assessor': FaceQualityAssessor(),
        'preprocessor': FacePreprocessor(target_size),
        'tracker': FaceTracker()
    }

if __name__ == "__main__":
    # Example usage
    import cv2
    
    # Load test image
    image = cv2.imread("test_image.jpg")
    
    # Create face detection pipeline
    pipeline = create_face_detection_pipeline()
    
    # Detect faces
    faces = pipeline['detector'].detect_faces(image)
    print(f"Detected {len(faces)} faces")
    
    # Draw face boxes
    result_image = pipeline['detector'].draw_face_boxes(image, faces)
    
    # Display result
    cv2.imshow("Face Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
