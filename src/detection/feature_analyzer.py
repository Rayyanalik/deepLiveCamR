"""
Feature-based Deepfake Detection

This module analyzes facial features and inconsistencies to detect deepfakes.
"""

import cv2
import numpy as np
import dlib
import face_recognition
from typing import List, Tuple, Dict, Optional
import mediapipe as mp
from scipy import stats
import matplotlib.pyplot as plt


class FeatureAnalyzer:
    """
    Analyzes facial features to detect deepfake characteristics.
    """
    
    def __init__(self):
        """Initialize the feature analyzer."""
        # Initialize face detection models
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        
        # MediaPipe for additional face analysis
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Feature extraction parameters
        self.eye_aspect_ratio_threshold = 0.25
        self.mouth_aspect_ratio_threshold = 0.5
        
    def extract_facial_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from an image.
        
        Args:
            image: Input image
            
        Returns:
            Array of facial landmarks or None if no face detected
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            return None
        
        # Use the first detected face
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Convert to numpy array
        landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        return landmarks_array
    
    def calculate_eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) to detect blinking inconsistencies.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Eye aspect ratio
        """
        # Left eye landmarks (indices 36-41)
        left_eye = landmarks[36:42]
        
        # Right eye landmarks (indices 42-47)
        right_eye = landmarks[42:48]
        
        # Calculate EAR for both eyes
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        
        # Average EAR
        ear = (left_ear + right_ear) / 2.0
        
        return ear
    
    def _calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """Calculate EAR for a single eye."""
        # Vertical eye landmarks
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal eye landmark
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def calculate_mouth_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) to detect mouth movement inconsistencies.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Mouth aspect ratio
        """
        # Mouth landmarks (indices 48-67)
        mouth = landmarks[48:68]
        
        # Vertical mouth landmarks
        A = np.linalg.norm(mouth[2] - mouth[10])
        B = np.linalg.norm(mouth[4] - mouth[8])
        
        # Horizontal mouth landmark
        C = np.linalg.norm(mouth[0] - mouth[6])
        
        # MAR formula
        mar = (A + B) / (2.0 * C)
        
        return mar
    
    def analyze_face_symmetry(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Analyze facial symmetry which can be inconsistent in deepfakes.
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Dictionary with symmetry metrics
        """
        symmetry_metrics = {}
        
        # Eye symmetry
        left_eye_center = np.mean(landmarks[36:42], axis=0)
        right_eye_center = np.mean(landmarks[42:48], axis=0)
        eye_symmetry = np.linalg.norm(left_eye_center - right_eye_center)
        symmetry_metrics['eye_symmetry'] = eye_symmetry
        
        # Eyebrow symmetry
        left_eyebrow = landmarks[17:22]
        right_eyebrow = landmarks[22:27]
        left_eyebrow_center = np.mean(left_eyebrow, axis=0)
        right_eyebrow_center = np.mean(right_eyebrow, axis=0)
        eyebrow_symmetry = np.linalg.norm(left_eyebrow_center - right_eyebrow_center)
        symmetry_metrics['eyebrow_symmetry'] = eyebrow_symmetry
        
        # Mouth symmetry
        mouth_left = landmarks[48:55]
        mouth_right = landmarks[55:62]
        mouth_left_center = np.mean(mouth_left, axis=0)
        mouth_right_center = np.mean(mouth_right, axis=0)
        mouth_symmetry = np.linalg.norm(mouth_left_center - mouth_right_center)
        symmetry_metrics['mouth_symmetry'] = mouth_symmetry
        
        return symmetry_metrics
    
    def detect_face_swapping_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """
        Detect artifacts specific to face swapping.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with artifact detection scores
        """
        artifacts = {}
        
        # Extract landmarks
        landmarks = self.extract_facial_landmarks(image)
        if landmarks is None:
            return {'no_face_detected': 1.0}
        
        # Analyze color inconsistencies around face boundary
        artifacts['color_inconsistency'] = self._analyze_color_inconsistency(image, landmarks)
        
        # Analyze edge artifacts
        artifacts['edge_artifacts'] = self._analyze_edge_artifacts(image, landmarks)
        
        # Analyze lighting inconsistencies
        artifacts['lighting_inconsistency'] = self._analyze_lighting_inconsistency(image, landmarks)
        
        return artifacts
    
    def _analyze_color_inconsistency(self, image: np.ndarray, landmarks: np.ndarray) -> float:
        """Analyze color inconsistencies around face boundary."""
        # Create face mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillPoly(mask, [hull], 255)
        
        # Dilate mask to get boundary region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        boundary_mask = dilated_mask - mask
        
        # Get boundary pixels
        boundary_pixels = image[boundary_mask > 0]
        
        if len(boundary_pixels) == 0:
            return 0.0
        
        # Calculate color variance (higher variance indicates inconsistency)
        color_variance = np.var(boundary_pixels, axis=0).mean()
        
        # Normalize to 0-1 scale
        inconsistency_score = min(color_variance / 1000.0, 1.0)
        
        return inconsistency_score
    
    def _analyze_edge_artifacts(self, image: np.ndarray, landmarks: np.ndarray) -> float:
        """Analyze edge artifacts around face region."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Create face mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillPoly(mask, [hull], 255)
        
        # Dilate mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Count edge pixels in face region
        face_edges = cv2.bitwise_and(edges, dilated_mask)
        edge_count = np.sum(face_edges > 0)
        
        # Normalize by face area
        face_area = np.sum(dilated_mask > 0)
        if face_area == 0:
            return 0.0
        
        edge_density = edge_count / face_area
        
        # Higher edge density indicates potential artifacts
        artifact_score = min(edge_density * 10, 1.0)
        
        return artifact_score
    
    def _analyze_lighting_inconsistency(self, image: np.ndarray, landmarks: np.ndarray) -> float:
        """Analyze lighting inconsistencies in face region."""
        # Convert to LAB color space for better lighting analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Create face mask
        mask = np.zeros(l_channel.shape, dtype=np.uint8)
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillPoly(mask, [hull], 255)
        
        # Get face region lighting values
        face_lighting = l_channel[mask > 0]
        
        if len(face_lighting) == 0:
            return 0.0
        
        # Calculate lighting variance
        lighting_variance = np.var(face_lighting)
        
        # Normalize to 0-1 scale
        inconsistency_score = min(lighting_variance / 1000.0, 1.0)
        
        return inconsistency_score
    
    def analyze_frame_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive feature analysis of a single frame.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with all feature analysis results
        """
        results = {}
        
        # Extract landmarks
        landmarks = self.extract_facial_landmarks(image)
        if landmarks is None:
            results['no_face_detected'] = 1.0
            return results
        
        # Calculate facial ratios
        results['eye_aspect_ratio'] = self.calculate_eye_aspect_ratio(landmarks)
        results['mouth_aspect_ratio'] = self.calculate_mouth_aspect_ratio(landmarks)
        
        # Analyze symmetry
        symmetry = self.analyze_face_symmetry(landmarks)
        results.update(symmetry)
        
        # Detect artifacts
        artifacts = self.detect_face_swapping_artifacts(image)
        results.update(artifacts)
        
        return results
    
    def calculate_deepfake_probability(self, feature_results: Dict[str, float]) -> float:
        """
        Calculate probability that the frame contains a deepfake.
        
        Args:
            feature_results: Results from feature analysis
            
        Returns:
            Probability score (0-1, higher = more likely deepfake)
        """
        if 'no_face_detected' in feature_results:
            return 0.0  # Can't analyze without face
        
        # Weighted combination of different indicators
        weights = {
            'color_inconsistency': 0.3,
            'edge_artifacts': 0.25,
            'lighting_inconsistency': 0.2,
            'eye_symmetry': 0.1,
            'eyebrow_symmetry': 0.1,
            'mouth_symmetry': 0.05
        }
        
        probability = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in feature_results:
                probability += feature_results[feature] * weight
                total_weight += weight
        
        if total_weight > 0:
            probability /= total_weight
        
        return min(probability, 1.0)
