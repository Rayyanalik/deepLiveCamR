"""
Face Swap Generation Module

This module implements face swapping capabilities using various AI models
for creating realistic deepfake videos.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import mediapipe as mp
from PIL import Image
import face_recognition
import dlib


class FaceSwapGenerator:
    """
    Advanced face swap generator using multiple techniques for realistic results.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the face swap generator.
        
        Args:
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize face detection models
        self.face_detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
        
        # Face recognition for identity matching
        self.known_faces = {}
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image using multiple methods for robustness.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        faces = []
        
        # Method 1: dlib face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dlib_faces = self.face_detector(gray)
        
        for face in dlib_faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            faces.append((x, y, w, h))
        
        # Method 2: OpenCV Haar Cascades (backup)
        if not faces:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in cv_faces:
                faces.append((x, y, w, h))
        
        return faces
    
    def extract_face_landmarks(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract facial landmarks for precise face alignment.
        
        Args:
            image: Input image
            face_box: Face bounding box (x, y, w, h)
            
        Returns:
            Array of facial landmarks
        """
        x, y, w, h = face_box
        face_roi = image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect landmarks using dlib
        shape = self.predictor(gray_roi, dlib.rectangle(0, 0, w, h))
        landmarks = np.array([[p.x + x, p.y + y] for p in shape.parts()])
        
        return landmarks
    
    def align_face(self, image: np.ndarray, landmarks: np.ndarray, 
                   target_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align face to a standard pose for better swapping results.
        
        Args:
            image: Input image
            landmarks: Facial landmarks
            target_size: Target face size
            
        Returns:
            Aligned face image and transformation matrix
        """
        # Define reference landmarks (frontal face)
        ref_landmarks = np.array([
            [30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
            [33.5493, 92.3655], [62.7299, 92.2041]
        ]) * target_size[0] / 112
        
        # Get key landmarks (eyes, nose, mouth corners)
        key_landmarks = landmarks[[36, 45, 30, 48, 54]]  # Left eye, right eye, nose, left mouth, right mouth
        
        # Calculate transformation matrix
        transform_matrix = cv2.getAffineTransform(
            key_landmarks[:3].astype(np.float32),
            ref_landmarks[:3].astype(np.float32)
        )
        
        # Apply transformation
        aligned_face = cv2.warpAffine(image, transform_matrix, target_size)
        
        return aligned_face, transform_matrix
    
    def blend_faces(self, source_face: np.ndarray, target_face: np.ndarray, 
                   mask: np.ndarray) -> np.ndarray:
        """
        Blend source face onto target face using a mask for seamless integration.
        
        Args:
            source_face: Source face image
            target_face: Target face image
            mask: Blending mask
            
        Returns:
            Blended face image
        """
        # Ensure mask is 3-channel
        if len(mask.shape) == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Normalize mask
        mask = mask.astype(np.float32) / 255.0
        
        # Apply Gaussian blur to mask for smoother blending
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Blend faces
        blended = source_face * mask + target_face * (1 - mask)
        
        return blended.astype(np.uint8)
    
    def create_face_mask(self, landmarks: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a face mask based on facial landmarks.
        
        Args:
            landmarks: Facial landmarks
            image_shape: Shape of the target image
            
        Returns:
            Face mask as binary image
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # Create convex hull from landmarks
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillPoly(mask, [hull], 255)
        
        # Apply morphological operations for smoother mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def swap_faces(self, source_image: np.ndarray, target_image: np.ndarray,
                   source_face_idx: int = 0, target_face_idx: int = 0) -> np.ndarray:
        """
        Perform face swap between source and target images.
        
        Args:
            source_image: Image containing the face to swap
            target_image: Image where the face will be placed
            source_face_idx: Index of face to use from source image
            target_face_idx: Index of face to replace in target image
            
        Returns:
            Image with swapped face
        """
        # Detect faces in both images
        source_faces = self.detect_faces(source_image)
        target_faces = self.detect_faces(target_image)
        
        if not source_faces or not target_faces:
            raise ValueError("No faces detected in one or both images")
        
        if source_face_idx >= len(source_faces) or target_face_idx >= len(target_faces):
            raise ValueError("Face index out of range")
        
        # Get face regions
        source_face_box = source_faces[source_face_idx]
        target_face_box = target_faces[target_face_idx]
        
        # Extract landmarks
        source_landmarks = self.extract_face_landmarks(source_image, source_face_box)
        target_landmarks = self.extract_face_landmarks(target_image, target_face_box)
        
        # Align faces
        source_aligned, _ = self.align_face(source_image, source_landmarks)
        target_aligned, transform_matrix = self.align_face(target_image, target_landmarks)
        
        # Create face mask
        mask = self.create_face_mask(target_landmarks, target_image.shape)
        
        # Resize source face to match target face
        x, y, w, h = target_face_box
        source_resized = cv2.resize(source_aligned, (w, h))
        
        # Create inverse transformation to map back to original image
        inv_transform = cv2.invertAffineTransform(transform_matrix)
        
        # Transform source face back to original coordinates
        source_transformed = cv2.warpAffine(source_resized, inv_transform, 
                                          (target_image.shape[1], target_image.shape[0]))
        
        # Blend faces
        result = self.blend_faces(source_transformed, target_image, mask)
        
        return result
    
    def generate_mundane_task_video(self, source_face_path: str, 
                                   target_video_path: str, 
                                   output_path: str) -> str:
        """
        Generate a video of a person performing mundane tasks with swapped face.
        
        Args:
            source_face_path: Path to source face image
            target_video_path: Path to target video
            output_path: Path to save output video
            
        Returns:
            Path to generated video
        """
        # Load source face
        source_image = cv2.imread(source_face_path)
        if source_image is None:
            raise ValueError(f"Could not load source image: {source_face_path}")
        
        # Open target video
        cap = cv2.VideoCapture(target_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open target video: {target_video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Perform face swap
                swapped_frame = self.swap_faces(source_image, frame)
                out.write(swapped_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Progress update every 30 frames
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                out.write(frame)  # Write original frame if swap fails
            
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Video generation complete: {output_path}")
        return output_path
