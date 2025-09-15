"""
Comprehensive Deepfake Detection System

This module combines multiple detection methods for robust deepfake identification.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import time
from .feature_analyzer import FeatureAnalyzer
import mediapipe as mp
from scipy import stats
import matplotlib.pyplot as plt


class DeepfakeDetector:
    """
    Comprehensive deepfake detector combining multiple analysis methods.
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the deepfake detector.
        
        Args:
            device: Device to run inference on
        """
        self.device = self._get_device(device)
        
        # Initialize analyzers
        self.feature_analyzer = FeatureAnalyzer()
        
        # Detection thresholds
        self.thresholds = {
            'feature_analysis': 0.5,
            'temporal_consistency': 0.4,
            'blink_detection': 0.3,
            'overall': 0.6
        }
        
        # Performance tracking
        self.detection_history = []
        self.frame_count = 0
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def analyze_video(self, video_path: str, sample_rate: int = 30) -> Dict[str, any]:
        """
        Analyze a video file for deepfake characteristics.
        
        Args:
            video_path: Path to video file
            sample_rate: Frame sampling rate (analyze every Nth frame)
            
        Returns:
            Dictionary with analysis results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Analyzing video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        
        # Analysis results
        frame_results = []
        temporal_features = []
        blink_patterns = []
        
        frame_idx = 0
        analyzed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on sample_rate
            if frame_idx % sample_rate == 0:
                # Analyze frame
                frame_analysis = self.feature_analyzer.analyze_frame_features(frame)
                frame_results.append(frame_analysis)
                
                # Extract temporal features
                temporal_feature = self._extract_temporal_features(frame)
                if temporal_feature is not None:
                    temporal_features.append(temporal_feature)
                
                # Analyze blink patterns
                blink_feature = self._analyze_blink_pattern(frame)
                if blink_feature is not None:
                    blink_patterns.append(blink_feature)
                
                analyzed_frames += 1
                
                if analyzed_frames % 10 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({analyzed_frames} frames analyzed)")
            
            frame_idx += 1
        
        cap.release()
        
        # Compile results
        results = self._compile_analysis_results(frame_results, temporal_features, blink_patterns)
        results['video_info'] = {
            'total_frames': total_frames,
            'analyzed_frames': analyzed_frames,
            'fps': fps,
            'duration': duration,
            'sample_rate': sample_rate
        }
        
        return results
    
    def _extract_temporal_features(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract features that can be analyzed temporally.
        
        Args:
            frame: Input frame
            
        Returns:
            Feature vector or None if no face detected
        """
        landmarks = self.feature_analyzer.extract_facial_landmarks(frame)
        if landmarks is None:
            return None
        
        # Extract key facial measurements
        features = []
        
        # Eye measurements
        left_eye_ear = self.feature_analyzer._calculate_ear(landmarks[36:42])
        right_eye_ear = self.feature_analyzer._calculate_ear(landmarks[42:48])
        features.extend([left_eye_ear, right_eye_ear])
        
        # Mouth measurements
        mar = self.feature_analyzer.calculate_mouth_aspect_ratio(landmarks)
        features.append(mar)
        
        # Facial symmetry
        symmetry = self.feature_analyzer.analyze_face_symmetry(landmarks)
        features.extend([symmetry['eye_symmetry'], symmetry['eyebrow_symmetry'], symmetry['mouth_symmetry']])
        
        return np.array(features)
    
    def _analyze_blink_pattern(self, frame: np.ndarray) -> Optional[float]:
        """
        Analyze blink pattern for inconsistencies.
        
        Args:
            frame: Input frame
            
        Returns:
            Blink analysis score or None if no face detected
        """
        landmarks = self.feature_analyzer.extract_facial_landmarks(frame)
        if landmarks is None:
            return None
        
        # Calculate eye aspect ratio
        ear = self.feature_analyzer.calculate_eye_aspect_ratio(landmarks)
        
        # Determine if eyes are closed (EAR below threshold)
        is_blinking = ear < self.feature_analyzer.eye_aspect_ratio_threshold
        
        return float(is_blinking)
    
    def _compile_analysis_results(self, frame_results: List[Dict], 
                                 temporal_features: List[np.ndarray],
                                 blink_patterns: List[float]) -> Dict[str, any]:
        """
        Compile analysis results into final detection scores.
        
        Args:
            frame_results: Results from frame-by-frame analysis
            temporal_features: Temporal feature vectors
            blink_patterns: Blink pattern analysis
            
        Returns:
            Compiled analysis results
        """
        results = {}
        
        # Frame-level analysis
        if frame_results:
            frame_scores = []
            for frame_result in frame_results:
                if 'no_face_detected' not in frame_result:
                    score = self.feature_analyzer.calculate_deepfake_probability(frame_result)
                    frame_scores.append(score)
            
            if frame_scores:
                results['frame_analysis'] = {
                    'mean_score': np.mean(frame_scores),
                    'std_score': np.std(frame_scores),
                    'max_score': np.max(frame_scores),
                    'min_score': np.min(frame_scores),
                    'scores': frame_scores
                }
        
        # Temporal consistency analysis
        if len(temporal_features) > 1:
            temporal_consistency = self._analyze_temporal_consistency(temporal_features)
            results['temporal_consistency'] = temporal_consistency
        
        # Blink pattern analysis
        if len(blink_patterns) > 1:
            blink_analysis = self._analyze_blink_patterns(blink_patterns)
            results['blink_analysis'] = blink_analysis
        
        # Overall detection score
        overall_score = self._calculate_overall_score(results)
        results['overall_score'] = overall_score
        results['is_deepfake'] = overall_score > self.thresholds['overall']
        
        return results
    
    def _analyze_temporal_consistency(self, temporal_features: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze temporal consistency of facial features.
        
        Args:
            temporal_features: List of temporal feature vectors
            
        Returns:
            Temporal consistency analysis results
        """
        if len(temporal_features) < 2:
            return {'consistency_score': 0.0}
        
        # Convert to numpy array
        features_array = np.array(temporal_features)
        
        # Calculate variance across time for each feature
        feature_variances = np.var(features_array, axis=0)
        
        # Higher variance indicates less temporal consistency (potential deepfake)
        consistency_score = np.mean(feature_variances)
        
        # Normalize to 0-1 scale
        consistency_score = min(consistency_score * 10, 1.0)
        
        return {
            'consistency_score': consistency_score,
            'feature_variances': feature_variances.tolist(),
            'mean_variance': np.mean(feature_variances)
        }
    
    def _analyze_blink_patterns(self, blink_patterns: List[float]) -> Dict[str, float]:
        """
        Analyze blink patterns for naturalness.
        
        Args:
            blink_patterns: List of blink indicators
            
        Returns:
            Blink pattern analysis results
        """
        if len(blink_patterns) < 2:
            return {'blink_score': 0.0}
        
        # Calculate blink frequency
        total_blinks = sum(blink_patterns)
        blink_frequency = total_blinks / len(blink_patterns)
        
        # Normal blink frequency is around 0.1-0.2 (10-20% of frames)
        # Too high or too low frequency indicates potential deepfake
        normal_frequency_range = (0.05, 0.25)
        
        if normal_frequency_range[0] <= blink_frequency <= normal_frequency_range[1]:
            blink_score = 0.0  # Normal blink pattern
        else:
            # Calculate deviation from normal range
            if blink_frequency < normal_frequency_range[0]:
                deviation = normal_frequency_range[0] - blink_frequency
            else:
                deviation = blink_frequency - normal_frequency_range[1]
            
            blink_score = min(deviation * 5, 1.0)  # Scale to 0-1
        
        return {
            'blink_score': blink_score,
            'blink_frequency': blink_frequency,
            'total_blinks': total_blinks,
            'total_frames': len(blink_patterns)
        }
    
    def _calculate_overall_score(self, results: Dict[str, any]) -> float:
        """
        Calculate overall deepfake probability score.
        
        Args:
            results: Analysis results
            
        Returns:
            Overall probability score (0-1)
        """
        scores = []
        weights = []
        
        # Frame analysis score
        if 'frame_analysis' in results:
            scores.append(results['frame_analysis']['mean_score'])
            weights.append(0.4)
        
        # Temporal consistency score
        if 'temporal_consistency' in results:
            scores.append(results['temporal_consistency']['consistency_score'])
            weights.append(0.3)
        
        # Blink pattern score
        if 'blink_analysis' in results:
            scores.append(results['blink_analysis']['blink_score'])
            weights.append(0.3)
        
        if not scores:
            return 0.0
        
        # Weighted average
        overall_score = np.average(scores, weights=weights)
        
        return min(overall_score, 1.0)
    
    def detect_in_real_time(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Detect deepfake characteristics in a single frame for real-time analysis.
        
        Args:
            frame: Input frame
            
        Returns:
            Detection results for the frame
        """
        # Analyze frame features
        frame_analysis = self.feature_analyzer.analyze_frame_features(frame)
        
        # Calculate deepfake probability
        probability = self.feature_analyzer.calculate_deepfake_probability(frame_analysis)
        
        # Store in history for temporal analysis
        self.detection_history.append(probability)
        self.frame_count += 1
        
        # Keep only recent history (last 30 frames)
        if len(self.detection_history) > 30:
            self.detection_history.pop(0)
        
        # Calculate temporal consistency if we have enough history
        temporal_score = 0.0
        if len(self.detection_history) > 5:
            temporal_score = np.std(self.detection_history)
        
        # Overall score
        overall_score = (probability + temporal_score) / 2.0
        
        return {
            'frame_probability': probability,
            'temporal_score': temporal_score,
            'overall_score': overall_score,
            'is_deepfake': overall_score > self.thresholds['overall'],
            'frame_count': self.frame_count
        }
    
    def generate_detection_report(self, results: Dict[str, any]) -> str:
        """
        Generate a human-readable detection report.
        
        Args:
            results: Analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 50)
        report.append("DEEPFAKE DETECTION REPORT")
        report.append("=" * 50)
        
        # Video information
        if 'video_info' in results:
            video_info = results['video_info']
            report.append(f"Video: {video_info['total_frames']} frames, {video_info['fps']:.1f} FPS")
            report.append(f"Duration: {video_info['duration']:.1f} seconds")
            report.append(f"Analyzed: {video_info['analyzed_frames']} frames")
            report.append("")
        
        # Overall result
        overall_score = results.get('overall_score', 0.0)
        is_deepfake = results.get('is_deepfake', False)
        
        report.append(f"OVERALL RESULT: {'DEEPFAKE DETECTED' if is_deepfake else 'AUTHENTIC'}")
        report.append(f"Confidence Score: {overall_score:.3f}")
        report.append("")
        
        # Detailed analysis
        if 'frame_analysis' in results:
            frame_analysis = results['frame_analysis']
            report.append("Frame Analysis:")
            report.append(f"  Mean Score: {frame_analysis['mean_score']:.3f}")
            report.append(f"  Std Deviation: {frame_analysis['std_score']:.3f}")
            report.append(f"  Max Score: {frame_analysis['max_score']:.3f}")
            report.append("")
        
        if 'temporal_consistency' in results:
            temporal = results['temporal_consistency']
            report.append("Temporal Consistency:")
            report.append(f"  Consistency Score: {temporal['consistency_score']:.3f}")
            report.append(f"  Mean Variance: {temporal['mean_variance']:.3f}")
            report.append("")
        
        if 'blink_analysis' in results:
            blink = results['blink_analysis']
            report.append("Blink Pattern Analysis:")
            report.append(f"  Blink Score: {blink['blink_score']:.3f}")
            report.append(f"  Blink Frequency: {blink['blink_frequency']:.3f}")
            report.append(f"  Total Blinks: {blink['total_blinks']}/{blink['total_frames']}")
            report.append("")
        
        report.append("=" * 50)
        
        return "\n".join(report)
