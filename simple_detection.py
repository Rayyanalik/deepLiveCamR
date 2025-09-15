"""
Simple Deepfake Detection Script
Works with basic dependencies for analyzing videos you create with online tools
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path


class SimpleDeepfakeDetector:
    """
    Simple deepfake detector using basic computer vision techniques.
    """
    
    def __init__(self):
        """Initialize the simple detector."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces(self, image):
        """Detect faces in an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def analyze_frame(self, frame):
        """Analyze a single frame for deepfake characteristics."""
        # Detect faces
        faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            return {'no_face_detected': True, 'score': 0.0}
        
        # Analyze face region
        face_scores = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Analyze color consistency
            color_score = self._analyze_color_consistency(face_roi)
            
            # Analyze edge artifacts
            edge_score = self._analyze_edge_artifacts(face_roi)
            
            # Analyze lighting consistency
            lighting_score = self._analyze_lighting_consistency(face_roi)
            
            # Combine scores
            face_score = (color_score + edge_score + lighting_score) / 3.0
            face_scores.append(face_score)
        
        # Return average score
        avg_score = np.mean(face_scores) if face_scores else 0.0
        
        return {
            'no_face_detected': False,
            'face_count': len(faces),
            'score': avg_score,
            'individual_scores': face_scores
        }
    
    def _analyze_color_consistency(self, face_roi):
        """Analyze color consistency in face region."""
        # Convert to LAB color space
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        
        # Calculate color variance
        color_variance = np.var(lab, axis=(0, 1))
        
        # Higher variance indicates potential artifacts
        inconsistency_score = np.mean(color_variance) / 1000.0
        
        return min(inconsistency_score, 1.0)
    
    def _analyze_edge_artifacts(self, face_roi):
        """Analyze edge artifacts in face region."""
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels
        edge_count = np.sum(edges > 0)
        
        # Normalize by face area
        face_area = face_roi.shape[0] * face_roi.shape[1]
        edge_density = edge_count / face_area
        
        # Higher edge density indicates potential artifacts
        artifact_score = min(edge_density * 10, 1.0)
        
        return artifact_score
    
    def _analyze_lighting_consistency(self, face_roi):
        """Analyze lighting consistency in face region."""
        # Convert to LAB color space
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate lighting variance
        lighting_variance = np.var(l_channel)
        
        # Higher variance indicates inconsistent lighting
        inconsistency_score = min(lighting_variance / 1000.0, 1.0)
        
        return inconsistency_score
    
    def analyze_video(self, video_path, sample_rate=30):
        """Analyze a video file for deepfake characteristics."""
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return None
        
        print(f"🔍 Analyzing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        
        # Analyze frames
        frame_scores = []
        frame_count = 0
        analyzed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames based on sample_rate
            if frame_count % sample_rate == 0:
                analysis = self.analyze_frame(frame)
                
                if not analysis['no_face_detected']:
                    frame_scores.append(analysis['score'])
                    analyzed_frames += 1
                    
                    if analyzed_frames % 10 == 0:
                        progress = (frame_count / total_frames) * 100
                        print(f"Progress: {progress:.1f}% ({analyzed_frames} frames analyzed)")
            
            frame_count += 1
        
        cap.release()
        
        if not frame_scores:
            print("❌ No faces detected in video")
            return None
        
        # Calculate overall results
        mean_score = np.mean(frame_scores)
        std_score = np.std(frame_scores)
        max_score = np.max(frame_scores)
        min_score = np.min(frame_scores)
        
        # Determine if it's likely a deepfake
        is_deepfake = mean_score > 0.3  # Threshold for detection
        
        results = {
            'video_path': video_path,
            'analyzed_frames': analyzed_frames,
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'mean_score': mean_score,
            'std_score': std_score,
            'max_score': max_score,
            'min_score': min_score,
            'is_deepfake': is_deepfake,
            'frame_scores': frame_scores
        }
        
        return results
    
    def generate_report(self, results):
        """Generate a human-readable report."""
        if not results:
            return "No analysis results available."
        
        report = []
        report.append("=" * 50)
        report.append("DEEPFAKE DETECTION REPORT")
        report.append("=" * 50)
        
        # Video information
        report.append(f"Video: {results['video_path']}")
        report.append(f"Duration: {results['duration']:.1f} seconds")
        report.append(f"FPS: {results['fps']:.1f}")
        report.append(f"Analyzed: {results['analyzed_frames']}/{results['total_frames']} frames")
        report.append("")
        
        # Detection results
        report.append(f"OVERALL RESULT: {'DEEPFAKE DETECTED' if results['is_deepfake'] else 'AUTHENTIC'}")
        report.append(f"Detection Score: {results['mean_score']:.3f}")
        report.append("")
        
        # Detailed analysis
        report.append("Detailed Analysis:")
        report.append(f"  Mean Score: {results['mean_score']:.3f}")
        report.append(f"  Std Deviation: {results['std_score']:.3f}")
        report.append(f"  Max Score: {results['max_score']:.3f}")
        report.append(f"  Min Score: {results['min_score']:.3f}")
        report.append("")
        
        # Interpretation
        score = results['mean_score']
        if score < 0.1:
            interpretation = "Very likely authentic"
        elif score < 0.3:
            interpretation = "Likely authentic"
        elif score < 0.5:
            interpretation = "Uncertain - may be synthetic"
        elif score < 0.7:
            interpretation = "Likely synthetic"
        else:
            interpretation = "Very likely synthetic"
        
        report.append(f"Interpretation: {interpretation}")
        report.append("")
        
        report.append("⚠️  Important Notes:")
        report.append("  • This is a basic analysis using computer vision techniques")
        report.append("  • Results should be interpreted with caution")
        report.append("  • Higher scores indicate more synthetic characteristics")
        report.append("  • This analysis is for research purposes only")
        
        report.append("=" * 50)
        
        return "\n".join(report)


def main():
    """Main function for simple deepfake detection."""
    print("🔍 Simple Deepfake Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = SimpleDeepfakeDetector()
    
    while True:
        print("\nSelect an option:")
        print("1. Analyze a video file")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ").strip()
        
        if choice == "1":
            video_path = input("Enter path to video file: ").strip()
            
            if not video_path:
                print("❌ No video path provided")
                continue
            
            if not os.path.exists(video_path):
                print(f"❌ Video file not found: {video_path}")
                continue
            
            # Analyze video
            results = detector.analyze_video(video_path)
            
            if results:
                # Generate and display report
                report = detector.generate_report(results)
                print("\n" + report)
                
                # Save report to file
                report_path = f"{video_path}_detection_report.txt"
                with open(report_path, 'w') as f:
                    f.write(report)
                print(f"\n✅ Report saved: {report_path}")
            else:
                print("❌ Analysis failed")
        
        elif choice == "2":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
