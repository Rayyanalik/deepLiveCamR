"""
Create a sample face image for testing the real-time demo.
"""

import cv2
import numpy as np
from pathlib import Path

def create_sample_face():
    """Create a simple sample face image."""
    # Create a simple face image
    face = np.ones((300, 300, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Face outline
    cv2.ellipse(face, (150, 150), (80, 100), 0, 0, 360, (200, 180, 160), -1)
    
    # Eyes
    cv2.circle(face, (130, 130), 15, (0, 0, 0), -1)
    cv2.circle(face, (170, 130), 15, (0, 0, 0), -1)
    
    # Nose
    cv2.ellipse(face, (150, 150), (8, 15), 0, 0, 360, (180, 160, 140), -1)
    
    # Mouth
    cv2.ellipse(face, (150, 180), (25, 10), 0, 0, 180, (0, 0, 0), 2)
    
    # Hair
    cv2.ellipse(face, (150, 100), (90, 40), 0, 0, 180, (100, 80, 60), -1)
    
    # Save the face
    output_path = Path("data/source_faces/my_face.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path), face)
    print(f"✅ Sample face created: {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    create_sample_face()
