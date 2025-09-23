"""
Synthetic data generation tools for deepfake detection research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import os
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""
    output_dir: str = "data/synthetic"
    num_samples: int = 1000
    image_size: Tuple[int, int] = (224, 224)
    face_size_range: Tuple[int, int] = (80, 150)
    background_types: List[str] = None
    lighting_conditions: List[str] = None
    compression_levels: List[int] = None
    noise_levels: List[float] = None
    augmentation_prob: float = 0.5
    
    def __post_init__(self):
        if self.background_types is None:
            self.background_types = ["solid", "gradient", "texture", "pattern"]
        if self.lighting_conditions is None:
            self.lighting_conditions = ["normal", "bright", "dim", "shadowed"]
        if self.compression_levels is None:
            self.compression_levels = [10, 30, 50, 70, 90]
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1]

class FaceGenerator:
    """Generate synthetic faces for research purposes."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.face_templates = self._create_face_templates()
        
    def _create_face_templates(self) -> List[np.ndarray]:
        """Create basic face templates."""
        templates = []
        
        # Template 1: Simple oval face
        face1 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.ellipse(face1, (100, 100), (80, 100), 0, 0, 360, (220, 180, 140), -1)
        cv2.ellipse(face1, (100, 100), (80, 100), 0, 0, 360, (0, 0, 0), 2)
        templates.append(face1)
        
        # Template 2: More detailed face
        face2 = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.ellipse(face2, (100, 100), (85, 105), 0, 0, 360, (210, 170, 130), -1)
        cv2.ellipse(face2, (100, 100), (85, 105), 0, 0, 360, (0, 0, 0), 2)
        
        # Add eyes
        cv2.circle(face2, (80, 80), 8, (0, 0, 0), -1)
        cv2.circle(face2, (120, 80), 8, (0, 0, 0), -1)
        
        # Add nose
        cv2.ellipse(face2, (100, 100), (3, 8), 0, 0, 360, (0, 0, 0), -1)
        
        # Add mouth
        cv2.ellipse(face2, (100, 120), (15, 8), 0, 0, 180, (0, 0, 0), 2)
        
        templates.append(face2)
        
        return templates
    
    def generate_face(self, template_idx: int = None) -> np.ndarray:
        """Generate a synthetic face."""
        if template_idx is None:
            template_idx = random.randint(0, len(self.face_templates) - 1)
        
        face = self.face_templates[template_idx].copy()
        
        # Add random variations
        face = self._add_face_variations(face)
        
        return face
    
    def _add_face_variations(self, face: np.ndarray) -> np.ndarray:
        """Add random variations to face."""
        # Random color variations
        color_shift = np.random.randint(-20, 20, 3)
        face = np.clip(face.astype(np.int16) + color_shift, 0, 255).astype(np.uint8)
        
        # Random brightness
        brightness = random.uniform(0.8, 1.2)
        face = np.clip(face * brightness, 0, 255).astype(np.uint8)
        
        # Random contrast
        contrast = random.uniform(0.9, 1.1)
        face = np.clip((face - 128) * contrast + 128, 0, 255).astype(np.uint8)
        
        return face

class BackgroundGenerator:
    """Generate synthetic backgrounds."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
    
    def generate_background(self, bg_type: str = None) -> np.ndarray:
        """Generate a synthetic background."""
        if bg_type is None:
            bg_type = random.choice(self.config.background_types)
        
        if bg_type == "solid":
            return self._generate_solid_background()
        elif bg_type == "gradient":
            return self._generate_gradient_background()
        elif bg_type == "texture":
            return self._generate_texture_background()
        elif bg_type == "pattern":
            return self._generate_pattern_background()
        else:
            return self._generate_solid_background()
    
    def _generate_solid_background(self) -> np.ndarray:
        """Generate solid color background."""
        color = np.random.randint(50, 200, 3)
        bg = np.full((self.config.image_size[1], self.config.image_size[0], 3), color, dtype=np.uint8)
        return bg
    
    def _generate_gradient_background(self) -> np.ndarray:
        """Generate gradient background."""
        bg = np.zeros((self.config.image_size[1], self.config.image_size[0], 3), dtype=np.uint8)
        
        # Random gradient direction
        direction = random.choice(["horizontal", "vertical", "diagonal", "radial"])
        
        if direction == "horizontal":
            for x in range(self.config.image_size[0]):
                intensity = int(255 * x / self.config.image_size[0])
                bg[:, x] = [intensity, intensity//2, intensity//3]
        
        elif direction == "vertical":
            for y in range(self.config.image_size[1]):
                intensity = int(255 * y / self.config.image_size[1])
                bg[y, :] = [intensity, intensity//2, intensity//3]
        
        elif direction == "diagonal":
            for y in range(self.config.image_size[1]):
                for x in range(self.config.image_size[0]):
                    intensity = int(255 * (x + y) / (self.config.image_size[0] + self.config.image_size[1]))
                    bg[y, x] = [intensity, intensity//2, intensity//3]
        
        else:  # radial
            center_x, center_y = self.config.image_size[0] // 2, self.config.image_size[1] // 2
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            for y in range(self.config.image_size[1]):
                for x in range(self.config.image_size[0]):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    intensity = int(255 * (1 - dist / max_dist))
                    bg[y, x] = [intensity, intensity//2, intensity//3]
        
        return bg
    
    def _generate_texture_background(self) -> np.ndarray:
        """Generate textured background."""
        bg = np.random.randint(100, 200, (self.config.image_size[1], self.config.image_size[0], 3), dtype=np.uint8)
        
        # Add texture patterns
        pattern_type = random.choice(["noise", "lines", "dots", "waves"])
        
        if pattern_type == "noise":
            noise = np.random.randint(-20, 20, bg.shape, dtype=np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        elif pattern_type == "lines":
            for i in range(0, self.config.image_size[1], 10):
                cv2.line(bg, (0, i), (self.config.image_size[0], i), (0, 0, 0), 1)
        
        elif pattern_type == "dots":
            for _ in range(50):
                x = random.randint(0, self.config.image_size[0]-1)
                y = random.randint(0, self.config.image_size[1]-1)
                cv2.circle(bg, (x, y), 2, (0, 0, 0), -1)
        
        else:  # waves
            for x in range(self.config.image_size[0]):
                y = int(self.config.image_size[1]//2 + 20 * np.sin(x * 0.1))
                if 0 <= y < self.config.image_size[1]:
                    cv2.line(bg, (x, y-1), (x, y+1), (0, 0, 0), 1)
        
        return bg
    
    def _generate_pattern_background(self) -> np.ndarray:
        """Generate patterned background."""
        bg = np.full((self.config.image_size[1], self.config.image_size[0], 3), 200, dtype=np.uint8)
        
        pattern_type = random.choice(["checkerboard", "stripes", "circles", "triangles"])
        
        if pattern_type == "checkerboard":
            square_size = 20
            for y in range(0, self.config.image_size[1], square_size):
                for x in range(0, self.config.image_size[0], square_size):
                    if (x // square_size + y // square_size) % 2 == 0:
                        cv2.rectangle(bg, (x, y), (x+square_size, y+square_size), (100, 100, 100), -1)
        
        elif pattern_type == "stripes":
            stripe_width = 10
            for x in range(0, self.config.image_size[0], stripe_width * 2):
                cv2.rectangle(bg, (x, 0), (x+stripe_width, self.config.image_size[1]), (100, 100, 100), -1)
        
        elif pattern_type == "circles":
            for _ in range(20):
                center = (random.randint(50, self.config.image_size[0]-50), 
                        random.randint(50, self.config.image_size[1]-50))
                radius = random.randint(10, 30)
                cv2.circle(bg, center, radius, (100, 100, 100), -1)
        
        else:  # triangles
            for _ in range(15):
                pts = np.array([
                    [random.randint(0, self.config.image_size[0]), random.randint(0, self.config.image_size[1])],
                    [random.randint(0, self.config.image_size[0]), random.randint(0, self.config.image_size[1])],
                    [random.randint(0, self.config.image_size[0]), random.randint(0, self.config.image_size[1])]
                ], np.int32)
                cv2.fillPoly(bg, [pts], (100, 100, 100))
        
        return bg

class ArtifactGenerator:
    """Generate various artifacts that might indicate deepfakes."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
    
    def add_compression_artifacts(self, image: np.ndarray, quality: int = None) -> np.ndarray:
        """Add compression artifacts to image."""
        if quality is None:
            quality = random.choice(self.config.compression_levels)
        
        # Simulate JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        return compressed_img
    
    def add_noise(self, image: np.ndarray, noise_level: float = None) -> np.ndarray:
        """Add noise to image."""
        if noise_level is None:
            noise_level = random.choice(self.config.noise_levels)
        
        if noise_level == 0:
            return image
        
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def add_blur(self, image: np.ndarray, blur_type: str = None) -> np.ndarray:
        """Add blur to image."""
        if blur_type is None:
            blur_type = random.choice(["gaussian", "motion", "radial"])
        
        if blur_type == "gaussian":
            kernel_size = random.choice([3, 5, 7])
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        elif blur_type == "motion":
            kernel_size = random.randint(5, 15)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            return cv2.filter2D(image, -1, kernel)
        
        else:  # radial
            center = (image.shape[1]//2, image.shape[0]//2)
            radius = random.randint(20, 50)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            
            blurred = cv2.GaussianBlur(image, (15, 15), 0)
            result = image.copy()
            result[mask > 0] = blurred[mask > 0]
            return result
    
    def add_lighting_artifacts(self, image: np.ndarray, lighting: str = None) -> np.ndarray:
        """Add lighting artifacts to image."""
        if lighting is None:
            lighting = random.choice(self.config.lighting_conditions)
        
        if lighting == "normal":
            return image
        
        elif lighting == "bright":
            bright_image = np.clip(image.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
            return bright_image
        
        elif lighting == "dim":
            dim_image = np.clip(image.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
            return dim_image
        
        else:  # shadowed
            shadowed_image = image.copy()
            # Add shadow overlay
            overlay = shadowed_image.copy()
            cv2.rectangle(overlay, (50, 50), (150, 150), (0, 0, 0), -1)
            cv2.addWeighted(shadowed_image, 0.8, overlay, 0.2, 0, shadowed_image)
            return shadowed_image

class SyntheticDatasetGenerator:
    """Main class for generating synthetic datasets."""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.face_generator = FaceGenerator(config)
        self.background_generator = BackgroundGenerator(config)
        self.artifact_generator = ArtifactGenerator(config)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "real"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "fake"), exist_ok=True)
    
    def generate_real_samples(self, num_samples: int = None) -> List[Dict]:
        """Generate synthetic 'real' samples."""
        if num_samples is None:
            num_samples = self.config.num_samples // 2
        
        samples = []
        
        for i in range(num_samples):
            # Generate face
            face = self.face_generator.generate_face()
            
            # Generate background
            background = self.background_generator.generate_background()
            
            # Composite face onto background
            composite_image = self._composite_face_background(face, background)
            
            # Add some artifacts (but keep them subtle for 'real' samples)
            if random.random() < 0.3:  # 30% chance of artifacts
                composite_image = self.artifact_generator.add_compression_artifacts(composite_image, 80)
            
            if random.random() < 0.2:  # 20% chance of noise
                composite_image = self.artifact_generator.add_noise(composite_image, 0.01)
            
            # Save image
            filename = f"real_{i:06d}.jpg"
            filepath = os.path.join(self.config.output_dir, "real", filename)
            cv2.imwrite(filepath, composite_image)
            
            # Store metadata
            sample_info = {
                'filename': filename,
                'class': 'real',
                'filepath': filepath,
                'face_template': random.randint(0, len(self.face_generator.face_templates) - 1),
                'background_type': random.choice(self.config.background_types),
                'lighting': random.choice(self.config.lighting_conditions),
                'has_artifacts': random.random() < 0.3
            }
            samples.append(sample_info)
        
        return samples
    
    def generate_fake_samples(self, num_samples: int = None) -> List[Dict]:
        """Generate synthetic 'fake' samples with more artifacts."""
        if num_samples is None:
            num_samples = self.config.num_samples // 2
        
        samples = []
        
        for i in range(num_samples):
            # Generate face
            face = self.face_generator.generate_face()
            
            # Generate background
            background = self.background_generator.generate_background()
            
            # Composite face onto background
            composite_image = self._composite_face_background(face, background)
            
            # Add more artifacts for 'fake' samples
            composite_image = self.artifact_generator.add_compression_artifacts(composite_image)
            composite_image = self.artifact_generator.add_noise(composite_image)
            
            if random.random() < 0.5:  # 50% chance of blur
                composite_image = self.artifact_generator.add_blur(composite_image)
            
            composite_image = self.artifact_generator.add_lighting_artifacts(composite_image)
            
            # Save image
            filename = f"fake_{i:06d}.jpg"
            filepath = os.path.join(self.config.output_dir, "fake", filename)
            cv2.imwrite(filepath, composite_image)
            
            # Store metadata
            sample_info = {
                'filename': filename,
                'class': 'fake',
                'filepath': filepath,
                'face_template': random.randint(0, len(self.face_generator.face_templates) - 1),
                'background_type': random.choice(self.config.background_types),
                'lighting': random.choice(self.config.lighting_conditions),
                'has_artifacts': True,
                'compression_level': random.choice(self.config.compression_levels),
                'noise_level': random.choice(self.config.noise_levels)
            }
            samples.append(sample_info)
        
        return samples
    
    def _composite_face_background(self, face: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Composite face onto background."""
        # Resize face to random size
        face_size = random.randint(*self.config.face_size_range)
        face_resized = cv2.resize(face, (face_size, face_size))
        
        # Random position on background
        max_x = self.config.image_size[0] - face_size
        max_y = self.config.image_size[1] - face_size
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        
        # Create composite
        composite = background.copy()
        
        # Simple alpha blending (assuming face has some transparency)
        face_region = composite[y:y+face_size, x:x+face_size]
        
        # Blend face with background
        alpha = 0.8  # Face opacity
        blended = cv2.addWeighted(face_region, alpha, face_resized, 1-alpha, 0)
        composite[y:y+face_size, x:x+face_size] = blended
        
        return composite
    
    def generate_dataset(self) -> Dict:
        """Generate complete synthetic dataset."""
        print("🎨 Generating synthetic dataset...")
        
        # Generate samples
        real_samples = self.generate_real_samples()
        fake_samples = self.generate_fake_samples()
        
        # Combine all samples
        all_samples = real_samples + fake_samples
        
        # Create metadata
        metadata = {
            'total_samples': len(all_samples),
            'real_samples': len(real_samples),
            'fake_samples': len(fake_samples),
            'image_size': self.config.image_size,
            'face_size_range': self.config.face_size_range,
            'background_types': self.config.background_types,
            'lighting_conditions': self.config.lighting_conditions,
            'compression_levels': self.config.compression_levels,
            'noise_levels': self.config.noise_levels,
            'samples': all_samples
        }
        
        # Save metadata
        metadata_path = os.path.join(self.config.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create dataset split
        self._create_dataset_splits(all_samples)
        
        print(f"✅ Dataset generated successfully!")
        print(f"📁 Output directory: {self.config.output_dir}")
        print(f"📊 Total samples: {len(all_samples)}")
        print(f"   - Real: {len(real_samples)}")
        print(f"   - Fake: {len(fake_samples)}")
        
        return metadata
    
    def _create_dataset_splits(self, samples: List[Dict]):
        """Create train/validation/test splits."""
        random.shuffle(samples)
        
        total_samples = len(samples)
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        train_samples = samples[:train_size]
        val_samples = samples[train_size:train_size + val_size]
        test_samples = samples[train_size + val_size:]
        
        # Save splits
        splits = {
            'train': train_samples,
            'validation': val_samples,
            'test': test_samples
        }
        
        for split_name, split_samples in splits.items():
            split_path = os.path.join(self.config.output_dir, f"{split_name}.json")
            with open(split_path, 'w') as f:
                json.dump(split_samples, f, indent=2)
        
        print(f"📋 Dataset splits created:")
        print(f"   - Train: {len(train_samples)} samples")
        print(f"   - Validation: {len(val_samples)} samples")
        print(f"   - Test: {len(test_samples)} samples")
    
    def visualize_samples(self, num_samples: int = 8):
        """Visualize generated samples."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Load some samples
        real_dir = os.path.join(self.config.output_dir, "real")
        fake_dir = os.path.join(self.config.output_dir, "fake")
        
        real_files = [f for f in os.listdir(real_dir) if f.endswith('.jpg')][:4]
        fake_files = [f for f in os.listdir(fake_dir) if f.endswith('.jpg')][:4]
        
        for i, filename in enumerate(real_files + fake_files):
            if i < 4:
                filepath = os.path.join(real_dir, filename)
                label = "Real"
            else:
                filepath = os.path.join(fake_dir, filename)
                label = "Fake"
            
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(image)
            axes[i].set_title(f"{label} Sample")
            axes[i].axis('off')
        
        plt.suptitle("Synthetic Dataset Samples", fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.config.output_dir, "sample_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Sample visualization saved to: {viz_path}")

def create_synthetic_dataset(output_dir: str = "data/synthetic",
                           num_samples: int = 1000,
                           image_size: Tuple[int, int] = (224, 224)) -> Dict:
    """Convenience function to create synthetic dataset."""
    
    config = SyntheticConfig(
        output_dir=output_dir,
        num_samples=num_samples,
        image_size=image_size
    )
    
    generator = SyntheticDatasetGenerator(config)
    metadata = generator.generate_dataset()
    generator.visualize_samples()
    
    return metadata

if __name__ == "__main__":
    # Example usage
    metadata = create_synthetic_dataset(
        output_dir="data/synthetic_demo",
        num_samples=100,
        image_size=(224, 224)
    )
    
    print("Dataset generation complete!")
