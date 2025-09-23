"""
Advanced data augmentation techniques for deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Optional, Union, Callable
import random
import math
from scipy import ndimage
from skimage import exposure, filters, segmentation
import matplotlib.pyplot as plt

class DeepfakeSpecificAugmentation:
    """Augmentation techniques specifically designed for deepfake detection."""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def compression_artifact_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Simulate compression artifacts that might be present in deepfakes."""
        if random.random() > self.prob:
            return image
        
        # Random JPEG compression quality
        quality = random.randint(10, 90)
        
        # Encode and decode to simulate compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        compressed_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        return compressed_img
    
    def face_landmark_distortion(self, image: np.ndarray) -> np.ndarray:
        """Apply subtle distortions that might occur in face generation."""
        if random.random() > self.prob:
            return image
        
        # Create a subtle warping effect
        h, w = image.shape[:2]
        
        # Generate random displacement field
        displacement_x = np.random.normal(0, 2, (h, w))
        displacement_y = np.random.normal(0, 2, (h, w))
        
        # Apply Gaussian blur to make displacement smoother
        displacement_x = cv2.GaussianBlur(displacement_x, (15, 15), 0)
        displacement_y = cv2.GaussianBlur(displacement_y, (15, 15), 0)
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        x_coords = x_coords.astype(np.float32) + displacement_x
        y_coords = y_coords.astype(np.float32) + displacement_y
        
        # Apply remapping
        distorted = cv2.remap(image, x_coords, y_coords, cv2.INTER_LINEAR)
        
        return distorted
    
    def lighting_inconsistency(self, image: np.ndarray) -> np.ndarray:
        """Simulate lighting inconsistencies that might appear in deepfakes."""
        if random.random() > self.prob:
            return image
        
        # Create a gradient lighting effect
        h, w = image.shape[:2]
        
        # Random lighting direction
        direction = random.choice(['horizontal', 'vertical', 'diagonal', 'radial'])
        
        if direction == 'horizontal':
            gradient = np.linspace(0.7, 1.3, w)
            gradient = np.tile(gradient, (h, 1))
        elif direction == 'vertical':
            gradient = np.linspace(0.7, 1.3, h)
            gradient = np.tile(gradient.reshape(-1, 1), (1, w))
        elif direction == 'diagonal':
            gradient = np.zeros((h, w))
            for i in range(h):
                for j in range(w):
                    gradient[i, j] = 0.7 + 0.6 * (i + j) / (h + w)
        else:  # radial
            center_x, center_y = w // 2, h // 2
            gradient = np.zeros((h, w))
            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    max_dist = np.sqrt(center_x**2 + center_y**2)
                    gradient[i, j] = 0.7 + 0.6 * (1 - dist / max_dist)
        
        # Apply lighting gradient
        lit_image = image.astype(np.float32) * gradient[:, :, np.newaxis]
        lit_image = np.clip(lit_image, 0, 255).astype(np.uint8)
        
        return lit_image
    
    def color_space_inconsistency(self, image: np.ndarray) -> np.ndarray:
        """Simulate color space inconsistencies."""
        if random.random() > self.prob:
            return image
        
        # Convert to different color spaces and back
        color_space = random.choice(['HSV', 'LAB', 'YUV'])
        
        if color_space == 'HSV':
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # Add noise to hue channel
            hsv[:, :, 0] = hsv[:, :, 0] + np.random.normal(0, 5, hsv[:, :, 0].shape)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        elif color_space == 'LAB':
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            # Add noise to A and B channels
            lab[:, :, 1] = lab[:, :, 1] + np.random.normal(0, 3, lab[:, :, 1].shape)
            lab[:, :, 2] = lab[:, :, 2] + np.random.normal(0, 3, lab[:, :, 2].shape)
            lab = np.clip(lab, 0, 255)
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        else:  # YUV
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            # Add noise to U and V channels
            yuv[:, :, 1] = yuv[:, :, 1] + np.random.normal(0, 3, yuv[:, :, 1].shape)
            yuv[:, :, 2] = yuv[:, :, 2] + np.random.normal(0, 3, yuv[:, :, 2].shape)
            yuv = np.clip(yuv, 0, 255)
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        
        return result
    
    def temporal_inconsistency(self, image: np.ndarray) -> np.ndarray:
        """Simulate temporal inconsistencies in video deepfakes."""
        if random.random() > self.prob:
            return image
        
        # Create ghosting effect
        ghost_intensity = random.uniform(0.1, 0.3)
        
        # Apply motion blur
        kernel_size = random.randint(5, 15)
        angle = random.uniform(0, 360)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1
        kernel = kernel / kernel_size
        
        # Rotate kernel
        center = (kernel_size//2, kernel_size//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        # Apply blur
        blurred = cv2.filter2D(image, -1, kernel)
        
        # Blend with original
        result = cv2.addWeighted(image, 1-ghost_intensity, blurred, ghost_intensity, 0)
        
        return result

class AdvancedAugmentationPipeline:
    """Advanced augmentation pipeline combining multiple techniques."""
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (224, 224),
                 deepfake_specific: bool = True,
                 standard_augmentation: bool = True,
                 prob: float = 0.5):
        
        self.input_size = input_size
        self.deepfake_specific = deepfake_specific
        self.standard_augmentation = standard_augmentation
        self.prob = prob
        
        # Initialize augmentation components
        self.deepfake_aug = DeepfakeSpecificAugmentation(prob)
        
        # Standard augmentations
        self.standard_transforms = A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.RandomShadow(p=0.3),
            A.RandomSunFlare(p=0.2),
            A.RandomRain(p=0.2),
            A.RandomSnow(p=0.2),
            A.RandomFog(p=0.2)
        ])
        
        # Normalization
        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Apply augmentation pipeline."""
        
        # Apply deepfake-specific augmentations
        if self.deepfake_specific:
            image = self.deepfake_aug.compression_artifact_augmentation(image)
            image = self.deepfake_aug.face_landmark_distortion(image)
            image = self.deepfake_aug.lighting_inconsistency(image)
            image = self.deepfake_aug.color_space_inconsistency(image)
            image = self.deepfake_aug.temporal_inconsistency(image)
        
        # Apply standard augmentations
        if self.standard_augmentation:
            augmented = self.standard_transforms(image=image)
            image = augmented['image']
        
        # Normalize and convert to tensor
        normalized = self.normalize(image=image)
        return normalized['image']

class MixUpAugmentation:
    """MixUp augmentation for deepfake detection."""
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp augmentation."""
        if random.random() > self.prob:
            return images, labels
        
        batch_size = images.size(0)
        if batch_size < 2:
            return images, labels
        
        # Generate random permutation
        indices = torch.randperm(batch_size)
        
        # Generate mixing weights
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Mix labels (for binary classification)
        mixed_labels = lam * labels.float() + (1 - lam) * labels[indices].float()
        
        return mixed_images, mixed_labels

class CutMixAugmentation:
    """CutMix augmentation for deepfake detection."""
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation."""
        if random.random() > self.prob:
            return images, labels
        
        batch_size = images.size(0)
        if batch_size < 2:
            return images, labels
        
        # Generate random permutation
        indices = torch.randperm(batch_size)
        
        # Generate mixing weights
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Calculate cut size
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Calculate bounding box
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Mix labels
        mixed_labels = lam * labels.float() + (1 - lam) * labels[indices].float()
        
        return mixed_images, mixed_labels

class AdversarialAugmentation:
    """Adversarial augmentation to improve robustness."""
    
    def __init__(self, epsilon: float = 0.03, prob: float = 0.3):
        self.epsilon = epsilon
        self.prob = prob
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Apply adversarial augmentation."""
        if random.random() > self.prob:
            return images
        
        # Generate random noise
        noise = torch.randn_like(images) * self.epsilon
        
        # Add noise
        adversarial_images = images + noise
        
        # Clip to valid range
        adversarial_images = torch.clamp(adversarial_images, 0, 1)
        
        return adversarial_images

class AugmentationVisualizer:
    """Visualize augmentation effects."""
    
    def __init__(self, augmentation_pipeline: AdvancedAugmentationPipeline):
        self.augmentation_pipeline = augmentation_pipeline
    
    def visualize_augmentations(self, 
                             original_image: np.ndarray, 
                             num_samples: int = 8,
                             save_path: str = None):
        """Visualize different augmentation effects."""
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Apply augmentations
        for i in range(1, num_samples):
            # Apply augmentation pipeline
            augmented_tensor = self.augmentation_pipeline(original_image)
            
            # Convert back to numpy for visualization
            augmented_image = augmented_tensor.permute(1, 2, 0).numpy()
            
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            augmented_image = augmented_image * std + mean
            augmented_image = np.clip(augmented_image, 0, 1)
            
            axes[i].imshow(augmented_image)
            axes[i].set_title(f'Augmented {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def create_augmentation_pipeline(input_size: Tuple[int, int] = (224, 224),
                               deepfake_specific: bool = True,
                               standard_augmentation: bool = True,
                               prob: float = 0.5) -> AdvancedAugmentationPipeline:
    """Create augmentation pipeline."""
    return AdvancedAugmentationPipeline(
        input_size=input_size,
        deepfake_specific=deepfake_specific,
        standard_augmentation=standard_augmentation,
        prob=prob
    )

def create_training_augmentations(input_size: Tuple[int, int] = (224, 224)) -> Dict[str, Callable]:
    """Create comprehensive training augmentations."""
    return {
        'image_augmentation': create_augmentation_pipeline(input_size, prob=0.8),
        'mixup': MixUpAugmentation(alpha=0.2, prob=0.5),
        'cutmix': CutMixAugmentation(alpha=1.0, prob=0.5),
        'adversarial': AdversarialAugmentation(epsilon=0.03, prob=0.3)
    }

if __name__ == "__main__":
    # Example usage
    pipeline = create_augmentation_pipeline()
    
    # Load a sample image
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Visualize augmentations
    visualizer = AugmentationVisualizer(pipeline)
    visualizer.visualize_augmentations(sample_image, save_path="augmentation_examples.png")
    
    print("Augmentation visualization complete!")
