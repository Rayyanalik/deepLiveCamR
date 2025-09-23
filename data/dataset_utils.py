"""
Dataset utilities for deepfake detection research.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
from PIL import Image
import os
import json
from typing import List, Dict, Tuple, Optional, Union, Callable
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class DeepfakeDataset(Dataset):
    """PyTorch dataset for deepfake detection."""
    
    def __init__(self, 
                 data_dir: str,
                 metadata_file: str = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 class_names: List[str] = None):
        
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.class_names = class_names or ["real", "fake"]
        
        # Load metadata if provided
        if metadata_file:
            self.metadata = self._load_metadata(metadata_file)
            self.samples = self._load_samples_from_metadata()
        else:
            self.samples = self._load_samples_from_directory()
        
        print(f"📊 Dataset loaded: {len(self.samples)} samples")
        print(f"   - Real: {sum(1 for s in self.samples if s[1] == 0)}")
        print(f"   - Fake: {sum(1 for s in self.samples if s[1] == 1)}")
    
    def _load_metadata(self, metadata_file: str) -> Dict:
        """Load dataset metadata."""
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def _load_samples_from_metadata(self) -> List[Tuple[str, int]]:
        """Load samples from metadata file."""
        samples = []
        
        for sample_info in self.metadata['samples']:
            filepath = sample_info['filepath']
            class_name = sample_info['class']
            class_idx = self.class_names.index(class_name)
            samples.append((filepath, class_idx))
        
        return samples
    
    def _load_samples_from_directory(self) -> List[Tuple[str, int]]:
        """Load samples from directory structure."""
        samples = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"⚠️ Warning: Class directory {class_dir} not found")
                continue
            
            for file_path in class_dir.glob("*.jpg"):
                samples.append((str(file_path), class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, class_idx = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(filepath).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            # Return a dummy image
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            class_idx = self.target_transform(class_idx)
        
        return image, class_idx
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        class_counts = Counter([sample[1] for sample in self.samples])
        total_samples = len(self.samples)
        
        weights = []
        for class_idx in range(len(self.class_names)):
            count = class_counts.get(class_idx, 1)
            weight = total_samples / (len(self.class_names) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)

class VideoDataset(Dataset):
    """Dataset for video-based deepfake detection."""
    
    def __init__(self, 
                 data_dir: str,
                 sequence_length: int = 16,
                 frame_interval: int = 1,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.frame_interval = frame_interval
        self.transform = transform
        self.target_transform = target_transform
        
        self.samples = self._load_video_samples()
        
        print(f"📹 Video dataset loaded: {len(self.samples)} video sequences")
    
    def _load_video_samples(self) -> List[Tuple[str, int]]:
        """Load video samples from directory."""
        samples = []
        
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            class_idx = 0 if class_name == "real" else 1
            
            for video_file in class_dir.glob("*.mp4"):
                samples.append((str(video_file), class_idx))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, class_idx = self.samples[idx]
        
        # Extract frames from video
        frames = self._extract_frames(video_path)
        
        # Apply transforms
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # Stack frames into tensor
        frames_tensor = torch.stack(frames)
        
        if self.target_transform:
            class_idx = self.target_transform(class_idx)
        
        return frames_tensor, class_idx
    
    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        frame_count = 0
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
        
        cap.release()
        
        # Pad with last frame if needed
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])
        
        return frames

class DatasetAnalyzer:
    """Analyze dataset characteristics and statistics."""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.analysis_results = {}
    
    def analyze_dataset(self) -> Dict:
        """Perform comprehensive dataset analysis."""
        print("🔍 Analyzing dataset...")
        
        # Basic statistics
        self._analyze_basic_stats()
        
        # Class distribution
        self._analyze_class_distribution()
        
        # Image characteristics
        self._analyze_image_characteristics()
        
        # Quality assessment
        self._analyze_quality()
        
        return self.analysis_results
    
    def _analyze_basic_stats(self):
        """Analyze basic dataset statistics."""
        total_samples = len(self.dataset)
        
        # Sample a subset for analysis
        sample_size = min(1000, total_samples)
        sample_indices = np.random.choice(total_samples, sample_size, replace=False)
        
        self.analysis_results['basic_stats'] = {
            'total_samples': total_samples,
            'analyzed_samples': sample_size,
            'classes': len(self.dataset.class_names) if hasattr(self.dataset, 'class_names') else 2
        }
    
    def _analyze_class_distribution(self):
        """Analyze class distribution."""
        class_counts = Counter()
        
        for i in range(len(self.dataset)):
            _, class_idx = self.dataset[i]
            class_counts[class_idx] += 1
        
        total_samples = len(self.dataset)
        class_distribution = {
            idx: {
                'count': count,
                'percentage': count / total_samples * 100
            }
            for idx, count in class_counts.items()
        }
        
        self.analysis_results['class_distribution'] = class_distribution
    
    def _analyze_image_characteristics(self):
        """Analyze image characteristics."""
        sample_size = min(100, len(self.dataset))
        sample_indices = np.random.choice(len(self.dataset), sample_size, replace=False)
        
        widths, heights, channels = [], [], []
        
        for idx in sample_indices:
            image, _ = self.dataset[idx]
            
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    height, width = image.shape[1], image.shape[2]
                    channel = image.shape[0]
                else:
                    height, width = image.shape[0], image.shape[1]
                    channel = 1
            else:
                height, width = image.shape[:2]
                channel = image.shape[2] if len(image.shape) == 3 else 1
            
            widths.append(width)
            heights.append(height)
            channels.append(channel)
        
        self.analysis_results['image_characteristics'] = {
            'width': {
                'mean': np.mean(widths),
                'std': np.std(widths),
                'min': np.min(widths),
                'max': np.max(widths)
            },
            'height': {
                'mean': np.mean(heights),
                'std': np.std(heights),
                'min': np.min(heights),
                'max': np.max(heights)
            },
            'channels': {
                'mean': np.mean(channels),
                'most_common': Counter(channels).most_common(1)[0][0]
            }
        }
    
    def _analyze_quality(self):
        """Analyze image quality metrics."""
        sample_size = min(50, len(self.dataset))
        sample_indices = np.random.choice(len(self.dataset), sample_size, replace=False)
        
        brightness_values = []
        contrast_values = []
        
        for idx in sample_indices:
            image, _ = self.dataset[idx]
            
            if isinstance(image, torch.Tensor):
                image = image.numpy()
            
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Calculate brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            brightness_values.append(brightness)
            contrast_values.append(contrast)
        
        self.analysis_results['quality_metrics'] = {
            'brightness': {
                'mean': np.mean(brightness_values),
                'std': np.std(brightness_values),
                'min': np.min(brightness_values),
                'max': np.max(brightness_values)
            },
            'contrast': {
                'mean': np.mean(contrast_values),
                'std': np.std(contrast_values),
                'min': np.min(contrast_values),
                'max': np.max(contrast_values)
            }
        }
    
    def visualize_analysis(self, save_path: str = None):
        """Visualize dataset analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Class distribution
        class_dist = self.analysis_results['class_distribution']
        classes = list(class_dist.keys())
        counts = [class_dist[c]['count'] for c in classes]
        
        axes[0, 0].bar(classes, counts, color=['#1f77b4', '#ff7f0e'])
        axes[0, 0].set_title('Class Distribution')
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Count')
        
        # Image size distribution
        img_chars = self.analysis_results['image_characteristics']
        widths = [img_chars['width']['mean']] * 2
        heights = [img_chars['height']['mean']] * 2
        
        axes[0, 1].scatter(widths, heights, c=['#1f77b4', '#ff7f0e'], s=100)
        axes[0, 1].set_title('Average Image Dimensions')
        axes[0, 1].set_xlabel('Width')
        axes[0, 1].set_ylabel('Height')
        
        # Quality metrics
        quality = self.analysis_results['quality_metrics']
        brightness = quality['brightness']['mean']
        contrast = quality['contrast']['mean']
        
        axes[1, 0].scatter([brightness], [contrast], c=['#2ca02c'], s=100)
        axes[1, 0].set_title('Average Quality Metrics')
        axes[1, 0].set_xlabel('Brightness')
        axes[1, 0].set_ylabel('Contrast')
        
        # Dataset summary
        basic_stats = self.analysis_results['basic_stats']
        summary_text = f"""
        Total Samples: {basic_stats['total_samples']}
        Classes: {basic_stats['classes']}
        Analyzed: {basic_stats['analyzed_samples']}
        
        Image Size: {img_chars['width']['mean']:.0f}x{img_chars['height']['mean']:.0f}
        Brightness: {brightness:.1f}
        Contrast: {contrast:.1f}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Dataset Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class DataLoaderFactory:
    """Factory for creating data loaders with different configurations."""
    
    @staticmethod
    def create_image_transforms(input_size: Tuple[int, int] = (224, 224),
                              augmentation: bool = True,
                              normalization: str = "imagenet") -> A.Compose:
        """Create image transforms for training/validation."""
        
        # Define normalization values
        if normalization == "imagenet":
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif normalization == "custom":
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        
        if augmentation:
            # Training transforms with augmentation
            transforms = A.Compose([
                A.Resize(input_size[0], input_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            # Validation transforms without augmentation
            transforms = A.Compose([
                A.Resize(input_size[0], input_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        
        return transforms
    
    @staticmethod
    def create_dataloaders(dataset: Dataset,
                          batch_size: int = 32,
                          train_split: float = 0.8,
                          val_split: float = 0.1,
                          test_split: float = 0.1,
                          num_workers: int = 4,
                          pin_memory: bool = True) -> Dict[str, DataLoader]:
        """Create train/validation/test data loaders."""
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return {
            'train': train_loader,
            'validation': val_loader,
            'test': test_loader
        }

def create_dataset_from_directory(data_dir: str,
                                metadata_file: str = None,
                                input_size: Tuple[int, int] = (224, 224),
                                augmentation: bool = True) -> DeepfakeDataset:
    """Convenience function to create dataset from directory."""
    
    # Create transforms
    transform = DataLoaderFactory.create_image_transforms(
        input_size=input_size,
        augmentation=augmentation
    )
    
    # Create dataset
    dataset = DeepfakeDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        transform=transform
    )
    
    return dataset

def analyze_dataset_quality(dataset: Dataset, save_path: str = None) -> Dict:
    """Convenience function to analyze dataset quality."""
    analyzer = DatasetAnalyzer(dataset)
    results = analyzer.analyze_dataset()
    analyzer.visualize_analysis(save_path)
    return results

if __name__ == "__main__":
    # Example usage
    dataset = create_dataset_from_directory("data/synthetic")
    results = analyze_dataset_quality(dataset, "dataset_analysis.png")
    print("Dataset analysis complete!")
