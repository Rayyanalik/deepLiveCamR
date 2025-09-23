"""
Data package for deepfake detection research platform.
"""

from .synthetic_generator import (
    SyntheticConfig,
    FaceGenerator,
    BackgroundGenerator,
    ArtifactGenerator,
    SyntheticDatasetGenerator,
    create_synthetic_dataset
)

from .dataset_utils import (
    DeepfakeDataset,
    VideoDataset,
    DatasetAnalyzer,
    DataLoaderFactory,
    create_dataset_from_directory,
    analyze_dataset_quality
)

from .augmentation import (
    DeepfakeSpecificAugmentation,
    AdvancedAugmentationPipeline,
    MixUpAugmentation,
    CutMixAugmentation,
    AdversarialAugmentation,
    AugmentationVisualizer,
    create_augmentation_pipeline,
    create_training_augmentations
)

__all__ = [
    # Synthetic Generation
    "SyntheticConfig",
    "FaceGenerator",
    "BackgroundGenerator", 
    "ArtifactGenerator",
    "SyntheticDatasetGenerator",
    "create_synthetic_dataset",
    
    # Dataset Utilities
    "DeepfakeDataset",
    "VideoDataset",
    "DatasetAnalyzer",
    "DataLoaderFactory",
    "create_dataset_from_directory",
    "analyze_dataset_quality",
    
    # Augmentation
    "DeepfakeSpecificAugmentation",
    "AdvancedAugmentationPipeline",
    "MixUpAugmentation",
    "CutMixAugmentation",
    "AdversarialAugmentation",
    "AugmentationVisualizer",
    "create_augmentation_pipeline",
    "create_training_augmentations"
]
