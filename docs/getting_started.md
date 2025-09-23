# Getting Started with Deepfake Detection Research Platform

Welcome to the Deepfake Detection Research Platform! This guide will help you get started with using the platform for research and educational purposes.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Usage](#basic-usage)
4. [Web Interface](#web-interface)
5. [Real-time Detection](#real-time-detection)
6. [Dataset Creation](#dataset-creation)
7. [Model Training](#model-training)
8. [Evaluation](#evaluation)
9. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd deepfake-detection-platform
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install PyTorch (choose appropriate version for your system)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Verify Installation

```bash
# Test installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import streamlit; print('Streamlit installed successfully')"
```

## Quick Start

### 1. Generate Synthetic Dataset

```python
from data import create_synthetic_dataset

# Create a small synthetic dataset for testing
metadata = create_synthetic_dataset(
    output_dir="data/synthetic_demo",
    num_samples=100,
    image_size=(224, 224)
)

print("Dataset created successfully!")
```

### 2. Load and Test a Model

```python
from models import create_cnn_model
from data import create_dataset_from_directory
from evaluation import DetectionMetrics

# Create a simple CNN model
model = create_cnn_model(model_name="resnet50", pretrained=True)

# Load dataset
dataset = create_dataset_from_directory("data/synthetic_demo")

# Test the model
metrics = DetectionMetrics()
# ... (run evaluation code)
```

### 3. Launch Web Interface

```bash
# Launch Streamlit app
streamlit run web_interface/streamlit_app.py

# Or launch Gradio app
python web_interface/gradio_app.py
```

## Basic Usage

### Model Creation

```python
from models import create_cnn_model, create_transformer_model, create_ensemble_model

# CNN models
cnn_model = create_cnn_model(model_name="resnet50", pretrained=True)
efficientnet_model = create_cnn_model(model_name="efficientnet_b0", pretrained=True)

# Transformer models
vit_model = create_transformer_model(model_type="vit_base", pretrained=True)
swin_model = create_transformer_model(model_type="swin_base", pretrained=True)

# Ensemble models
ensemble_model = create_ensemble_model(ensemble_type="weighted_average")
```

### Dataset Loading

```python
from data import create_dataset_from_directory, DataLoaderFactory

# Create dataset
dataset = create_dataset_from_directory(
    data_dir="data/synthetic_demo",
    input_size=(224, 224),
    augmentation=True
)

# Create data loaders
dataloaders = DataLoaderFactory.create_dataloaders(
    dataset=dataset,
    batch_size=32,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1
)

train_loader = dataloaders['train']
val_loader = dataloaders['validation']
test_loader = dataloaders['test']
```

### Model Evaluation

```python
from evaluation import DetectionMetrics, BenchmarkSuite

# Initialize metrics
metrics = DetectionMetrics()

# Evaluate model
for batch_idx, (data, target) in enumerate(test_loader):
    # Run inference
    with torch.no_grad():
        output = model(data)
        predictions = torch.argmax(output, dim=1)
    
    # Update metrics
    metrics.update(predictions, target, output)

# Get results
results = metrics.compute_metrics()
print(f"Accuracy: {results['accuracy']:.3f}")
print(f"F1-Score: {results['f1_score']:.3f}")
```

## Web Interface

### Streamlit Interface

The Streamlit interface provides a comprehensive web-based platform for:

- **Model Selection**: Choose from various detection models
- **Image Analysis**: Upload and analyze images
- **Real-time Detection**: Monitor live camera feed
- **Performance Analysis**: View model performance metrics
- **Model Comparison**: Compare different models

#### Launching Streamlit

```bash
streamlit run web_interface/streamlit_app.py
```

#### Features

1. **Image Detection Tab**
   - Upload images for analysis
   - Real-time camera input
   - Confidence threshold adjustment
   - Face detection visualization

2. **Real-time Detection Tab**
   - Live camera monitoring
   - Performance metrics display
   - FPS and latency tracking

3. **Performance Analysis Tab**
   - Model performance visualization
   - Detailed metrics display
   - Model complexity analysis

4. **Model Comparison Tab**
   - Side-by-side model comparison
   - Performance benchmarking
   - Interactive visualizations

### Gradio Interface

The Gradio interface provides a simpler, more focused interface for quick testing.

#### Launching Gradio

```bash
python web_interface/gradio_app.py
```

#### Features

- **Simple Upload Interface**: Easy image upload and analysis
- **Model Configuration**: Quick model selection and loading
- **Results Display**: Clear visualization of detection results
- **Performance Metrics**: Basic performance information

## Real-time Detection

### Command Line Interface

```bash
# Basic real-time detection
python web_interface/real_time_detector.py

# With specific model
python web_interface/real_time_detector.py --model-type cnn --model-name resnet50

# With custom settings
python web_interface/real_time_detector.py \
    --model-type transformer \
    --model-name vit_base \
    --confidence-threshold 0.7 \
    --target-fps 30 \
    --camera 0
```

### Programmatic Usage

```python
from web_interface import RealTimeDetector

# Create detector
detector = RealTimeDetector(
    model_type="cnn",
    model_name="resnet50",
    confidence_threshold=0.5,
    target_fps=30
)

# Run detection
detector.run(camera_index=0)
```

### Controls

- **'q'**: Quit detection
- **'s'**: Save current frame
- **Mouse**: Interact with interface

## Dataset Creation

### Synthetic Dataset Generation

```python
from data import SyntheticConfig, SyntheticDatasetGenerator

# Configure dataset generation
config = SyntheticConfig(
    output_dir="data/my_synthetic_dataset",
    num_samples=1000,
    image_size=(224, 224),
    face_size_range=(80, 150),
    background_types=["solid", "gradient", "texture"],
    lighting_conditions=["normal", "bright", "dim"],
    compression_levels=[10, 30, 50, 70, 90],
    noise_levels=[0.0, 0.01, 0.02, 0.05, 0.1]
)

# Generate dataset
generator = SyntheticDatasetGenerator(config)
metadata = generator.generate_dataset()

# Visualize samples
generator.visualize_samples(num_samples=8)
```

### Custom Dataset Loading

```python
from data import DeepfakeDataset

# Load custom dataset
dataset = DeepfakeDataset(
    data_dir="path/to/your/dataset",
    metadata_file="metadata.json",  # Optional
    transform=your_transforms,
    class_names=["real", "fake"]
)
```

## Model Training

### Basic Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Setup training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
```

### Advanced Training with Augmentation

```python
from data import create_training_augmentations

# Create training augmentations
augmentations = create_training_augmentations(input_size=(224, 224))

# Use in training loop
for batch_idx, (data, target) in enumerate(train_loader):
    # Apply augmentations
    data, target = augmentations['mixup'](data, target)
    data = augmentations['adversarial'](data)
    
    # Continue training...
```

## Evaluation

### Comprehensive Evaluation

```python
from evaluation import BenchmarkSuite, ModelAnalyzer

# Create benchmark suite
benchmark_suite = BenchmarkSuite()

# Benchmark model
results = benchmark_suite.benchmark_model(
    model=model,
    dataloader=test_loader,
    device=device,
    model_name="my_model"
)

# Analyze model
analyzer = ModelAnalyzer()
complexity = analyzer.analyze_model_complexity(model)
confidence_analysis = analyzer.analyze_prediction_confidence(
    model=model,
    dataloader=test_loader,
    device=device
)
```

### Visualization

```python
from evaluation import create_visualization_suite

# Create visualization suite
visualizers = create_visualization_suite()

# Create performance plots
performance_plot = visualizers['detection_visualizer'].plot_metrics_comparison(
    metrics_data=results,
    metrics=['accuracy', 'precision', 'recall', 'f1_score']
)

# Create interactive visualizations
interactive_plot = visualizers['interactive_visualizer'].create_model_comparison_dashboard(
    model_results=comparison_results
)
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache
torch.cuda.empty_cache()
```

#### 2. Model Loading Errors

```python
# Check model compatibility
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Load with proper device
model = model.to(device)
model.eval()
```

#### 3. Dataset Loading Issues

```python
# Check dataset structure
import os
print("Dataset structure:")
for root, dirs, files in os.walk("data/synthetic_demo"):
    level = root.replace("data/synthetic_demo", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f"{subindent}{file}")
```

#### 4. Web Interface Issues

```bash
# Check if ports are available
netstat -an | grep 8501  # Streamlit
netstat -an | grep 7860  # Gradio

# Try different ports
streamlit run web_interface/streamlit_app.py --server.port 8502
```

### Performance Optimization

#### 1. Speed Up Inference

```python
# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    output = model(input_tensor)

# Use TensorRT (if available)
import torch_tensorrt
model_trt = torch_tensorrt.compile(model)
```

#### 2. Reduce Memory Usage

```python
# Use smaller model
model = create_cnn_model(model_name="efficientnet_b0")

# Use gradient accumulation
accumulation_steps = 4
for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Getting Help

#### Resources

1. **Documentation**: Check the docs/ directory for detailed guides
2. **Issues**: Report issues on the project repository
3. **Community**: Join the research community discussions
4. **Examples**: Look at example scripts and notebooks

#### Support

- **Email**: research@deepfake-detection.org
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides in docs/
- **Tutorials**: Step-by-step tutorials available

## Next Steps

### Learning Path

1. **Start Simple**: Begin with basic CNN models
2. **Explore Datasets**: Try different synthetic datasets
3. **Experiment**: Test various augmentation techniques
4. **Compare Models**: Use the comparison tools
5. **Advanced Features**: Explore ensemble methods and transformers

### Research Directions

1. **Model Improvement**: Develop better detection algorithms
2. **Real-time Optimization**: Optimize for live applications
3. **Robustness**: Improve resistance to adversarial attacks
4. **Generalization**: Better cross-domain performance
5. **Explainability**: Understand model decisions

### Contributing

1. **Code Contributions**: Submit improvements and bug fixes
2. **Documentation**: Help improve documentation
3. **Testing**: Test on different platforms and datasets
4. **Research**: Share research findings and insights
5. **Community**: Help build the research community

Welcome to the Deepfake Detection Research Platform! We hope this guide helps you get started with your research journey.
