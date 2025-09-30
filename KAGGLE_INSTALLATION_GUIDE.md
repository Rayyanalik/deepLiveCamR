# 🔧 Complete Installation Guide for Kaggle GPU

**Purpose**: All required packages and dependencies for running AI deepfake research files on Kaggle GPU  
**Target**: Kaggle Notebooks with GPU enabled  
**Date**: 2025-09-22  

---

## 🚀 **QUICK INSTALLATION (Copy & Paste)**

### **Cell 1: Core PyTorch with CUDA**
```python
# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install torch-audio
```

### **Cell 2: Hugging Face Ecosystem**
```python
# Hugging Face packages for text-to-video
!pip install diffusers>=0.20.0
!pip install transformers>=4.30.0
!pip install accelerate>=0.20.0
!pip install huggingface-hub>=0.15.0
```

### **Cell 3: Computer Vision & Face Detection**
```python
# Computer vision and face detection
!pip install opencv-python>=4.8.0
!pip install mediapipe>=0.10.0
!pip install face-recognition>=1.3.0
!pip install Pillow>=9.5.0
```

### **Cell 4: Deep Learning Models**
```python
# Advanced deep learning models
!pip install timm>=0.9.0
!pip install albumentations>=1.3.0
!pip install scikit-learn>=1.3.0
```

### **Cell 5: Video Processing**
```python
# Video processing and I/O
!pip install imageio>=2.31.0
!pip install imageio-ffmpeg>=0.4.8
!pip install moviepy
```

### **Cell 6: Data Processing & Visualization**
```python
# Data processing and visualization
!pip install numpy>=1.24.0
!pip install pandas>=2.0.0
!pip install matplotlib>=3.7.0
!pip install seaborn>=0.12.0
!pip install plotly>=5.15.0
```

### **Cell 7: Utilities & Performance**
```python
# Utilities and performance optimization
!pip install tqdm>=4.65.0
!pip install requests>=2.31.0
!pip install python-dotenv>=1.0.0
!pip install psutil
```

### **Cell 8: Optional Performance Boosters**
```python
# Optional but recommended for better performance
!pip install xformers>=0.0.20
!pip install flash-attn>=2.3.0
!pip install bitsandbytes
```

---

## 📋 **FILE-SPECIFIC DEPENDENCIES**

### **For `pyramid_working_research_system.py`:**
```python
# Required packages
!pip install diffusers transformers accelerate
!pip install opencv-python numpy
!pip install torch torchvision
```

### **For `models/cnn_detector.py`:**
```python
# Required packages
!pip install torch torchvision
!pip install timm
!pip install albumentations
!pip install scikit-learn
```

### **For `models/transformer_detector.py`:**
```python
# Required packages
!pip install torch torchvision
!pip install transformers
!pip install timm
!pip install einops
```

### **For `models/ensemble_detector.py`:**
```python
# Required packages
!pip install torch torchvision
!pip install scikit-learn
!pip install joblib
```

### **For `evaluation/benchmark.py`:**
```python
# Required packages
!pip install torch torchvision
!pip install psutil
!pip install matplotlib seaborn
```

### **For `utils/preprocessing.py`:**
```python
# Required packages
!pip install opencv-python
!pip install mediapipe
!pip install face-recognition
!pip install albumentations
!pip install Pillow
```

---

## 🎯 **COMPLETE KAGGLE NOTEBOOK SETUP**

### **Step 1: Environment Setup**
```python
# Cell 1: Environment and GPU check
import os
import torch
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

### **Step 2: Install All Packages**
```python
# Cell 2: Install all required packages
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install diffusers transformers accelerate huggingface-hub
!pip install opencv-python mediapipe face-recognition Pillow
!pip install timm albumentations scikit-learn
!pip install imageio imageio-ffmpeg moviepy
!pip install numpy pandas matplotlib seaborn plotly
!pip install tqdm requests python-dotenv psutil
!pip install xformers flash-attn bitsandbytes
```

### **Step 3: Verify Installation**
```python
# Cell 3: Verify all packages are installed
import torch
import diffusers
import transformers
import cv2
import mediapipe
import face_recognition
import timm
import albumentations
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ All packages installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Diffusers version: {diffusers.__version__}")
print(f"Transformers version: {transformers.__version__}")
```

### **Step 4: Set API Tokens**
```python
# Cell 4: Set your API tokens
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'
# Add other API tokens as needed
```

### **Step 5: Upload Your Files**
```python
# Cell 5: Upload your research files
# Method 1: Direct upload (for small files)
from google.colab import files
uploaded = files.upload()

# Method 2: Use Kaggle dataset (recommended)
# Create a dataset with all your files and add it to the notebook
```

### **Step 6: Run Your Research**
```python
# Cell 6: Run your research system
import sys
sys.path.append('/kaggle/working')

# Import your research system
from CORE_RESEARCH_FILES.pyramid_working_research_system import PyramidWorkingResearchSystem

# Initialize and run
system = PyramidWorkingResearchSystem(os.getenv('HF_TOKEN'))
prompts = ["A person cooking dinner in the kitchen"]
videos, results = system.generate_research_videos(prompts)
```

---

## ⚡ **PERFORMANCE OPTIMIZATION**

### **GPU Memory Management:**
```python
# Clear GPU cache regularly
import torch
torch.cuda.empty_cache()

# Optimize GPU settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

### **Memory Monitoring:**
```python
# Monitor GPU memory usage
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

print_gpu_memory()
```

---

## 🚨 **TROUBLESHOOTING**

### **Common Issues:**

1. **CUDA Out of Memory:**
   ```python
   torch.cuda.empty_cache()
   # Reduce batch size or model size
   ```

2. **Package Conflicts:**
   ```python
   # Restart runtime and reinstall
   !pip install --upgrade --force-reinstall package_name
   ```

3. **Import Errors:**
   ```python
   # Check if package is installed
   !pip list | grep package_name
   ```

4. **GPU Not Detected:**
   ```python
   # Verify GPU is enabled in Kaggle settings
   import torch
   print(torch.cuda.is_available())
   ```

---

## 📊 **EXPECTED INSTALLATION TIME**

- **Core PyTorch**: 2-3 minutes
- **Hugging Face packages**: 3-5 minutes
- **Computer Vision**: 2-4 minutes
- **Deep Learning models**: 4-6 minutes
- **Video processing**: 1-2 minutes
- **Data processing**: 1-2 minutes
- **Utilities**: 1 minute

**Total Installation Time**: 15-25 minutes

---

## 🎯 **QUICK START COMMANDS**

### **One-Cell Installation (All Packages):**
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
pip install diffusers transformers accelerate huggingface-hub && \
pip install opencv-python mediapipe face-recognition Pillow && \
pip install timm albumentations scikit-learn && \
pip install imageio imageio-ffmpeg moviepy && \
pip install numpy pandas matplotlib seaborn plotly && \
pip install tqdm requests python-dotenv psutil && \
pip install xformers flash-attn bitsandbytes
```

### **Verify Installation:**
```python
import torch, diffusers, transformers, cv2, mediapipe, face_recognition, timm, albumentations, imageio, numpy, pandas, matplotlib, seaborn, plotly, tqdm, requests
print("✅ All packages installed successfully!")
```

---

**Ready to run your AI deepfake research on Kaggle GPU! 🚀**
