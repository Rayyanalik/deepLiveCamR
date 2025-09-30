# 🚀 Google Colab Setup Guide for AI Deepfake Research

**Purpose**: Complete setup guide for running AI deepfake research on Google Colab with GPU  
**Target**: Google Colab Notebooks with GPU enabled  
**Date**: 2025-09-22  

---

## 🎯 **QUICK START (5 Minutes)**

### **Step 1: Enable GPU in Colab**
1. Go to: https://colab.research.google.com/
2. Create new notebook
3. Go to **Runtime** → **Change runtime type**
4. Set **Hardware accelerator** to **GPU (T4)**
5. Click **Save**

### **Step 2: One-Cell Installation**
```python
# Copy this entire block into one Colab cell and run:
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install diffusers transformers accelerate huggingface-hub
!pip install opencv-python mediapipe face-recognition Pillow
!pip install timm albumentations scikit-learn
!pip install imageio imageio-ffmpeg moviepy
!pip install numpy pandas matplotlib seaborn plotly
!pip install tqdm requests python-dotenv psutil
!pip install xformers flash-attn bitsandbytes

# Set environment variables
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print("✅ Colab GPU setup complete!")
```

### **Step 3: Upload Your Files**
```python
# Upload your research files
from google.colab import files
uploaded = files.upload()

# Or use Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

### **Step 4: Run Your Research**
```python
# Set your Hugging Face token
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'

# Import and run your system
import sys
sys.path.append('/content')

from CORE_RESEARCH_FILES.pyramid_working_research_system import PyramidWorkingResearchSystem

# Initialize and run
system = PyramidWorkingResearchSystem(os.getenv('HF_TOKEN'))
prompts = ["A person cooking dinner in the kitchen"]
videos, results = system.generate_research_videos(prompts)
```

---

## 🔧 **DETAILED SETUP GUIDE**

### **Cell 1: Environment Setup**
```python
# Check Colab environment
import os
import torch
import warnings
warnings.filterwarnings('ignore')

print("🚀 Google Colab AI Deepfake Research Setup")
print("=" * 50)

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ GPU not available - enable GPU in Runtime settings")

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

### **Cell 2: Install Core PyTorch**
```python
# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
print("✅ PyTorch installed")
```

### **Cell 3: Install Hugging Face Packages**
```python
# Install Hugging Face ecosystem
!pip install diffusers>=0.20.0
!pip install transformers>=4.30.0
!pip install accelerate>=0.20.0
!pip install huggingface-hub>=0.15.0
print("✅ Hugging Face packages installed")
```

### **Cell 4: Install Computer Vision**
```python
# Install computer vision packages
!pip install opencv-python>=4.8.0
!pip install mediapipe>=0.10.0
!pip install face-recognition>=1.3.0
!pip install Pillow>=9.5.0
print("✅ Computer vision packages installed")
```

### **Cell 5: Install Deep Learning Models**
```python
# Install deep learning model packages
!pip install timm>=0.9.0
!pip install albumentations>=1.3.0
!pip install scikit-learn>=1.3.0
print("✅ Deep learning packages installed")
```

### **Cell 6: Install Video Processing**
```python
# Install video processing packages
!pip install imageio>=2.31.0
!pip install imageio-ffmpeg>=0.4.8
!pip install moviepy
print("✅ Video processing packages installed")
```

### **Cell 7: Install Data Processing**
```python
# Install data processing packages
!pip install numpy>=1.24.0
!pip install pandas>=2.0.0
!pip install matplotlib>=3.7.0
!pip install seaborn>=0.12.0
!pip install plotly>=5.15.0
print("✅ Data processing packages installed")
```

### **Cell 8: Install Utilities**
```python
# Install utility packages
!pip install tqdm>=4.65.0
!pip install requests>=2.31.0
!pip install python-dotenv>=1.0.0
!pip install psutil
print("✅ Utility packages installed")
```

### **Cell 9: Install Performance Boosters**
```python
# Install performance optimization packages
!pip install xformers>=0.0.20
!pip install flash-attn>=2.3.0
!pip install bitsandbytes
print("✅ Performance packages installed")
```

### **Cell 10: Verify Installation**
```python
# Verify all packages are installed
try:
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
    import plotly
    
    print("✅ All packages installed successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Diffusers version: {diffusers.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
```

---

## 📁 **FILE UPLOAD METHODS**

### **Method 1: Direct Upload**
```python
# Upload files directly to Colab
from google.colab import files
uploaded = files.upload()

# List uploaded files
for filename in uploaded.keys():
    print(f"Uploaded: {filename}")
```

### **Method 2: Google Drive Integration**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy files from Drive to Colab
!cp -r /content/drive/MyDrive/your_project_folder/* /content/
```

### **Method 3: GitHub Integration**
```python
# Clone from GitHub
!git clone https://github.com/yourusername/your-repo.git
!cp -r your-repo/* /content/
```

### **Method 4: Zip File Upload**
```python
# Upload zip file
from google.colab import files
uploaded = files.upload()

# Extract zip file
import zipfile
with zipfile.ZipFile('your_project.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')
```

---

## 🎬 **RUNNING YOUR RESEARCH SYSTEM**

### **Cell 11: Set API Tokens**
```python
# Set your API tokens
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'
# Add other API tokens as needed
```

### **Cell 12: Import and Initialize System**
```python
# Import your research system
import sys
sys.path.append('/content')

from CORE_RESEARCH_FILES.pyramid_working_research_system import PyramidWorkingResearchSystem

# Initialize system
system = PyramidWorkingResearchSystem(os.getenv('HF_TOKEN'))
print("✅ Research system initialized")
```

### **Cell 13: Generate Videos**
```python
# Generate research videos
prompts = [
    "A person cooking dinner in the kitchen",
    "Someone reading a book on the couch",
    "A person watering plants in the garden"
]

print("🎬 Starting video generation...")
videos, results = system.generate_research_videos(prompts)
print(f"✅ Generated {len(videos)} videos")
```

### **Cell 14: Download Results**
```python
# Download generated videos and reports
from google.colab import files
import os

# Download videos
for video in videos:
    if os.path.exists(video['video_path']):
        files.download(video['video_path'])

# Download reports
for file in os.listdir('pyramid_working_research_output/reports/'):
    files.download(f'pyramid_working_research_output/reports/{file}')
```

---

## ⚡ **COLAB GPU OPTIMIZATION**

### **GPU Memory Management**
```python
# Clear GPU cache regularly
import torch
torch.cuda.empty_cache()

# Monitor GPU memory
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

print_gpu_memory()
```

### **Performance Optimization**
```python
# Optimize GPU settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Set mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

---

## 🚨 **TROUBLESHOOTING**

### **Common Issues:**

1. **GPU Not Available:**
   - Go to Runtime → Change runtime type → GPU
   - Restart runtime if needed

2. **CUDA Out of Memory:**
   ```python
   torch.cuda.empty_cache()
   # Reduce batch size or model size
   ```

3. **Package Installation Errors:**
   ```python
   # Restart runtime and reinstall
   !pip install --upgrade --force-reinstall package_name
   ```

4. **File Upload Issues:**
   ```python
   # Use Google Drive instead
   from google.colab import drive
   drive.mount('/content/drive')
   ```

---

## 📊 **COLAB vs KAGGLE COMPARISON**

| Feature | Google Colab | Kaggle |
|---------|-------------|--------|
| **GPU Access** | T4 (Free), V100/A100 (Pro) | T4/P100 (Free), V100/A100 (Pro) |
| **Session Time** | 12 hours (Free), 24 hours (Pro) | 9 hours (Free), 20 hours (Pro) |
| **Setup Difficulty** | ⭐⭐ Easy | ⭐⭐⭐ Medium |
| **File Upload** | ⭐⭐⭐ Very Easy | ⭐⭐ Easy |
| **Performance** | ⭐⭐⭐ Excellent | ⭐⭐⭐ Excellent |
| **Cost** | Free/Pro $10/month | Free/Pro $5/month |

---

## 🎯 **QUICK START COMMANDS**

### **One-Cell Complete Setup:**
```python
# Copy this entire block into one Colab cell:
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
pip install diffusers transformers accelerate huggingface-hub && \
pip install opencv-python mediapipe face-recognition Pillow && \
pip install timm albumentations scikit-learn && \
pip install imageio imageio-ffmpeg moviepy && \
pip install numpy pandas matplotlib seaborn plotly && \
pip install tqdm requests python-dotenv psutil && \
pip install xformers flash-attn bitsandbytes

import os, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(f"CUDA available: {torch.cuda.is_available()}")
print("✅ Colab setup complete!")
```

### **Upload and Run:**
```python
# Upload files
from google.colab import files
files.upload()

# Set token and run
os.environ['HF_TOKEN'] = 'your_token_here'
from CORE_RESEARCH_FILES.pyramid_working_research_system import PyramidWorkingResearchSystem
system = PyramidWorkingResearchSystem(os.getenv('HF_TOKEN'))
videos, results = system.generate_research_videos(["A person cooking dinner"])
```

---

**🚀 Ready to run your AI deepfake research on Google Colab GPU!**
