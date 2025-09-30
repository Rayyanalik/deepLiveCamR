"""
One-Cell Installation Script for Kaggle GPU
Copy and paste this entire cell into your Kaggle notebook
"""

# =============================================================================
# KAGGLE GPU - ONE CELL INSTALLATION
# =============================================================================

# Install all required packages for AI deepfake research
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install diffusers>=0.20.0 transformers>=4.30.0 accelerate>=0.20.0 huggingface-hub>=0.15.0
!pip install opencv-python>=4.8.0 mediapipe>=0.10.0 face-recognition>=1.3.0 Pillow>=9.5.0
!pip install timm>=0.9.0 albumentations>=1.3.0 scikit-learn>=1.3.0
!pip install imageio>=2.31.0 imageio-ffmpeg>=0.4.8 moviepy
!pip install numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 seaborn>=0.12.0 plotly>=5.15.0
!pip install tqdm>=4.65.0 requests>=2.31.0 python-dotenv>=1.0.0 psutil
!pip install xformers>=0.0.20 flash-attn>=2.3.0 bitsandbytes

# Set environment variables
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Verify all packages are installed
try:
    import diffusers, transformers, cv2, mediapipe, face_recognition, timm, albumentations, imageio, numpy, pandas, matplotlib, seaborn, plotly, tqdm, requests
    print("✅ All packages installed successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Import error: {e}")

print("🚀 Ready for AI deepfake research on Kaggle GPU!")
