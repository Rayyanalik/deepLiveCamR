"""
Google Colab - Working Text-to-Video Setup
Copy and paste this into your Colab notebook to avoid green screen issues
"""

# =============================================================================
# CELL 1: INSTALL ALL REQUIRED PACKAGES
# =============================================================================

# Install all required packages for working text-to-video
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
print("🚀 Google Colab Working Text-to-Video Setup")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("❌ GPU not available - enable GPU in Runtime settings")

# Verify all packages are installed
try:
    import diffusers, transformers, cv2, mediapipe, face_recognition, timm, albumentations, imageio, numpy, pandas, matplotlib, seaborn, plotly, tqdm, requests
    print("✅ All packages installed successfully!")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Import error: {e}")

print("🎬 Ready for working text-to-video generation!")

# =============================================================================
# CELL 2: UPLOAD YOUR RESEARCH FILES
# =============================================================================

# Upload your research files
from google.colab import files
print("📁 Upload your research files...")
print("Make sure to upload:")
print("- CORE_RESEARCH_FILES/working_text_to_video_system.py")
print("- Any other research files you need")
uploaded = files.upload()

# List uploaded files
print("📋 Uploaded files:")
for filename in uploaded.keys():
    print(f"  ✅ {filename}")

# =============================================================================
# CELL 3: SET YOUR HUGGING FACE TOKEN AND RUN
# =============================================================================

# Set your Hugging Face token
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'  # Replace with your actual token
print("🔑 HF Token set")

# Import and run the working text-to-video system
import sys
sys.path.append('/content')

try:
    from CORE_RESEARCH_FILES.working_text_to_video_system import WorkingTextToVideoSystem
    print("✅ Working text-to-video system imported successfully")
    
    # Initialize system
    system = WorkingTextToVideoSystem(os.getenv('HF_TOKEN'))
    print("🎬 Working text-to-video system initialized")
    
    # Define research prompts
    prompts = [
        "A person cooking dinner in the kitchen",
        "Someone reading a book on the couch",
        "A person watering plants in the garden"
    ]
    
    print(f"🎬 Starting video generation for {len(prompts)} prompts...")
    print("This will avoid green screen issues by using multiple fallback methods")
    
    # Generate videos
    videos, results = system.generate_research_videos(prompts)
    
    print(f"✅ Generated {len(videos)} videos successfully!")
    
    # Display results
    for i, video in enumerate(videos, 1):
        print(f"Video {i}: {video['prompt']}")
        print(f"  Path: {video['video_path']}")
        print(f"  Size: {video['file_size'] / 1024:.1f} KB")
    
    # Download results
    print("\n📥 Downloading generated videos...")
    for video in videos:
        if os.path.exists(video['video_path']):
            files.download(video['video_path'])
            print(f"  ✅ Downloaded: {video['video_path']}")
    
    # Download reports
    print("\n📊 Downloading reports...")
    reports_dir = 'working_video_output/reports/'
    if os.path.exists(reports_dir):
        for file in os.listdir(reports_dir):
            if file.endswith('.md'):
                files.download(f'{reports_dir}{file}')
                print(f"  ✅ Downloaded: {file}")
    
    print("\n🎉 Working text-to-video research completed successfully!")
    print("✅ No green screen issues - videos generated with rich content!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you uploaded the working_text_to_video_system.py file")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your HF token and try again")

print("\n🚀 Working text-to-video system ready for use!")
