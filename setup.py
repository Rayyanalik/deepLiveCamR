"""
Setup script for Deepfake Generation and Detection System

This script helps set up the environment and download required models.
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import tarfile
from pathlib import Path


def install_requirements():
    """Install required Python packages."""
    print("📦 Installing Python requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def download_dlib_model():
    """Download dlib face landmark model."""
    print("📥 Downloading dlib face landmark model...")
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "shape_predictor_68_face_landmarks.dat"
    
    if model_path.exists():
        print("✅ dlib model already exists")
        return True
    
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_path = models_dir / "shape_predictor_68_face_landmarks.dat.bz2"
    
    try:
        print("Downloading compressed model...")
        urllib.request.urlretrieve(model_url, compressed_path)
        
        print("Extracting model...")
        import bz2
        with bz2.BZ2File(compressed_path, 'rb') as source:
            with open(model_path, 'wb') as target:
                target.write(source.read())
        
        # Clean up compressed file
        compressed_path.unlink()
        
        print("✅ dlib model downloaded and extracted")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download dlib model: {e}")
        print("Please manually download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return False


def create_sample_files():
    """Create sample files and directories."""
    print("📁 Creating sample files and directories...")
    
    # Create directories
    directories = [
        "data/source_videos",
        "data/generated", 
        "data/test_data",
        "models",
        "notebooks",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create sample README for data directory
    data_readme = Path("data/README.md")
    if not data_readme.exists():
        data_readme.write_text("""
# Data Directory

This directory contains the data used for deepfake generation and detection.

## Structure

- `source_videos/`: Original videos and images used as source material
- `generated/`: Generated deepfake videos
- `test_data/`: Test datasets for evaluation

## Required Files

To use the system, you need:

1. **Source Face Image** (`source_videos/source_face.jpg`): A clear image of the face to be swapped
2. **Target Video** (`source_videos/target_video.mp4`): A video containing the target person performing mundane tasks

## Sample Data

You can use any publicly available videos or images for testing. Make sure you have the rights to use the content.

## Ethical Guidelines

- Only use content you have permission to use
- Clearly mark generated content as synthetic
- Follow ethical AI research guidelines
- Do not use for malicious purposes
""")
    
    print("✅ Sample files and directories created")


def check_system_requirements():
    """Check system requirements."""
    print("🔍 Checking system requirements...")
    
    requirements = {
        "Python": sys.version_info >= (3, 8),
        "OpenCV": True,  # Will be checked during import
        "PyTorch": True,  # Will be checked during import
    }
    
    # Check Python version
    if not requirements["Python"]:
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available, will use CPU")
    except ImportError:
        print("⚠️  PyTorch not installed yet")
    
    return True


def create_virtual_environment():
    """Create a virtual environment for the project."""
    print("🐍 Creating virtual environment...")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    try:
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created")
        print("To activate: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Deepfake Generation and Detection System Setup")
    print("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met")
        return
    
    # Create virtual environment (optional)
    create_virtual_env = input("Create virtual environment? (y/n): ").lower().strip() == 'y'
    if create_virtual_env:
        create_virtual_environment()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return
    
    # Download models
    if not download_dlib_model():
        print("⚠️  dlib model download failed, but you can continue")
    
    # Create sample files
    create_sample_files()
    
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add source face image to: data/source_videos/source_face.jpg")
    print("2. Add target video to: data/source_videos/target_video.mp4")
    print("3. Run: python example_usage.py")
    print("\nFor real-time webcam simulation:")
    print("1. Install OBS Studio")
    print("2. Add OBS Virtual Camera plugin")
    print("3. Run the real-time simulation")
    
    print("\n⚠️  Important Notes:")
    print("- This is for research and educational purposes only")
    print("- Follow ethical guidelines when using deepfake technology")
    print("- Ensure you have permission to use any source materials")
    print("- Always clearly mark generated content as synthetic")


if __name__ == "__main__":
    main()
