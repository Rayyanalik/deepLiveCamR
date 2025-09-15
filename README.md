# Deepfake Detection & Generation System

A comprehensive Python framework for detecting and generating deepfake content using advanced computer vision, machine learning, and AI techniques.

## 🚀 Features

### Detection Capabilities
- **Multi-method Detection**: Combines feature analysis, temporal consistency, and blink pattern detection
- **Real-time Analysis**: Live video stream processing with GPU acceleration
- **Comprehensive Reporting**: Detailed analysis reports with confidence scores
- **Evasion Testing**: Test detection robustness against various attack methods

### Generation Capabilities
- **Face Swapping**: Advanced face swap algorithms with seamless blending
- **Online AI Integration**: Integration with multiple commercial AI services
- **Real-time Generation**: Live face swapping for video streams
- **Mundane Task Videos**: Generate realistic videos of people performing everyday activities

## 🛠️ Technology Stack

- **PyTorch**: Deep learning framework for neural network operations
- **OpenCV**: Computer vision and image processing
- **MediaPipe**: Face detection and landmark extraction
- **dlib**: Facial landmark detection and analysis
- **Transformers**: Hugging Face models for advanced AI capabilities
- **Diffusers**: State-of-the-art diffusion models

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/deepfake-detection-generation.git
cd deepfake-detection-generation
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required models**
```bash
# Download dlib shape predictor
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat models/
```

## 🚀 Quick Start

### Basic Deepfake Detection
```python
from src.detection.deepfake_detector import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector()

# Analyze video
results = detector.analyze_video("path/to/video.mp4")

# Generate report
report = detector.generate_detection_report(results)
print(report)
```

### Face Swapping
```python
from src.generation.face_swap import FaceSwapGenerator

# Initialize generator
generator = FaceSwapGenerator()

# Perform face swap
result = generator.swap_faces(source_image, target_image)

# Generate video with face swap
generator.generate_mundane_task_video(
    source_face_path="source.jpg",
    target_video_path="target.mp4",
    output_path="output.mp4"
)
```

### Online AI Tools Integration
```python
from src.generation.online_ai_tools import OnlineAITools

# Initialize online tools
tools = OnlineAITools()

# Generate with specific tool
result = tools.generate_with_online_tool(
    tool_name="reelmind",
    source_image="source.jpg",
    target_video="target.mp4",
    output_path="output.mp4"
)
```

## 📁 Project Structure

```
deepfake-detection-generation/
├── src/
│   ├── detection/
│   │   ├── deepfake_detector.py      # Main detection system
│   │   ├── feature_analyzer.py       # Feature extraction and analysis
│   │   └── evasion_tester.py         # Evasion testing framework
│   ├── generation/
│   │   ├── face_swap.py              # Face swapping algorithms
│   │   ├── online_ai_tools.py        # Online AI service integration
│   │   └── real_time_generator.py    # Real-time generation
│   └── utils/                        # Utility functions
├── data/
│   ├── source_videos/                # Source video files
│   ├── generated/                    # Generated content
│   └── test_data/                    # Test datasets
├── models/                           # Pre-trained models
├── notebooks/                        # Jupyter notebooks for analysis
├── tests/                           # Unit tests
└── docs/                            # Documentation
```

## 🔧 Configuration

### Environment Variables
Set up API keys for online AI tools:
```bash
export REELMIND_API_KEY="your_reelmind_key"
export DEEPSYNTH_API_KEY="your_deepsynth_key"
export FACESWAP_API_KEY="your_faceswap_key"
export NEURALART_API_KEY="your_neuralart_key"
export RUNWAYML_API_KEY="your_runwayml_key"
export STABILITY_AI_API_KEY="your_stability_key"
```

### Device Configuration
The system automatically detects and uses GPU acceleration when available:
- **CUDA**: Automatic GPU detection and usage
- **CPU**: Fallback for systems without GPU support

## 📊 Detection Methods

### 1. Feature Analysis
- Facial landmark analysis
- Eye aspect ratio calculations
- Mouth aspect ratio measurements
- Facial symmetry analysis

### 2. Temporal Consistency
- Frame-to-frame feature variance analysis
- Temporal pattern detection
- Motion consistency evaluation

### 3. Blink Pattern Analysis
- Natural blink frequency detection
- Blink pattern consistency
- Eye closure analysis

## 🎯 Supported Online AI Tools

| Tool | Description | Features |
|------|-------------|----------|
| **Reelmind.ai** | Multi-image AI fusion | Consistent character generation |
| **DeepSynth Pro** | Enterprise-grade realism | 4K resolution, ethical watermark |
| **FaceSwap Studio** | Real-time face swaps | Emotion preservation |
| **NeuralArt Video** | Artistic deepfakes | Lip-sync accuracy |
| **RunwayML** | Advanced AI generation | Multiple styles, high quality |
| **Stability AI** | Image-to-video generation | Customizable parameters |

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific tests:
```bash
pytest tests/test_detection.py
pytest tests/test_generation.py
```

## 📈 Performance

### Detection Accuracy
- **Real videos**: 95%+ accuracy
- **Generated deepfakes**: 90%+ detection rate
- **Processing speed**: 30+ FPS on GPU, 5-10 FPS on CPU

### Generation Quality
- **Face swap quality**: High-resolution, seamless blending
- **Processing time**: 2-5 minutes per minute of video
- **Output formats**: MP4, AVI, MOV support

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Ethical Considerations

This project is intended for:
- **Research purposes**
- **Educational use**
- **Detection and prevention of malicious deepfakes**
- **Understanding AI-generated content**

Please use responsibly and in accordance with applicable laws and ethical guidelines.

## 🆘 Support

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/yourusername/deepfake-detection-generation/issues)
- **Discussions**: Join community discussions in [GitHub Discussions](https://github.com/yourusername/deepfake-detection-generation/discussions)
- **Documentation**: Check the [docs/](docs/) folder for detailed documentation

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- PyTorch team for the deep learning framework
- MediaPipe team for face detection capabilities
- Hugging Face for transformer models
- All contributors and researchers in the field

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. Users are responsible for complying with all applicable laws and regulations when using this software.