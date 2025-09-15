# Real-time Deepfake Generation Guide

This guide explains how to create videos of people doing mundane tasks in near real-time that mimic a webcam feed.

## 🎯 Overview

The real-time generation system allows you to:
- Generate videos of people performing mundane tasks (typing, reading, drinking coffee, etc.)
- Output the generated content as a virtual webcam
- Achieve near real-time performance with low latency
- Seamlessly integrate with video conferencing applications

## 🚀 Quick Start

### 1. Installation

```bash
# Install real-time specific requirements
pip install -r requirements_realtime.txt

# Install virtual camera driver (Windows)
# Download and install OBS Virtual Camera or similar

# For macOS/Linux, pyvirtualcam should work out of the box
```

### 2. Basic Usage

```python
from src.realtime.realtime_generator import RealtimeGenerator

# Initialize generator
generator = RealtimeGenerator(target_fps=30)

# Start generation with a source face
result = generator.start_generation(
    source_face_path="path/to/your/face.jpg",
    task_name='typing',
    virtual_camera=True
)

# The generated content will now appear as a webcam
# You can use it in Zoom, Teams, Discord, etc.
```

### 3. Run Demo

```bash
python realtime_demo.py
```

## 🎭 Available Mundane Tasks

The system includes several pre-built mundane task templates:

| Task | Description | Duration | FPS |
|------|-------------|----------|-----|
| `typing` | Person typing on keyboard | 2.0s | 15 |
| `reading` | Person reading a book | 3.0s | 13.3 |
| `drinking_coffee` | Person drinking coffee | 2.5s | 10 |
| `looking_at_phone` | Person looking at phone | 2.8s | 12.5 |
| `nodding` | Person nodding | 1.5s | 13.3 |
| `smiling` | Person smiling | 2.0s | 15 |
| `yawning` | Person yawning | 2.0s | 12.5 |

## 🔧 Configuration

### Performance Optimization

```python
# For high-performance systems
generator = RealtimeGenerator(
    device='cuda',  # Use GPU if available
    target_fps=60   # Higher FPS for smoother output
)

# For lower-end systems
generator = RealtimeGenerator(
    device='cpu',
    target_fps=15   # Lower FPS to reduce load
)
```

### Virtual Camera Settings

```python
from src.realtime.virtual_camera import VirtualCamera

# Create custom virtual camera
camera = VirtualCamera(
    width=1280,   # Higher resolution
    height=720,
    fps=30
)

# Start with custom device name
camera.start(device_name="DeepfakeCam")
```

## 📱 Integration with Video Apps

### Zoom Integration

1. Start the real-time generator
2. Open Zoom
3. Go to Settings > Video
4. Select "DeepfakeCam" or your virtual camera
5. The generated content will appear as your webcam feed

### Discord Integration

1. Start the generator
2. Open Discord
3. Go to User Settings > Voice & Video
4. Select your virtual camera
5. Start a video call

### Teams Integration

1. Start the generator
2. Open Microsoft Teams
3. Go to Settings > Devices
4. Select your virtual camera
5. Join a meeting

## 🎨 Creating Custom Tasks

### Method 1: Using Frame Sequences

```python
from src.realtime.mundane_tasks import MundaneTaskTemplates
import cv2

# Load your custom frames
frames = []
for i in range(30):
    frame = cv2.imread(f"custom_task/frame_{i:03d}.jpg")
    frames.append(frame)

# Add custom task
tasks = MundaneTaskTemplates()
tasks.add_custom_task(
    name="custom_action",
    frames=frames,
    description="Custom mundane action"
)

# Use in generator
generator.change_task("custom_action")
```

### Method 2: Programmatic Generation

```python
import numpy as np
import cv2

def create_custom_task():
    frames = []
    for i in range(20):
        # Create frame programmatically
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
        
        # Add animated elements
        x = 300 + int(20 * np.sin(i * 0.3))
        y = 200 + int(10 * np.cos(i * 0.2))
        cv2.circle(frame, (x, y), 15, (100, 100, 100), -1)
        
        frames.append(frame)
    
    return frames

# Add to tasks
frames = create_custom_task()
tasks.add_custom_task("animated_circle", frames, "Animated circle task")
```

## ⚡ Performance Optimization

### GPU Acceleration

```python
# Check if CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Use GPU for better performance
generator = RealtimeGenerator(device='cuda')
```

### Memory Management

```python
# For long-running sessions
generator.set_target_fps(15)  # Reduce FPS to save memory

# Monitor performance
stats = generator.get_performance_stats()
print(f"Current FPS: {stats['current_fps']:.1f}")
print(f"Memory usage: {stats.get('memory_usage', 'N/A')}")
```

### Latency Reduction

```python
# Use smaller frame sizes for lower latency
camera = VirtualCamera(width=320, height=240, fps=30)

# Reduce processing complexity
generator = RealtimeGenerator(target_fps=15)
```

## 🛠️ Troubleshooting

### Common Issues

#### Virtual Camera Not Working

```bash
# Windows: Install OBS Studio and enable Virtual Camera
# macOS: pyvirtualcam should work automatically
# Linux: May need additional drivers

# Check available devices
python -c "import pyvirtualcam; print(pyvirtualcam.Camera.devices)"
```

#### Low Performance

```python
# Reduce FPS
generator.set_target_fps(15)

# Use CPU instead of GPU
generator = RealtimeGenerator(device='cpu')

# Check system resources
import psutil
print(f"CPU usage: {psutil.cpu_percent()}%")
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

#### Face Detection Issues

```python
# Ensure source face is clear and well-lit
# Use high-resolution source images
# Check face detection in the source image

from src.generation.face_swap import FaceSwapGenerator
generator = FaceSwapGenerator()
faces = generator.detect_faces(source_image)
print(f"Faces detected: {len(faces)}")
```

### Performance Monitoring

```python
# Real-time performance monitoring
import time

start_time = time.time()
frame_count = 0

while generator.is_running:
    stats = generator.get_performance_stats()
    current_fps = stats.get('current_fps', 0)
    
    if current_fps < target_fps * 0.8:  # 80% of target
        print(f"⚠️  Low FPS: {current_fps:.1f}")
    
    time.sleep(1)
```

## 📊 Advanced Features

### Multi-Camera Setup

```python
from src.realtime.virtual_camera import CameraManager

# Create multiple virtual cameras
manager = CameraManager()
manager.create_camera("cam1", 640, 480, 30)
manager.create_camera("cam2", 1280, 720, 30)

# Start both cameras
manager.start_camera("cam1")
manager.start_camera("cam2")

# Send frames to different cameras
manager.send_frame_to_camera("cam1", frame1)
manager.send_frame_to_camera("cam2", frame2)
```

### Recording and Streaming

```python
# Record the generated content
result = generator.start_generation(
    source_face_path="face.jpg",
    task_name='typing',
    virtual_camera=True,
    output_path="output/recorded_session.mp4"
)

# The video will be saved while also streaming to virtual camera
```

### Dynamic Task Switching

```python
# Change tasks during generation
generator.change_task('reading')
time.sleep(5)
generator.change_task('drinking_coffee')
time.sleep(5)
generator.change_task('typing')
```

## 🔒 Security and Ethics

### Responsible Use

- Only use with consent from the person whose face you're using
- Don't use for malicious purposes
- Be transparent about the use of deepfake technology
- Respect privacy and intellectual property rights

### Technical Safeguards

```python
# Add watermarks to generated content
def add_watermark(frame):
    cv2.putText(frame, "GENERATED", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return frame

# Apply watermark in generation loop
frame = generator._generate_frame()
frame = add_watermark(frame)
```

## 📈 Future Enhancements

### Planned Features

- Real-time emotion detection and transfer
- Voice synthesis integration
- Advanced facial expression mapping
- Multi-person scene generation
- Real-time style transfer
- Advanced post-processing effects

### Contributing

To contribute to the real-time generation system:

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Submit a pull request

## 📚 Additional Resources

- [PyVirtualCam Documentation](https://github.com/letmaik/pyvirtualcam)
- [OpenCV Real-time Processing](https://opencv.org/)
- [PyTorch Optimization Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html)

---

**⚠️ Disclaimer**: This technology should be used responsibly and ethically. Always obtain proper consent and follow applicable laws and regulations.
