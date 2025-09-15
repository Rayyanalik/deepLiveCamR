"""
Comprehensive Test Script for Deepfake Generation and Detection System

This script tests all components of the system to ensure proper functionality.
"""

import os
import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from generation.face_swap import FaceSwapGenerator
from generation.real_time_generator import RealTimeGenerator
from detection.deepfake_detector import DeepfakeDetector
from detection.feature_analyzer import FeatureAnalyzer


def create_test_data():
    """Create test data for system testing."""
    print("🧪 Creating test data...")
    
    # Create test directories
    test_dir = Path("data/test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(test_dir / "test_image.jpg"), test_image)
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(test_dir / "test_video.mp4"), fourcc, 30.0, (640, 480))
    
    for i in range(90):  # 3 seconds at 30 FPS
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add some variation to make it more realistic
        frame[:, :, 0] = (frame[:, :, 0] + i * 2) % 255
        out.write(frame)
    
    out.release()
    
    print("✅ Test data created")
    return test_dir


def test_face_swap_generator():
    """Test face swap generator functionality."""
    print("\n🔍 Testing Face Swap Generator...")
    
    try:
        # Initialize generator
        generator = FaceSwapGenerator(device='auto')
        print(f"✅ Generator initialized on device: {generator.device}")
        
        # Test face detection
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = generator.detect_faces(test_image)
        print(f"✅ Face detection test completed: {len(faces)} faces detected")
        
        # Test landmark extraction (will fail without real face, but should not crash)
        try:
            landmarks = generator.extract_face_landmarks(test_image)
            print(f"✅ Landmark extraction test: {'Success' if landmarks is not None else 'No face detected (expected)'}")
        except Exception as e:
            print(f"⚠️  Landmark extraction test: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Face swap generator test failed: {e}")
        return False


def test_feature_analyzer():
    """Test feature analyzer functionality."""
    print("\n🔍 Testing Feature Analyzer...")
    
    try:
        # Initialize analyzer
        analyzer = FeatureAnalyzer()
        print("✅ Feature analyzer initialized")
        
        # Test with random image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test feature analysis
        results = analyzer.analyze_frame_features(test_image)
        print(f"✅ Feature analysis test completed: {len(results)} features analyzed")
        
        # Test deepfake probability calculation
        probability = analyzer.calculate_deepfake_probability(results)
        print(f"✅ Deepfake probability calculation: {probability:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature analyzer test failed: {e}")
        return False


def test_deepfake_detector():
    """Test deepfake detector functionality."""
    print("\n🔍 Testing Deepfake Detector...")
    
    try:
        # Initialize detector
        detector = DeepfakeDetector(device='auto')
        print(f"✅ Detector initialized on device: {detector.device}")
        
        # Test real-time detection
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect_in_real_time(test_image)
        print(f"✅ Real-time detection test: {result['overall_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Deepfake detector test failed: {e}")
        return False


def test_video_analysis():
    """Test video analysis functionality."""
    print("\n🔍 Testing Video Analysis...")
    
    try:
        # Create test data
        test_dir = create_test_data()
        test_video = test_dir / "test_video.mp4"
        
        # Initialize detector
        detector = DeepfakeDetector(device='auto')
        
        # Test video analysis
        print(f"Analyzing test video: {test_video}")
        results = detector.analyze_video(str(test_video), sample_rate=10)
        
        print(f"✅ Video analysis completed:")
        print(f"  - Overall score: {results.get('overall_score', 0.0):.3f}")
        print(f"  - Is deepfake: {results.get('is_deepfake', False)}")
        
        # Generate report
        report = detector.generate_detection_report(results)
        print("✅ Detection report generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Video analysis test failed: {e}")
        return False


def test_real_time_generator():
    """Test real-time generator functionality."""
    print("\n🔍 Testing Real-Time Generator...")
    
    try:
        # Create test data
        test_dir = create_test_data()
        test_image = test_dir / "test_image.jpg"
        test_video = test_dir / "test_video.mp4"
        
        # Initialize generator
        generator = RealTimeGenerator(str(test_image), device='auto')
        print("✅ Real-time generator initialized")
        
        # Test performance stats
        stats = generator.get_performance_stats()
        print(f"✅ Performance stats: {stats}")
        
        # Test processing start/stop
        generator.start_processing()
        time.sleep(1)  # Let it run briefly
        generator.stop_processing()
        print("✅ Processing start/stop test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time generator test failed: {e}")
        return False


def test_system_integration():
    """Test system integration."""
    print("\n🔍 Testing System Integration...")
    
    try:
        # Create test data
        test_dir = create_test_data()
        
        # Test complete pipeline
        print("Testing complete pipeline...")
        
        # 1. Generate deepfake (will fail without real faces, but should not crash)
        generator = FaceSwapGenerator(device='auto')
        print("✅ Generator ready")
        
        # 2. Detect deepfake
        detector = DeepfakeDetector(device='auto')
        print("✅ Detector ready")
        
        # 3. Analyze test video
        test_video = test_dir / "test_video.mp4"
        results = detector.analyze_video(str(test_video), sample_rate=15)
        print("✅ Video analysis completed")
        
        # 4. Generate report
        report = detector.generate_detection_report(results)
        print("✅ Report generation completed")
        
        print("✅ System integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False


def run_performance_benchmark():
    """Run performance benchmarks."""
    print("\n⚡ Running Performance Benchmarks...")
    
    try:
        # Initialize components
        generator = FaceSwapGenerator(device='auto')
        detector = DeepfakeDetector(device='auto')
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Benchmark face detection
        start_time = time.time()
        for _ in range(10):
            generator.detect_faces(test_image)
        face_detection_time = (time.time() - start_time) / 10
        print(f"✅ Face detection: {face_detection_time*1000:.1f}ms per frame")
        
        # Benchmark feature analysis
        start_time = time.time()
        for _ in range(10):
            detector.detect_in_real_time(test_image)
        feature_analysis_time = (time.time() - start_time) / 10
        print(f"✅ Feature analysis: {feature_analysis_time*1000:.1f}ms per frame")
        
        # Calculate theoretical FPS
        total_time = face_detection_time + feature_analysis_time
        theoretical_fps = 1.0 / total_time
        print(f"✅ Theoretical FPS: {theoretical_fps:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Deepfake Generation and Detection System Tests")
    print("=" * 60)
    
    tests = [
        ("Face Swap Generator", test_face_swap_generator),
        ("Feature Analyzer", test_feature_analyzer),
        ("Deepfake Detector", test_deepfake_detector),
        ("Video Analysis", test_video_analysis),
        ("Real-Time Generator", test_real_time_generator),
        ("System Integration", test_system_integration),
        ("Performance Benchmark", run_performance_benchmark)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n📝 Next Steps:")
    print("1. Add real source face image to: data/source_videos/source_face.jpg")
    print("2. Add real target video to: data/source_videos/target_video.mp4")
    print("3. Run: python example_usage.py")
    print("4. For real-time testing, install OBS Studio and virtual camera plugin")


if __name__ == "__main__":
    main()
