"""
Evasion Testing Example

This script demonstrates how to test deepfake evasion capabilities
against detection algorithms.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from detection.evasion_tester import EvasionTester
from detection.deepfake_detector import DeepfakeDetector


def create_test_deepfake_image():
    """Create a test image that simulates a deepfake for testing."""
    print("Creating test deepfake image...")
    
    # Create a simple face-like image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some face-like features
    # Face outline
    cv2.ellipse(image, (320, 240), (150, 200), 0, 0, 360, (200, 180, 160), -1)
    
    # Eyes
    cv2.circle(image, (280, 200), 20, (50, 50, 50), -1)
    cv2.circle(image, (360, 200), 20, (50, 50, 50), -1)
    
    # Nose
    cv2.ellipse(image, (320, 240), (10, 30), 0, 0, 360, (180, 160, 140), -1)
    
    # Mouth
    cv2.ellipse(image, (320, 280), (30, 15), 0, 0, 180, (100, 50, 50), -1)
    
    # Add some artifacts that might be detected
    # Color inconsistency
    image[100:150, 100:150] = [255, 0, 0]  # Red patch
    
    # Edge artifacts
    cv2.rectangle(image, (500, 100), (600, 200), (0, 255, 0), 2)
    
    return image


def test_evasion_techniques():
    """Test various evasion techniques."""
    print("🧪 Testing Deepfake Evasion Techniques")
    print("=" * 50)
    
    # Create test image
    test_image = create_test_deepfake_image()
    
    # Save test image
    cv2.imwrite("data/test_data/test_deepfake.jpg", test_image)
    print("✅ Test deepfake image created")
    
    # Initialize evasion tester
    tester = EvasionTester(device='auto')
    print("✅ Evasion tester initialized")
    
    # Test all techniques
    print("\n🔍 Testing all evasion techniques...")
    results = tester.test_all_techniques(test_image, intensities=[0.3, 0.6, 0.9])
    
    # Generate report
    print("\n📊 Generating evasion report...")
    report = tester.generate_evasion_report(results)
    print(report)
    
    # Save report
    with open("data/test_data/evasion_report.txt", "w") as f:
        f.write(report)
    print("✅ Evasion report saved")
    
    # Find best combination
    print("\n🏆 Finding best evasion combination...")
    best_combination = tester.find_best_evasion_combination(test_image)
    
    if best_combination:
        print(f"Best combination: {best_combination['techniques']}")
        print(f"Effectiveness: {best_combination['effectiveness']:.3f}")
        print(f"Original score: {best_combination['original_score']:.3f}")
        print(f"Modified score: {best_combination['modified_score']:.3f}")
    else:
        print("No effective evasion combination found")
    
    # Plot results
    try:
        print("\n📈 Plotting evasion results...")
        tester.plot_evasion_results(results)
        print("✅ Evasion plots generated")
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")
    
    return results


def test_real_deepfake_evasion(video_path: str):
    """Test evasion on a real deepfake video."""
    print(f"\n🎬 Testing evasion on real deepfake: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    # Initialize components
    detector = DeepfakeDetector(device='auto')
    tester = EvasionTester(device='auto')
    
    # Analyze original video
    print("Analyzing original video...")
    original_results = detector.analyze_video(video_path, sample_rate=30)
    original_score = original_results.get('overall_score', 0.0)
    print(f"Original detection score: {original_score:.3f}")
    
    # Extract frames for evasion testing
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for i in range(10):  # Test first 10 frames
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    
    cap.release()
    
    if not frames:
        print("❌ Could not extract frames from video")
        return None
    
    # Test evasion on frames
    print(f"Testing evasion on {len(frames)} frames...")
    
    frame_results = []
    for i, frame in enumerate(frames):
        print(f"Testing frame {i+1}/{len(frames)}...")
        
        # Test best individual technique
        best_technique = None
        best_effectiveness = 0
        
        for technique in tester.evasion_techniques:
            try:
                result = tester.test_single_technique(frame, technique, 0.5)
                if result['evasion_effectiveness'] > best_effectiveness:
                    best_effectiveness = result['evasion_effectiveness']
                    best_technique = technique
            except Exception as e:
                print(f"  Error testing {technique}: {e}")
        
        frame_results.append({
            'frame': i,
            'best_technique': best_technique,
            'best_effectiveness': best_effectiveness
        })
    
    # Analyze results
    avg_effectiveness = np.mean([r['best_effectiveness'] for r in frame_results])
    successful_frames = sum(1 for r in frame_results if r['best_effectiveness'] > 0.1)
    
    print(f"\n📊 Evasion Results:")
    print(f"  Average effectiveness: {avg_effectiveness:.3f}")
    print(f"  Successful frames: {successful_frames}/{len(frames)}")
    print(f"  Success rate: {successful_frames/len(frames)*100:.1f}%")
    
    # Find most effective technique
    technique_counts = {}
    for result in frame_results:
        if result['best_technique']:
            technique_counts[result['best_technique']] = technique_counts.get(result['best_technique'], 0) + 1
    
    if technique_counts:
        most_effective = max(technique_counts.items(), key=lambda x: x[1])
        print(f"  Most effective technique: {most_effective[0]} ({most_effective[1]} frames)")
    
    return frame_results


def main():
    """Main function to run evasion tests."""
    print("🚀 Deepfake Evasion Testing System")
    print("=" * 60)
    
    # Create test data directory
    test_dir = Path("data/test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Test evasion techniques
    print("\n1. Testing Evasion Techniques")
    evasion_results = test_evasion_techniques()
    
    # Test on real deepfake if available
    deepfake_video = "data/generated/deepfake_output.mp4"
    if os.path.exists(deepfake_video):
        print("\n2. Testing on Real Deepfake Video")
        real_results = test_real_deepfake_evasion(deepfake_video)
    else:
        print("\n2. Real Deepfake Video Not Found")
        print(f"   Expected: {deepfake_video}")
        print("   Generate a deepfake video first using example_usage.py")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 EVASION TESTING SUMMARY")
    print("=" * 60)
    
    if evasion_results:
        print("✅ Evasion technique testing completed")
        
        # Count successful techniques
        successful_techniques = 0
        total_tests = 0
        
        for technique, results in evasion_results.items():
            if results:
                total_tests += len(results)
                successful_tests = sum(1 for r in results if r['evasion_success'])
                successful_techniques += successful_tests
        
        success_rate = (successful_techniques / total_tests * 100) if total_tests > 0 else 0
        print(f"📊 Overall evasion success rate: {success_rate:.1f}%")
        
        if success_rate > 50:
            print("🎯 High evasion success rate - detection may be vulnerable")
        elif success_rate > 25:
            print("⚠️  Moderate evasion success rate - detection has some vulnerabilities")
        else:
            print("🛡️  Low evasion success rate - detection appears robust")
    
    print("\n💡 Key Findings:")
    print("  • Evasion testing helps identify detection vulnerabilities")
    print("  • Some techniques may reduce detection accuracy")
    print("  • Combining techniques can improve evasion effectiveness")
    print("  • Detection algorithms should be continuously improved")
    
    print("\n⚠️  Important Notes:")
    print("  • These tests are for research and educational purposes only")
    print("  • Evasion techniques may reduce video quality")
    print("  • Always follow ethical guidelines when using deepfake technology")
    print("  • Detection algorithms continue to evolve and improve")


if __name__ == "__main__":
    main()
