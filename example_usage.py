"""
Example Usage of Deepfake Generation and Detection System

This script demonstrates how to use the deepfake generation and detection modules.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from generation.face_swap import FaceSwapGenerator
from generation.real_time_generator import RealTimeGenerator
from generation.online_ai_tools import OnlineAITools
from detection.deepfake_detector import DeepfakeDetector


def create_sample_data():
    """Create sample data directories and download models if needed."""
    print("Setting up sample data...")
    
    # Create data directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "source_videos").mkdir(exist_ok=True)
    (data_dir / "generated").mkdir(exist_ok=True)
    (data_dir / "test_data").mkdir(exist_ok=True)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Check if dlib model exists
    dlib_model = models_dir / "shape_predictor_68_face_landmarks.dat"
    if not dlib_model.exists():
        print("⚠️  dlib face landmark model not found!")
        print("Please download it from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract and place it in the models/ directory")
        return False
    
    print("✅ Sample data setup complete")
    return True


def generate_deepfake_video():
    """Generate a deepfake video from source materials."""
    print("\n" + "="*50)
    print("GENERATING DEEPFAKE VIDEO")
    print("="*50)
    
    # Check if we have the required files
    source_face = "data/source_videos/source_face.jpg"
    target_video = "data/source_videos/target_video.mp4"
    
    if not os.path.exists(source_face):
        print(f"❌ Source face image not found: {source_face}")
        print("Please add a source face image to generate deepfakes")
        return None
    
    if not os.path.exists(target_video):
        print(f"❌ Target video not found: {target_video}")
        print("Please add a target video to perform face swapping")
        return None
    
    try:
        # Initialize face swap generator
        print("Initializing face swap generator...")
        generator = FaceSwapGenerator(device='auto')
        
        # Generate deepfake video
        output_path = "data/generated/deepfake_output.mp4"
        print(f"Generating deepfake video: {output_path}")
        
        result_path = generator.generate_mundane_task_video(
            source_face_path=source_face,
            target_video_path=target_video,
            output_path=output_path
        )
        
        print(f"✅ Deepfake video generated: {result_path}")
        return result_path
        
    except Exception as e:
        print(f"❌ Error generating deepfake video: {e}")
        return None


def detect_deepfake_video(video_path: str):
    """Detect deepfake characteristics in a video."""
    print("\n" + "="*50)
    print("DETECTING DEEPFAKE CHARACTERISTICS")
    print("="*50)
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    try:
        # Initialize detector
        print("Initializing deepfake detector...")
        detector = DeepfakeDetector(device='auto')
        
        # Analyze video
        print(f"Analyzing video: {video_path}")
        results = detector.analyze_video(video_path, sample_rate=30)
        
        # Generate report
        report = detector.generate_detection_report(results)
        print("\n" + report)
        
        # Save results
        import json
        results_file = f"{video_path}_detection_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            json_results[key][k] = v.tolist()
                        else:
                            json_results[key][k] = v
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Detection results saved: {results_file}")
        return results
        
    except Exception as e:
        print(f"❌ Error detecting deepfake: {e}")
        return None


def simulate_real_time_webcam():
    """Simulate real-time webcam with deepfake generation."""
    print("\n" + "="*50)
    print("REAL-TIME WEBCAM SIMULATION")
    print("="*50)
    
    source_face = "data/source_videos/source_face.jpg"
    target_video = "data/source_videos/target_video.mp4"
    
    if not os.path.exists(source_face):
        print(f"❌ Source face image not found: {source_face}")
        print("Please add a source face image for real-time generation")
        return
    
    if not os.path.exists(target_video):
        print(f"❌ Target video not found: {target_video}")
        print("Please add a target video for real-time simulation")
        return
    
    try:
        # Initialize real-time generator
        print("Initializing real-time generator...")
        generator = RealTimeGenerator(source_face, device='auto')
        
        print("Starting webcam simulation...")
        print("Press 'q' to quit")
        print("The virtual camera will be available as 'DeepfakeCam'")
        
        # Simulate webcam from video
        generator.simulate_webcam_from_video(target_video)
        
    except Exception as e:
        print(f"❌ Error in real-time simulation: {e}")


def generate_deepfake_with_online_tools():
    """Generate deepfake using online AI tools."""
    print("\n" + "="*50)
    print("GENERATING DEEPFAKE WITH ONLINE AI TOOLS")
    print("="*50)
    
    # Check for source files
    source_face = "data/source_videos/source_face.jpg"
    target_video = "data/source_videos/target_video.mp4"
    
    if not os.path.exists(source_face):
        print(f"❌ Source face image not found: {source_face}")
        print("Please add a source face image to generate deepfakes")
        return
    
    if not os.path.exists(target_video):
        print(f"❌ Target video not found: {target_video}")
        print("Please add a target video to perform face swapping")
        return
    
    try:
        # Initialize online tools
        online_tools = OnlineAITools()
        
        # Get available tools
        tool_info = online_tools.get_tool_info()
        available_tools = [tool for tool, info in tool_info.items() if info['has_api_key']]
        
        if not available_tools:
            print("❌ No online AI tools available (no API keys set)")
            print("Please set up API keys for online tools:")
            print("  - Reelmind.ai: https://reelmind.ai/api")
            print("  - DeepSynth Pro: https://deepsynth.com/api")
            print("  - FaceSwap Studio: https://faceswapstudio.com/api")
            print("  - NeuralArt Video: https://neuralart.com/api")
            print("  - RunwayML: https://runwayml.com/api")
            print("  - Stability AI: https://stability.ai/api")
            return
        
        # Show available tools
        print("Available online AI tools:")
        for i, tool in enumerate(available_tools, 1):
            info = tool_info[tool]
            print(f"  {i}. {info['name']} - {info['description']}")
        
        # Get user selection
        try:
            choice = int(input(f"\nSelect tool (1-{len(available_tools)}): ")) - 1
            if 0 <= choice < len(available_tools):
                selected_tool = available_tools[choice]
            else:
                print("❌ Invalid selection")
                return
        except ValueError:
            print("❌ Invalid input")
            return
        
        # Generate deepfake
        output_path = f"data/generated/online_{selected_tool}_deepfake.mp4"
        print(f"\nGenerating deepfake with {tool_info[selected_tool]['name']}...")
        
        result_path = online_tools.generate_with_online_tool(
            selected_tool, source_face, target_video, output_path
        )
        
        if result_path:
            print(f"✅ Deepfake generated: {result_path}")
            
            # Analyze the generated deepfake
            print("\nAnalyzing generated deepfake...")
            detector = DeepfakeDetector(device='auto')
            results = detector.analyze_video(result_path, sample_rate=30)
            
            # Show results
            overall_score = results.get('overall_score', 0.0)
            is_deepfake = results.get('is_deepfake', False)
            
            print(f"\nAnalysis Results:")
            print(f"Detection Score: {overall_score:.3f}")
            print(f"Detected as Deepfake: {'YES' if is_deepfake else 'NO'}")
            
            # Generate report
            report = detector.generate_detection_report(results)
            print("\nDetailed Report:")
            print(report)
            
        else:
            print("❌ Failed to generate deepfake")
            
    except Exception as e:
        print(f"❌ Error generating deepfake with online tools: {e}")


def compare_online_ai_tools():
    """Compare multiple online AI tools."""
    print("\n" + "="*50)
    print("COMPARING ONLINE AI TOOLS")
    print("="*50)
    
    # Check for source files
    source_face = "data/source_videos/source_face.jpg"
    target_video = "data/source_videos/target_video.mp4"
    
    if not os.path.exists(source_face):
        print(f"❌ Source face image not found: {source_face}")
        print("Please add a source face image to generate deepfakes")
        return
    
    if not os.path.exists(target_video):
        print(f"❌ Target video not found: {target_video}")
        print("Please add a target video to perform face swapping")
        return
    
    try:
        # Initialize online tools
        online_tools = OnlineAITools()
        
        # Compare all available tools
        print("Comparing all available online AI tools...")
        output_dir = "data/generated/online_comparison"
        results = online_tools.compare_online_tools(source_face, target_video, output_dir)
        
        if not results:
            print("❌ No tools available or all failed")
            return
        
        print(f"\n✅ Generated {len(results)} deepfakes for comparison")
        
        # Analyze generated deepfakes
        print("\nAnalyzing generated deepfakes...")
        detector = DeepfakeDetector(device='auto')
        
        comparison_results = {}
        for tool_name, video_path in results.items():
            print(f"Analyzing {tool_name}...")
            analysis = detector.analyze_video(video_path, sample_rate=30)
            
            overall_score = analysis.get('overall_score', 0.0)
            is_deepfake = analysis.get('is_deepfake', False)
            
            comparison_results[tool_name] = {
                'score': overall_score,
                'detected': is_deepfake,
                'video_path': video_path
            }
            
            print(f"  Score: {overall_score:.3f}, Detected: {'YES' if is_deepfake else 'NO'}")
        
        # Generate comparison report
        print("\n📊 Comparison Summary:")
        print("=" * 40)
        
        # Sort by detection score (higher = more detectable)
        sorted_results = sorted(comparison_results.items(), 
                              key=lambda x: x[1]['score'], reverse=True)
        
        print("Tools ranked by detectability (highest to lowest):")
        for i, (tool_name, results) in enumerate(sorted_results, 1):
            score = results['score']
            detected = results['detected']
            status = "DETECTED" if detected else "NOT DETECTED"
            print(f"  {i}. {tool_name}: {score:.3f} ({status})")
        
        # Find best evasion tool
        undetected_tools = [tool for tool, results in comparison_results.items() 
                           if not results['detected']]
        
        if undetected_tools:
            print(f"\n🏆 Best Evasion Tools (not detected):")
            for tool in undetected_tools:
                print(f"  ✅ {tool}")
        else:
            print(f"\n⚠️  All tools were detected as deepfakes")
        
        print(f"\n💡 Key Insights:")
        print(f"  - Online AI tools show varying detection resistance")
        print(f"  - Some tools may be better at evading detection")
        print(f"  - Detection algorithms can identify most synthetic content")
        print(f"  - Tool selection affects detection vulnerability")
        
    except Exception as e:
        print(f"❌ Error comparing online AI tools: {e}")


def test_detection_evasion():
    """Test how well generated deepfakes can evade detection."""
    print("\n" + "="*50)
    print("TESTING DETECTION EVASION")
    print("="*50)
    
    generated_video = "data/generated/deepfake_output.mp4"
    
    if not os.path.exists(generated_video):
        print(f"❌ Generated deepfake video not found: {generated_video}")
        print("Please generate a deepfake video first")
        return
    
    try:
        # Initialize detector
        detector = DeepfakeDetector(device='auto')
        
        # Analyze the generated deepfake
        print("Analyzing generated deepfake for evasion capabilities...")
        results = detector.analyze_video(generated_video, sample_rate=15)
        
        # Check if it was detected
        is_detected = results.get('is_deepfake', False)
        confidence = results.get('overall_score', 0.0)
        
        print(f"\nEvasion Test Results:")
        print(f"Deepfake Detected: {'YES' if is_detected else 'NO'}")
        print(f"Detection Confidence: {confidence:.3f}")
        
        if is_detected:
            print("❌ Deepfake was successfully detected")
            print("Consider improving generation techniques for better evasion")
        else:
            print("✅ Deepfake successfully evaded detection!")
            print("The generation method shows good evasion capabilities")
        
        # Generate detailed report
        report = detector.generate_detection_report(results)
        print("\nDetailed Analysis:")
        print(report)
        
    except Exception as e:
        print(f"❌ Error testing evasion: {e}")


def main():
    """Main function to run the complete pipeline."""
    print("🤖 Deepfake Generation and Detection System")
    print("=" * 50)
    
    # Setup
    if not create_sample_data():
        print("❌ Setup failed. Please check requirements.")
        return
    
    # Menu
    while True:
        print("\nSelect an option:")
        print("1. Generate Deepfake Video (Local)")
        print("2. Generate Deepfake Video (Online AI Tools)")
        print("3. Detect Deepfake Characteristics")
        print("4. Simulate Real-Time Webcam")
        print("5. Test Detection Evasion")
        print("6. Compare Online AI Tools")
        print("7. Run Complete Pipeline")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == "1":
            generate_deepfake_video()
            
        elif choice == "2":
            generate_deepfake_with_online_tools()
            
        elif choice == "3":
            video_path = input("Enter video path to analyze: ").strip()
            if video_path:
                detect_deepfake_video(video_path)
                
        elif choice == "4":
            simulate_real_time_webcam()
            
        elif choice == "5":
            test_detection_evasion()
            
        elif choice == "6":
            compare_online_ai_tools()
            
        elif choice == "7":
            print("\n🚀 Running Complete Pipeline...")
            
            # Generate deepfake
            deepfake_path = generate_deepfake_video()
            
            if deepfake_path:
                # Detect deepfake
                detect_deepfake_video(deepfake_path)
                
                # Test evasion
                test_detection_evasion()
            
        elif choice == "8":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
