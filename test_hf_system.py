"""
Test Hugging Face Text-to-Video System
Quick test to verify the system works with fallback models
"""

import os
import sys
sys.path.append('CORE_RESEARCH_FILES')

from huggingface_text_to_video_system import HuggingFaceTextToVideoSystem

def test_hf_system():
    """Test the Hugging Face text-to-video system."""
    print("🧪 Testing Hugging Face Text-to-Video System")
    print("=" * 50)
    
    # Get HF token
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print("❌ Please set your HF_TOKEN environment variable")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        return
    
    # Initialize system
    system = HuggingFaceTextToVideoSystem(hf_token)
    
    # Test with a simple prompt
    test_prompt = "a boy playing football on a green field"
    print(f"\n🎬 Testing with prompt: '{test_prompt}'")
    
    # Generate video
    video_path = system.generate_video(test_prompt)
    
    if video_path:
        print(f"✅ Video generated successfully: {video_path}")
        
        # Test detection
        detection_result = system.test_detection_on_video(video_path)
        print(f"🔍 Detection result: {detection_result['detection']['label']}")
        print(f"   Confidence: {detection_result['detection']['confidence']:.2f}")
        
        # Create report
        videos = [{'prompt': test_prompt, 'video_path': video_path, 'file_size': os.path.getsize(video_path), 'generated_at': os.path.getctime(video_path)}]
        results = [detection_result]
        report_path = system.create_research_report(videos, results)
        print(f"📋 Report created: {report_path}")
        
    else:
        print("❌ Video generation failed")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    test_hf_system()
