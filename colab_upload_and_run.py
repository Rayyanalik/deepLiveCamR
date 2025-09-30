"""
Google Colab - Upload Files and Run Research System
Copy and paste this into your Colab notebook after the setup cell
"""

# =============================================================================
# GOOGLE COLAB - UPLOAD FILES AND RUN RESEARCH
# =============================================================================

# Upload your research files
from google.colab import files
print("📁 Upload your research files...")
uploaded = files.upload()

# List uploaded files
print("📋 Uploaded files:")
for filename in uploaded.keys():
    print(f"  ✅ {filename}")

# Set your Hugging Face token
import os
os.environ['HF_TOKEN'] = 'your_huggingface_token_here'  # Replace with your actual token
print("🔑 HF Token set")

# Import and initialize research system
import sys
sys.path.append('/content')

try:
    from CORE_RESEARCH_FILES.pyramid_working_research_system import PyramidWorkingResearchSystem
    print("✅ Research system imported successfully")
    
    # Initialize system
    system = PyramidWorkingResearchSystem(os.getenv('HF_TOKEN'))
    print("🎬 Research system initialized")
    
    # Define research prompts
    prompts = [
        "A person cooking dinner in the kitchen",
        "Someone reading a book on the couch",
        "A person watering plants in the garden"
    ]
    
    print(f"🎬 Starting video generation for {len(prompts)} prompts...")
    
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
    reports_dir = 'pyramid_working_research_output/reports/'
    if os.path.exists(reports_dir):
        for file in os.listdir(reports_dir):
            if file.endswith('.md'):
                files.download(f'{reports_dir}{file}')
                print(f"  ✅ Downloaded: {file}")
    
    print("\n🎉 Research completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you uploaded all the required files")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your HF token and try again")

print("\n🚀 Research system ready for use!")
