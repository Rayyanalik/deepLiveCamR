#!/usr/bin/env python3
"""
LIA-X Style UI Launcher
Starts the Flask backend and serves the UI
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['flask', 'flask-cors', 'opencv-python', 'pillow', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def start_server():
    """Start the Flask server."""
    print("🚀 Starting LIA-X Style UI...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return False
    
    # Change to the web_interface directory
    web_interface_dir = Path(__file__).parent / "web_interface"
    os.chdir(web_interface_dir)
    
    print("📁 Working directory:", os.getcwd())
    print("🌐 Starting Flask server...")
    print("📱 UI will be available at: http://localhost:5000")
    print("🔧 API endpoints available at: http://localhost:5000/api/")
    print("\n⏳ Starting server in 3 seconds...")
    print("   Press Ctrl+C to stop the server")
    
    # Wait a moment
    time.sleep(3)
    
    try:
        # Start the Flask server
        subprocess.run([sys.executable, "lia_x_backend.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        return False
    
    return True

def main():
    """Main function."""
    print("🎬 LIA-X Style UI Launcher")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not os.path.exists("web_interface"):
        print("❌ Please run this script from the project root directory")
        print("   Current directory:", os.getcwd())
        return
    
    # Start the server
    success = start_server()
    
    if success:
        print("✅ Server started successfully")
    else:
        print("❌ Failed to start server")

if __name__ == "__main__":
    main()
