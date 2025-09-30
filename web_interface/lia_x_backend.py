"""
LIA-X Style Backend
Flask backend for the LIA-X style UI
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import json
import time
import cv2
import numpy as np
from PIL import Image
import io
import base64

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from CORE_RESEARCH_FILES.huggingface_text_to_video_system import HuggingFaceTextToVideoSystem
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: HuggingFace system not available")

app = Flask(__name__)
CORS(app)

# Global variables
hf_system = None
current_session = {
    'source_image': None,
    'driving_video': None,
    'controls': {
        'yaw1': 0, 'yaw2': 0, 'pitch': 0,
        'roll1': 0, 'roll2': 0, 'neck': 0
    }
}

def initialize_hf_system():
    """Initialize the Hugging Face system if available."""
    global hf_system
    if HF_AVAILABLE:
        hf_token = os.getenv('HF_TOKEN', '')
        if hf_token:
            try:
                hf_system = HuggingFaceTextToVideoSystem(hf_token)
                print("✅ Hugging Face system initialized")
                return True
            except Exception as e:
                print(f"❌ Failed to initialize HF system: {e}")
                return False
        else:
            print("⚠️ HF_TOKEN not set, using mock system")
            return False
    return False

@app.route('/')
def index():
    """Serve the main UI."""
    return send_file('lia_x_style_ui.html')

@app.route('/api/upload/source', methods=['POST'])
def upload_source():
    """Handle source image/video upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the file
        filename = f"source_{int(time.time())}.{file.filename.split('.')[-1]}"
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Store in session
        current_session['source_image'] = filepath
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Source file uploaded successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/driving', methods=['POST'])
def upload_driving():
    """Handle driving video upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the file
        filename = f"driving_{int(time.time())}.{file.filename.split('.')[-1]}"
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Store in session
        current_session['driving_video'] = filepath
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Driving video uploaded successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/controls', methods=['POST'])
def update_controls():
    """Update control parameters."""
    try:
        data = request.get_json()
        controls = data.get('controls', {})
        
        # Update session controls
        current_session['controls'].update(controls)
        
        return jsonify({
            'success': True,
            'controls': current_session['controls'],
            'message': 'Controls updated successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/animate', methods=['POST'])
def animate():
    """Generate animation based on current session."""
    try:
        if not current_session['source_image']:
            return jsonify({'error': 'No source image provided'}), 400
        
        # Simulate processing time
        time.sleep(2)
        
        # Generate mock results
        timestamp = int(time.time())
        
        # Create mock edited image
        edited_image_path = f"outputs/edited_{timestamp}.jpg"
        os.makedirs('outputs', exist_ok=True)
        
        # Create a simple mock image
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        img[:] = [50, 50, 100]  # Dark blue background
        
        # Add text
        cv2.putText(img, "Edited Image", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Controls: {current_session['controls']}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imwrite(edited_image_path, img)
        
        # Create mock animated video
        video_path = f"outputs/animated_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 8, (400, 400))
        
        for i in range(24):  # 3 seconds at 8 fps
            frame = img.copy()
            cv2.putText(frame, f"Frame {i+1}/24", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            out.write(frame)
        
        out.release()
        
        return jsonify({
            'success': True,
            'edited_image': edited_image_path,
            'animated_video': video_path,
            'message': 'Animation generated successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_text_to_video', methods=['POST'])
def generate_text_to_video():
    """Generate video from text prompt using Hugging Face."""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        if not hf_system:
            return jsonify({'error': 'Hugging Face system not available'}), 500
        
        # Generate video
        video_path = hf_system.generate_video(prompt)
        
        if video_path:
            return jsonify({
                'success': True,
                'video_path': video_path,
                'message': 'Video generated successfully'
            })
        else:
            return jsonify({'error': 'Failed to generate video'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status."""
    return jsonify({
        'hf_system_available': hf_system is not None,
        'current_session': current_session,
        'timestamp': time.time()
    })

@app.route('/api/clear', methods=['POST'])
def clear_session():
    """Clear current session."""
    global current_session
    current_session = {
        'source_image': None,
        'driving_video': None,
        'controls': {
            'yaw1': 0, 'yaw2': 0, 'pitch': 0,
            'roll1': 0, 'roll2': 0, 'neck': 0
        }
    }
    
    return jsonify({
        'success': True,
        'message': 'Session cleared successfully'
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'hf_available': HF_AVAILABLE,
        'hf_initialized': hf_system is not None
    })

if __name__ == '__main__':
    # Initialize systems
    print("🚀 Starting LIA-X Style Backend...")
    print("=" * 50)
    
    # Initialize Hugging Face system
    hf_initialized = initialize_hf_system()
    
    if hf_initialized:
        print("✅ Hugging Face system ready")
    else:
        print("⚠️ Using mock system (HF not available)")
    
    print("🌐 Starting Flask server...")
    print("📱 UI will be available at: http://localhost:5000")
    print("🔧 API endpoints available at: http://localhost:5000/api/")
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Start the server
    app.run(debug=True, host='0.0.0.0', port=5000)
