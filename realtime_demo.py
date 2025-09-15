"""
Real-time Deepfake Generation Demo

This demo shows how to create videos of people doing mundane tasks
in real-time, mimicking a webcam feed.
"""

import cv2
import numpy as np
import time
from src.realtime.realtime_generator import RealtimeGenerator
from src.realtime.virtual_camera import VirtualCamera, CameraManager
from src.realtime.webcam_simulator import WebcamSimulator, WebcamManager
from src.realtime.mundane_tasks import MundaneTaskTemplates


def main():
    """Main demo function."""
    print("🎬 Real-time Deepfake Generation Demo")
    print("=" * 50)
    
    # Initialize components
    generator = RealtimeGenerator(target_fps=30)
    camera_manager = CameraManager()
    webcam_manager = WebcamManager()
    
    # Get available tasks
    tasks = generator.get_available_tasks()
    print(f"📋 Available mundane tasks: {list(tasks['tasks'].keys())}")
    
    # Example source face (you'll need to provide your own)
    source_face_path = "data/source_faces/example_face.jpg"
    
    # Check if source face exists
    import os
    if not os.path.exists(source_face_path):
        print(f"⚠️  Source face not found: {source_face_path}")
        print("Please provide a source face image for face swapping.")
        return
    
    print(f"👤 Using source face: {source_face_path}")
    
    # Start real-time generation
    print("\n🚀 Starting real-time generation...")
    result = generator.start_generation(
        source_face_path=source_face_path,
        task_name='typing',
        virtual_camera=True,
        output_path="output/realtime_demo.mp4"
    )
    
    if not result['success']:
        print(f"❌ Failed to start generation: {result['error']}")
        return
    
    print(f"✅ Generation started successfully!")
    print(f"   Task: {result['task_name']}")
    print(f"   Virtual Camera: {result['virtual_camera']}")
    print(f"   Output: {result['output_path']}")
    print(f"   Target FPS: {result['target_fps']}")
    
    # Run for demonstration
    print("\n📹 Running for 30 seconds...")
    print("Press Ctrl+C to stop early")
    
    try:
        start_time = time.time()
        while time.time() - start_time < 30:
            # Get performance stats
            stats = generator.get_performance_stats()
            print(f"\r🎯 FPS: {stats.get('current_fps', 0):.1f} | "
                  f"Frames: {stats.get('frame_count', 0)} | "
                  f"Time: {stats.get('elapsed_time', 0):.1f}s", end="")
            
            time.sleep(1)
        
        print("\n\n🔄 Changing task to 'drinking_coffee'...")
        generator.change_task('drinking_coffee')
        
        # Run for another 20 seconds
        start_time = time.time()
        while time.time() - start_time < 20:
            stats = generator.get_performance_stats()
            print(f"\r☕ FPS: {stats.get('current_fps', 0):.1f} | "
                  f"Frames: {stats.get('frame_count', 0)} | "
                  f"Time: {stats.get('elapsed_time', 0):.1f}s", end="")
            
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping generation...")
    
    # Stop generation
    stop_result = generator.stop_generation()
    if stop_result['success']:
        print(f"✅ Generation stopped successfully!")
        print(f"   Total frames: {stop_result['performance_stats']['total_frames']}")
        print(f"   Average FPS: {stop_result['performance_stats']['fps']:.1f}")
        print(f"   Total time: {stop_result['total_time']:.1f}s")
    
    print("\n🎉 Demo completed!")


def demo_virtual_camera():
    """Demo virtual camera functionality."""
    print("\n📷 Virtual Camera Demo")
    print("=" * 30)
    
    # Create virtual camera
    camera = VirtualCamera(width=640, height=480, fps=30)
    
    # Start camera
    result = camera.start()
    if not result['success']:
        print(f"❌ Failed to start virtual camera: {result['error']}")
        return
    
    print(f"✅ Virtual camera started: {result['device']}")
    
    # Generate some test frames
    print("🎬 Generating test frames...")
    for i in range(100):
        # Create a simple test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some visual content
        cv2.putText(frame, f"Frame {i}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Send to virtual camera
        success = camera.send_frame(frame)
        if not success:
            print(f"⚠️  Frame {i} dropped")
        
        time.sleep(1/30)  # 30 FPS
    
    # Stop camera
    stop_result = camera.stop()
    if stop_result['success']:
        stats = stop_result['statistics']
        print(f"✅ Virtual camera stopped!")
        print(f"   Total frames: {stats['total_frames']}")
        print(f"   Dropped frames: {stats['dropped_frames']}")
        print(f"   Actual FPS: {stats['actual_fps']:.1f}")


def demo_mundane_tasks():
    """Demo mundane task templates."""
    print("\n🎭 Mundane Tasks Demo")
    print("=" * 25)
    
    # Initialize task templates
    tasks = MundaneTaskTemplates()
    
    # Get all available tasks
    all_tasks = tasks.get_all_tasks()
    print(f"📋 Available tasks: {all_tasks['total_tasks']}")
    
    for task_name, task_info in all_tasks['tasks'].items():
        print(f"   • {task_name}: {task_info['description']}")
        print(f"     Frames: {task_info['frame_count']}, "
              f"Duration: {task_info['loop_duration']}s, "
              f"FPS: {task_info['fps']:.1f}")
    
    # Demo a specific task
    print(f"\n🎬 Demo: Typing task")
    typing_task = tasks.get_task('typing')
    if typing_task:
        print(f"   Description: {typing_task.description}")
        print(f"   Frame count: {typing_task.frame_count}")
        
        # Show a few frames
        for i in range(5):
            frame = typing_task.get_current_frame()
            if frame is not None:
                print(f"   Frame {i}: {frame.shape}")
                typing_task.update_frame()


def demo_webcam_simulation():
    """Demo webcam simulation."""
    print("\n🖥️  Webcam Simulation Demo")
    print("=" * 30)
    
    # Create webcam simulator
    simulator = WebcamSimulator(width=640, height=480, fps=30)
    
    # Frame generator function
    def generate_frame():
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Simulated Webcam", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame
    
    # Start simulation
    result = simulator.start_simulation(generate_frame)
    if not result['success']:
        print(f"❌ Failed to start simulation: {result['error']}")
        return
    
    print(f"✅ Webcam simulation started!")
    
    # Start recording
    recording_result = simulator.start_recording("output/webcam_simulation.mp4")
    if recording_result['success']:
        print(f"📹 Recording started: {recording_result['output_path']}")
    
    # Run simulation
    print("🎬 Running simulation for 10 seconds...")
    start_time = time.time()
    
    while time.time() - start_time < 10:
        # Get frame from simulation
        frame = simulator.get_frame()
        if frame is not None:
            print(f"\r📺 Frame received: {frame.shape}", end="")
        
        time.sleep(0.1)
    
    # Stop recording
    stop_recording = simulator.stop_recording()
    if stop_recording['success']:
        print(f"\n✅ Recording saved: {stop_recording['output_path']}")
    
    # Stop simulation
    stop_result = simulator.stop_simulation()
    if stop_result['success']:
        stats = stop_result['statistics']
        print(f"✅ Simulation stopped!")
        print(f"   Total frames: {stats['total_frames']}")
        print(f"   Actual FPS: {stats['actual_fps']:.1f}")


if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs("output", exist_ok=True)
    
    # Run demos
    demo_mundane_tasks()
    demo_virtual_camera()
    demo_webcam_simulation()
    
    # Main demo
    main()
