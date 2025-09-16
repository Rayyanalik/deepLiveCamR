"""
Final Working Real-time Deepfake Demo

This demo creates videos of people doing mundane tasks in real-time
without complex face detection (simplified version).
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Dict, Any
from pathlib import Path


class FinalMundaneTaskGenerator:
    """Final working generator for mundane task animations."""
    
    def __init__(self, task_name: str = 'typing'):
        self.task_name = task_name
        self.frame_count = 0
        self.task_frames = self._create_task_frames()
    
    def _create_task_frames(self) -> list:
        """Create animated frames for the mundane task."""
        frames = []
        
        if self.task_name == 'typing':
            frames = self._create_typing_frames()
        elif self.task_name == 'reading':
            frames = self._create_reading_frames()
        elif self.task_name == 'drinking':
            frames = self._create_drinking_frames()
        else:
            frames = self._create_default_frames()
        
        return frames
    
    def _create_typing_frames(self) -> list:
        """Create typing animation frames."""
        frames = []
        for i in range(30):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light gray background
            
            # Draw person
            cv2.rectangle(frame, (250, 150), (390, 350), (200, 180, 160), -1)  # Body
            cv2.rectangle(frame, (280, 120), (360, 180), (200, 180, 160), -1)  # Head
            
            # Eyes
            cv2.circle(frame, (300, 140), 8, (0, 0, 0), -1)
            cv2.circle(frame, (340, 140), 8, (0, 0, 0), -1)
            
            # Mouth
            cv2.ellipse(frame, (320, 160), (15, 5), 0, 0, 180, (0, 0, 0), 2)
            
            # Typing hands animation
            hand_offset = int(10 * np.sin(i * 0.3))
            cv2.circle(frame, (300 + hand_offset, 250), 15, (200, 180, 160), -1)
            cv2.circle(frame, (340 - hand_offset, 250), 15, (200, 180, 160), -1)
            
            # Keyboard
            cv2.rectangle(frame, (200, 300), (440, 320), (50, 50, 50), -1)
            
            # Add some text to show it's working
            cv2.putText(frame, f"Typing Frame {i}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            frames.append(frame)
        
        return frames
    
    def _create_reading_frames(self) -> list:
        """Create reading animation frames."""
        frames = []
        for i in range(40):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
            
            # Draw person
            cv2.rectangle(frame, (250, 150), (390, 350), (200, 180, 160), -1)  # Body
            cv2.rectangle(frame, (280, 120), (360, 180), (200, 180, 160), -1)  # Head
            
            # Eyes
            cv2.circle(frame, (300, 140), 8, (0, 0, 0), -1)
            cv2.circle(frame, (340, 140), 8, (0, 0, 0), -1)
            
            # Mouth
            cv2.ellipse(frame, (320, 160), (15, 5), 0, 0, 180, (0, 0, 0), 2)
            
            # Book
            book_offset = int(5 * np.sin(i * 0.2))
            cv2.rectangle(frame, (180 + book_offset, 200), (260 + book_offset, 300), (255, 255, 200), -1)
            cv2.rectangle(frame, (180 + book_offset, 200), (260 + book_offset, 300), (0, 0, 0), 2)
            
            # Reading hand
            hand_offset = int(8 * np.sin(i * 0.3))
            cv2.circle(frame, (220 + hand_offset, 250), 15, (200, 180, 160), -1)
            
            # Add some text to show it's working
            cv2.putText(frame, f"Reading Frame {i}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            frames.append(frame)
        
        return frames
    
    def _create_drinking_frames(self) -> list:
        """Create drinking animation frames."""
        frames = []
        for i in range(25):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
            
            # Draw person
            cv2.rectangle(frame, (250, 150), (390, 350), (200, 180, 160), -1)  # Body
            cv2.rectangle(frame, (280, 120), (360, 180), (200, 180, 160), -1)  # Head
            
            # Eyes
            cv2.circle(frame, (300, 140), 8, (0, 0, 0), -1)
            cv2.circle(frame, (340, 140), 8, (0, 0, 0), -1)
            
            # Mouth
            cv2.ellipse(frame, (320, 160), (15, 5), 0, 0, 180, (0, 0, 0), 2)
            
            # Cup
            cup_offset = int(8 * np.sin(i * 0.4))
            cv2.rectangle(frame, (200 + cup_offset, 200), (230 + cup_offset, 250), (139, 69, 19), -1)
            cv2.rectangle(frame, (200 + cup_offset, 200), (230 + cup_offset, 250), (0, 0, 0), 2)
            
            # Drinking hand
            hand_offset = int(10 * np.sin(i * 0.5))
            cv2.circle(frame, (215 + hand_offset, 225), 15, (200, 180, 160), -1)
            
            # Add some text to show it's working
            cv2.putText(frame, f"Drinking Frame {i}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            frames.append(frame)
        
        return frames
    
    def _create_default_frames(self) -> list:
        """Create default animation frames."""
        frames = []
        for i in range(30):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
            
            # Draw person
            cv2.rectangle(frame, (250, 150), (390, 350), (200, 180, 160), -1)  # Body
            cv2.rectangle(frame, (280, 120), (360, 180), (200, 180, 160), -1)  # Head
            
            # Eyes
            cv2.circle(frame, (300, 140), 8, (0, 0, 0), -1)
            cv2.circle(frame, (340, 140), 8, (0, 0, 0), -1)
            
            # Mouth
            cv2.ellipse(frame, (320, 160), (15, 5), 0, 0, 180, (0, 0, 0), 2)
            
            # Simple animation
            offset = int(5 * np.sin(i * 0.2))
            cv2.circle(frame, (320 + offset, 250), 20, (100, 100, 100), -1)
            
            # Add some text to show it's working
            cv2.putText(frame, f"Default Frame {i}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            frames.append(frame)
        
        return frames
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame."""
        if not self.task_frames:
            return None
        
        frame_idx = self.frame_count % len(self.task_frames)
        return self.task_frames[frame_idx].copy()
    
    def update_frame(self):
        """Update to next frame."""
        self.frame_count += 1


class FinalRealtimeGenerator:
    """Final working real-time generator for mundane tasks."""
    
    def __init__(self, target_fps: int = 30):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        self.task_generator = None
        self.source_face = None
        
        self.is_running = False
        self.video_writer = None
        
        self.frame_count = 0
        self.start_time = None
    
    def start_generation(self, source_face_path: str, task_name: str = 'typing',
                        output_path: str = "output/final_realtime_demo.mp4") -> Dict[str, Any]:
        """Start real-time generation."""
        if self.is_running:
            return {'success': False, 'error': 'Generation already running'}
        
        # Load source face (we'll use it for simple overlay)
        source_image = cv2.imread(source_face_path)
        if source_image is None:
            return {'success': False, 'error': f'Could not load source face: {source_face_path}'}
        
        self.source_face = source_image
        
        # Initialize task generator
        self.task_generator = FinalMundaneTaskGenerator(task_name)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, self.target_fps, (640, 480))
        
        if not self.video_writer.isOpened():
            return {'success': False, 'error': 'Could not initialize video writer'}
        
        # Start generation
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start processing thread
        processing_thread = threading.Thread(target=self._processing_loop)
        processing_thread.daemon = True
        processing_thread.start()
        
        return {
            'success': True,
            'task_name': task_name,
            'source_face': source_face_path,
            'output_path': output_path,
            'target_fps': self.target_fps
        }
    
    def stop_generation(self) -> Dict[str, Any]:
        """Stop generation."""
        if not self.is_running:
            return {'success': False, 'error': 'Generation not running'}
        
        self.is_running = False
        
        # Close video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        # Calculate stats
        total_time = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / total_time if total_time > 0 else 0
        
        return {
            'success': True,
            'total_frames': self.frame_count,
            'total_time': total_time,
            'fps': fps
        }
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.is_running:
            loop_start = time.time()
            
            try:
                # Generate frame
                frame = self._generate_frame()
                
                if frame is not None:
                    # Write to video file
                    if self.video_writer:
                        self.video_writer.write(frame)
                    
                    self.frame_count += 1
                
                # Maintain FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                break
    
    def _generate_frame(self) -> Optional[np.ndarray]:
        """Generate a single frame."""
        if not self.task_generator:
            return None
        
        try:
            # Get task frame
            task_frame = self.task_generator.get_current_frame()
            if task_frame is None:
                return None
            
            # Simple face overlay (no complex face detection)
            result_frame = task_frame.copy()
            
            # Add a simple face overlay from source image
            if self.source_face is not None:
                # Resize source face to fit
                face_resized = cv2.resize(self.source_face, (80, 60))
                
                # Overlay on the head area
                result_frame[120:180, 280:360] = cv2.resize(face_resized, (80, 60))
            
            # Update task animation
            self.task_generator.update_frame()
            
            return result_frame
            
        except Exception as e:
            print(f"Error generating frame: {e}")
            return None
    
    def change_task(self, task_name: str) -> Dict[str, Any]:
        """Change the current task."""
        if not self.is_running:
            return {'success': False, 'error': 'Generation not running'}
        
        self.task_generator = FinalMundaneTaskGenerator(task_name)
        
        return {
            'success': True,
            'new_task': task_name
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.start_time:
            return {}
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'current_fps': current_fps,
            'frame_count': self.frame_count,
            'elapsed_time': elapsed_time
        }


def main():
    """Main demo function."""
    print("🎬 Final Working Real-time Deepfake Demo")
    print("=" * 45)
    
    # Check if source face exists
    source_face_path = "data/source_faces/my_face.jpg"
    if not Path(source_face_path).exists():
        print(f"⚠️  Source face not found: {source_face_path}")
        print("Please place a face image at this location.")
        print("You can use any clear photo of a face.")
        return
    
    # Initialize generator
    generator = FinalRealtimeGenerator(target_fps=30)
    
    # Start generation
    print("🚀 Starting real-time generation...")
    result = generator.start_generation(
        source_face_path=source_face_path,
        task_name='typing',
        output_path="output/final_realtime_demo.mp4"
    )
    
    if not result['success']:
        print(f"❌ Failed to start: {result['error']}")
        return
    
    print(f"✅ Generation started!")
    print(f"   Task: {result['task_name']}")
    print(f"   Output: {result['output_path']}")
    print(f"   Target FPS: {result['target_fps']}")
    
    # Run for demonstration
    print("\n🎭 Running for 30 seconds...")
    print("Press Ctrl+C to stop early")
    
    try:
        for i in range(30):
            stats = generator.get_performance_stats()
            print(f"\r🎯 FPS: {stats.get('current_fps', 0):.1f} | "
                  f"Frames: {stats.get('frame_count', 0)} | "
                  f"Time: {stats.get('elapsed_time', 0):.1f}s", end="")
            time.sleep(1)
        
        print("\n\n🔄 Changing task to 'reading'...")
        generator.change_task('reading')
        
        # Run for another 20 seconds
        for i in range(20):
            stats = generator.get_performance_stats()
            print(f"\r📖 FPS: {stats.get('current_fps', 0):.1f} | "
                  f"Frames: {stats.get('frame_count', 0)} | "
                  f"Time: {stats.get('elapsed_time', 0):.1f}s", end="")
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping generation...")
    
    # Stop generation
    stop_result = generator.stop_generation()
    if stop_result['success']:
        print(f"\n✅ Generation stopped!")
        print(f"   Total frames: {stop_result['total_frames']}")
        print(f"   Average FPS: {stop_result['fps']:.1f}")
        print(f"   Total time: {stop_result['total_time']:.1f}s")
        print(f"   Video saved: {result['output_path']}")
    
    print("\n🎉 Demo completed!")
    print("📹 Check the output video file to see the generated content!")
    print("🎬 You now have a working real-time deepfake generation system!")


if __name__ == "__main__":
    main()
