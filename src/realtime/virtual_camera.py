"""
Virtual Camera Module

This module provides virtual camera functionality for real-time
video generation, allowing the generated content to appear as a webcam.
"""

import cv2
import numpy as np
import pyvirtualcam
import threading
import time
import queue
from typing import Optional, Tuple, Dict, Any
import subprocess
import platform


class VirtualCamera:
    """
    Virtual camera for real-time video output.
    """
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize virtual camera.
        
        Args:
            width: Camera width
            height: Camera height
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_time = 1.0 / fps
        
        self.camera = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.dropped_frames = 0
        
        # Threading
        self.camera_thread = None
        
    def start(self, device_name: str = None) -> Dict[str, Any]:
        """
        Start virtual camera.
        
        Args:
            device_name: Custom device name
            
        Returns:
            Start results
        """
        if self.is_running:
            return {'success': False, 'error': 'Virtual camera already running'}
        
        try:
            # Initialize virtual camera
            self.camera = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                device=device_name
            )
            
            self.is_running = True
            self.start_time = time.time()
            self.frame_count = 0
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self._camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            return {
                'success': True,
                'device': self.camera.device,
                'width': self.width,
                'height': self.height,
                'fps': self.fps
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def stop(self) -> Dict[str, Any]:
        """
        Stop virtual camera.
        
        Returns:
            Stop results with statistics
        """
        if not self.is_running:
            return {'success': False, 'error': 'Virtual camera not running'}
        
        self.is_running = False
        
        # Wait for thread to finish
        if self.camera_thread:
            self.camera_thread.join(timeout=2.0)
        
        # Close camera
        if self.camera:
            self.camera.close()
            self.camera = None
        
        # Calculate statistics
        stats = self._get_statistics()
        
        return {
            'success': True,
            'statistics': stats
        }
    
    def send_frame(self, frame: np.ndarray) -> bool:
        """
        Send frame to virtual camera.
        
        Args:
            frame: Frame to send (BGR format)
            
        Returns:
            True if frame queued successfully
        """
        if not self.is_running or not self.camera:
            return False
        
        try:
            # Resize frame if needed
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Try to add to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(rgb_frame)
                return True
            except queue.Full:
                self.dropped_frames += 1
                return False
                
        except Exception as e:
            print(f"Error sending frame: {e}")
            return False
    
    def _camera_loop(self):
        """Main camera loop for sending frames."""
        while self.is_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Send to virtual camera
                self.camera.send(frame)
                self.frame_count += 1
                
                # Maintain FPS
                elapsed = time.time() - self.start_time if self.start_time else 0
                expected_frames = int(elapsed * self.fps)
                
                if self.frame_count < expected_frames:
                    # We're behind, skip frame timing
                    continue
                elif self.frame_count > expected_frames:
                    # We're ahead, wait a bit
                    time.sleep(self.frame_time * 0.5)
                
            except queue.Empty:
                # No frame available, continue
                continue
            except Exception as e:
                print(f"Error in camera loop: {e}")
                break
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get camera statistics."""
        if not self.start_time:
            return {}
        
        elapsed_time = time.time() - self.start_time
        actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'total_frames': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'actual_fps': actual_fps,
            'target_fps': self.fps,
            'elapsed_time': elapsed_time,
            'queue_size': self.frame_queue.qsize()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get virtual camera status.
        
        Returns:
            Status information
        """
        return {
            'is_running': self.is_running,
            'device': self.camera.device if self.camera else None,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'statistics': self._get_statistics() if self.is_running else {}
        }


class WebcamSimulator:
    """
    Webcam simulator that can replace real webcam input.
    """
    
    def __init__(self, virtual_camera: VirtualCamera):
        """
        Initialize webcam simulator.
        
        Args:
            virtual_camera: Virtual camera instance
        """
        self.virtual_camera = virtual_camera
        self.is_simulating = False
        
    def start_simulation(self, source_video: str = None) -> Dict[str, Any]:
        """
        Start webcam simulation.
        
        Args:
            source_video: Optional source video to simulate
            
        Returns:
            Simulation start results
        """
        if self.is_simulating:
            return {'success': False, 'error': 'Simulation already running'}
        
        # Start virtual camera
        camera_result = self.virtual_camera.start()
        if not camera_result['success']:
            return camera_result
        
        self.is_simulating = True
        
        return {
            'success': True,
            'virtual_camera': camera_result,
            'simulation_type': 'generated' if source_video is None else 'video'
        }
    
    def stop_simulation(self) -> Dict[str, Any]:
        """
        Stop webcam simulation.
        
        Returns:
            Simulation stop results
        """
        if not self.is_simulating:
            return {'success': False, 'error': 'Simulation not running'}
        
        # Stop virtual camera
        camera_result = self.virtual_camera.stop()
        
        self.is_simulating = False
        
        return {
            'success': True,
            'virtual_camera': camera_result
        }
    
    def send_generated_frame(self, frame: np.ndarray) -> bool:
        """
        Send generated frame to virtual camera.
        
        Args:
            frame: Generated frame
            
        Returns:
            True if successful
        """
        if not self.is_simulating:
            return False
        
        return self.virtual_camera.send_frame(frame)


class CameraManager:
    """
    Manager for multiple virtual cameras and webcam simulation.
    """
    
    def __init__(self):
        """Initialize camera manager."""
        self.cameras = {}
        self.active_cameras = set()
        
    def create_camera(self, name: str, width: int = 640, height: int = 480, 
                     fps: int = 30) -> Dict[str, Any]:
        """
        Create a new virtual camera.
        
        Args:
            name: Camera name
            width: Camera width
            height: Camera height
            fps: Frames per second
            
        Returns:
            Camera creation results
        """
        if name in self.cameras:
            return {'success': False, 'error': f'Camera {name} already exists'}
        
        try:
            camera = VirtualCamera(width, height, fps)
            self.cameras[name] = camera
            
            return {
                'success': True,
                'camera_name': name,
                'width': width,
                'height': height,
                'fps': fps
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def start_camera(self, name: str) -> Dict[str, Any]:
        """
        Start a virtual camera.
        
        Args:
            name: Camera name
            
        Returns:
            Camera start results
        """
        if name not in self.cameras:
            return {'success': False, 'error': f'Camera {name} not found'}
        
        camera = self.cameras[name]
        result = camera.start()
        
        if result['success']:
            self.active_cameras.add(name)
        
        return result
    
    def stop_camera(self, name: str) -> Dict[str, Any]:
        """
        Stop a virtual camera.
        
        Args:
            name: Camera name
            
        Returns:
            Camera stop results
        """
        if name not in self.cameras:
            return {'success': False, 'error': f'Camera {name} not found'}
        
        camera = self.cameras[name]
        result = camera.stop()
        
        if result['success']:
            self.active_cameras.discard(name)
        
        return result
    
    def send_frame_to_camera(self, name: str, frame: np.ndarray) -> bool:
        """
        Send frame to specific camera.
        
        Args:
            name: Camera name
            frame: Frame to send
            
        Returns:
            True if successful
        """
        if name not in self.cameras:
            return False
        
        camera = self.cameras[name]
        return camera.send_frame(frame)
    
    def get_camera_status(self, name: str) -> Dict[str, Any]:
        """
        Get camera status.
        
        Args:
            name: Camera name
            
        Returns:
            Camera status
        """
        if name not in self.cameras:
            return {'success': False, 'error': f'Camera {name} not found'}
        
        camera = self.cameras[name]
        return camera.get_status()
    
    def list_cameras(self) -> Dict[str, Any]:
        """
        List all cameras.
        
        Returns:
            Camera list
        """
        cameras_info = {}
        for name, camera in self.cameras.items():
            cameras_info[name] = {
                'width': camera.width,
                'height': camera.height,
                'fps': camera.fps,
                'is_running': camera.is_running,
                'is_active': name in self.active_cameras
            }
        
        return {
            'total_cameras': len(self.cameras),
            'active_cameras': len(self.active_cameras),
            'cameras': cameras_info
        }
    
    def cleanup(self):
        """Cleanup all cameras."""
        for name in list(self.active_cameras):
            self.stop_camera(name)
        
        self.cameras.clear()
        self.active_cameras.clear()
