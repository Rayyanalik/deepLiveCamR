"""
Real-time Video Generation Module

This module generates videos of people doing mundane tasks in real-time,
mimicking webcam functionality with virtual camera output.
"""

import cv2
import numpy as np
import torch
import threading
import time
import queue
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import mediapipe as mp
from PIL import Image
import pyvirtualcam
from .mundane_tasks import MundaneTaskTemplates
from ..generation.face_swap import FaceSwapGenerator


class RealtimeGenerator:
    """
    Real-time generator for mundane task videos with face swapping.
    """
    
    def __init__(self, device: str = 'auto', target_fps: int = 30):
        """
        Initialize real-time generator.
        
        Args:
            device: Device to use for processing
            target_fps: Target frames per second
        """
        self.device = self._get_device(device)
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        # Initialize components
        self.face_swap_generator = FaceSwapGenerator(device=device)
        self.task_templates = MundaneTaskTemplates()
        
        # Real-time processing
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.performance_stats = {
            'fps': 0,
            'avg_processing_time': 0,
            'frame_drops': 0,
            'total_frames': 0
        }
        
        # Current state
        self.current_task = None
        self.current_source_face = None
        self.virtual_camera = None
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def start_generation(self, source_face_path: str, task_name: str = 'typing',
                        virtual_camera: bool = True, output_path: str = None) -> Dict[str, Any]:
        """
        Start real-time video generation.
        
        Args:
            source_face_path: Path to source face image
            task_name: Name of mundane task to perform
            virtual_camera: Whether to output to virtual camera
            output_path: Optional path to save video file
            
        Returns:
            Generation start results
        """
        if self.is_running:
            return {'success': False, 'error': 'Generation already running'}
        
        # Load source face
        source_image = cv2.imread(source_face_path)
        if source_image is None:
            return {'success': False, 'error': f'Could not load source face: {source_face_path}'}
        
        self.current_source_face = source_image
        
        # Get task template
        task_template = self.task_templates.get_task(task_name)
        if not task_template:
            return {'success': False, 'error': f'Unknown task: {task_name}'}
        
        self.current_task = task_template
        
        # Initialize virtual camera if requested
        if virtual_camera:
            try:
                self.virtual_camera = pyvirtualcam.Camera(
                    width=640, height=480, fps=self.target_fps
                )
                print(f"Virtual camera initialized: {self.virtual_camera.device}")
            except Exception as e:
                print(f"Warning: Could not initialize virtual camera: {e}")
                virtual_camera = False
        
        # Setup video writer if output path provided
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, self.target_fps, (640, 480))
        
        # Start generation
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start processing thread
        processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(video_writer, virtual_camera)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        return {
            'success': True,
            'task_name': task_name,
            'source_face': source_face_path,
            'virtual_camera': virtual_camera,
            'output_path': output_path,
            'target_fps': self.target_fps
        }
    
    def stop_generation(self) -> Dict[str, Any]:
        """
        Stop real-time generation.
        
        Returns:
            Stop results with performance stats
        """
        if not self.is_running:
            return {'success': False, 'error': 'Generation not running'}
        
        self.is_running = False
        
        # Close virtual camera
        if self.virtual_camera:
            self.virtual_camera.close()
            self.virtual_camera = None
        
        # Calculate final stats
        if self.start_time:
            total_time = time.time() - self.start_time
            self.performance_stats['fps'] = self.frame_count / total_time if total_time > 0 else 0
            self.performance_stats['total_frames'] = self.frame_count
        
        return {
            'success': True,
            'performance_stats': self.performance_stats.copy(),
            'total_time': time.time() - self.start_time if self.start_time else 0
        }
    
    def _processing_loop(self, video_writer=None, virtual_camera=False):
        """Main processing loop for real-time generation."""
        frame_times = []
        
        while self.is_running:
            loop_start = time.time()
            
            try:
                # Generate frame
                frame = self._generate_frame()
                
                if frame is not None:
                    # Resize to standard webcam resolution
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Send to virtual camera
                    if virtual_camera and self.virtual_camera:
                        # Convert BGR to RGB for virtual camera
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.virtual_camera.send(rgb_frame)
                    
                    # Write to video file
                    if video_writer:
                        video_writer.write(frame)
                    
                    self.frame_count += 1
                    
                    # Update performance stats
                    processing_time = time.time() - loop_start
                    frame_times.append(processing_time)
                    
                    if len(frame_times) > 30:  # Keep last 30 frames
                        frame_times.pop(0)
                    
                    self.performance_stats['avg_processing_time'] = np.mean(frame_times)
                
                # Maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                self.performance_stats['frame_drops'] += 1
        
        # Cleanup
        if video_writer:
            video_writer.release()
    
    def _generate_frame(self) -> Optional[np.ndarray]:
        """
        Generate a single frame with face swap.
        
        Returns:
            Generated frame or None if error
        """
        if not self.current_task or not self.current_source_face:
            return None
        
        try:
            # Get current task frame
            task_frame = self.current_task.get_current_frame()
            if task_frame is None:
                return None
            
            # Perform face swap
            swapped_frame = self.face_swap_generator.swap_faces(
                self.current_source_face, task_frame
            )
            
            # Update task animation
            self.current_task.update_frame()
            
            return swapped_frame
            
        except Exception as e:
            print(f"Error generating frame: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.
        
        Returns:
            Performance statistics
        """
        if self.start_time:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            stats = self.performance_stats.copy()
            stats['current_fps'] = current_fps
            stats['elapsed_time'] = elapsed_time
            
            return stats
        
        return self.performance_stats.copy()
    
    def change_task(self, task_name: str) -> Dict[str, Any]:
        """
        Change the current mundane task.
        
        Args:
            task_name: Name of new task
            
        Returns:
            Task change results
        """
        if not self.is_running:
            return {'success': False, 'error': 'Generation not running'}
        
        task_template = self.task_templates.get_task(task_name)
        if not task_template:
            return {'success': False, 'error': f'Unknown task: {task_name}'}
        
        self.current_task = task_template
        
        return {
            'success': True,
            'new_task': task_name,
            'task_description': task_template.description
        }
    
    def change_source_face(self, source_face_path: str) -> Dict[str, Any]:
        """
        Change the source face for swapping.
        
        Args:
            source_face_path: Path to new source face image
            
        Returns:
            Face change results
        """
        if not self.is_running:
            return {'success': False, 'error': 'Generation not running'}
        
        source_image = cv2.imread(source_face_path)
        if source_image is None:
            return {'success': False, 'error': f'Could not load source face: {source_face_path}'}
        
        self.current_source_face = source_image
        
        return {
            'success': True,
            'new_source_face': source_face_path
        }
    
    def get_available_tasks(self) -> Dict[str, Any]:
        """
        Get list of available mundane tasks.
        
        Returns:
            Available tasks information
        """
        return self.task_templates.get_all_tasks()
    
    def set_target_fps(self, fps: int) -> Dict[str, Any]:
        """
        Set target FPS for generation.
        
        Args:
            fps: Target frames per second
            
        Returns:
            FPS change results
        """
        if fps < 1 or fps > 60:
            return {'success': False, 'error': 'FPS must be between 1 and 60'}
        
        self.target_fps = fps
        self.frame_time = 1.0 / fps
        
        # Update virtual camera FPS if running
        if self.virtual_camera:
            try:
                self.virtual_camera.close()
                self.virtual_camera = pyvirtualcam.Camera(
                    width=640, height=480, fps=fps
                )
            except Exception as e:
                print(f"Warning: Could not update virtual camera FPS: {e}")
        
        return {
            'success': True,
            'new_fps': fps,
            'frame_time': self.frame_time
        }
    
    def add_custom_task(self, task_name: str, frames: List[np.ndarray], 
                       description: str = "") -> Dict[str, Any]:
        """
        Add a custom mundane task.
        
        Args:
            task_name: Name of the task
            frames: List of frames for the task
            description: Description of the task
            
        Returns:
            Task addition results
        """
        if not frames:
            return {'success': False, 'error': 'No frames provided'}
        
        success = self.task_templates.add_custom_task(task_name, frames, description)
        
        if success:
            return {
                'success': True,
                'task_name': task_name,
                'frame_count': len(frames),
                'description': description
            }
        else:
            return {'success': False, 'error': 'Failed to add custom task'}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current generation status.
        
        Returns:
            Current status information
        """
        return {
            'is_running': self.is_running,
            'current_task': self.current_task.name if self.current_task else None,
            'current_source_face': self.current_source_face is not None,
            'virtual_camera': self.virtual_camera is not None,
            'target_fps': self.target_fps,
            'frame_count': self.frame_count,
            'performance_stats': self.get_performance_stats()
        }
