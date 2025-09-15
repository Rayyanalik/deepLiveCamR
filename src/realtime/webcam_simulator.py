"""
Webcam Simulator Module

This module provides webcam simulation functionality for real-time
deepfake generation, allowing applications to see the generated content
as if it's coming from a real webcam.
"""

import cv2
import numpy as np
import threading
import time
import queue
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import json


class WebcamSimulator:
    """
    Simulates a webcam by providing generated video frames.
    """
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initialize webcam simulator.
        
        Args:
            width: Simulated webcam width
            height: Simulated webcam height
            fps: Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_time = 1.0 / fps
        
        # Simulation state
        self.is_running = False
        self.is_recording = False
        
        # Frame generation
        self.frame_generator = None
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Threading
        self.simulation_thread = None
        self.recording_thread = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.dropped_frames = 0
        
        # Recording
        self.video_writer = None
        self.recording_path = None
        
    def start_simulation(self, frame_generator: Callable[[], np.ndarray]) -> Dict[str, Any]:
        """
        Start webcam simulation.
        
        Args:
            frame_generator: Function that generates frames
            
        Returns:
            Simulation start results
        """
        if self.is_running:
            return {'success': False, 'error': 'Simulation already running'}
        
        self.frame_generator = frame_generator
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        return {
            'success': True,
            'width': self.width,
            'height': self.height,
            'fps': self.fps
        }
    
    def stop_simulation(self) -> Dict[str, Any]:
        """
        Stop webcam simulation.
        
        Returns:
            Simulation stop results
        """
        if not self.is_running:
            return {'success': False, 'error': 'Simulation not running'}
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Calculate statistics
        stats = self._get_statistics()
        
        return {
            'success': True,
            'statistics': stats
        }
    
    def start_recording(self, output_path: str) -> Dict[str, Any]:
        """
        Start recording simulation to video file.
        
        Args:
            output_path: Path to save recorded video
            
        Returns:
            Recording start results
        """
        if self.is_recording:
            return {'success': False, 'error': 'Recording already active'}
        
        if not self.is_running:
            return {'success': False, 'error': 'Simulation not running'}
        
        try:
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            
            if not self.video_writer.isOpened():
                return {'success': False, 'error': 'Could not initialize video writer'}
            
            self.is_recording = True
            self.recording_path = output_path
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            return {
                'success': True,
                'output_path': output_path,
                'width': self.width,
                'height': self.height,
                'fps': self.fps
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def stop_recording(self) -> Dict[str, Any]:
        """
        Stop recording simulation.
        
        Returns:
            Recording stop results
        """
        if not self.is_recording:
            return {'success': False, 'error': 'Recording not active'}
        
        self.is_recording = False
        
        # Wait for recording thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        # Release video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        output_path = self.recording_path
        self.recording_path = None
        
        return {
            'success': True,
            'output_path': output_path
        }
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from simulation.
        
        Returns:
            Latest frame or None if not available
        """
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _simulation_loop(self):
        """Main simulation loop."""
        while self.is_running:
            loop_start = time.time()
            
            try:
                # Generate frame
                if self.frame_generator:
                    frame = self.frame_generator()
                    
                    if frame is not None:
                        # Resize frame if needed
                        if frame.shape[:2] != (self.height, self.width):
                            frame = cv2.resize(frame, (self.width, self.height))
                        
                        # Add to queue (non-blocking)
                        try:
                            self.frame_queue.put_nowait(frame)
                            self.frame_count += 1
                        except queue.Full:
                            self.dropped_frames += 1
                
                # Maintain FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Error in simulation loop: {e}")
                break
    
    def _recording_loop(self):
        """Recording loop for saving frames to video."""
        while self.is_recording and self.is_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Write to video file
                if self.video_writer:
                    self.video_writer.write(frame)
                
            except queue.Empty:
                # No frame available, continue
                continue
            except Exception as e:
                print(f"Error in recording loop: {e}")
                break
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
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
            'queue_size': self.frame_queue.qsize(),
            'is_recording': self.is_recording
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get simulation status.
        
        Returns:
            Status information
        """
        return {
            'is_running': self.is_running,
            'is_recording': self.is_recording,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'statistics': self._get_statistics() if self.is_running else {},
            'recording_path': self.recording_path
        }


class WebcamReplacer:
    """
    Replaces real webcam input with generated content.
    """
    
    def __init__(self, simulator: WebcamSimulator):
        """
        Initialize webcam replacer.
        
        Args:
            simulator: Webcam simulator instance
        """
        self.simulator = simulator
        self.is_replacing = False
        
    def start_replacement(self, frame_generator: Callable[[], np.ndarray]) -> Dict[str, Any]:
        """
        Start replacing webcam input.
        
        Args:
            frame_generator: Function that generates frames
            
        Returns:
            Replacement start results
        """
        if self.is_replacing:
            return {'success': False, 'error': 'Replacement already active'}
        
        # Start simulation
        result = self.simulator.start_simulation(frame_generator)
        if not result['success']:
            return result
        
        self.is_replacing = True
        
        return {
            'success': True,
            'simulation': result
        }
    
    def stop_replacement(self) -> Dict[str, Any]:
        """
        Stop webcam replacement.
        
        Returns:
            Replacement stop results
        """
        if not self.is_replacing:
            return {'success': False, 'error': 'Replacement not active'}
        
        # Stop simulation
        result = self.simulator.stop_simulation()
        
        self.is_replacing = False
        
        return {
            'success': True,
            'simulation': result
        }
    
    def get_replacement_frame(self) -> Optional[np.ndarray]:
        """
        Get frame for webcam replacement.
        
        Returns:
            Replacement frame or None
        """
        if not self.is_replacing:
            return None
        
        return self.simulator.get_frame()


class WebcamManager:
    """
    Manager for webcam simulation and replacement.
    """
    
    def __init__(self):
        """Initialize webcam manager."""
        self.simulators = {}
        self.active_simulators = set()
        
    def create_simulator(self, name: str, width: int = 640, height: int = 480, 
                        fps: int = 30) -> Dict[str, Any]:
        """
        Create a new webcam simulator.
        
        Args:
            name: Simulator name
            width: Simulator width
            height: Simulator height
            fps: Frames per second
            
        Returns:
            Simulator creation results
        """
        if name in self.simulators:
            return {'success': False, 'error': f'Simulator {name} already exists'}
        
        try:
            simulator = WebcamSimulator(width, height, fps)
            self.simulators[name] = simulator
            
            return {
                'success': True,
                'simulator_name': name,
                'width': width,
                'height': height,
                'fps': fps
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def start_simulation(self, name: str, frame_generator: Callable[[], np.ndarray]) -> Dict[str, Any]:
        """
        Start simulation with specific simulator.
        
        Args:
            name: Simulator name
            frame_generator: Frame generation function
            
        Returns:
            Simulation start results
        """
        if name not in self.simulators:
            return {'success': False, 'error': f'Simulator {name} not found'}
        
        simulator = self.simulators[name]
        result = simulator.start_simulation(frame_generator)
        
        if result['success']:
            self.active_simulators.add(name)
        
        return result
    
    def stop_simulation(self, name: str) -> Dict[str, Any]:
        """
        Stop simulation with specific simulator.
        
        Args:
            name: Simulator name
            
        Returns:
            Simulation stop results
        """
        if name not in self.simulators:
            return {'success': False, 'error': f'Simulator {name} not found'}
        
        simulator = self.simulators[name]
        result = simulator.stop_simulation()
        
        if result['success']:
            self.active_simulators.discard(name)
        
        return result
    
    def get_simulator_status(self, name: str) -> Dict[str, Any]:
        """
        Get simulator status.
        
        Args:
            name: Simulator name
            
        Returns:
            Simulator status
        """
        if name not in self.simulators:
            return {'success': False, 'error': f'Simulator {name} not found'}
        
        simulator = self.simulators[name]
        return simulator.get_status()
    
    def list_simulators(self) -> Dict[str, Any]:
        """
        List all simulators.
        
        Returns:
            Simulator list
        """
        simulators_info = {}
        for name, simulator in self.simulators.items():
            simulators_info[name] = {
                'width': simulator.width,
                'height': simulator.height,
                'fps': simulator.fps,
                'is_running': simulator.is_running,
                'is_recording': simulator.is_recording,
                'is_active': name in self.active_simulators
            }
        
        return {
            'total_simulators': len(self.simulators),
            'active_simulators': len(self.active_simulators),
            'simulators': simulators_info
        }
    
    def cleanup(self):
        """Cleanup all simulators."""
        for name in list(self.active_simulators):
            self.stop_simulation(name)
        
        self.simulators.clear()
        self.active_simulators.clear()
