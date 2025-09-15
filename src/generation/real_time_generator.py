"""
Real-Time Deepfake Generator

This module implements real-time deepfake generation to simulate webcam feeds.
"""

import cv2
import numpy as np
import torch
import threading
import time
import queue
from typing import Optional, Callable, Tuple
import pyvirtualcam
from .face_swap import FaceSwapGenerator


class RealTimeGenerator:
    """
    Real-time deepfake generator that simulates webcam feeds.
    """
    
    def __init__(self, source_face_path: str, device: str = 'auto'):
        """
        Initialize the real-time generator.
        
        Args:
            source_face_path: Path to source face image
            device: Device to run inference on
        """
        self.source_face_path = source_face_path
        self.face_swap = FaceSwapGenerator(device=device)
        
        # Load source face
        self.source_image = cv2.imread(source_face_path)
        if self.source_image is None:
            raise ValueError(f"Could not load source image: {source_face_path}")
        
        # Threading and queue management
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.processing_thread = None
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with face swapping.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with face swap
        """
        try:
            # Perform face swap
            swapped_frame = self.face_swap.swap_faces(self.source_image, frame)
            return swapped_frame
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame  # Return original frame if processing fails
    
    def _processing_worker(self):
        """Worker thread for processing frames."""
        while self.is_running:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=0.1)
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Update FPS counter
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.last_fps_time = current_time
                
                # Put processed frame back (for virtual camera)
                if hasattr(self, 'processed_queue'):
                    try:
                        self.processed_queue.put_nowait(processed_frame)
                    except queue.Full:
                        # Remove oldest frame if queue is full
                        try:
                            self.processed_queue.get_nowait()
                            self.processed_queue.put_nowait(processed_frame)
                        except queue.Empty:
                            pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing worker: {e}")
    
    def start_processing(self):
        """Start the processing thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processed_queue = queue.Queue(maxsize=10)
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def simulate_webcam_from_video(self, video_path: str, 
                                  virtual_cam_name: str = "DeepfakeCam") -> None:
        """
        Simulate webcam feed from a video file with real-time face swapping.
        
        Args:
            video_path: Path to source video
            virtual_cam_name: Name for virtual camera
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup virtual camera
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, 
                               backend=pyvirtualcam.Backend.OBS) as cam:
            print(f"Virtual camera '{cam.device}' started")
            print(f"Streaming {width}x{height} at {fps} FPS")
            
            # Start processing thread
            self.start_processing()
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        # Loop video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    # Add frame to processing queue
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Skip frame if queue is full
                        continue
                    
                    # Get processed frame
                    try:
                        processed_frame = self.processed_queue.get(timeout=0.1)
                        
                        # Convert BGR to RGB for virtual camera
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Send to virtual camera
                        cam.send(rgb_frame)
                        
                        # Display FPS
                        if self.current_fps > 0:
                            cv2.putText(processed_frame, f"FPS: {self.current_fps}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Show preview (optional)
                        cv2.imshow('Deepfake Preview', processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                    except queue.Empty:
                        # Send original frame if no processed frame available
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cam.send(rgb_frame)
                        
            except KeyboardInterrupt:
                print("Stopping webcam simulation...")
            finally:
                self.stop_processing()
                cap.release()
                cv2.destroyAllWindows()
    
    def simulate_webcam_from_live(self, source_cam_index: int = 0,
                                 virtual_cam_name: str = "DeepfakeCam") -> None:
        """
        Simulate webcam feed from live camera with real-time face swapping.
        
        Args:
            source_cam_index: Index of source camera
            virtual_cam_name: Name for virtual camera
        """
        # Open source camera
        cap = cv2.VideoCapture(source_cam_index)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {source_cam_index}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Setup virtual camera
        with pyvirtualcam.Camera(width=width, height=height, fps=fps,
                               backend=pyvirtualcam.Backend.OBS) as cam:
            print(f"Virtual camera '{cam.device}' started")
            print(f"Streaming {width}x{height} at {fps} FPS")
            
            # Start processing thread
            self.start_processing()
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    # Add frame to processing queue
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        continue
                    
                    # Get processed frame
                    try:
                        processed_frame = self.processed_queue.get(timeout=0.1)
                        
                        # Convert BGR to RGB for virtual camera
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Send to virtual camera
                        cam.send(rgb_frame)
                        
                        # Display FPS
                        if self.current_fps > 0:
                            cv2.putText(processed_frame, f"FPS: {self.current_fps}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Show preview
                        cv2.imshow('Deepfake Preview', processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                    except queue.Empty:
                        # Send original frame if no processed frame available
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        cam.send(rgb_frame)
                        
            except KeyboardInterrupt:
                print("Stopping live webcam simulation...")
            finally:
                self.stop_processing()
                cap.release()
                cv2.destroyAllWindows()
    
    def get_performance_stats(self) -> dict:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'fps': self.current_fps,
            'queue_size': self.frame_queue.qsize(),
            'processed_queue_size': getattr(self, 'processed_queue', queue.Queue()).qsize(),
            'is_running': self.is_running
        }
