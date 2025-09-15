"""
Mundane Task Templates Module

This module provides templates for various mundane tasks that can be
performed in real-time video generation.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import math
import random


class MundaneTaskTemplate:
    """
    Template for a mundane task with frame generation.
    """
    
    def __init__(self, name: str, frames: List[np.ndarray], 
                 description: str = "", loop_duration: float = 5.0):
        """
        Initialize mundane task template.
        
        Args:
            name: Name of the task
            frames: List of frames for the task
            description: Description of the task
            loop_duration: Duration of one complete loop in seconds
        """
        self.name = name
        self.frames = frames
        self.description = description
        self.loop_duration = loop_duration
        self.current_frame_idx = 0
        self.frame_count = len(frames)
        self.fps = self.frame_count / loop_duration if loop_duration > 0 else 30
        
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the current frame for the task.
        
        Returns:
            Current frame or None if no frames
        """
        if not self.frames:
            return None
        
        return self.frames[self.current_frame_idx].copy()
    
    def update_frame(self):
        """Update to the next frame in the sequence."""
        if self.frames:
            self.current_frame_idx = (self.current_frame_idx + 1) % self.frame_count
    
    def reset(self):
        """Reset to the first frame."""
        self.current_frame_idx = 0


class MundaneTaskTemplates:
    """
    Collection of mundane task templates.
    """
    
    def __init__(self):
        """Initialize mundane task templates."""
        self.tasks = {}
        self._create_default_tasks()
    
    def _create_default_tasks(self):
        """Create default mundane task templates."""
        # Typing task
        self._create_typing_task()
        
        # Reading task
        self._create_reading_task()
        
        # Drinking coffee task
        self._create_drinking_coffee_task()
        
        # Looking at phone task
        self._create_phone_task()
        
        # Nodding task
        self._create_nodding_task()
        
        # Smiling task
        self._create_smiling_task()
        
        # Yawning task
        self._create_yawning_task()
    
    def _create_typing_task(self):
        """Create typing task template."""
        frames = []
        base_frame = self._create_base_frame()
        
        # Create typing animation frames
        for i in range(30):
            frame = base_frame.copy()
            
            # Add typing hand movement
            hand_pos = self._get_typing_hand_position(i)
            cv2.circle(frame, hand_pos, 15, (100, 100, 100), -1)
            
            # Add subtle head movement
            head_offset = int(2 * math.sin(i * 0.2))
            cv2.rectangle(frame, (300 + head_offset, 100), (340 + head_offset, 140), (200, 180, 160), -1)
            
            frames.append(frame)
        
        self.tasks['typing'] = MundaneTaskTemplate(
            'typing', frames, 'Person typing on keyboard', 2.0
        )
    
    def _create_reading_task(self):
        """Create reading task template."""
        frames = []
        base_frame = self._create_base_frame()
        
        # Create reading animation frames
        for i in range(40):
            frame = base_frame.copy()
            
            # Add book/paper
            book_pos = (200 + int(5 * math.sin(i * 0.1)), 200)
            cv2.rectangle(frame, book_pos, (book_pos[0] + 80, book_pos[1] + 100), (255, 255, 200), -1)
            cv2.rectangle(frame, book_pos, (book_pos[0] + 80, book_pos[1] + 100), (0, 0, 0), 2)
            
            # Add reading hand
            hand_pos = (book_pos[0] + 40, book_pos[1] + 50)
            cv2.circle(frame, hand_pos, 12, (200, 180, 160), -1)
            
            # Add subtle eye movement
            eye_offset = int(3 * math.sin(i * 0.3))
            cv2.circle(frame, (320 + eye_offset, 120), 8, (0, 0, 0), -1)
            cv2.circle(frame, (360 + eye_offset, 120), 8, (0, 0, 0), -1)
            
            frames.append(frame)
        
        self.tasks['reading'] = MundaneTaskTemplate(
            'reading', frames, 'Person reading a book', 3.0
        )
    
    def _create_drinking_coffee_task(self):
        """Create drinking coffee task template."""
        frames = []
        base_frame = self._create_base_frame()
        
        # Create drinking animation frames
        for i in range(25):
            frame = base_frame.copy()
            
            # Add coffee cup
            cup_pos = (250 + int(10 * math.sin(i * 0.4)), 180)
            cv2.rectangle(frame, cup_pos, (cup_pos[0] + 30, cup_pos[1] + 40), (139, 69, 19), -1)
            cv2.rectangle(frame, cup_pos, (cup_pos[0] + 30, cup_pos[1] + 40), (0, 0, 0), 2)
            
            # Add steam
            for j in range(3):
                steam_x = cup_pos[0] + 15 + j * 5
                steam_y = cup_pos[1] - 10 - j * 3
                cv2.circle(frame, (steam_x, steam_y), 2, (200, 200, 200), -1)
            
            # Add drinking hand
            hand_pos = (cup_pos[0] + 15, cup_pos[1] + 20)
            cv2.circle(frame, hand_pos, 12, (200, 180, 160), -1)
            
            # Add subtle head tilt
            head_tilt = int(2 * math.sin(i * 0.5))
            cv2.rectangle(frame, (300 + head_tilt, 100), (340 + head_tilt, 140), (200, 180, 160), -1)
            
            frames.append(frame)
        
        self.tasks['drinking_coffee'] = MundaneTaskTemplate(
            'drinking_coffee', frames, 'Person drinking coffee', 2.5
        )
    
    def _create_phone_task(self):
        """Create looking at phone task template."""
        frames = []
        base_frame = self._create_base_frame()
        
        # Create phone looking animation frames
        for i in range(35):
            frame = base_frame.copy()
            
            # Add phone
            phone_pos = (280 + int(8 * math.sin(i * 0.2)), 150)
            cv2.rectangle(frame, phone_pos, (phone_pos[0] + 40, phone_pos[1] + 60), (50, 50, 50), -1)
            cv2.rectangle(frame, phone_pos, (phone_pos[0] + 40, phone_pos[1] + 60), (0, 0, 0), 2)
            
            # Add phone screen
            screen_pos = (phone_pos[0] + 5, phone_pos[1] + 5)
            cv2.rectangle(frame, screen_pos, (screen_pos[0] + 30, screen_pos[1] + 50), (100, 100, 255), -1)
            
            # Add scrolling effect
            scroll_offset = int(5 * math.sin(i * 0.3))
            cv2.rectangle(frame, (screen_pos[0] + 2, screen_pos[1] + 10 + scroll_offset), 
                         (screen_pos[0] + 28, screen_pos[1] + 20 + scroll_offset), (255, 255, 255), -1)
            
            # Add phone holding hand
            hand_pos = (phone_pos[0] + 20, phone_pos[1] + 30)
            cv2.circle(frame, hand_pos, 12, (200, 180, 160), -1)
            
            frames.append(frame)
        
        self.tasks['looking_at_phone'] = MundaneTaskTemplate(
            'looking_at_phone', frames, 'Person looking at phone', 2.8
        )
    
    def _create_nodding_task(self):
        """Create nodding task template."""
        frames = []
        base_frame = self._create_base_frame()
        
        # Create nodding animation frames
        for i in range(20):
            frame = base_frame.copy()
            
            # Add nodding motion
            nod_offset = int(8 * math.sin(i * 0.5))
            cv2.rectangle(frame, (300, 100 + nod_offset), (340, 140 + nod_offset), (200, 180, 160), -1)
            
            # Add subtle facial expression change
            mouth_y = 160 + nod_offset
            cv2.ellipse(frame, (320, mouth_y), (15, 5), 0, 0, 180, (0, 0, 0), 2)
            
            frames.append(frame)
        
        self.tasks['nodding'] = MundaneTaskTemplate(
            'nodding', frames, 'Person nodding', 1.5
        )
    
    def _create_smiling_task(self):
        """Create smiling task template."""
        frames = []
        base_frame = self._create_base_frame()
        
        # Create smiling animation frames
        for i in range(30):
            frame = base_frame.copy()
            
            # Add smiling motion
            smile_intensity = 0.5 + 0.5 * math.sin(i * 0.3)
            mouth_width = int(20 * smile_intensity)
            mouth_height = int(8 * smile_intensity)
            
            cv2.ellipse(frame, (320, 160), (mouth_width, mouth_height), 0, 0, 180, (0, 0, 0), 2)
            
            # Add eye squinting
            eye_squint = int(2 * smile_intensity)
            cv2.ellipse(frame, (320, 120), (8 + eye_squint, 6), 0, 0, 180, (0, 0, 0), 2)
            cv2.ellipse(frame, (360, 120), (8 + eye_squint, 6), 0, 0, 180, (0, 0, 0), 2)
            
            frames.append(frame)
        
        self.tasks['smiling'] = MundaneTaskTemplate(
            'smiling', frames, 'Person smiling', 2.0
        )
    
    def _create_yawning_task(self):
        """Create yawning task template."""
        frames = []
        base_frame = self._create_base_frame()
        
        # Create yawning animation frames
        for i in range(25):
            frame = base_frame.copy()
            
            # Add yawning motion
            yawn_intensity = 0.3 + 0.7 * math.sin(i * 0.4)
            mouth_width = int(15 * yawn_intensity)
            mouth_height = int(25 * yawn_intensity)
            
            cv2.ellipse(frame, (320, 160), (mouth_width, mouth_height), 0, 0, 180, (0, 0, 0), 2)
            
            # Add closed eyes
            eye_close = int(4 * yawn_intensity)
            cv2.ellipse(frame, (320, 120), (8, eye_close), 0, 0, 180, (0, 0, 0), 2)
            cv2.ellipse(frame, (360, 120), (8, eye_close), 0, 0, 180, (0, 0, 0), 2)
            
            frames.append(frame)
        
        self.tasks['yawning'] = MundaneTaskTemplate(
            'yawning', frames, 'Person yawning', 2.0
        )
    
    def _create_base_frame(self) -> np.ndarray:
        """
        Create a base frame with a simple background and face.
        
        Returns:
            Base frame
        """
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add simple face
        cv2.rectangle(frame, (300, 100), (340, 140), (200, 180, 160), -1)  # Face
        cv2.circle(frame, (320, 120), 8, (0, 0, 0), -1)  # Left eye
        cv2.circle(frame, (360, 120), 8, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(frame, (320, 160), (15, 5), 0, 0, 180, (0, 0, 0), 2)  # Mouth
        
        # Add simple body
        cv2.rectangle(frame, (280, 200), (360, 400), (100, 100, 150), -1)  # Shirt
        
        return frame
    
    def _get_typing_hand_position(self, frame_idx: int) -> Tuple[int, int]:
        """Get typing hand position for given frame."""
        base_x = 350
        base_y = 250
        
        # Simulate typing motion
        x_offset = int(20 * math.sin(frame_idx * 0.3))
        y_offset = int(10 * math.cos(frame_idx * 0.2))
        
        return (base_x + x_offset, base_y + y_offset)
    
    def get_task(self, task_name: str) -> Optional[MundaneTaskTemplate]:
        """
        Get a task template by name.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Task template or None if not found
        """
        return self.tasks.get(task_name)
    
    def get_all_tasks(self) -> Dict[str, Any]:
        """
        Get information about all available tasks.
        
        Returns:
            Dictionary with task information
        """
        tasks_info = {}
        for name, template in self.tasks.items():
            tasks_info[name] = {
                'name': name,
                'description': template.description,
                'frame_count': template.frame_count,
                'loop_duration': template.loop_duration,
                'fps': template.fps
            }
        
        return {
            'total_tasks': len(self.tasks),
            'tasks': tasks_info
        }
    
    def add_custom_task(self, task_name: str, frames: List[np.ndarray], 
                       description: str = "") -> bool:
        """
        Add a custom task template.
        
        Args:
            task_name: Name of the task
            frames: List of frames for the task
            description: Description of the task
            
        Returns:
            True if successful, False otherwise
        """
        if not frames or task_name in self.tasks:
            return False
        
        try:
            template = MundaneTaskTemplate(task_name, frames, description)
            self.tasks[task_name] = template
            return True
        except Exception as e:
            print(f"Error adding custom task: {e}")
            return False
    
    def remove_task(self, task_name: str) -> bool:
        """
        Remove a task template.
        
        Args:
            task_name: Name of the task to remove
            
        Returns:
            True if successful, False otherwise
        """
        if task_name in self.tasks:
            del self.tasks[task_name]
            return True
        return False
    
    def save_tasks(self, file_path: str) -> bool:
        """
        Save tasks to file.
        
        Args:
            file_path: Path to save tasks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tasks_data = {}
            for name, template in self.tasks.items():
                # Convert frames to base64 for serialization
                frames_b64 = []
                for frame in template.frames:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frames_b64.append(buffer.tobytes().hex())
                
                tasks_data[name] = {
                    'description': template.description,
                    'loop_duration': template.loop_duration,
                    'frames': frames_b64
                }
            
            with open(file_path, 'w') as f:
                json.dump(tasks_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving tasks: {e}")
            return False
    
    def load_tasks(self, file_path: str) -> bool:
        """
        Load tasks from file.
        
        Args:
            file_path: Path to load tasks from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                tasks_data = json.load(f)
            
            for name, data in tasks_data.items():
                # Convert base64 frames back to numpy arrays
                frames = []
                for frame_b64 in data['frames']:
                    frame_bytes = bytes.fromhex(frame_b64)
                    frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    frames.append(frame)
                
                template = MundaneTaskTemplate(
                    name, frames, data['description'], data['loop_duration']
                )
                self.tasks[name] = template
            
            return True
        except Exception as e:
            print(f"Error loading tasks: {e}")
            return False
