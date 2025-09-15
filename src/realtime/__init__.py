"""
Real-time Video Generation Module

This module handles real-time generation of mundane task videos
with webcam-like functionality and virtual camera output.
"""

from .realtime_generator import RealtimeGenerator
from .virtual_camera import VirtualCamera
from .mundane_tasks import MundaneTaskTemplates
from .webcam_simulator import WebcamSimulator

__all__ = [
    'RealtimeGenerator',
    'VirtualCamera',
    'MundaneTaskTemplates',
    'WebcamSimulator'
]
