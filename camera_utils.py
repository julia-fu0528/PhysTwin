#!/usr/bin/env python3
"""
Utility functions for camera name to index mapping.
Provides a consistent interface for camera selection across all scripts.
"""

import os
import json
from typing import Optional, Tuple


def find_camera_index(episode_path: str, camera_name: Optional[str] = None) -> Tuple[int, str]:
    """
    Find the camera index for a given camera name in an episode.
    
    Args:
        episode_path: Path to the episode directory (e.g., /path/to/object/episode_0)
        camera_name: Name of the camera (e.g., 'brics-odroid-022_cam1'). 
                     If None, returns index 0 (first camera).
    
    Returns:
        Tuple of (camera_index, actual_camera_name):
            - camera_index: The index of the camera in the metadata cameras list
            - actual_camera_name: The actual camera name (either the matched name or the fallback)
    
    Falls back to first camera (index 0) if:
        - camera_name is None
        - metadata.json not found
        - camera_name not found in the cameras list
    """
    metadata_path = os.path.join(episode_path, "metadata.json")
    
    # Fallback to first camera if no camera name specified
    if camera_name is None:
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                cameras = metadata.get('cameras', [])
                if cameras:
                    return 0, cameras[0]
            except Exception:
                pass
        return 0, "camera_0"
    
    # Try to find the camera by name
    if not os.path.exists(metadata_path):
        print(f"Warning: metadata.json not found at {metadata_path}, falling back to first camera")
        return 0, camera_name
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        cameras = metadata.get('cameras', [])
        if not cameras:
            print(f"Warning: No cameras found in metadata.json, falling back to first camera")
            return 0, camera_name
        
        # Find the camera index
        if camera_name in cameras:
            cam_idx = cameras.index(camera_name)
            print(f"Found camera '{camera_name}' at index {cam_idx}")
            return cam_idx, camera_name
        else:
            print(f"Warning: Camera '{camera_name}' not found in metadata.json")
            print(f"Available cameras: {cameras}")
            print(f"Falling back to first camera: {cameras[0]}")
            return 0, cameras[0]
    
    except Exception as e:
        print(f"Error loading metadata.json: {e}, falling back to first camera")
        return 0, camera_name


def get_camera_list(episode_path: str) -> list:
    """
    Get the list of all cameras for an episode.
    
    Args:
        episode_path: Path to the episode directory
    
    Returns:
        List of camera names, or empty list if not found
    """
    metadata_path = os.path.join(episode_path, "metadata.json")
    
    if not os.path.exists(metadata_path):
        return []
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata.get('cameras', [])
    except Exception:
        return []
