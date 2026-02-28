#!/usr/bin/env python3
"""
Helper script to show the mapping between camera indices and camera names.
This helps identify which camera index corresponds to which camera folder.

The camera ordering is read from metadata.json, which is the authoritative source
used by inference_warp.py and the evaluation scripts.
"""

import argparse
import glob
import os
import json


def load_camera_list(base_path, case_name=None):
    """
    Load the camera list from metadata.json.
    
    Args:
        base_path: Path to the data directory
        case_name: Episode name (e.g., 'episode_0'). If None, uses the first available episode.
    
    Returns:
        tuple: (case_name, camera_list) or (None, None) if not found
    """
    # If no case name provided, find the first episode
    if case_name is None:
        episodes = sorted(glob.glob(f"{base_path}/episode_*"))
        if not episodes:
            print(f"No episodes found in {base_path}")
            return None, None
        case_name = os.path.basename(episodes[0])
        print(f"Using first available episode: {case_name}\n")
    
    metadata_path = f"{base_path}/{case_name}/metadata.json"
    
    # Check if metadata exists
    if not os.path.exists(metadata_path):
        print(f"Metadata not found: {metadata_path}")
        return None, None
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if 'cameras' not in metadata:
            print(f"'cameras' field not found in {metadata_path}")
            return None, None
        
        return case_name, metadata['cameras']
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None, None


def show_camera_mapping(base_path, case_name=None):
    """
    Display the mapping between camera indices and camera names.
    
    Args:
        base_path: Path to the data directory
        case_name: Episode name (e.g., 'episode_0'). If None, uses the first available episode.
    """
    case_name, cameras = load_camera_list(base_path, case_name)
    
    if cameras is None:
        return
    
    print(f"Camera Index Mapping for: {case_name}")
    print(f"(Source: metadata.json)")
    print("=" * 60)
    print(f"{'Index':<8} | {'Camera Name':<40}")
    print("-" * 60)
    
    for idx, cam_name in enumerate(cameras):
        print(f"{idx:<8} | {cam_name:<40}")
    
    print("=" * 60)
    print(f"\nTotal cameras: {len(cameras)}")
    

def find_camera_index(base_path, camera_name, case_name=None):
    """
    Find the index of a specific camera by name.
    
    Args:
        base_path: Path to the data directory
        camera_name: Name of the camera to find (e.g., 'brics-odroid-022_cam1')
        case_name: Episode name (e.g., 'episode_0'). If None, uses the first available episode.
    
    Returns:
        int or None: Camera index if found, None otherwise
    """
    case_name, cameras = load_camera_list(base_path, case_name)
    
    if cameras is None:
        return None
    
    # Find matching camera
    try:
        return cameras.index(camera_name)
    except ValueError:
        return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show camera index mapping or find index for a specific camera"
    )
    parser.add_argument(
        "--base_path", 
        type=str, 
        required=True, 
        help="Path to the data directory"
    )
    parser.add_argument(
        "--case_name", 
        type=str, 
        default=None, 
        help="Episode name (e.g., 'episode_0'). If not provided, uses first available episode"
    )
    parser.add_argument(
        "--find_camera", 
        type=str, 
        default=None, 
        help="Find the index of a specific camera name (e.g., 'brics-odroid-022_cam1')"
    )
    
    args = parser.parse_args()
    
    if args.find_camera:
        # Find specific camera
        idx = find_camera_index(args.base_path, args.find_camera, args.case_name)
        if idx is not None:
            print(f"\n✓ Camera '{args.find_camera}' found at index: {idx}")
            print(f"\nUsage example:")
            print(f"  python script_inference.py --base_path {args.base_path} --ep_idx 0 --cam_idx {idx}")
        else:
            print(f"\n✗ Camera '{args.find_camera}' not found")
            print(f"\nShowing all available cameras:\n")
            show_camera_mapping(args.base_path, args.case_name)
    else:
        # Show all mappings
        show_camera_mapping(args.base_path, args.case_name)
