#!/usr/bin/env python3
"""
Example showing how to use the 3D particle in your interactive playground.
The particle will be properly projected from 3D world coordinates to 2D screen coordinates.
"""

import numpy as np

def example_3d_particle_usage():
    """Example of how to use the 3D particle functionality"""
    
    print("=== 3D Particle Usage Example ===")
    
    # Example: Set different 3D particle positions
    particle_positions = [
        [0.0, 0.0, 0.5],    # 0.5m above origin
        [0.2, 0.1, 0.3],    # Offset position
        [0.0, 0.0, 1.0],    # 1m above origin
        [-0.1, 0.2, 0.4],   # Negative X, positive Y
    ]
    
    print("You can set the 3D particle position using:")
    print("trainer.set_3d_particle_position([x, y, z])")
    print()
    
    for i, pos in enumerate(particle_positions):
        print(f"Example {i+1}: Position {pos}")
        print(f"  trainer.set_3d_particle_position({pos})")
        print()
    
    print("The particle will be:")
    print("- Projected from 3D world coordinates to 2D screen coordinates")
    print("- Displayed as a red circle where it appears on screen")
    print("- Only visible if it's in front of the camera and within screen bounds")
    print("- Labeled with its 3D coordinates")

def integrate_with_interactive_playground():
    """How to integrate this with your interactive playground"""
    
    print("\n=== Integration with Interactive Playground ===")
    
    code_example = '''
# In your interactive_playground.py or wherever you initialize the trainer:

# Set ground transform (if you have one)
ground_transform = np.eye(4)
ground_transform[:3, :3] = R_world_avg
ground_transform[:3, 3] = t_world_avg
trainer.set_ground_transform(ground_transform)

# Set 3D particle position
trainer.set_3d_particle_position([0.0, 0.0, 0.5])  # 0.5m above origin

# Run interactive playground
trainer.interactive_playground(
    best_model_path,
    gaussians_path,
    n_ctrl_parts=1
)
'''
    
    print(code_example)

if __name__ == "__main__":
    example_3d_particle_usage()
    integrate_with_interactive_playground()
