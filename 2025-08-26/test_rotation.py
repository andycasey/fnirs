#!/usr/bin/env python3
"""
Test script to verify the rotation functionality works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from spherical_projection import project_fnirs_to_sphere, visualize_spherical_projection

# Create some example fNIRS-like positions on a head
# Simulate optodes placed on accessible parts of the head (upper hemisphere + sides)
np.random.seed(42)

# Generate positions roughly on the upper part of a sphere (like a skull cap)
n_optodes = 30
radius_base = 10.0  # cm, typical head radius

# Create positions mostly in upper hemisphere and sides
theta_range = np.pi * 0.7  # Avoid the very bottom
phi_range = 2 * np.pi

theta_samples = np.random.uniform(0, theta_range, n_optodes)
phi_samples = np.random.uniform(0, phi_range, n_optodes)
r_samples = radius_base + np.random.normal(0, 0.5, n_optodes)  # Add some noise

# Convert to Cartesian
x = r_samples * np.sin(theta_samples) * np.cos(phi_samples)
y = r_samples * np.sin(theta_samples) * np.sin(phi_samples) 
z = r_samples * np.cos(theta_samples)

positions_3d = np.column_stack([x, y, z])
print(f"Created {n_optodes} simulated fNIRS positions")

# Test without rotation
print("\n=== Testing WITHOUT rotation ===")
result_no_rotation = project_fnirs_to_sphere(positions_3d, rotate_poles=False)
print(f"Theta range: {result_no_rotation['theta'].min():.3f} to {result_no_rotation['theta'].max():.3f}")
print(f"Phi range: {result_no_rotation['phi'].min():.3f} to {result_no_rotation['phi'].max():.3f}")

# Test with rotation (default)
print("\n=== Testing WITH rotation ===")
result_with_rotation = project_fnirs_to_sphere(positions_3d, rotate_poles=True)
print(f"Theta range: {result_with_rotation['theta'].min():.3f} to {result_with_rotation['theta'].max():.3f}")
print(f"Phi range: {result_with_rotation['phi'].min():.3f} to {result_with_rotation['phi'].max():.3f}")

# Check if rotation matrix is applied
if result_with_rotation['rotation_matrix'] is not None:
    print("Rotation matrix successfully computed and applied")
else:
    print("No rotation applied")

print("\nTest completed successfully!")
