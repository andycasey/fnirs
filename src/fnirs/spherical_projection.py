#!/usr/bin/env python3
"""
Functions for projecting 3D fNIRS coordinates onto a sphere for spherical harmonics modeling.
"""

import numpy as np
from typing import Tuple, Optional, Union

def project_to_sphere(positions_3d: np.ndarray, 
                     center: Optional[np.ndarray] = None,
                     radius: Optional[float] = None,
                     method: str = 'radial') -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Project 3D positions onto a sphere.
    
    Parameters
    ----------
    positions_3d : np.ndarray
        3D positions with shape (N, 3)
    center : np.ndarray, optional
        Center of the sphere. If None, uses centroid of positions
    radius : float, optional
        Radius of the sphere. If None, uses mean distance from center
    method : str
        Projection method:
        - 'radial': Project along radial lines from center
        - 'nearest': Project to nearest point on sphere surface
        
    Returns
    -------
    projected_positions : np.ndarray
        Projected positions on sphere surface (N, 3)
    sphere_radius : float
        Radius of the fitted sphere
    sphere_center : np.ndarray
        Center of the fitted sphere
    """
    positions_3d = np.array(positions_3d)
    
    # Determine sphere center
    if center is None:
        sphere_center = np.mean(positions_3d, axis=0)
    else:
        sphere_center = np.array(center)
    
    # Center the positions
    centered_positions = positions_3d - sphere_center
    
    # Calculate distances from center
    distances = np.linalg.norm(centered_positions, axis=1)
    
    # Determine sphere radius
    if radius is None:
        sphere_radius = np.mean(distances)
    else:
        sphere_radius = radius
    
    if method == 'radial':
        # Project along radial lines from center
        # Normalize to unit vectors and scale by sphere radius
        unit_vectors = centered_positions / distances[:, np.newaxis]
        projected_centered = unit_vectors * sphere_radius
        
    elif method == 'nearest':
        # Project to nearest point on sphere (same as radial for convex surfaces)
        unit_vectors = centered_positions / distances[:, np.newaxis]
        projected_centered = unit_vectors * sphere_radius
        
    else:
        raise ValueError(f"Unknown projection method: {method}")
    
    # Translate back to sphere center
    projected_positions = projected_centered + sphere_center
    
    return projected_positions, sphere_radius, sphere_center


def cartesian_to_spherical(positions_3d: np.ndarray, 
                          center: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert 3D Cartesian coordinates to spherical coordinates (r, theta, phi).
    
    Parameters
    ----------
    positions_3d : np.ndarray
        3D positions with shape (N, 3) [x, y, z]
    center : np.ndarray, optional
        Origin for spherical coordinate system. If None, uses (0, 0, 0)
        
    Returns
    -------
    r : np.ndarray
        Radial distances from center
    theta : np.ndarray  
        Polar angles (colatitude) in radians [0, π]
    phi : np.ndarray
        Azimuthal angles (longitude) in radians [0, 2π]
        
    Notes
    -----
    Uses physics convention:
    - theta: angle from positive z-axis (colatitude) 
    - phi: angle from positive x-axis in xy-plane (azimuth)
    """
    positions_3d = np.array(positions_3d)
    
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.array(center)
    
    # Center the positions
    centered = positions_3d - center
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
    
    # Calculate spherical coordinates
    r = np.linalg.norm(centered, axis=1)
    theta = np.arccos(z / r)  # Polar angle (0 to π)
    phi = np.arctan2(y, x)    # Azimuthal angle (-π to π)
    
    # Convert phi to [0, 2π] range
    phi = np.where(phi < 0, phi + 2*np.pi, phi)
    
    return r, theta, phi


def spherical_to_cartesian(r: np.ndarray, 
                          theta: np.ndarray, 
                          phi: np.ndarray,
                          center: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert spherical coordinates to 3D Cartesian coordinates.
    
    Parameters
    ----------
    r : np.ndarray
        Radial distances
    theta : np.ndarray
        Polar angles (colatitude) in radians [0, π]
    phi : np.ndarray
        Azimuthal angles (longitude) in radians [0, 2π]
    center : np.ndarray, optional
        Origin for coordinate system. If None, uses (0, 0, 0)
        
    Returns
    -------
    positions_3d : np.ndarray
        3D Cartesian positions with shape (N, 3)
    """
    if center is None:
        center = np.array([0.0, 0.0, 0.0])
    else:
        center = np.array(center)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)  
    z = r * np.cos(theta)
    
    positions_3d = np.column_stack([x, y, z]) + center
    
    return positions_3d


def fit_sphere_to_head(positions_3d: np.ndarray, 
                      method: str = 'least_squares') -> Tuple[np.ndarray, float]:
    """
    Fit a sphere to 3D head surface positions (fNIRS optode positions).
    
    Parameters
    ----------
    positions_3d : np.ndarray
        3D optode positions with shape (N, 3)
    method : str
        Fitting method:
        - 'centroid': Use centroid as center, mean distance as radius
        - 'least_squares': Minimize sum of squared distance errors
        
    Returns
    -------
    center : np.ndarray
        Center of fitted sphere
    radius : float
        Radius of fitted sphere
    """
    positions_3d = np.array(positions_3d)
    
    if method == 'centroid':
        center = np.mean(positions_3d, axis=0)
        distances = np.linalg.norm(positions_3d - center, axis=1)
        radius = np.mean(distances)
        
    elif method == 'least_squares':
        # Minimize ||pos - center||^2 - r^2 for all positions
        from scipy.optimize import minimize
        
        # Initial guess: centroid method
        center_init = np.mean(positions_3d, axis=0)
        distances_init = np.linalg.norm(positions_3d - center_init, axis=1)
        radius_init = np.mean(distances_init)
        
        def objective(params):
            center = params[:3]
            radius = params[3]
            distances = np.linalg.norm(positions_3d - center, axis=1)
            return np.sum((distances - radius) ** 2)
        
        result = minimize(objective, 
                         np.concatenate([center_init, [radius_init]]),
                         method='L-BFGS-B')
        
        center = result.x[:3]
        radius = result.x[3]
        
    else:
        raise ValueError(f"Unknown fitting method: {method}")
    
    return center, radius


def _compute_optimal_rotation(projected_positions: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Compute optimal rotation matrix to place poles and phi boundaries away from data positions.
    
    For fNIRS data on human skulls, we want to rotate the coordinate system so that:
    1. The poles (theta=0, theta=π) are in regions with no data
    2. The phi boundary (phi=0/2π) is in a region with no data
    
    Strategy:
    - Find the centroid of the data positions on the sphere
    - Rotate so this centroid is at theta=π/2, phi=π (opposite of typical "front")
    - This places poles at unused regions and phi boundary away from dense data
    
    Parameters
    ----------
    projected_positions : np.ndarray
        Positions projected onto sphere surface
    center : np.ndarray
        Center of the sphere
        
    Returns
    -------
    rotation_matrix : np.ndarray
        3x3 rotation matrix to apply to coordinates
    """
    # Center the positions
    centered_positions = projected_positions - center
    
    # Find the centroid direction of the data on the sphere
    # This represents the "center" of where the data is located
    data_centroid = np.mean(centered_positions, axis=0)
    data_centroid = data_centroid / np.linalg.norm(data_centroid)
    
    # We want to rotate so that:
    # 1. data_centroid -> direction (0, -1, 0) which corresponds to theta=π/2, phi=3π/2
    # This puts the north pole (theta=0) and south pole (theta=π) away from data
    # And puts phi=0 boundary away from the main data region
    
    target_direction = np.array([0.0, -1.0, 0.0])
    
    # Compute rotation matrix to align data_centroid with target_direction
    # Using Rodrigues' rotation formula
    v = np.cross(data_centroid, target_direction)
    s = np.linalg.norm(v)
    c = np.dot(data_centroid, target_direction)
    
    if s < 1e-10:  # Vectors are already aligned or opposite
        if c > 0:  # Same direction
            rotation_matrix = np.eye(3)
        else:  # Opposite direction - rotate 180° around any perpendicular axis
            # Find a perpendicular axis
            if abs(data_centroid[0]) < 0.9:
                perp = np.array([1.0, 0.0, 0.0])
            else:
                perp = np.array([0.0, 1.0, 0.0])
            perp = perp - np.dot(perp, data_centroid) * data_centroid
            perp = perp / np.linalg.norm(perp)
            # 180° rotation around perpendicular axis
            rotation_matrix = 2 * np.outer(perp, perp) - np.eye(3)
    else:
        # General case - use Rodrigues' formula
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    
    return rotation_matrix


def inverse_rotation_transform(positions: np.ndarray, 
                              rotation_matrix: np.ndarray, 
                              center: np.ndarray) -> np.ndarray:
    """
    Apply inverse rotation to transform positions back to original coordinate system.
    
    Parameters
    ----------
    positions : np.ndarray
        Positions in rotated coordinate system
    rotation_matrix : np.ndarray
        Original rotation matrix (3x3)
    center : np.ndarray
        Sphere center
        
    Returns
    -------
    original_positions : np.ndarray
        Positions in original coordinate system
    """
    if rotation_matrix is None:
        return positions
    
    # Apply inverse rotation (transpose of rotation matrix)
    centered_pos = positions - center
    original_centered_pos = centered_pos @ rotation_matrix  # Inverse is transpose
    original_positions = original_centered_pos + center
    
    return original_positions

def project_fnirs_to_sphere(positions_3d: np.ndarray,
                           fit_method: str = 'least_squares',
                           projection_method: str = 'radial',
                           rotate_poles: bool = True) -> dict:
    """
    Complete pipeline to project fNIRS positions onto sphere and convert to spherical coordinates.
    
    Parameters
    ----------
    positions_3d : np.ndarray
        3D fNIRS optode positions with shape (N, 3)
    fit_method : str
        Method to fit sphere ('centroid' or 'least_squares')
    projection_method : str
        Method to project onto sphere ('radial')
    rotate_poles : bool
        If True, rotate coordinate system to place poles and phi boundaries 
        away from data positions to avoid singularities
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'sphere_center': Center of fitted sphere
        - 'sphere_radius': Radius of fitted sphere  
        - 'projected_positions': 3D positions on sphere surface
        - 'theta': Polar angles (colatitude) [0, π]
        - 'phi': Azimuthal angles (longitude) [0, 2π]
        - 'original_positions': Original input positions
        - 'projection_errors': Distance from original to projected positions
        - 'rotation_matrix': Rotation matrix applied (if rotate_poles=True)
    """
    positions_3d = np.array(positions_3d)
    
    # Fit sphere to the positions
    center, radius = fit_sphere_to_head(positions_3d, method=fit_method)
    
    # Project positions onto sphere
    projected_pos, _, _ = project_to_sphere(positions_3d, 
                                          center=center, 
                                          radius=radius,
                                          method=projection_method)
    
    # Determine optimal rotation to avoid poles/boundaries at data positions
    rotation_matrix = None
    rotated_projected_pos = projected_pos
    
    if rotate_poles:
        rotation_matrix = _compute_optimal_rotation(projected_pos, center)
        # Apply rotation to projected positions (centered at sphere center)
        centered_pos = projected_pos - center
        rotated_centered_pos = centered_pos @ rotation_matrix.T
        rotated_projected_pos = rotated_centered_pos + center
    
    # Convert to spherical coordinates
    r, theta, phi = cartesian_to_spherical(rotated_projected_pos, center=center)
    
    # Calculate projection errors
    projection_errors = np.linalg.norm(positions_3d - projected_pos, axis=1)
    
    result = {
        'sphere_center': center,
        'sphere_radius': radius,
        'projected_positions': rotated_projected_pos,
        'theta': theta,
        'phi': phi,
        'r': r,  # Should be constant = radius
        'original_positions': positions_3d,
        'projection_errors': projection_errors,
        'fit_method': fit_method,
        'projection_method': projection_method,
        'rotation_matrix': rotation_matrix,
        'unrotated_projected_positions': projected_pos
    }
    
    return result


def visualize_spherical_projection(result: dict, 
                                 data_values: Optional[np.ndarray] = None,
                                 title: str = "fNIRS Spherical Projection"):
    """
    Visualize the spherical projection results.
    
    Parameters
    ----------
    result : dict
        Result from project_fnirs_to_sphere()
    data_values : np.ndarray, optional
        Data values to color-code the points
    title : str
        Plot title
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15, 5))
    
    # Original 3D positions
    ax1 = fig.add_subplot(131, projection='3d')
    pos_orig = result['original_positions']
    
    if data_values is not None:
        scatter = ax1.scatter(pos_orig[:, 0], pos_orig[:, 1], pos_orig[:, 2], 
                            c=data_values, cmap='RdBu_r', s=50)
        plt.colorbar(scatter, ax=ax1, shrink=0.8)
    else:
        ax1.scatter(pos_orig[:, 0], pos_orig[:, 1], pos_orig[:, 2], s=50)
    
    ax1.set_xlabel('X (cm)')
    ax1.set_ylabel('Y (cm)')  
    ax1.set_zlabel('Z (cm)')
    ax1.set_title('Original 3D Positions')
    
    # Projected positions on sphere
    ax2 = fig.add_subplot(132, projection='3d')
    pos_proj = result['projected_positions']
    center = result['sphere_center']
    radius = result['sphere_radius']
    
    if data_values is not None:
        scatter = ax2.scatter(pos_proj[:, 0], pos_proj[:, 1], pos_proj[:, 2], 
                            c=data_values, cmap='RdBu_r', s=50)
        plt.colorbar(scatter, ax=ax2, shrink=0.8)
    else:
        ax2.scatter(pos_proj[:, 0], pos_proj[:, 1], pos_proj[:, 2], s=50)
    
    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + center[1] 
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax2.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.3, color='gray')
    
    ax2.set_xlabel('X (cm)')
    ax2.set_ylabel('Y (cm)')
    ax2.set_zlabel('Z (cm)')
    ax2.set_title('Projected on Sphere')
    
    # Spherical coordinates (theta-phi map)
    ax3 = fig.add_subplot(133)
    theta = result['theta'] 
    phi = result['phi']
    
    if data_values is not None:
        scatter = ax3.scatter(phi, theta, c=data_values, cmap='RdBu_r', s=50)
        plt.colorbar(scatter, ax=ax3)
    else:
        ax3.scatter(phi, theta, s=50)
    
    ax3.set_xlabel('Phi (azimuth) [radians]')
    ax3.set_ylabel('Theta (colatitude) [radians]')
    ax3.set_title('Spherical Coordinates')
    ax3.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Spherical Projection Summary:")
    print(f"  Sphere center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}] cm")
    print(f"  Sphere radius: {radius:.2f} cm")
    print(f"  Projection errors: {result['projection_errors'].min():.3f} to {result['projection_errors'].max():.3f} cm")
    print(f"  Mean projection error: {np.mean(result['projection_errors']):.3f} cm")
    print(f"  Theta range: {theta.min():.3f} to {theta.max():.3f} radians")
    print(f"  Phi range: {phi.min():.3f} to {phi.max():.3f} radians")
    
    if result['rotation_matrix'] is not None:
        print(f"  Coordinate system rotated to avoid poles/boundaries at data positions")
    else:
        print(f"  Using standard coordinate system (no rotation applied)")


