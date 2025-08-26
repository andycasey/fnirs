#!/usr/bin/env python3
"""
Example of spherical harmonics modeling for fNIRS hemoglobin concentration data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from spherical_projection import (project_fnirs_to_sphere, 
                                create_spherical_harmonics_basis,
                                visualize_spherical_projection)
from load_fnirs_part_ii import load_hemodynamic_data


def fit_spherical_harmonics_model(data: np.ndarray, 
                                 theta: np.ndarray, 
                                 phi: np.ndarray,
                                 max_degree: int = 4,
                                 regularization: float = 1e-6) -> dict:
    """
    Fit spherical harmonics model to data on a sphere.
    
    Parameters
    ----------
    data : np.ndarray
        Data values at spherical positions, shape (N,) or (N, T) for time series
    theta : np.ndarray
        Polar angles (colatitude) [0, π]
    phi : np.ndarray  
        Azimuthal angles (longitude) [0, 2π]
    max_degree : int
        Maximum degree for spherical harmonics
    regularization : float
        L2 regularization parameter
        
    Returns
    -------
    result : dict
        Fitting results containing coefficients, reconstructed data, etc.
    """
    data = np.array(data)
    if data.ndim == 1:
        data = data[:, np.newaxis]
        single_timepoint = True
    else:
        single_timepoint = False
        
    n_positions, n_timepoints = data.shape
    
    # Create real spherical harmonics basis
    basis, basis_labels = create_spherical_harmonics_basis(theta, phi, max_degree)
    n_basis = basis.shape[1]
    
    # Fit coefficients for each timepoint
    coefficients = np.zeros((n_basis, n_timepoints))
    reconstructed = np.zeros_like(data)
    
    for t in range(n_timepoints):
        # Add L2 regularization
        A = np.concatenate([basis, 
                           regularization * np.eye(n_basis)], axis=0)
        b = np.concatenate([data[:, t],
                           np.zeros(n_basis)])
        
        # Solve regularized least squares
        coef, _ = nnls(A, b)
        coefficients[:, t] = coef
        
        # Reconstruct data
        reconstructed[:, t] = basis @ coef
    
    # Compute reconstruction error
    mse = np.mean((data - reconstructed) ** 2)
    r_squared = 1 - np.sum((data - reconstructed) ** 2) / np.sum((data - np.mean(data)) ** 2)
    
    result = {
        'coefficients': coefficients,
        'reconstructed': reconstructed[:, 0] if single_timepoint else reconstructed,
        'original': data[:, 0] if single_timepoint else data,
        'basis_matrix': basis,
        'basis_labels': basis_labels,
        'mse': mse,
        'r_squared': r_squared,
        'max_degree': max_degree,
        'n_basis_functions': n_basis,
        'regularization': regularization
    }
    
    return result


def analyze_temporal_spherical_harmonics(hemo_data, 
                                       sphere_result,
                                       chromophore: str = 'HbO',
                                       max_degree: int = 4,
                                       time_subsample: int = 10):
    """
    Analyze temporal evolution of spherical harmonics coefficients.
    
    Parameters
    ----------
    hemo_data : HemodynamicData
        Loaded hemodynamic data
    sphere_result : dict
        Result from project_fnirs_to_sphere
    chromophore : str
        Which chromophore to analyze ('HbO', 'HbR', 'HbT')
    max_degree : int
        Maximum spherical harmonics degree
    time_subsample : int
        Temporal subsampling factor (for computational efficiency)
        
    Returns
    -------
    temporal_result : dict
        Analysis results
    """
    
    # Get data
    if chromophore == 'HbO':
        data_full = hemo_data.get_hbo_data()
    elif chromophore == 'HbR':
        data_full = hemo_data.get_hbr_data()
    elif chromophore == 'HbT':
        data_full = hemo_data.get_hbt_data()
    else:
        raise ValueError(f"Unknown chromophore: {chromophore}")
    
    # Subsample time for efficiency
    time_indices = np.arange(0, data_full.shape[0], time_subsample)
    data = data_full[time_indices, :]  # Shape: (subsampled_time, channels)
    time_subsampled = hemo_data.time[time_indices]
    
    print(f"Analyzing {chromophore} temporal evolution:")
    print(f"  Original time series: {data_full.shape}")
    print(f"  Subsampled: {data.shape}")
    print(f"  Time range: {time_subsampled[0]:.1f} to {time_subsampled[-1]:.1f}s")
    
    # Fit spherical harmonics model
    theta = sphere_result['theta']
    phi = sphere_result['phi']
    
    print(f"  Fitting spherical harmonics (degree {max_degree})...")
    sh_result = fit_spherical_harmonics_model(data.T,  # Transpose to (channels, time)
                                            theta, phi,
                                            max_degree=max_degree)
    
    print(f"  Model performance: R² = {sh_result['r_squared']:.4f}")
    print(f"  MSE: {sh_result['mse']:.2e}")
    
    # Analyze coefficient evolution
    coefficients = sh_result['coefficients']  # Shape: (n_basis, time)
    
    # Compute temporal statistics for each basis function
    coef_temporal_mean = np.mean(coefficients, axis=1)
    coef_temporal_std = np.std(coefficients, axis=1)
    coef_temporal_range = np.ptp(coefficients, axis=1)  # peak-to-peak
    
    temporal_result = {
        'sh_result': sh_result,
        'time_subsampled': time_subsampled,
        'coefficients_temporal_mean': coef_temporal_mean,
        'coefficients_temporal_std': coef_temporal_std, 
        'coefficients_temporal_range': coef_temporal_range,
        'chromophore': chromophore,
        'max_degree': max_degree
    }
    
    return temporal_result


def plot_spherical_harmonics_analysis(temporal_result, sphere_result, n_time_snapshots: int = 4):
    """Plot spherical harmonics analysis results."""
    
    sh_result = temporal_result['sh_result']
    time_sub = temporal_result['time_subsampled']
    chromophore = temporal_result['chromophore']
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Coefficient evolution over time
    ax1 = plt.subplot(3, 4, 1)
    coefficients = sh_result['coefficients']
    n_basis = coefficients.shape[0]
    
    # Show first few basis functions
    for i in range(min([6, n_basis])):
        plt.plot(time_sub, coefficients[i, :], label=f'Basis {i}', alpha=0.7)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Coefficient magnitude')
    plt.title(f'{chromophore} SH Coefficients vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Coefficient magnitude distribution
    ax2 = plt.subplot(3, 4, 2)
    coef_magnitudes = temporal_result['coefficients_temporal_std']
    basis_indices = np.arange(len(coef_magnitudes))
    plt.bar(basis_indices, coef_magnitudes)
    plt.xlabel('Basis Function Index')
    plt.ylabel('Temporal Std Dev')
    plt.title('Temporal Variability by Basis')
    plt.grid(True, alpha=0.3)
    
    # 3. Model fit quality
    ax3 = plt.subplot(3, 4, 3)
    original = sh_result['original']
    reconstructed = sh_result['reconstructed']
    
    # Show fit for a sample timepoint
    sample_t = original.shape[1] // 2 if original.ndim > 1 else 0
    if original.ndim > 1:
        orig_sample = original[:, sample_t]
        recon_sample = reconstructed[:, sample_t]
    else:
        orig_sample = original
        recon_sample = reconstructed
        
    plt.scatter(orig_sample, recon_sample, alpha=0.6, s=30)
    data_range = [min([orig_sample.min(), recon_sample.min()]),
                  max([orig_sample.max(), recon_sample.max()])]
    plt.plot(data_range, data_range, 'r--', alpha=0.7)
    plt.xlabel('Original Data')
    plt.ylabel('Reconstructed Data') 
    plt.title(f'Model Fit (R² = {sh_result["r_squared"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    # 4. Spatial reconstruction snapshots
    theta = sphere_result['theta']
    phi = sphere_result['phi']
    
    # Show reconstructions at different timepoints
    for i, snapshot_idx in enumerate(np.linspace(0, original.shape[1]-1, n_time_snapshots, dtype=int)):
        ax = plt.subplot(3, 4, 5 + i)
        
        if original.ndim > 1:
            data_snapshot = original[:, snapshot_idx]
            recon_snapshot = reconstructed[:, snapshot_idx]
            time_point = time_sub[snapshot_idx]
        else:
            data_snapshot = original
            recon_snapshot = reconstructed
            time_point = time_sub[0]
            
        # Plot on spherical coordinates  
        scatter = plt.scatter(phi, theta, c=data_snapshot, cmap='RdBu_r', s=40)
        plt.colorbar(scatter, ax=ax)
        plt.xlabel('Phi (azimuth)')
        plt.ylabel('Theta (colatitude)')
        plt.title(f'Original t={time_point:.0f}s')
        plt.grid(True, alpha=0.3)
        
        # Reconstruction
        if i < n_time_snapshots - 1:
            ax_recon = plt.subplot(3, 4, 9 + i)
            scatter = plt.scatter(phi, theta, c=recon_snapshot, cmap='RdBu_r', s=40)
            plt.colorbar(scatter, ax=ax_recon)
            plt.xlabel('Phi (azimuth)')
            plt.ylabel('Theta (colatitude)')
            plt.title(f'Reconstructed t={time_point:.0f}s')
            plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Spherical Harmonics Analysis: {chromophore}')
    plt.tight_layout()
    plt.show()


def main():
    """Complete example of spherical harmonics modeling for fNIRS data."""
    
    print("=== fNIRS Spherical Harmonics Modeling Example ===")
    
    # Load data
    print("\n1. Loading fNIRS data...")
    hemo_data = load_hemodynamic_data("rsFC-fnirs-course/Data_for_Part_II.mat")
    
    # Project to sphere
    print("\n2. Projecting to sphere...")
    coords_3d = hemo_data.get_spatial_coordinates_3d()
    sphere_result = project_fnirs_to_sphere(coords_3d, fit_method='least_squares')
    
    print(f"   Fitted sphere: radius = {sphere_result['sphere_radius']:.1f} cm")
    print(f"   Mean projection error: {np.mean(sphere_result['projection_errors']):.3f} cm")
    
    # Analyze different chromophores
    print("\n3. Fitting spherical harmonics models...")
    
    results = {}
    for chromophore in ['HbO', 'HbR']:
        print(f"\n   Analyzing {chromophore}...")
        temporal_result = analyze_temporal_spherical_harmonics(
            hemo_data, sphere_result,
            chromophore=chromophore,
            max_degree=4,
            time_subsample=20  # Every 20th timepoint for efficiency
        )
        results[chromophore] = temporal_result
        
        # Plot results
        plot_spherical_harmonics_analysis(temporal_result, sphere_result)
    
    print("\n=== Summary ===")
    for chromophore, result in results.items():
        sh_r = result['sh_result']
        print(f"{chromophore}:")
        print(f"  Model R²: {sh_r['r_squared']:.4f}")
        print(f"  MSE: {sh_r['mse']:.2e}")
        print(f"  Basis functions: {sh_r['n_basis_functions']}")
        print(f"  Most variable basis: #{np.argmax(result['coefficients_temporal_std'])}")
    
    return results, sphere_result, hemo_data


if __name__ == "__main__":
    results, sphere_result, hemo_data = main()