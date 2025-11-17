#!/usr/bin/env python3
"""
Test script showing the difference between naive concatenation and proper real spherical harmonics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from spherical_projection import create_spherical_harmonics_basis, project_fnirs_to_sphere
from load_fnirs_part_ii import load_hemodynamic_data


def create_naive_basis(theta, phi, max_degree=2):
    """Create basis by naively concatenating real and imaginary parts (WRONG WAY)."""
    n_points = len(theta)
    n_basis = (max_degree + 1) ** 2
    basis_complex = np.zeros((n_points, n_basis), dtype=complex)
    
    col_idx = 0
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, phi, theta)
            basis_complex[:, col_idx] = Y_lm
            col_idx += 1
    
    # Naive approach: just concatenate real and imaginary parts
    basis_real = np.real(basis_complex)
    basis_imag = np.imag(basis_complex)
    naive_basis = np.concatenate([basis_real, basis_imag], axis=1)
    
    return naive_basis, basis_complex


def main():
    """Compare naive vs proper spherical harmonics basis."""
    
    # Load data
    hemo_data = load_hemodynamic_data("rsFC-fnirs-course/Data_for_Part_II.mat")
    coords_3d = hemo_data.get_spatial_coordinates_3d()
    sphere_result = project_fnirs_to_sphere(coords_3d)
    
    theta = sphere_result['theta']
    phi = sphere_result['phi']
    
    print("=== Comparison: Naive vs Proper Spherical Harmonics Basis ===")
    
    max_degree = 2  # Use degree 2 for clear comparison
    
    # Method 1: Naive approach (concatenate real/imag parts)
    naive_basis, complex_basis = create_naive_basis(theta, phi, max_degree)
    
    # Method 2: Proper real spherical harmonics
    proper_basis, labels = create_spherical_harmonics_basis(theta, phi, max_degree)
    
    print(f"\nComplex spherical harmonics shape: {complex_basis.shape}")
    print(f"Naive basis (real + imag concatenated): {naive_basis.shape}")
    print(f"Proper real basis: {proper_basis.shape}")
    
    print(f"\nExpected number of unique basis functions for degree {max_degree}: {(max_degree+1)**2}")
    
    # Check for redundancy in naive approach
    print(f"\n=== Checking for Redundant Columns ===")
    
    def check_redundancy(basis, name):
        correlation_matrix = np.corrcoef(basis.T)
        np.fill_diagonal(correlation_matrix, 0)
        
        # Find highly correlated pairs (> 0.99)
        high_corr = np.where(np.abs(correlation_matrix) > 0.99)
        redundant_pairs = [(i, j) for i, j in zip(high_corr[0], high_corr[1]) if i < j]
        
        print(f"{name}:")
        print(f"  Shape: {basis.shape}")
        print(f"  Max correlation between columns: {np.abs(correlation_matrix).max():.6f}")
        print(f"  Number of redundant pairs: {len(redundant_pairs)}")
        
        if len(redundant_pairs) > 0 and len(redundant_pairs) <= 5:
            for i, j in redundant_pairs[:5]:
                corr = correlation_matrix[i, j]
                print(f"    Columns {i} and {j}: correlation = {corr:.6f}")
        
        return len(redundant_pairs)
    
    naive_redundant = check_redundancy(naive_basis, "Naive approach")
    proper_redundant = check_redundancy(proper_basis, "Proper approach")
    
    # Visualize first few basis functions
    print(f"\n=== Visualization ===")
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Top row: naive basis
    for i in range(5):
        ax = axes[0, i]
        scatter = ax.scatter(phi, theta, c=naive_basis[:, i], cmap='RdBu_r', s=30)
        ax.set_title(f'Naive Basis {i+1}')
        ax.set_xlabel('Phi')
        ax.set_ylabel('Theta')
        plt.colorbar(scatter, ax=ax)
    
    # Bottom row: proper basis
    for i in range(5):
        ax = axes[1, i]
        scatter = ax.scatter(phi, theta, c=proper_basis[:, i], cmap='RdBu_r', s=30)
        ax.set_title(f'Proper: {labels[i]}')
        ax.set_xlabel('Phi') 
        ax.set_ylabel('Theta')
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.suptitle('Comparison: Naive vs Proper Spherical Harmonics Basis', y=1.02)
    plt.show()
    
    # Show the mathematical relationship
    print(f"\n=== Mathematical Explanation ===")
    print(f"The issue with naive concatenation:")
    print(f"1. Complex spherical harmonics have symmetry: Y_l^(-m) = (-1)^m * conj(Y_l^m)")
    print(f"2. When you concatenate real and imaginary parts naively, you get:")
    print(f"   - Real parts of Y_l^m and Y_l^(-m) are related")
    print(f"   - Imaginary parts of Y_l^m and Y_l^(-m) are related")
    print(f"3. This creates {naive_redundant} redundant basis functions")
    print(f"")
    print(f"The proper real spherical harmonics approach:")
    print(f"1. For m = 0: Use Y_l^0 (already real)")
    print(f"2. For m > 0: Create cos and sin components:")
    print(f"   - Y_l^m_cos = sqrt(2) * Re(Y_l^m)")
    print(f"   - Y_l^m_sin = sqrt(2) * Im(Y_l^m)")
    print(f"3. Skip m < 0 terms (they're redundant)")
    print(f"4. Results in exactly {(max_degree+1)**2} unique basis functions")
    
    return naive_basis, proper_basis, labels


if __name__ == "__main__":
    naive_basis, proper_basis, labels = main()