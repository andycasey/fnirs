import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
from typing import Tuple
from functools import partial


def associated_legendre_polynomial(l: int, m: int, x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the associated Legendre polynomial P_l^m(x) using JAX.
    
    Args:
        l: Degree of the polynomial
        m: Order of the polynomial (|m| <= l)
        x: Input values, typically cos(theta) where theta is colatitude
        
    Returns:
        Values of P_l^m(x)
    """
    # Handle the case where |m| > l
    m_abs = jnp.abs(m)
    
    # For |m| > l, return zeros
    result = jnp.where(m_abs > l, 0.0, _compute_legendre(l, m_abs, x))
    
    # Apply the sign correction for negative m
    sign_factor = jnp.where(m < 0, (-1)**m_abs, 1.0)
    
    return sign_factor * result


def _compute_legendre(l: int, m: int, x: jnp.ndarray) -> jnp.ndarray:
    """Helper function to compute P_l^m(x) for m >= 0."""
    # Start with P_0^0 = 1
    if l == 0 and m == 0:
        return jnp.ones_like(x)
    
    # For m > l, return 0
    if m > l:
        return jnp.zeros_like(x)
    
    # Initialize arrays for the recurrence relation
    # We'll compute all P_k^m for k from m to l
    sqrt_1_minus_x2 = jnp.sqrt(1 - x**2)
    
    # Base case: P_m^m
    pmm = jnp.ones_like(x)
    if m > 0:
        # P_m^m = (-1)^m * (2m-1)!! * (1-x^2)^(m/2)
        double_factorial = 1.0
        for i in range(1, m + 1):
            double_factorial *= (2 * i - 1)
        pmm = ((-1)**m) * double_factorial * (sqrt_1_minus_x2**m)
    
    if l == m:
        return pmm
    
    # Next case: P_{m+1}^m
    pmm1 = x * (2 * m + 1) * pmm
    
    if l == m + 1:
        return pmm1
    
    # General recurrence: P_l^m = ((2l-1)*x*P_{l-1}^m - (l+m-1)*P_{l-2}^m) / (l-m)
    pll_minus_2 = pmm
    pll_minus_1 = pmm1
    
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pll_minus_1 - (ll + m - 1) * pll_minus_2) / (ll - m)
        pll_minus_2 = pll_minus_1
        pll_minus_1 = pll
    
    return pll_minus_1


def spherical_harmonic(l: int, m: int, theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the spherical harmonic Y_l^m(theta, phi).
    
    Args:
        l: Degree (l >= 0)
        m: Order (-l <= m <= l)
        theta: Colatitude angles (0 to pi)
        phi: Azimuthal angles (0 to 2*pi)
        
    Returns:
        Complex values of Y_l^m(theta, phi)
    """
    # Normalization constant
    norm = jnp.sqrt((2 * l + 1) / (4 * jnp.pi) * 
                    factorial_ratio(l - jnp.abs(m), l + jnp.abs(m)))
    
    # Associated Legendre polynomial
    cos_theta = jnp.cos(theta)
    legendre = associated_legendre_polynomial(l, jnp.abs(m), cos_theta)
    
    # Exponential part
    exp_part = jnp.exp(1j * m * phi)
    
    # Apply sign for negative m
    sign = jnp.where(m < 0, (-1)**jnp.abs(m), 1.0)
    
    return sign * norm * legendre * exp_part


def factorial_ratio(n_minus: int, n_plus: int) -> float:
    """Compute (n_minus)! / (n_plus)! efficiently."""
    if n_minus > n_plus:
        return 1.0 / factorial_ratio(n_plus, n_minus)
    
    result = 1.0
    for i in range(n_minus + 1, n_plus + 1):
        result *= i
    return 1.0 / result


def spherical_harmonic_design_matrix(theta: jnp.ndarray, 
                                   phi: jnp.ndarray, 
                                   l_max: int,
                                   real_basis: bool = True) -> jnp.ndarray:
    """
    Create a design matrix for spherical harmonics up to degree l_max.
    
    Args:
        theta: Colatitude angles, shape (N,)
        phi: Azimuthal angles, shape (N,)
        l_max: Maximum degree of spherical harmonics
        real_basis: If True, use real spherical harmonics; if False, use complex
        
    Returns:
        Design matrix of shape (N, (l_max + 1)^2) containing spherical harmonic
        basis functions evaluated at the input points
    """
    n_points = len(theta)
    n_basis = (l_max + 1) ** 2
    
    if real_basis:
        design_matrix = jnp.zeros((n_points, n_basis), dtype=jnp.float64)
    else:
        design_matrix = jnp.zeros((n_points, n_basis), dtype=jnp.complex128)
    
    col_idx = 0
    
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Compute the spherical harmonic
            Y_lm = spherical_harmonic(l, m, theta, phi)
            
            if real_basis:
                if m == 0:
                    # m = 0: Y_l^0 is already real
                    Y_real = jnp.real(Y_lm)
                elif m > 0:
                    # m > 0: Real part of Y_l^m (cosine term)
                    Y_real = jnp.sqrt(2) * jnp.real(Y_lm)
                else:  # m < 0
                    # m < 0: Imaginary part of Y_l^|m| (sine term)
                    Y_real = jnp.sqrt(2) * jnp.imag(spherical_harmonic(l, -m, theta, phi))
                
                design_matrix = design_matrix.at[:, col_idx].set(Y_real)
            else:
                design_matrix = design_matrix.at[:, col_idx].set(Y_lm)
            
            col_idx += 1
    
    return design_matrix


def get_spherical_harmonic_indices(l_max: int) -> list:
    """
    Get the (l, m) indices corresponding to each column of the design matrix.
    
    Args:
        l_max: Maximum degree
        
    Returns:
        List of (l, m) tuples
    """
    indices = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            indices.append((l, m))
    return indices


# Example usage and helper functions
def create_spherical_grid(n_theta: int, n_phi: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create a regular grid on the sphere.
    
    Args:
        n_theta: Number of colatitude points
        n_phi: Number of azimuthal points
        
    Returns:
        theta, phi arrays flattened to 1D
    """
    theta = jnp.linspace(0, jnp.pi, n_theta)
    phi = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False)
    
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing='ij')
    
    return theta_grid.flatten(), phi_grid.flatten()


def fit_spherical_harmonics(data: jnp.ndarray,
                          theta: jnp.ndarray,
                          phi: jnp.ndarray,
                          l_max: int,
                          regularization: float = 0.0) -> jnp.ndarray:
    """
    Fit spherical harmonic coefficients to data on the sphere.
    
    Args:
        data: Data values at the given points, shape (N,)
        theta: Colatitude angles, shape (N,)
        phi: Azimuthal angles, shape (N,)
        l_max: Maximum degree for the fit
        regularization: L2 regularization parameter
        
    Returns:
        Spherical harmonic coefficients
    """
    # Create design matrix
    Y = spherical_harmonic_design_matrix(theta, phi, l_max, real_basis=True)
    
    # Solve least squares with optional regularization
    if regularization > 0:
        # Ridge regression
        A = Y.T @ Y + regularization * jnp.eye(Y.shape[1])
        b = Y.T @ data
        coeffs = jnp.linalg.solve(A, b)
    else:
        # Ordinary least squares
        coeffs = jnp.linalg.lstsq(Y, data, rcond=None)[0]
    
    return coeffs


if __name__ == "__main__":
    # Example: Create a design matrix for spherical harmonics up to degree 3
    l_max = 3
    
    # Create a grid on the sphere
    theta, phi = create_spherical_grid(20, 40)
    
    # Create the design matrix
    design_matrix = spherical_harmonic_design_matrix(theta, phi, l_max, real_basis=True)
    
    print(f"Design matrix shape: {design_matrix.shape}")
    print(f"Number of basis functions: {(l_max + 1)**2}")
    
    # Get the corresponding (l, m) indices
    indices = get_spherical_harmonic_indices(l_max)
    print(f"Basis function indices (l, m): {indices}")
    
    # Example: fit to some synthetic data
    # Create synthetic data (e.g., Y_2^1 spherical harmonic)
    true_Y21 = spherical_harmonic(2, 1, theta, phi)
    synthetic_data = jnp.real(true_Y21) + 0.1 * jax.random.normal(jax.random.PRNGKey(42), shape=theta.shape)
    
    # Fit spherical harmonics
    coeffs = fit_spherical_harmonics(synthetic_data, theta, phi, l_max)
    
    # The coefficient for Y_2^1 should be close to 1, others close to 0
    print(f"Fitted coefficients: {coeffs}")
    
    # Reconstruct the signal
    reconstructed = design_matrix @ coeffs
    mse = jnp.mean((synthetic_data - reconstructed)**2)
    print(f"Reconstruction MSE: {mse:.6f}")