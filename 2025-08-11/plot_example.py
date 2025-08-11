import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List

# Import the spherical harmonics functions from the previous code
# (You would import these from your spherical harmonics module)

def associated_legendre_polynomial(l: int, m: int, x: jnp.ndarray) -> jnp.ndarray:
    """Compute the associated Legendre polynomial P_l^m(x) using JAX."""
    m_abs = jnp.abs(m)
    result = jnp.where(m_abs > l, 0.0, _compute_legendre(l, m_abs, x))
    sign_factor = jnp.where(m < 0, (-1)**m_abs, 1.0)
    return sign_factor * result

def _compute_legendre(l: int, m: int, x: jnp.ndarray) -> jnp.ndarray:
    """Helper function to compute P_l^m(x) for m >= 0."""
    if l == 0 and m == 0:
        return jnp.ones_like(x)
    
    if m > l:
        return jnp.zeros_like(x)
    
    sqrt_1_minus_x2 = jnp.sqrt(1 - x**2)
    
    pmm = jnp.ones_like(x)
    if m > 0:
        double_factorial = 1.0
        for i in range(1, m + 1):
            double_factorial *= (2 * i - 1)
        pmm = ((-1)**m) * double_factorial * (sqrt_1_minus_x2**m)
    
    if l == m:
        return pmm
    
    pmm1 = x * (2 * m + 1) * pmm
    
    if l == m + 1:
        return pmm1
    
    pll_minus_2 = pmm
    pll_minus_1 = pmm1
    
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pll_minus_1 - (ll + m - 1) * pll_minus_2) / (ll - m)
        pll_minus_2 = pll_minus_1
        pll_minus_1 = pll
    
    return pll_minus_1

def factorial_ratio(n_minus: int, n_plus: int) -> float:
    """Compute (n_minus)! / (n_plus)! efficiently."""
    if n_minus > n_plus:
        return 1.0 / factorial_ratio(n_plus, n_minus)
    
    result = 1.0
    for i in range(n_minus + 1, n_plus + 1):
        result *= i
    return 1.0 / result

def spherical_harmonic(l: int, m: int, theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Compute the spherical harmonic Y_l^m(theta, phi)."""
    norm = jnp.sqrt((2 * l + 1) / (4 * jnp.pi) * 
                    factorial_ratio(l - jnp.abs(m), l + jnp.abs(m)))
    
    cos_theta = jnp.cos(theta)
    legendre = associated_legendre_polynomial(l, jnp.abs(m), cos_theta)
    
    exp_part = jnp.exp(1j * m * phi)
    
    sign = jnp.where(m < 0, (-1)**jnp.abs(m), 1.0)
    
    return sign * norm * legendre * exp_part

def real_spherical_harmonic(l: int, m: int, theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Compute real spherical harmonics."""
    Y_lm = spherical_harmonic(l, jnp.abs(m), theta, phi)
    
    if m == 0:
        return jnp.real(Y_lm)
    elif m > 0:
        return jnp.sqrt(2) * jnp.real(Y_lm)
    else:  # m < 0
        return jnp.sqrt(2) * jnp.imag(Y_lm)

# Visualization functions

def create_sphere_coordinates(n_theta: int = 100, n_phi: int = 200) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create coordinates for sphere visualization."""
    theta = jnp.linspace(0, jnp.pi, n_theta)
    phi = jnp.linspace(0, 2 * jnp.pi, n_phi)
    
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing='ij')
    
    return theta, phi, theta_grid, phi_grid

def spherical_to_cartesian(theta: jnp.ndarray, phi: jnp.ndarray, r: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert spherical coordinates to Cartesian."""
    if r is None:
        r = jnp.ones_like(theta)
    
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    
    return x, y, z

def plot_spherical_harmonic_2d(l: int, m: int, title_suffix: str = "", ax=None):
    """Plot a 2D projection of a spherical harmonic (Mollweide projection)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': 'mollweide'})
    
    # Create coordinate grid
    theta, phi, theta_grid, phi_grid = create_sphere_coordinates(100, 200)
    
    # Compute the spherical harmonic
    Y_lm = real_spherical_harmonic(l, m, theta_grid, phi_grid)
    
    # Convert to longitude/latitude for Mollweide projection
    lon = phi_grid - jnp.pi  # Convert [0, 2π] to [-π, π]
    lat = jnp.pi/2 - theta_grid  # Convert colatitude to latitude
    
    # Plot
    im = ax.contourf(lon, lat, Y_lm, levels=20, cmap='RdBu_r', extend='both')
    ax.set_title(f'Y_{l}^{m} {title_suffix}', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    
    return im

def plot_spherical_harmonic_3d(l: int, m: int, title_suffix: str = "", ax=None, colorbar=True):
    """Plot a 3D visualization of a spherical harmonic on a sphere."""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Create coordinate grid (lower resolution for 3D)
    theta, phi, theta_grid, phi_grid = create_sphere_coordinates(50, 100)
    
    # Compute the spherical harmonic
    Y_lm = real_spherical_harmonic(l, m, theta_grid, phi_grid)
    
    # Normalize for visualization
    Y_norm = Y_lm / jnp.max(jnp.abs(Y_lm)) if jnp.max(jnp.abs(Y_lm)) > 0 else Y_lm
    
    # Convert to Cartesian coordinates
    # Use the harmonic values to modulate the radius slightly for better visualization
    r = 1 + 0.3 * Y_norm  # Base radius 1, modulated by harmonic
    x, y, z = spherical_to_cartesian(theta_grid, phi_grid, r)
    
    # Color by the harmonic values
    colors = cm.RdBu_r((Y_norm + 1) / 2)  # Normalize to [0, 1] for colormap
    
    # Plot
    surf = ax.plot_surface(x, y, z, facecolors=colors, alpha=0.8, 
                          linewidth=0, antialiased=True, shade=False)
    
    # Set equal aspect ratio
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_box_aspect([1,1,1])
    
    ax.set_title(f'Y_{l}^{m} {title_suffix}', fontsize=14, pad=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if colorbar:
        # Create a ScalarMappable for the colorbar
        sm = cm.ScalarMappable(cmap='RdBu_r')
        sm.set_array(Y_lm)
        plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    
    return surf

def plot_harmonic_gallery_2d(l_max: int = 3):
    """Create a gallery of 2D spherical harmonic plots."""
    # Count total harmonics
    harmonics = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            harmonics.append((l, m))
    
    n_harmonics = len(harmonics)
    n_cols = min([4, n_harmonics])
    n_rows = (n_harmonics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows),
                            subplot_kw={'projection': 'mollweide'})
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (l, m) in enumerate(harmonics):
        row = i // n_cols
        col = i % n_cols
        
        if n_rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        
        im = plot_spherical_harmonic_2d(l, m, ax=ax)
        
        # Add colorbar to each subplot
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide unused subplots
    for i in range(n_harmonics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle(f'Real Spherical Harmonics up to l = {l_max}', fontsize=16, y=0.98)
    return fig

def plot_harmonic_gallery_3d(harmonics_list: List[Tuple[int, int]]):
    """Create a gallery of 3D spherical harmonic plots."""
    n_harmonics = len(harmonics_list)
    n_cols = min([3, n_harmonics])
    n_rows = (n_harmonics + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(6*n_cols, 6*n_rows))
    
    for i, (l, m) in enumerate(harmonics_list):
        ax = fig.add_subplot(n_rows, n_cols, i+1, projection='3d')
        plot_spherical_harmonic_3d(l, m, ax=ax, colorbar=False)
    
    plt.tight_layout()
    return fig

def interactive_harmonic_explorer():
    """Create an interactive plot to explore individual harmonics."""
    # Select some interesting harmonics to showcase
    showcase_harmonics = [
        (0, 0),   # Monopole
        (1, 0),   # Dipole (z)
        (1, 1),   # Dipole (x)
        (1, -1),  # Dipole (y)
        (2, 0),   # Quadrupole (z²)
        (2, 1),   # Quadrupole (xz)
        (2, 2),   # Quadrupole (x²-y²)
        (3, 0),   # Octupole
        (3, 2),   # Octupole
    ]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12),
                                                 gridspec_kw={'height_ratios': [1, 1]})
    
    # Make the left plots Mollweide projections
    ax1.remove()
    ax3.remove()
    ax1 = plt.subplot(2, 2, 1, projection='mollweide')
    ax3 = plt.subplot(2, 2, 3, projection='mollweide')
    
    # Make the right plots 3D
    ax2.remove()
    ax4.remove()
    ax2 = plt.subplot(2, 2, 2, projection='3d')
    ax4 = plt.subplot(2, 2, 4, projection='3d')
    
    # Plot two different harmonics
    l1, m1 = showcase_harmonics[4]  # (2, 0)
    l2, m2 = showcase_harmonics[6]  # (2, 2)
    
    # Top row: First harmonic
    im1 = plot_spherical_harmonic_2d(l1, m1, f"(l={l1}, m={m1})", ax1)
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    plot_spherical_harmonic_3d(l1, m1, f"(l={l1}, m={m1})", ax2)
    
    # Bottom row: Second harmonic
    im2 = plot_spherical_harmonic_2d(l2, m2, f"(l={l2}, m={m2})", ax3)
    plt.colorbar(im2, ax=ax3, shrink=0.8)
    
    plot_spherical_harmonic_3d(l2, m2, f"(l={l2}, m={m2})", ax4)
    
    plt.tight_layout()
    return fig

def demonstrate_orthogonality():
    """Demonstrate the orthogonality of spherical harmonics."""
    print("Demonstrating Spherical Harmonic Orthogonality")
    print("=" * 50)
    
    # Create a high-resolution grid
    theta, phi, theta_grid, phi_grid = create_sphere_coordinates(100, 200)
    
    # Select a few harmonics to test
    harmonics = [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    
    # Compute all harmonics
    Y_values = {}
    for l, m in harmonics:
        Y_values[(l, m)] = real_spherical_harmonic(l, m, theta_grid, phi_grid)
    
    # Test orthogonality by computing inner products
    # ∫ Y_l^m * Y_l'^m' dΩ = δ_ll' δ_mm'
    sin_theta = jnp.sin(theta_grid)
    dtheta = jnp.pi / (theta_grid.shape[0] - 1)
    dphi = 2 * jnp.pi / theta_grid.shape[1]
    
    print("Inner products (should be ~1 for same harmonics, ~0 for different):")
    print("(l₁,m₁) × (l₂,m₂) = inner product")
    print("-" * 40)
    
    for i, (l1, m1) in enumerate(harmonics):
        for j, (l2, m2) in enumerate(harmonics):
            if j <= i:  # Only compute upper triangle
                Y1 = Y_values[(l1, m1)]
                Y2 = Y_values[(l2, m2)]
                
                # Numerical integration over the sphere
                integrand = Y1 * Y2 * sin_theta
                inner_product = jnp.sum(integrand) * dtheta * dphi
                
                print(f"({l1},{m1:2}) × ({l2},{m2:2}) = {inner_product:8.1f}")
    
    return Y_values

# Main execution
if __name__ == "__main__":
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    
    print("Spherical Harmonics Visualization Examples")
    print("=" * 50)
    
    # Example 1: Gallery of 2D projections
    print("\n1. Creating gallery of 2D spherical harmonics...")
    fig1 = plot_harmonic_gallery_2d(l_max=2)
    plt.show()
    
    # Example 2: Interactive explorer with both 2D and 3D views
    print("\n2. Creating interactive harmonic explorer...")
    fig2 = interactive_harmonic_explorer()
    plt.show()
    
    # Example 3: 3D gallery of selected harmonics
    print("\n3. Creating 3D gallery of selected harmonics...")
    selected_harmonics = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    fig3 = plot_harmonic_gallery_3d(selected_harmonics)
    plt.show()
    
    # Example 4: Demonstrate orthogonality
    print("\n4. Demonstrating orthogonality...")
    Y_values = demonstrate_orthogonality()
    
    # Example 5: Individual harmonic with detailed analysis
    print("\n5. Detailed analysis of Y₂² harmonic...")
    fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Remove and recreate with proper projections
    ax1.remove()
    ax2.remove()
    ax1 = plt.subplot(1, 2, 1, projection='mollweide')
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    
    # Plot Y₂²
    l, m = 2, 2
    im = plot_spherical_harmonic_2d(l, m, "Real Spherical Harmonic", ax1)
    plt.colorbar(im, ax=ax1, shrink=0.8)
    
    plot_spherical_harmonic_3d(l, m, "3D Visualization", ax2)
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization complete! The plots show:")
    print("- 2D Mollweide projections (like world maps)")
    print("- 3D sphere visualizations with radius modulation")
    print("- Color coding represents harmonic amplitude")
    print("- Red/blue typically represent positive/negative values")
    
    # Additional analysis
    print(f"\nNote: Y₂² represents the (x²-y²) quadrupole moment")
    print("This pattern has 4-fold symmetry around the z-axis")
    print("and is commonly seen in atomic orbitals and gravitational fields.")