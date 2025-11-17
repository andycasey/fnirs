#!/usr/bin/env python
"""
Create a static visualization showing spherical harmonics arranged by (l, m).

This creates a single figure showing different spherical harmonic basis functions
colored by their values on the sphere surface.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm_y
import matplotlib.cm as cm


def create_sphere_mesh(n_theta=50, n_phi=50):
    """Create a mesh of points on a unit sphere."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    # Convert to Cartesian coordinates
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)

    return x, y, z, theta_grid, phi_grid


def compute_real_spherical_harmonic(l, m, theta, phi):
    """
    Compute real spherical harmonic Y_l^m.

    Uses the same convention as the fnirs package.
    """
    Y = sph_harm_y(l, abs(m), phi, theta)

    if m == 0:
        return Y.real
    elif m > 0:
        return np.sqrt(2) * (-1)**m * Y.real
    else:  # m < 0
        return np.sqrt(2) * (-1)**m * Y.imag


def create_static_figure(max_l=3, output_file='spherical_harmonics_static.png', dpi=150):
    """
    Create a static figure showing spherical harmonics.

    Parameters
    ----------
    max_l : int
        Maximum degree of spherical harmonics to show
    output_file : str
        Output filename
    dpi : int
        Resolution in dots per inch
    """
    print("="*60)
    print("Spherical Harmonics Static Visualization")
    print("="*60)
    print(f"Maximum l: {max_l}")
    print(f"Output: {output_file}")
    print()

    # Create sphere mesh
    print("Creating sphere mesh...")
    x, y, z, theta_grid, phi_grid = create_sphere_mesh(n_theta=80, n_phi=80)

    # Count total number of modes
    n_modes = sum(2*l + 1 for l in range(max_l + 1))

    # Create list of (l, m) pairs
    lm_pairs = []
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            lm_pairs.append((l, m))

    # Determine grid layout
    n_cols = 2*max_l + 1  # One column per possible m value
    n_rows = max_l + 1    # One row per l value

    print(f"Creating figure with {n_modes} spherical harmonics...")
    print(f"Grid size: {n_rows} rows × {n_cols} columns")

    # Create figure with extra space at top for title
    fig = plt.figure(figsize=(2.5*n_cols, 2.5*n_rows + 0.5))
    fig.suptitle('Real Spherical Harmonics: $Y_l^m(\\theta, \\phi)$',
                 fontsize=20, fontweight='bold', y=0.96)

    # Add row labels (l values)
    fig.text(0.02, 0.5, 'Degree (l)', va='center', rotation='vertical',
             fontsize=16, fontweight='bold')

    # Add column labels (m values)
    fig.text(0.5, 0.02, 'Order (m)', ha='center',
             fontsize=16, fontweight='bold')

    # Create colormap
    cmap = cm.plasma

    # Plot each spherical harmonic
    print("Computing and plotting spherical harmonics...")
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            # Compute spherical harmonic
            Y = compute_real_spherical_harmonic(l, m, theta_grid, phi_grid)

            # Determine subplot position
            # Row: indexed by l
            # Column: centered based on m (range -l to l)
            row = l
            col = m + max_l  # Center at max_l

            # Create subplot
            subplot_idx = row * n_cols + col + 1
            ax = fig.add_subplot(n_rows, n_cols, subplot_idx, projection='3d')

            # Set aspect ratio and limits
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # Remove axis panes
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('none')
            ax.yaxis.pane.set_edgecolor('none')
            ax.zaxis.pane.set_edgecolor('none')

            # Add title
            ax.set_title(f'$Y_{{{l}}}^{{{m}}}$', fontsize=14, pad=5)

            # Add l and m labels for leftmost and bottom plots
            if m == -l:  # Leftmost in row
                ax.text2D(-0.15, 0.5, f'l={l}', transform=ax.transAxes,
                         fontsize=12, va='center', ha='right', fontweight='bold')

            if l == max_l:  # Bottom row
                ax.text2D(0.5, -0.15, f'm={m}', transform=ax.transAxes,
                         fontsize=12, va='top', ha='center', fontweight='bold')

            # Normalize Y for coloring
            Y_norm = (Y - Y.min()) / (Y.max() - Y.min() + 1e-10)
            colors = cmap(Y_norm)

            # Plot surface
            ax.plot_surface(x, y, z, facecolors=colors,
                          linewidth=0, antialiased=True, shade=False, alpha=0.9)

            # Set view angle
            ax.view_init(elev=20, azim=45)

    # Adjust layout to make room for colorbar on top right
    plt.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.08,
                       hspace=0.1, wspace=0.05)

    # Add colorbar on top right (horizontal orientation)
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.65, 0.94, 0.25, 0.015])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Normalized Value', fontsize=12, labelpad=5)
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')

    # Save figure
    print(f"\nSaving figure to {output_file}...")
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"✓ Figure saved (DPI: {dpi})")

    print("\n" + "="*60)
    print("Done! You can view the figure:")
    print(f"  open {output_file}")
    print("="*60)

    return fig


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    max_l = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'spherical_harmonics_static.png'
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 150

    fig = create_static_figure(max_l=max_l, output_file=output_file, dpi=dpi)

    # Optionally display
    if '--show' in sys.argv:
        plt.show()
