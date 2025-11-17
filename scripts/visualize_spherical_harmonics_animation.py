#!/usr/bin/env python
"""
Create an animated visualization showing how spherical harmonics combine
with temporal basis functions to represent spatial-temporal data.

This creates a figure with:
- Grid of spheres showing different spherical harmonics (colored by plasma)
- Time series plot showing the temporal coefficients
- Animation where sphere opacity changes to show temporal contributions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
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


def create_temporal_signal(t, n_modes, seed=42):
    """
    Create a synthetic temporal signal as a sum of sinusoids.

    Parameters
    ----------
    t : array
        Time points
    n_modes : int
        Number of spherical harmonic modes
    seed : int
        Random seed for reproducibility

    Returns
    -------
    signal : array (n_modes, len(t))
        Temporal coefficients for each spatial mode
    """
    np.random.seed(seed)
    signal = np.zeros((n_modes, len(t)))

    # Create interesting temporal patterns for each mode
    for i in range(n_modes):
        # Mix of different frequencies and phases
        freq1 = 0.5 + np.random.rand() * 2.0
        freq2 = 0.2 + np.random.rand() * 1.0
        phase1 = np.random.rand() * 2 * np.pi
        phase2 = np.random.rand() * 2 * np.pi
        amp1 = 0.5 + np.random.rand() * 0.5
        amp2 = 0.3 + np.random.rand() * 0.3

        signal[i] = (amp1 * np.sin(2*np.pi*freq1*t + phase1) +
                     amp2 * np.cos(2*np.pi*freq2*t + phase2))

    return signal


def setup_figure(max_l=2):
    """
    Set up the figure with subplots for spherical harmonics and time series.

    Parameters
    ----------
    max_l : int
        Maximum degree of spherical harmonics to show

    Returns
    -------
    fig : Figure
    axs : dict
        Dictionary of axes: 'spheres' (2D array), 'timeseries' (1D array)
    lm_pairs : list
        List of (l, m) pairs corresponding to each sphere subplot
    """
    # Count total number of modes
    n_modes = sum(2*l + 1 for l in range(max_l + 1))

    # Create list of (l, m) pairs
    lm_pairs = []
    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            lm_pairs.append((l, m))

    # Determine grid layout (aim for roughly square)
    n_cols = int(np.ceil(np.sqrt(n_modes)))
    n_rows = int(np.ceil(n_modes / n_cols))

    # Create figure
    fig = plt.figure(figsize=(16, 10))

    # Create grid: spheres on top, time series on bottom
    gs = fig.add_gridspec(n_rows + 1, n_cols,
                          height_ratios=[1]*n_rows + [0.5],
                          hspace=0.05, wspace=0.05,
                          left=0.05, right=0.95, top=0.95, bottom=0.05)

    # Create sphere subplots
    sphere_axs = []
    for i, (l, m) in enumerate(lm_pairs):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col], projection='3d')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f'$Y_{{{l}}}^{{{m}}}$', fontsize=12, pad=5)
        # Remove axis lines
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        sphere_axs.append(ax)

    # Create time series subplot (spans all columns)
    ts_ax = fig.add_subplot(gs[n_rows, :])
    ts_ax.set_xlabel('Time', fontsize=12)
    ts_ax.set_ylabel('Coefficient', fontsize=12)
    ts_ax.grid(True, alpha=0.3)

    return fig, {'spheres': sphere_axs, 'timeseries': ts_ax}, lm_pairs


def create_animation(max_l=2, duration=10.0, fps=30, output_file='spherical_harmonics_animation.mp4', optimize_size=False):
    """
    Create the full animation.

    Parameters
    ----------
    max_l : int
        Maximum degree of spherical harmonics
    duration : float
        Duration of animation in seconds
    fps : int
        Frames per second
    output_file : str
        Output filename
    """
    print("Setting up animation...")

    # Create sphere mesh
    x, y, z, theta_grid, phi_grid = create_sphere_mesh(n_theta=50, n_phi=50)

    # Set up figure
    fig, axs, lm_pairs = setup_figure(max_l=max_l)
    n_modes = len(lm_pairs)

    # Create time array
    n_frames = int(duration * fps)
    t = np.linspace(0, duration, n_frames)

    # Create temporal signals
    temporal_coeffs = create_temporal_signal(t, n_modes)

    # Compute spherical harmonics on mesh
    print("Computing spherical harmonics...")
    Y_values = []
    for l, m in lm_pairs:
        Y = compute_real_spherical_harmonic(l, m, theta_grid, phi_grid)
        Y_values.append(Y)

    # Create colormap
    cmap = cm.plasma

    # Plot initial spherical harmonics
    print("Creating initial sphere plots...")
    sphere_surfaces = []
    for i, (ax, Y) in enumerate(zip(axs['spheres'], Y_values)):
        # Normalize Y for coloring
        Y_norm = (Y - Y.min()) / (Y.max() - Y.min() + 1e-10)
        colors = cmap(Y_norm)

        # Initial alpha based on initial temporal coefficient
        alpha = 0.3 + 0.7 * abs(temporal_coeffs[i, 0]) / (abs(temporal_coeffs[i]).max() + 1e-10)
        colors[:, :, 3] = alpha

        surf = ax.plot_surface(x, y, z, facecolors=colors,
                              linewidth=0, antialiased=True, shade=False)
        sphere_surfaces.append(surf)

    # Plot time series
    print("Creating time series plot...")
    ts_lines = []
    ts_markers = []
    colors_ts = plt.cm.tab10(np.linspace(0, 1, min(n_modes, 10)))

    # Plot individual coefficients
    for i in range(n_modes):
        color = colors_ts[i % 10]
        line, = axs['timeseries'].plot(t, temporal_coeffs[i],
                                       color=color, alpha=0.5, linewidth=1.0,
                                       label=f"$Y_{{{lm_pairs[i][0]}}}^{{{lm_pairs[i][1]}}}$")
        marker, = axs['timeseries'].plot([], [], 'o', color=color, markersize=6)
        ts_lines.append(line)
        ts_markers.append(marker)

    # Compute and plot summed signal
    summed_signal = temporal_coeffs.sum(axis=0)
    sum_line, = axs['timeseries'].plot(t, summed_signal,
                                       color='black', linewidth=2,
                                       label='Sum (total signal)', zorder=100)
    sum_marker, = axs['timeseries'].plot([], [], 'o', color='black',
                                         markersize=10, zorder=101)

    # Add vertical line for current time
    time_line = axs['timeseries'].axvline(x=t[0], color='red', linestyle='--',
                                          linewidth=2, alpha=0.7, label='Current time',
                                          zorder=99)

    axs['timeseries'].set_xlim(t[0], t[-1])
    # Adjust y-limits to include summed signal
    y_min = min(temporal_coeffs.min(), summed_signal.min()) * 1.1
    y_max = max(temporal_coeffs.max(), summed_signal.max()) * 1.1
    axs['timeseries'].set_ylim(y_min, y_max)

    # Add legend (show sum and current time, plus a few harmonics if space allows)
    handles, labels = axs['timeseries'].get_legend_handles_labels()
    # Always show sum and current time
    important_handles = [sum_line, time_line]
    important_labels = ['Sum (total signal)', 'Current time']

    # Add a few individual harmonics if we have room
    if n_modes <= 6:
        # Show all harmonics
        for i in range(n_modes):
            important_handles.append(ts_lines[i])
            important_labels.append(f"$Y_{{{lm_pairs[i][0]}}}^{{{lm_pairs[i][1]}}}$")
    else:
        # Just show first 3 harmonics as examples
        for i in range(min(3, n_modes)):
            important_handles.append(ts_lines[i])
            important_labels.append(f"$Y_{{{lm_pairs[i][0]}}}^{{{lm_pairs[i][1]}}}$")
        if n_modes > 3:
            important_labels[-1] += " ..."

    axs['timeseries'].legend(important_handles, important_labels,
                             loc='upper right', fontsize=9, ncol=2, framealpha=0.9)

    # Add title
    fig.suptitle('Spatial-Temporal Decomposition: Spherical Harmonics × Time',
                 fontsize=16, fontweight='bold')

    def update(frame):
        """Update function for animation."""
        if frame % 10 == 0:
            print(f"Rendering frame {frame}/{n_frames}...")

        current_time = t[frame]

        # Update sphere opacities based on temporal coefficients
        for i, (surf, Y) in enumerate(zip(sphere_surfaces, Y_values)):
            # Remove old surface
            surf.remove()

            # Get current temporal coefficient
            coeff = temporal_coeffs[i, frame]
            max_coeff = abs(temporal_coeffs[i]).max()

            # Calculate alpha (opacity) based on coefficient magnitude
            alpha = 0.3 + 0.7 * abs(coeff) / (max_coeff + 1e-10)

            # Normalize Y for coloring
            Y_norm = (Y - Y.min()) / (Y.max() - Y.min() + 1e-10)
            colors = cmap(Y_norm)
            colors[:, :, 3] = alpha

            # Plot new surface
            new_surf = axs['spheres'][i].plot_surface(
                x, y, z, facecolors=colors,
                linewidth=0, antialiased=True, shade=False
            )
            sphere_surfaces[i] = new_surf

        # Update time series markers
        for i, marker in enumerate(ts_markers):
            marker.set_data([current_time], [temporal_coeffs[i, frame]])

        # Update sum marker
        sum_marker.set_data([current_time], [summed_signal[frame]])

        # Update vertical line
        time_line.set_xdata([current_time, current_time])

        return sphere_surfaces + ts_markers + [sum_marker, time_line]

    # Create animation
    print(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(fig, update, frames=n_frames,
                        interval=1000/fps, blit=False)

    # Save animation
    print(f"Saving animation to {output_file}...")

    # Check if ffmpeg is available
    try:
        writer = FFMpegWriter(fps=fps, bitrate=2000,
                             extra_args=['-vcodec', 'libx264'])
        anim.save(output_file, writer=writer, dpi=100)
        print(f"✓ Animation saved to {output_file} (using ffmpeg)")
    except (RuntimeError, FileNotFoundError):
        # Fallback to Pillow (creates GIF)
        output_file_gif = output_file.replace('.mp4', '.gif')
        print(f"  ffmpeg not available, saving as GIF: {output_file_gif}")

        # Use lower DPI for size optimization
        dpi_setting = 75 if optimize_size else 100
        writer = PillowWriter(fps=fps)
        anim.save(output_file_gif, writer=writer, dpi=dpi_setting)
        print(f"✓ Animation saved to {output_file_gif} (using Pillow, DPI: {dpi_setting})")
        output_file = output_file_gif

    return fig, anim


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    max_l = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'spherical_harmonics_animation.mp4'
    optimize_size = '--small' in sys.argv or '--optimize' in sys.argv

    print("="*60)
    print("Spherical Harmonics Animation Generator")
    print("="*60)
    print(f"Maximum l: {max_l}")
    print(f"Duration: {duration} seconds")
    print(f"Output: {output_file}")
    if optimize_size:
        print("Size optimization: ON (lower DPI)")
    print()

    fig, anim = create_animation(
        max_l=max_l,
        duration=duration,
        fps=30,
        output_file=output_file,
        optimize_size=optimize_size
    )

    print("\nDone! You can view the animation:")
    print(f"  open {output_file}")
