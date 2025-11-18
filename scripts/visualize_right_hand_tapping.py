#!/usr/bin/env python3
"""
Visualize left hemisphere model predictions during right-hand tapping task.

This script:
1. Loads the right-hand tapping SNIRF data
2. Filters for HbO channels in the left hemisphere
3. Fits a spatial-temporal model
4. Visualizes model predictions over time with tapping periods highlighted
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

from fnirs import (
    load_snirf_data,
    fit,
    project_fnirs_to_sphere,
    cartesian_to_spherical,
    create_spherical_harmonics_basis,
)


def main():
    # Load the right-hand tapping data
    snirf_path = repo_root / "scratch" / "2025-08-26" / "Processed_RightOnly.snirf"
    print(f"Loading data from: {snirf_path}")
    data = load_snirf_data(snirf_path)

    print(f"Total channels: {data.n_channels}")
    print(f"Time points: {data.n_timepoints}")
    print(f"Sampling frequency: {data.sampling_frequency:.2f} Hz")

    # Get HbO channels only
    hbo_channels = data.get_channels_by_data_type_label('HbO')
    print(f"HbO channels: {len(hbo_channels)}")

    # Get HbO channel indices and data
    hbo_indices = [ch.channel_idx for ch in hbo_channels]
    hbo_data = data.time_series[:, hbo_indices]  # [time x channels]

    # Get 3D coordinates for HbO channels
    hbo_coords_3d = np.array([ch.midpoint_3d for ch in hbo_channels])

    # Filter for left hemisphere
    # In standard neuroimaging coordinates, positive x is right, negative x is left
    # For right-hand tapping, we expect activation in the left hemisphere (negative x)
    # But let's check the coordinate system by looking at the center
    x_center = np.median(hbo_coords_3d[:, 0])
    print(f"X coordinate center: {x_center:.2f}")

    # Left hemisphere is typically x < center (or x < 0 if centered)
    left_hemisphere_mask = hbo_coords_3d[:, 0] < x_center
    n_left = left_hemisphere_mask.sum()
    print(f"Left hemisphere channels: {n_left}")

    # Filter data and coordinates for left hemisphere
    left_hbo_data = hbo_data[:, left_hemisphere_mask]  # [time x left_channels]
    left_coords_3d = hbo_coords_3d[left_hemisphere_mask]

    print(f"Left hemisphere data shape: {left_hbo_data.shape}")

    # Get stimulus information
    if data.stimulus and len(data.stimulus) > 0:
        stim = data.stimulus[0]
        print(f"\nStimulus: {stim.name}")
        print(f"Number of tapping blocks: {len(stim.onsets)}")
        print(f"Tapping duration: {stim.durations[0]:.1f} s")
        tapping_onsets = stim.onsets
        tapping_durations = stim.durations
    else:
        print("Warning: No stimulus information found")
        tapping_onsets = np.array([])
        tapping_durations = np.array([])

    # Project left hemisphere coordinates onto sphere
    print("\nProjecting coordinates onto sphere...")
    sphere_result = project_fnirs_to_sphere(left_coords_3d, fit_method='least_squares')
    θ, ϕ = sphere_result['theta'], sphere_result['phi']

    # Fit spatial-temporal model
    print("Fitting spatial-temporal model...")
    t = jnp.array(data.time)
    θ_jax, ϕ_jax = jnp.array(θ), jnp.array(ϕ)

    # Convert to μM and transpose to [channels x time]
    Y = jnp.array(left_hbo_data.T)

    # Use max_spherical_degree=4 to avoid having more basis functions than channels
    # (degree 4 gives (4+1)^2 = 25 basis functions)
    max_spherical_degree = 4
    n_fourier_components = len(t)

    X, f, *extras = fit(
        t, θ_jax, ϕ_jax, Y,
        max_spherical_degree=max_spherical_degree,
        n_fourier_components=n_fourier_components
    )

    # Get predictions [channels x time]
    Y_predicted = f(X)

    print(f"Model fitted. Predictions shape: {Y_predicted.shape}")

    # Filter for motor cortex region with specific anatomical constraints
    x_coords = left_coords_3d[:, 0]
    y_coords = left_coords_3d[:, 1]
    z_coords = left_coords_3d[:, 2]

    # Motor cortex selection criteria:
    # X: -75 to -10 mm (left hemisphere, excluding midline and far lateral)
    # Y: -50 to +50 mm (central anterior-posterior region)
    # Z: 40 to 80 mm (upper portion of head)
    motor_cortex_mask = (
        (x_coords >= -75) & (x_coords <= -10) &
        (y_coords >= -50) & (y_coords <= 50) &
        (z_coords >= 40) & (z_coords <= 80)
    )
    n_motor = motor_cortex_mask.sum()
    print(f"Motor cortex region channels: {n_motor}")
    print(f"  X range: [{x_coords[motor_cortex_mask].min():.1f}, {x_coords[motor_cortex_mask].max():.1f}] mm")
    print(f"  Y range: [{y_coords[motor_cortex_mask].min():.1f}, {y_coords[motor_cortex_mask].max():.1f}] mm")
    print(f"  Z range: [{z_coords[motor_cortex_mask].min():.1f}, {z_coords[motor_cortex_mask].max():.1f}] mm")

    # Compute spatial average only for motor cortex region
    Y_motor_avg = np.array(Y)[motor_cortex_mask, :].mean(axis=0)  # Average across motor channels
    Y_pred_motor_avg = np.array(Y_predicted)[motor_cortex_mask, :].mean(axis=0)  # Average across motor channels

    # Also keep the full left hemisphere average for comparison
    Y_spatial_avg = np.array(Y).mean(axis=0)  # Average across all left channels
    Y_pred_spatial_avg = np.array(Y_predicted).mean(axis=0)  # Average across all left channels

    # Create visualization with 5 panels (3 spatial projections + 2 time series)
    fig = plt.figure(figsize=(18, 12))

    # Create grid: left column for 2D projections, right column for time series
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.5], hspace=0.35, wspace=0.3)
    ax_xy = fig.add_subplot(gs[0, 0])  # XY projection (top-down view)
    ax_xz = fig.add_subplot(gs[1, 0])  # XZ projection (front view)
    ax_yz = fig.add_subplot(gs[2, 0])  # YZ projection (side view)
    ax1 = fig.add_subplot(gs[0, 1])    # Time series - spatial average
    ax2 = fig.add_subplot(gs[1:, 1], sharex=ax1)  # Time series - sample channels

    # Get all HbO channel coordinates
    all_hbo_coords = np.array([ch.midpoint_3d for ch in hbo_channels])
    motor_coords = left_coords_3d[motor_cortex_mask]

    # Helper function to plot 2D projections
    def plot_2d_projection(ax, coords_all, coords_left, coords_motor, xi, yi, xlabel, ylabel, title):
        # All channels
        ax.scatter(coords_all[:, xi], coords_all[:, yi],
                   c='lightgray', s=30, alpha=0.4, label=f'All HbO ({len(hbo_channels)})')
        # Left hemisphere
        ax.scatter(coords_left[:, xi], coords_left[:, yi],
                   c='gray', s=50, alpha=0.6, label=f'Left hem. ({len(left_coords_3d)})')
        # Motor cortex
        ax.scatter(coords_motor[:, xi], coords_motor[:, yi],
                   c='blue', s=100, alpha=0.8, edgecolors='darkblue', linewidths=2,
                   label=f'Motor ctx ({n_motor})')

        # Add boundary lines for motor cortex region
        if xi == 0:  # X axis present
            ax.axvline(x_center, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(-10, color='red', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(-75, color='red', linestyle=':', linewidth=1, alpha=0.5)

        if yi == 1 or xi == 1:  # Y axis present
            ax.axhline(-50, color='red', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(50, color='red', linestyle=':', linewidth=1, alpha=0.5)

        if yi == 2 or xi == 2:  # Z axis present
            ax.axhline(40, color='red', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(80, color='red', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        return ax

    # XY projection (top-down view)
    plot_2d_projection(ax_xy, all_hbo_coords, left_coords_3d, motor_coords,
                       0, 1, 'X (mm)', 'Y (mm)', 'XY View (Top-Down)')

    # XZ projection (front view)
    plot_2d_projection(ax_xz, all_hbo_coords, left_coords_3d, motor_coords,
                       0, 2, 'X (mm)', 'Z (mm)', 'XZ View (Front)')

    # YZ projection (side view)
    plot_2d_projection(ax_yz, all_hbo_coords, left_coords_3d, motor_coords,
                       1, 2, 'Y (mm)', 'Z (mm)', 'YZ View (Side)')

    # Add legend to first plot only
    ax_xy.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Plot 1: Motor cortex region average of data and model prediction
    ax1.plot(data.time, Y_motor_avg, 'k-', alpha=0.7, linewidth=1.5, label='Data (motor cortex avg)')
    ax1.plot(data.time, Y_pred_motor_avg, 'r-', linewidth=2, label='Model (motor cortex avg)')

    # Shade tapping periods
    for onset, duration in zip(tapping_onsets, tapping_durations):
        ax1.axvspan(onset, onset + duration, alpha=0.2, color='blue', label='Tapping' if onset == tapping_onsets[0] else '')

    ax1.set_ylabel('HbO Concentration (μM)', fontsize=12)
    ax1.set_title(f'Motor Cortex Region ({n_motor} channels): Spatial Average', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Individual channel predictions (sample from motor cortex)
    # Select a few representative channels from motor cortex region
    motor_indices = np.where(motor_cortex_mask)[0]
    n_display = min(5, len(motor_indices))
    np.random.seed(42)
    display_indices = np.random.choice(motor_indices, size=n_display, replace=False)

    # Highlight the sample channels in the 2D projections
    sample_coords = left_coords_3d[display_indices]
    colors = plt.cm.tab10(np.linspace(0, 1, n_display))

    for i, idx in enumerate(display_indices):
        # Add to all 2D projections with matching color
        ax_xy.scatter(sample_coords[i, 0], sample_coords[i, 1],
                      c=[colors[i]], s=150, alpha=1.0, edgecolors='black', linewidths=2,
                      marker='*', zorder=10)
        ax_xz.scatter(sample_coords[i, 0], sample_coords[i, 2],
                      c=[colors[i]], s=150, alpha=1.0, edgecolors='black', linewidths=2,
                      marker='*', zorder=10)
        ax_yz.scatter(sample_coords[i, 1], sample_coords[i, 2],
                      c=[colors[i]], s=150, alpha=1.0, edgecolors='black', linewidths=2,
                      marker='*', zorder=10)

    # Add legend entry for sample channels to XY plot
    ax_xy.scatter([], [], c='red', s=150, marker='*', edgecolors='black', linewidths=2,
                  label=f'Sample ch. ({n_display})')
    ax_xy.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Plot time series for sample channels
    for i, idx in enumerate(display_indices):
        # Normalize each channel for display
        y_norm = np.array(Y[idx])
        y_pred_norm = np.array(Y_predicted[idx])

        # Offset for visualization
        offset = i * 2

        ax2.plot(data.time, y_norm + offset, '-', color=colors[i], alpha=0.5, linewidth=1, label=f'Ch {idx} Data')
        ax2.plot(data.time, y_pred_norm + offset, '-', color=colors[i], linewidth=2, label=f'Ch {idx} Model')

    # Shade tapping periods
    for onset, duration in zip(tapping_onsets, tapping_durations):
        ax2.axvspan(onset, onset + duration, alpha=0.2, color='blue')

    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('HbO Concentration (μM, offset)', fontsize=12)
    ax2.set_title('Sample Motor Cortex Channels: Model vs Data', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle('Right-Hand Tapping: Left Motor Cortex Analysis', fontsize=14, fontweight='bold', y=0.98)

    # Save figure
    output_path = repo_root / "figures" / "right_hand_tapping_left_hemisphere.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Compute and print some statistics
    from sklearn.metrics import r2_score

    # Flatten for overall R²
    r2_overall = r2_score(np.array(Y).flatten(), np.array(Y_predicted).flatten())
    print(f"\nOverall R²: {r2_overall:.3f}")

    # Per-channel R²
    r2_per_channel = []
    for i in range(Y.shape[0]):
        r2 = r2_score(np.array(Y[i]), np.array(Y_predicted[i]))
        r2_per_channel.append(r2)

    r2_per_channel = np.array(r2_per_channel)
    print(f"Mean R² per channel: {r2_per_channel.mean():.3f} ± {r2_per_channel.std():.3f}")


if __name__ == "__main__":
    main()
