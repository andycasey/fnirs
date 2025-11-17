#!/usr/bin/env python3
"""
Example demonstrating spatial-temporal modeling of hemoglobin concentration changes.
"""

import numpy as np
import matplotlib.pyplot as plt
from load_fnirs_part_ii import load_hemodynamic_data, ChromophoreType
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata


def example_spatial_temporal_analysis():
    """Example analysis showing spatial-temporal modeling capabilities."""
    
    # Load the hemodynamic data
    data_path = "rsFC-fnirs-course/Data_for_Part_II.mat"
    hemo_data = load_hemodynamic_data(data_path)
    
    print("=== Spatial-Temporal Modeling Example ===")
    
    # 1. Extract spatial coordinates and hemoglobin data
    coords = hemo_data.get_spatial_coordinates()
    hbo_data = hemo_data.get_hbo_data()  # Shape: (time, channels)
    hbr_data = hemo_data.get_hbr_data()
    time = hemo_data.time
    
    print(f"Data dimensions: {hbo_data.shape[0]} timepoints × {hbo_data.shape[1]} channels")
    print(f"Spatial coverage: X=[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], Y=[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}] cm")
    
    # 2. Compute spatial correlation matrix
    print(f"\n=== Spatial Analysis ===")
    spatial_distances = cdist(coords, coords)
    
    # Compute temporal correlation between channels
    hbo_corr = np.corrcoef(hbo_data.T)  # Transpose to get channel×channel correlation
    
    # Example: Find channels within 2cm of a central location
    center_pos = np.array([0.0, 0.0, 0.0])  # Head center (3D to match coordinates)
    nearby_channels = hemo_data.get_channels_near_position(center_pos, radius=2.0)
    print(f"Channels within 2cm of center: {len(nearby_channels)} out of {len(hemo_data.channels)}")
    
    # 3. Temporal analysis
    print(f"\n=== Temporal Analysis ===") 
    
    # Compute temporal derivatives (rate of change)
    dt = 1.0 / hemo_data.probe.sampling_freq
    hbo_velocity = np.gradient(hbo_data, dt, axis=0)
    hbr_velocity = np.gradient(hbr_data, dt, axis=0)
    
    print(f"HbO velocity range: {hbo_velocity.min():.2e} to {hbo_velocity.max():.2e} [units/s]")
    print(f"HbR velocity range: {hbr_velocity.min():.2e} to {hbr_velocity.max():.2e} [units/s]")
    
    # 4. Example: Find time periods with largest spatial gradients
    # Compute spatial gradient magnitude at each time point
    spatial_grad_magnitudes = []
    
    for t_idx in range(0, len(time), 100):  # Sample every 100 timepoints for efficiency
        # Get HbO values at this timepoint
        hbo_t = hbo_data[t_idx, :]
        
        # Estimate spatial gradient using nearby points
        grad_mag = 0.0
        count = 0
        for i, coord_i in enumerate(coords):
            # Find nearby channels
            distances = np.linalg.norm(coords - coord_i, axis=1)
            nearby_idx = np.where((distances > 0) & (distances < 1.0))[0]  # Within 1cm
            
            if len(nearby_idx) > 0:
                # Estimate local gradient
                local_grad = np.std(hbo_t[nearby_idx])  # Simple gradient estimate
                grad_mag += local_grad
                count += 1
                
        if count > 0:
            spatial_grad_magnitudes.append(grad_mag / count)
        else:
            spatial_grad_magnitudes.append(0.0)
    
    sample_times = time[::100][:len(spatial_grad_magnitudes)]
    max_grad_idx = np.argmax(spatial_grad_magnitudes)
    max_grad_time = sample_times[max_grad_idx]
    
    print(f"\nSpatial gradient analysis:")
    print(f"  Max spatial gradient at t={max_grad_time:.1f}s")
    print(f"  Gradient magnitude: {spatial_grad_magnitudes[max_grad_idx]:.2e}")
    
    # 5. Example spatial interpolation for visualization
    print(f"\n=== Spatial Interpolation Example ===")
    
    # Create regular grid for interpolation
    X, Y = hemo_data.create_spatial_mesh(resolution=30)
    
    # Interpolate HbO data at a specific timepoint
    t_interp = int(len(time) * 0.3)  # Use 30% through the timeseries
    hbo_snapshot = hbo_data[t_interp, :]
    
    # Interpolate onto regular grid
    hbo_interpolated = griddata(
        coords[:, :2],  # 2D coordinates
        hbo_snapshot,   # Values to interpolate
        (X, Y),         # Grid points
        method='cubic',
        fill_value=0.0
    )
    
    print(f"Interpolated HbO data at t={time[t_interp]:.1f}s onto {X.shape} grid")
    print(f"Interpolated value range: {np.nanmin(hbo_interpolated):.2e} to {np.nanmax(hbo_interpolated):.2e}")
    
    # 6. Example: Channel selection for modeling
    print(f"\n=== Channel Selection for Modeling ===")
    
    # Get only long-separation channels (avoid short-separation contamination)
    long_channels = hemo_data.get_long_separation_channels(threshold=1.5)
    short_channels = hemo_data.get_short_separation_channels(threshold=1.5)
    
    print(f"Long-separation channels: {len(long_channels)}")
    print(f"Short-separation channels: {len(short_channels)}")
    
    # Note: In this dataset, all channels appear to be short separation!
    # This suggests it's a high-density probe design
    
    # Alternative: select channels by distance percentile
    distances = np.array([ch.distance for ch in hemo_data.channels])
    distance_threshold = np.percentile(distances, 75)  # Top 25% by distance
    far_channels = [ch for ch in hemo_data.channels if ch.distance >= distance_threshold]
    
    print(f"Channels with distance ≥ 75th percentile ({distance_threshold:.3f} cm): {len(far_channels)}")
    
    # 7. Summary for modeling
    print(f"\n=== Summary for Spatial-Temporal Modeling ===")
    print(f"Key data structures:")
    print(f"  - Spatial coordinates: {coords.shape} (channel midpoints)")
    print(f"  - HbO time series: {hbo_data.shape} (time × channels)")
    print(f"  - HbR time series: {hbr_data.shape} (time × channels)")  
    print(f"  - Time vector: {time.shape} (0 to {time[-1]:.1f}s)")
    print(f"")
    print(f"Modeling considerations:")
    print(f"  - All channels are short-separation (< 1.5cm)")
    print(f"  - High spatial density: {len(hemo_data.channels)} channels over {coords[:, 0].max() - coords[:, 0].min():.1f}×{coords[:, 1].max() - coords[:, 1].min():.1f} cm")
    print(f"  - Temporal resolution: {1/dt:.2f} Hz")
    print(f"  - Signal magnitudes: HbO ~{np.std(hbo_data):.1e}, HbR ~{np.std(hbr_data):.1e}")
    
    return hemo_data, coords, hbo_data, hbr_data, time


if __name__ == "__main__":
    hemo_data, coords, hbo_data, hbr_data, time = example_spatial_temporal_analysis()