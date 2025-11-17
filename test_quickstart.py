#!/usr/bin/env python
"""
Test script to verify the Quick Start instructions in README.md work correctly.
"""

print("="*60)
print("Testing Quick Start Instructions from README.md")
print("="*60)

print("\n1. Testing imports...")
try:
    from fnirs import (
        load_hemodynamic_data,
        project_fnirs_to_sphere,
        fit,
    )
    import jax.numpy as jnp
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

print("\n2. Loading data...")
try:
    hemo_data = load_hemodynamic_data("scratch/rsFC-fnirs-course/Data_for_Part_II.mat")
    print(f"✓ Data loaded successfully")
    print(f"  - {len(hemo_data.channels)} channels")
    print(f"  - {len(hemo_data.time)} time points")
    print(f"  - Time range: {hemo_data.time[0]:.2f} - {hemo_data.time[-1]:.2f} seconds")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    exit(1)

print("\n3. Projecting coordinates onto sphere...")
try:
    coords_3d = hemo_data.get_spatial_coordinates_3d()
    print(f"  - Got {len(coords_3d)} 3D coordinates")

    sphere_result = project_fnirs_to_sphere(coords_3d)
    θ, ϕ = sphere_result['theta'], sphere_result['phi']
    print(f"✓ Sphere projection successful")
    print(f"  - Sphere radius: {sphere_result['sphere_radius']:.2f} {hemo_data.probe.spatial_unit}")
    print(f"  - θ range: [{θ.min():.2f}, {θ.max():.2f}]")
    print(f"  - ϕ range: [{ϕ.min():.2f}, {ϕ.max():.2f}]")
except Exception as e:
    print(f"✗ Sphere projection failed: {e}")
    exit(1)

print("\n4. Fitting spatial-temporal model...")
try:
    # Prepare data
    t = jnp.array(hemo_data.time)
    Y = jnp.array(hemo_data.get_hbo_data().T)  # Shape: (n_channels, n_timepoints)
    print(f"  - Time points: {len(t)}")
    print(f"  - Data shape: {Y.shape}")

    # Fit model (use fewer components for faster testing)
    import time
    n_components = min(100, len(t))
    print(f"  - Using {n_components} Fourier components for testing")

    start_time = time.time()
    X, f, *extras = fit(
        t,
        jnp.array(θ),
        jnp.array(ϕ),
        Y,
        max_spherical_degree=5,
        n_fourier_components=n_components
    )
    elapsed = time.time() - start_time

    print(f"✓ Model fitting successful")
    print(f"  - Fit time: {elapsed:.2f} seconds")
    print(f"  - Coefficient matrix shape: {X.shape}")
except Exception as e:
    print(f"✗ Model fitting failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n5. Getting predictions...")
try:
    Y_predicted = f(X)
    print(f"✓ Predictions generated successfully")
    print(f"  - Predicted shape: {Y_predicted.shape}")

    # Calculate R²
    from sklearn.metrics import r2_score
    import numpy as np

    r2 = r2_score(np.array(Y).T, np.array(Y_predicted).T, multioutput='raw_values')
    print(f"  - R² (mean): {r2.mean():.3f}")
    print(f"  - R² (std): {r2.std():.3f}")
except Exception as e:
    print(f"✗ Prediction generation failed: {e}")
    exit(1)

print("\n" + "="*60)
print("✓ ALL QUICK START TESTS PASSED!")
print("="*60)
print("\nThe Quick Start instructions in README.md are working correctly.")
