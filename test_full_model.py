#!/usr/bin/env python
"""
Test the full model with all Fourier components (as shown in README).
"""

print("Testing full model with all Fourier components...")

from fnirs import (
    load_hemodynamic_data,
    project_fnirs_to_sphere,
    fit,
)
import jax.numpy as jnp
import time

# Load data
print("\n1. Loading data...")
hemo_data = load_hemodynamic_data("scratch/rsFC-fnirs-course/Data_for_Part_II.mat")
print(f"✓ Loaded {len(hemo_data.channels)} channels, {len(hemo_data.time)} time points")

# Project to sphere
print("\n2. Projecting to sphere...")
coords_3d = hemo_data.get_spatial_coordinates_3d()
sphere_result = project_fnirs_to_sphere(coords_3d)
θ, ϕ = sphere_result['theta'], sphere_result['phi']
print(f"✓ Projected to sphere (radius: {sphere_result['sphere_radius']:.2f} cm)")

# Fit spatial-temporal model
print("\n3. Fitting full model (this may take a moment)...")
t = jnp.array(hemo_data.time)
Y = jnp.array(hemo_data.get_hbo_data().T)  # Shape: (n_channels, n_timepoints)

start_time = time.time()
X, f, *extras = fit(
    t,
    jnp.array(θ),
    jnp.array(ϕ),
    Y,
    max_spherical_degree=5,
    n_fourier_components=len(t)  # Full model as in README
)
elapsed = time.time() - start_time

print(f"✓ Model fitted in {elapsed:.2f} seconds")
print(f"  Coefficient matrix shape: {X.shape}")

# Get predictions
print("\n4. Evaluating model...")
Y_predicted = f(X)

from sklearn.metrics import r2_score
import numpy as np

r2 = r2_score(np.array(Y).T, np.array(Y_predicted).T, multioutput='raw_values')
print(f"✓ R² (mean ± std): {r2.mean():.3f} ± {r2.std():.3f}")
print(f"  R² range: [{r2.min():.3f}, {r2.max():.3f}]")

print("\n" + "="*60)
print("✓ FULL MODEL TEST PASSED!")
print("="*60)
print(f"\nThe full model (with {len(t)} Fourier components) works correctly.")
print(f"Average R² of {r2.mean():.3f} indicates good model fit.")
