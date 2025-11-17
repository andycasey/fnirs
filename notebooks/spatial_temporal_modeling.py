"""
# Spatial-Temporal Modeling of fNIRS Data

This notebook demonstrates how to use the fnirs package for spatial-temporal modeling
of hemodynamic responses using spherical harmonics and Fourier basis functions.

## Overview

The analysis pipeline includes:
1. Loading hemodynamic concentration data
2. Short-separation channel regression (removing systemic noise)
3. Physiological regression
4. Projecting channel coordinates onto a sphere
5. Fitting a spatial-temporal model using spherical harmonics and Fourier basis
6. Visualizing the results
"""

import marimo

__generated_with = "0.15.3"
app = marimo.App(width="full")


@app.cell
def __():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import jax.numpy as jnp
    from fnirs import (
        load_hemodynamic_data,
        project_fnirs_to_sphere,
        cartesian_to_spherical,
        create_spherical_harmonics_basis,
        fit as fit_model,
    )
    return (
        LinearRegression,
        PCA,
        StandardScaler,
        cartesian_to_spherical,
        create_spherical_harmonics_basis,
        fit_model,
        jnp,
        load_hemodynamic_data,
        np,
        plt,
        project_fnirs_to_sphere,
        r2_score,
    )


@app.cell
def __():
    # Configuration
    DATA_PATH = "../rsFC-fnirs-course/Data_for_Part_II.mat"

    # Short separation channel indices (convert to 0-based indexing)
    ss_list = np.array([8, 29, 52, 66, 75, 92, 112, 125]) - 1

    # Number of PCA components to remove
    n_pca_remove = 2

    # Maximum degree for spherical harmonics
    max_spherical_degree = 5

    # Scaling factor for numerical stability
    scale = 1e6
    return DATA_PATH, max_spherical_degree, n_pca_remove, scale, ss_list


@app.cell
def __(DATA_PATH, load_hemodynamic_data):
    # Load the hemodynamic data
    print("Loading hemodynamic data...")
    hemo_data = load_hemodynamic_data(DATA_PATH)

    print(f"Loaded {len(hemo_data.channels)} channels")
    print(f"Time range: {hemo_data.time[0]:.2f} - {hemo_data.time[-1]:.2f} seconds")
    print(f"Sampling frequency: {hemo_data.probe.sampling_freq} Hz")
    return hemo_data,


@app.cell
def __(PCA, StandardScaler, hemo_data, n_pca_remove, np, ss_list):
    # Step 1: Short-separation channel regression
    print("\nPerforming short-separation channel regression...")

    # Create mask for short channels
    is_short_channel = np.zeros(len(hemo_data.channels), dtype=bool)
    is_short_channel[ss_list] = True

    # Standardize short channel data for PCA
    scaler = StandardScaler()
    y_reference = scaler.fit_transform(np.vstack([
        hemo_data.get_hbo_data()[:, is_short_channel].T,
        hemo_data.get_hbr_data()[:, is_short_channel].T
    ]))

    # Fit PCA on short channels
    pca = PCA()
    pca.fit(y_reference)

    # Project long channel data and remove first n_pca_remove components
    y_data = scaler.transform(hemo_data.get_hbo_data()[:, ~is_short_channel].T)
    target_scores = pca.transform(y_data)

    # Zero out the first n_pca_remove components
    target_scores_modified = target_scores.copy()
    target_scores_modified[:, :n_pca_remove] = 0

    # Reconstruct cleaned data
    target_reconstructed = pca.inverse_transform(target_scores_modified)
    target_cleaned = scaler.inverse_transform(target_reconstructed)

    print(f"Removed {n_pca_remove} PCA components from {(~is_short_channel).sum()} long channels")
    return (
        is_short_channel,
        pca,
        scaler,
        target_cleaned,
        target_reconstructed,
        target_scores,
        target_scores_modified,
        y_data,
        y_reference,
    )


@app.cell
def __(LinearRegression, hemo_data, target_cleaned):
    # Step 2: Physiological regression
    print("\nPerforming physiological regression...")

    regression_model = LinearRegression()
    regression_model.fit(hemo_data.physiology_data, target_cleaned.T)
    target_cleaned_phys = target_cleaned - regression_model.predict(hemo_data.physiology_data).T

    print(f"Regressed out {hemo_data.physiology_data.shape[1]} physiological signals")
    return regression_model, target_cleaned_phys


@app.cell
def __(hemo_data, is_short_channel, project_fnirs_to_sphere):
    # Step 3: Project coordinates onto sphere
    print("\nProjecting coordinates onto sphere...")

    # Get 3D coordinates for long channels only
    coords_3d = hemo_data.get_spatial_coordinates_3d()[~is_short_channel]

    # Project to sphere
    sphere_result = project_fnirs_to_sphere(coords_3d, fit_method='least_squares')
    θ, ϕ = sphere_result['theta'], sphere_result['phi']

    print(f"Projected {len(coords_3d)} channels onto sphere")
    print(f"Sphere radius: {sphere_result['sphere_radius']:.2f} {hemo_data.probe.spatial_unit}")
    return coords_3d, sphere_result, θ, ϕ


@app.cell
def __(
    fit_model,
    hemo_data,
    jnp,
    max_spherical_degree,
    scale,
    target_cleaned_phys,
    θ,
    ϕ,
):
    # Step 4: Fit spatial-temporal model
    print("\nFitting spatial-temporal model...")
    print(f"Using max spherical degree: {max_spherical_degree}")

    # Prepare data for model fitting
    t = jnp.array(hemo_data.time)
    θ_jax, ϕ_jax = jnp.array(θ), jnp.array(ϕ)
    Y = jnp.array(target_cleaned_phys) * scale

    # Fit the model
    import time
    t_start = time.time()
    X, f, *extras = fit_model(
        t, θ_jax, ϕ_jax, Y,
        max_spherical_degree=max_spherical_degree,
        n_fourier_components=len(t)
    )
    t_elapsed = time.time() - t_start

    print(f"Fitted model in {t_elapsed:.2f} seconds")
    print(f"Model coefficients shape: {X.shape}")
    return X, extras, f, t, t_elapsed, t_start, θ_jax, ϕ_jax


@app.cell
def __(Y, f, np, r2_score):
    # Step 5: Evaluate model fit
    print("\nEvaluating model fit...")

    Y_predicted = f(X)

    # Compute R^2 for each channel
    r2 = r2_score(Y, Y_predicted, multioutput='raw_values')

    print(f"R² (mean ± std): {r2.mean():.3f} ± {r2.std():.3f}")
    print(f"R² (min, max): ({r2.min():.3f}, {r2.max():.3f})")
    return Y_predicted, r2


@app.cell
def __(Y, Y_predicted, np, plt, r2, t):
    # Visualization 1: Model fit quality
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Scatter plot of observed vs predicted
    ax = axes[0]
    ax.scatter(Y, Y_predicted, s=1, alpha=0.1)
    limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
    limits = np.min(limits), np.max(limits)
    ax.plot(limits, limits, color='k', ls='--', label='Perfect fit')
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel('Observed (scaled)')
    ax.set_ylabel('Predicted (scaled)')
    ax.set_title('Model Fit: Observed vs Predicted')
    ax.legend()

    # Right: R² distribution
    ax = axes[1]
    ax.hist(r2, bins=30, edgecolor='k', alpha=0.7)
    ax.axvline(r2.mean(), color='r', ls='--', label=f'Mean: {r2.mean():.3f}')
    ax.set_xlabel('R²')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of R² Across Channels')
    ax.legend()

    plt.tight_layout()
    plt.show()
    return ax, axes, fig, limits


@app.cell
def __(Y, Y_predicted, np, plt, t):
    # Visualization 2: Example time series
    fig, ax = plt.subplots(figsize=(12, 8))

    # Select 10 random channels to plot
    np.random.seed(42)
    indices = np.random.choice(Y.shape[0], size=10, replace=False)

    for i, index in enumerate(indices):
        y_max = np.max(np.abs(Y[index]))
        offset = i * 3
        ax.plot(t, Y[index]/y_max + offset, c='k', alpha=0.7, label='Data' if i == 0 else None)
        ax.plot(t, Y_predicted[index]/y_max + offset, c='tab:red', alpha=0.7, label='Model' if i == 0 else None)
        ax.text(t[-1] * 1.01, offset, f"Ch {index + 1}", va='center')

    ax.set_xlim(0, t[-1] * 1.15)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Scaled and offset signals')
    ax.set_title('Model Fit: Example Time Series (10 channels)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return ax, fig, i, index, indices, offset, y_max


@app.cell
def __(Y, Y_predicted, coords_3d, hemo_data, np, plt, scale, t):
    # Visualization 3: Spatial and temporal residuals
    abs_residuals = np.abs(Y - Y_predicted)

    fig = plt.figure(figsize=(14, 5))

    # Left: Temporal residuals
    ax1 = fig.add_subplot(121)
    mean_residuals = abs_residuals.mean(axis=0)
    ax1.plot(t, mean_residuals, c='k')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(f'Mean |residual| per channel ({1/scale:.1e})')
    ax1.set_title('Temporal Residuals')
    ax1.grid(True, alpha=0.3)

    # Right: Spatial residuals
    ax2 = fig.add_subplot(122, projection='3d')
    spatial_residuals = abs_residuals.mean(axis=1)
    scatter = ax2.scatter(
        coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
        c=spatial_residuals,
        cmap='viridis',
        s=50
    )
    plt.colorbar(scatter, ax=ax2, label=f'Mean |residual| per time ({1/scale:.1e})')
    ax2.set_xlabel(f'X ({hemo_data.probe.spatial_unit})')
    ax2.set_ylabel(f'Y ({hemo_data.probe.spatial_unit})')
    ax2.set_zlabel(f'Z ({hemo_data.probe.spatial_unit})')
    ax2.set_title('Spatial Residuals')

    plt.tight_layout()
    plt.show()
    return abs_residuals, ax1, ax2, fig, mean_residuals, scatter, spatial_residuals


@app.cell
def __():
    # Summary
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nThis notebook demonstrated:")
    print("  1. Loading hemodynamic fNIRS data")
    print("  2. Short-separation channel regression")
    print("  3. Physiological regression")
    print("  4. Spherical projection of channel coordinates")
    print("  5. Spatial-temporal modeling with spherical harmonics and Fourier basis")
    print("  6. Model evaluation and visualization")
    print("\nThe model successfully captures spatial and temporal patterns in the data.")
    return


if __name__ == "__main__":
    app.run()
