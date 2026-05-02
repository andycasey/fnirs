#!/usr/bin/env python3
"""Command-line interface for fnirs package."""

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="fNIRS spatial-temporal modeling tools.")


@app.command()
def fit(
    data: Path = typer.Option(..., help="Path to .mat or .snirf file"),
    output: Path = typer.Option(..., help="Output directory for results"),
    max_degree: int = typer.Option(5, help="Max spherical harmonics degree"),
    n_fourier: Optional[int] = typer.Option(None, help="Number of frequency bins (default: all)"),
    chromophore: str = typer.Option("hbo", help="Chromophore: hbo, hbr, or hbt"),
    temporal_kernel: Optional[str] = typer.Option(None, help="Temporal kernel: matern12 or omit for none"),
    kernel_lengthscale: float = typer.Option(1.0, help="Kernel lengthscale"),
    kernel_variance: float = typer.Option(1.0, help="Kernel variance"),
    estimate_noise: bool = typer.Option(False, help="Enable IRLS noise estimation"),
    max_irls_iter: int = typer.Option(20, help="Max IRLS iterations"),
    irls_tol: float = typer.Option(1e-4, help="IRLS convergence tolerance"),
    seed: int = typer.Option(42, help="Random seed for train/test split"),
):
    """Fit a spatial-temporal model to fNIRS data."""
    import numpy as np
    import jax.numpy as jnp

    from fnirs.io import load_hemodynamic_data, load_snirf_data, ChromophoreType
    from fnirs.spherical_projection import project_fnirs_to_sphere
    from fnirs.model import fit as model_fit

    # Load data
    data_path = str(data)
    if data.suffix == ".snirf":
        nirs_data = load_snirf_data(data_path)
        Y = nirs_data.time_series.T  # (n_channels, n_timepoints)
        positions_3d = nirs_data.get_spatial_coordinates_3d()
        if positions_3d is None:
            positions_3d = nirs_data.get_spatial_coordinates_2d()
            # Pad to 3D
            positions_3d = np.column_stack([positions_3d, np.zeros(len(positions_3d))])
        t = jnp.array(nirs_data.time)
    else:
        hemo_data = load_hemodynamic_data(data_path)
        chrom_map = {"hbo": ChromophoreType.HbO, "hbr": ChromophoreType.HbR, "hbt": ChromophoreType.HbT}
        chrom = chrom_map[chromophore.lower()]
        Y = hemo_data.get_concentration_matrix(chrom).T  # (n_channels, n_timepoints)
        positions_3d = hemo_data.get_spatial_coordinates_3d()
        t = jnp.array(hemo_data.time)

    # Project to sphere
    proj = project_fnirs_to_sphere(positions_3d)
    theta = jnp.array(proj["theta"])
    phi = jnp.array(proj["phi"])
    Y = jnp.array(Y)

    # Fit model
    result = model_fit(
        t=t,
        θ=theta,
        ϕ=phi,
        Y=Y,
        max_spherical_degree=max_degree,
        n_fourier_components=n_fourier,
        estimate_noise=estimate_noise,
        max_irls_iter=max_irls_iter,
        irls_tol=irls_tol,
        temporal_kernel=temporal_kernel,
        kernel_lengthscale=kernel_lengthscale,
        kernel_variance=kernel_variance,
    )

    X_freq_full, predict_fn, _, ST, terms, noise_variance, n_iter = result

    # Save results
    output.mkdir(parents=True, exist_ok=True)

    save_dict = dict(
        X_freq_real=np.array(X_freq_full.real),
        X_freq_imag=np.array(X_freq_full.imag),
        ST=np.array(ST),
        n_timepoints=int(Y.shape[1]),
    )
    # Save terms as arrays
    terms_l = np.array([t[0] for t in terms])
    terms_m = np.array([t[1] for t in terms])
    save_dict["terms_l"] = terms_l
    save_dict["terms_m"] = terms_m

    if noise_variance is not None:
        save_dict["noise_variance"] = np.array(noise_variance)

    np.savez(output / "model.npz", **save_dict)

    # Save config
    config = dict(
        data=str(data),
        max_degree=max_degree,
        n_fourier=n_fourier,
        chromophore=chromophore,
        temporal_kernel=temporal_kernel,
        kernel_lengthscale=kernel_lengthscale,
        kernel_variance=kernel_variance,
        estimate_noise=estimate_noise,
        max_irls_iter=max_irls_iter,
        irls_tol=irls_tol,
        seed=seed,
    )
    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Print summary
    n_channels, n_timepoints = Y.shape
    n_spatial = ST.shape[1]
    n_freq = X_freq_full.shape[1]
    typer.echo(f"Fit complete.")
    typer.echo(f"  Channels: {n_channels}")
    typer.echo(f"  Timepoints: {n_timepoints}")
    typer.echo(f"  Spatial basis functions: {n_spatial}")
    typer.echo(f"  Frequency bins: {n_freq}")
    if noise_variance is not None:
        typer.echo(f"  IRLS iterations: {n_iter}")
        typer.echo(f"  Noise variance: min={float(noise_variance.min()):.6f}, max={float(noise_variance.max()):.6f}, median={float(jnp.median(noise_variance)):.6f}")
    typer.echo(f"  Results saved to: {output}")


@app.command()
def plot(
    model_dir: Path = typer.Option(..., help="Path to output from fnirs fit"),
    data: Optional[Path] = typer.Option(None, help="Original data file (for residuals)"),
    output: Optional[Path] = typer.Option(None, help="Directory for figures (default: model-dir/figures/)"),
):
    """Visualize a fitted model."""
    import numpy as np

    from fnirs.plotting import (
        plot_harmonics_timeseries,
        plot_residuals,
        plot_noise_variance,
        plot_power_spectrum,
        plot_spatial_snapshot,
    )

    # Load model
    model_data = np.load(model_dir / "model.npz")
    X_freq = model_data["X_freq_real"] + 1j * model_data["X_freq_imag"]
    ST = model_data["ST"]
    terms_l = model_data["terms_l"]
    terms_m = model_data["terms_m"]
    terms = list(zip(terms_l.tolist(), terms_m.tolist()))
    n_timepoints = int(model_data["n_timepoints"])
    noise_variance = model_data.get("noise_variance", None)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    # Output directory
    fig_dir = output if output else model_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Harmonics timeseries
    plot_harmonics_timeseries(X_freq, terms, n_timepoints, fig_dir / "harmonics_timeseries.png")
    typer.echo(f"Saved harmonics_timeseries.png")

    # 2. Residuals (needs original data)
    if data is not None:
        plot_residuals(X_freq, ST, n_timepoints, data, config, fig_dir / "residuals.png")
        typer.echo(f"Saved residuals.png")

    # 3. Noise variance
    if noise_variance is not None:
        plot_noise_variance(noise_variance, fig_dir / "noise_variance.png")
        typer.echo(f"Saved noise_variance.png")

    # 4. Power spectrum
    if data is not None:
        plot_power_spectrum(X_freq, ST, n_timepoints, data, config, fig_dir / "power_spectrum.png")
        typer.echo(f"Saved power_spectrum.png")

    # 5. Spatial snapshot
    plot_spatial_snapshot(X_freq, terms, n_timepoints, config, fig_dir / "spatial_snapshot.png")
    typer.echo(f"Saved spatial_snapshot.png")

    typer.echo(f"All figures saved to: {fig_dir}")


if __name__ == "__main__":
    app()
