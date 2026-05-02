#!/usr/bin/env python3
"""Command-line interface for fnirs package."""

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="fNIRS spatial-temporal modeling tools.")


@app.command()
def fit(
    data: Path = typer.Argument(..., help="Path to .mat or .snirf file"),
    output: Path = typer.Argument(..., help="Output directory for results"),
    max_degree: int = typer.Option(5, help="Max spherical harmonics degree"),
    n_fourier: Optional[int] = typer.Option(None, help="Number of frequency bins (default: all)"),
    chromophore: str = typer.Option("hbo", help="Chromophore: hbo, hbr, or hbt"),
    temporal_kernel: Optional[str] = typer.Option(None, help="Temporal kernel: matern12 or omit for none"),
    kernel_lengthscale: float = typer.Option(1.0, help="Kernel lengthscale"),
    kernel_variance: float = typer.Option(1.0, help="Kernel variance"),
    estimate_noise: bool = typer.Option(False, help="Enable IRLS noise estimation"),
    max_irls_iter: int = typer.Option(20, help="Max IRLS iterations"),
    irls_tol: float = typer.Option(1e-4, help="IRLS convergence tolerance"),
    irls_var_clip_ratio: float = typer.Option(100.0, help="Clip per-channel σ² to ±this factor of the median each IRLS iteration (prevents runaway when the basis can't represent some channels)."),
    spatial_ridge: float = typer.Option(0.0, help="Tikhonov ridge added to ST^T ST in every frequency bin. Use to tame ill-conditioning when channels cluster on a small patch (typical for max_degree ≥ 4). 0 = unbiased OLS; 1e-3 to 1e-2 is a sensible non-zero range."),
    short_pca_components: int = typer.Option(0, help="Run PCA on short-separation channels (top-k components) and regress them out of the long channels before fitting. 0 = disabled. Only applies to .snirf/.lob inputs."),
    seed: int = typer.Option(42, help="Random seed for train/test split"),
    plots: bool = typer.Option(True, "--plots/--no-plots", help="Run the plotting stage after fitting (default: on)."),
    include_short_channels: bool = typer.Option(False, "--include-short-channels", help="Include short-separation channels in the fit (default: excluded; they measure superficial physiology, not cortex)."),
    seed_channel_index: int = typer.Option(6, help="Channel index used to compute mean seed-vs-rest correlation."),
    mav_scale: bool = typer.Option(False, "--mav-scale", help="Per-channel divide signal by its mean absolute value before fitting."),
    bandpass: Optional[str] = typer.Option(None, "--bandpass", help="Zero-phase Butterworth bandpass 'low,high' in Hz, applied before short-PCA regression. Use 'low,' for highpass-only or ',high' for lowpass-only. Example: 0.01,0.1"),
):
    """Fit a spatial-temporal model to fNIRS data."""
    _run_fit(
        data=data, output=output, max_degree=max_degree, n_fourier=n_fourier,
        chromophore=chromophore, temporal_kernel=temporal_kernel,
        kernel_lengthscale=kernel_lengthscale, kernel_variance=kernel_variance,
        estimate_noise=estimate_noise, max_irls_iter=max_irls_iter,
        irls_tol=irls_tol, irls_var_clip_ratio=irls_var_clip_ratio, seed=seed,
        include_short_channels=include_short_channels, plots=plots,
        seed_channel_index=seed_channel_index, spatial_ridge=spatial_ridge,
        short_pca_components=short_pca_components, mav_scale=mav_scale,
        bandpass=bandpass,
    )


def _parse_bandpass(spec: str):
    """Parse '<low>,<high>' (either may be empty) into (low_hz, high_hz)."""
    if spec is None:
        return None, None
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"--bandpass expected 'low,high', got {spec!r}")
    low = float(parts[0]) if parts[0].strip() else None
    high = float(parts[1]) if parts[1].strip() else None
    if low is None and high is None:
        raise ValueError("--bandpass requires at least one of low or high")
    return low, high


def _apply_bandpass(Y, dt, low_hz, high_hz, order=4):
    """Zero-phase Butterworth bandpass along time axis (axis=1)."""
    import numpy as np
    from scipy.signal import butter, filtfilt
    fs = 1.0 / float(dt)
    nyq = 0.5 * fs
    if low_hz is not None and high_hz is not None:
        b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    elif low_hz is not None:
        b, a = butter(order, low_hz / nyq, btype="high")
    elif high_hz is not None:
        b, a = butter(order, high_hz / nyq, btype="low")
    else:
        return Y
    return filtfilt(b, a, np.asarray(Y), axis=1)


def _short_channel_pca_regress(Y_long, Y_short, n_components):
    """Regress top-k PCA temporal components of Y_short out of Y_long.

    Returns (Y_long_clean, V_top) where V_top has shape (k, n_t).
    Centering is on each per-channel mean before PCA and before regression.
    """
    import numpy as np
    Y_short_c = Y_short - Y_short.mean(axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(Y_short_c, full_matrices=False)
    k = min(int(n_components), Vt.shape[0])
    V_top = Vt[:k]  # rows are orthonormal temporal components
    Y_long_c = Y_long - Y_long.mean(axis=1, keepdims=True)
    beta = V_top @ Y_long_c.T  # (k, n_long)
    Y_long_clean = Y_long - beta.T @ V_top
    return Y_long_clean, V_top


def _run_fit(
    data: Path, output: Path, max_degree: int, n_fourier: Optional[int],
    chromophore: str, temporal_kernel: Optional[str],
    kernel_lengthscale: float, kernel_variance: float,
    estimate_noise: bool, max_irls_iter: int, irls_tol: float, seed: int,
    include_short_channels: bool, plots: bool,
    irls_var_clip_ratio: float = 100.0,
    seed_channel_index: int = 6,
    spatial_ridge: float = 0.0,
    short_pca_components: int = 0,
    mav_scale: bool = False,
    bandpass: Optional[str] = None,
) -> dict:
    """Run a single fit; save model + config; return summary metrics."""
    import numpy as np
    import jax.numpy as jnp

    from fnirs.io import load_hemodynamic_data, load_snirf_data, load_lob_data, ChromophoreType
    from fnirs.spherical_projection import project_fnirs_to_sphere
    from fnirs.model import fit as model_fit

    data_path = str(data)
    short_pca_basis = None
    Y_short_raw = None
    if data.suffix in (".snirf", ".lob"):
        nirs_data = load_snirf_data(data_path) if data.suffix == ".snirf" else load_lob_data(data_path)
        label_map = {"hbo": "HbO", "hbr": "HbR", "hbt": "HbT"}
        target_label = label_map.get(chromophore.lower())
        if target_label is None:
            raise typer.BadParameter(f"Unknown chromophore: {chromophore!r} (expected hbo/hbr/hbt)")
        selected = nirs_data.get_channels_by_data_type_label(target_label)
        if not selected:
            available = sorted({ch.measurement_info.data_type_label for ch in nirs_data.channels})
            raise typer.BadParameter(
                f"No channels with data_type_label={target_label!r} in {data_path}. "
                f"Available labels: {available}"
            )
        n_short = sum(1 for ch in selected if ch.is_short_separation)
        short_channels_in_label = [ch for ch in selected if ch.is_short_separation]
        if not include_short_channels:
            selected = [ch for ch in selected if not ch.is_short_separation]
            if not selected:
                raise typer.BadParameter(
                    f"All {target_label} channels are short-separation; pass --include-short-channels to fit them."
                )
        ch_indices = np.array([ch.channel_idx for ch in selected])
        n_total = len(nirs_data.channels)
        short_msg = f", excluded {n_short} short" if (n_short and not include_short_channels) else (f", including {n_short} short" if include_short_channels and n_short else "")
        typer.echo(f"Selected {len(ch_indices)} {target_label} channels (of {n_total} total{short_msg})")
        Y = nirs_data.time_series[:, ch_indices].T
        positions_3d_all = nirs_data.get_spatial_coordinates_3d()
        if positions_3d_all is None:
            positions_2d_all = nirs_data.get_spatial_coordinates_2d()
            positions_3d_all = np.column_stack([positions_2d_all, np.zeros(len(positions_2d_all))])
        positions_3d = positions_3d_all[ch_indices]
        t = jnp.array(nirs_data.time)

        if bandpass is not None:
            low_hz, high_hz = _parse_bandpass(bandpass)
            dt_data = float(np.asarray(nirs_data.time)[1] - np.asarray(nirs_data.time)[0])
            Y = _apply_bandpass(Y, dt_data, low_hz, high_hz)
            typer.echo(f"Applied bandpass [{low_hz}, {high_hz}] Hz (fs={1.0/dt_data:.3f} Hz) to {Y.shape[0]} channels.")

        short_pca_basis = None
        Y_short_raw = None
        if short_pca_components > 0:
            if include_short_channels:
                typer.echo("Warning: --short-pca-components ignored when --include-short-channels is set.")
            elif not short_channels_in_label:
                typer.echo(f"Warning: --short-pca-components={short_pca_components} requested but no short {target_label} channels available.")
            else:
                short_idx = np.array([ch.channel_idx for ch in short_channels_in_label])
                Y_short_raw = nirs_data.time_series[:, short_idx].T
                if bandpass is not None:
                    low_hz, high_hz = _parse_bandpass(bandpass)
                    dt_data = float(np.asarray(nirs_data.time)[1] - np.asarray(nirs_data.time)[0])
                    Y_short_raw = _apply_bandpass(Y_short_raw, dt_data, low_hz, high_hz)
                Y, short_pca_basis = _short_channel_pca_regress(Y, Y_short_raw, short_pca_components)
                typer.echo(f"Regressed top-{short_pca_basis.shape[0]} PCA components of {len(short_idx)} short channels out of {Y.shape[0]} long channels.")
    else:
        hemo_data = load_hemodynamic_data(data_path)
        chrom_map = {"hbo": ChromophoreType.HbO, "hbr": ChromophoreType.HbR, "hbt": ChromophoreType.HbT}
        chrom = chrom_map[chromophore.lower()]
        Y = hemo_data.get_concentration_matrix(chrom).T
        positions_3d = hemo_data.get_spatial_coordinates_3d()
        t = jnp.array(hemo_data.time)

    proj = project_fnirs_to_sphere(positions_3d)
    theta = jnp.array(proj["theta"])
    phi = jnp.array(proj["phi"])

    if mav_scale:
        Y_np = np.asarray(Y)
        mav = np.mean(np.abs(Y_np), axis=1, keepdims=True)
        scale = np.where(mav > 0, mav, 1.0)
        Y = Y_np / scale
        typer.echo(f"Per-channel divided Y by mean absolute value.")
    Y = jnp.array(Y)

    result = model_fit(
        t=t, θ=theta, ϕ=phi, Y=Y,
        max_spherical_degree=max_degree,
        n_fourier_components=n_fourier,
        estimate_noise=estimate_noise,
        max_irls_iter=max_irls_iter,
        irls_tol=irls_tol,
        irls_var_clip_ratio=irls_var_clip_ratio,
        temporal_kernel=temporal_kernel,
        kernel_lengthscale=kernel_lengthscale,
        kernel_variance=kernel_variance,
        spatial_ridge=spatial_ridge,
    )

    X_freq_full, predict_fn, _, ST, terms, noise_variance, n_iter, noise_variance_history = result

    output.mkdir(parents=True, exist_ok=True)

    Y_np = np.array(Y)
    Y_hat = np.fft.irfft(np.array(ST) @ np.array(X_freq_full), n=int(Y.shape[1]), axis=1)
    residuals = Y_np - Y_hat
    residual_rms_per_channel = np.sqrt(np.mean(residuals ** 2, axis=1))
    residual_rms = float(np.mean(residual_rms_per_channel))

    def _channel_corr(M):
        Mc = M - M.mean(axis=1, keepdims=True)
        std = Mc.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        Mn = Mc / std
        return (Mn @ Mn.T) / Mn.shape[1]

    save_dict = dict(
        X_freq_real=np.array(X_freq_full.real),
        X_freq_imag=np.array(X_freq_full.imag),
        ST=np.array(ST),
        n_timepoints=int(Y.shape[1]),
        correlation_data=_channel_corr(Y_np),
        correlation_fit=_channel_corr(Y_hat),
        correlation_residual=_channel_corr(residuals),
    )
    terms_l = np.array([t[0] for t in terms])
    terms_m = np.array([t[1] for t in terms])
    save_dict["terms_l"] = terms_l
    save_dict["terms_m"] = terms_m
    if noise_variance is not None:
        save_dict["noise_variance"] = np.array(noise_variance)
    if noise_variance_history is not None:
        save_dict["noise_variance_history"] = np.array(noise_variance_history)
    if short_pca_basis is not None:
        save_dict["short_pca_basis"] = np.asarray(short_pca_basis)
    if Y_short_raw is not None:
        save_dict["short_channels_raw"] = np.asarray(Y_short_raw)
    save_dict["time"] = np.asarray(t)
    np.savez(output / "model.npz", **save_dict)

    config = dict(
        data=str(data.resolve()),
        max_degree=max_degree,
        n_fourier=n_fourier,
        chromophore=chromophore,
        temporal_kernel=temporal_kernel,
        kernel_lengthscale=kernel_lengthscale,
        kernel_variance=kernel_variance,
        estimate_noise=estimate_noise,
        max_irls_iter=max_irls_iter,
        irls_tol=irls_tol,
        irls_var_clip_ratio=irls_var_clip_ratio,
        seed=seed,
        include_short_channels=include_short_channels,
        spatial_ridge=spatial_ridge,
        short_pca_components=short_pca_components,
        mav_scale=mav_scale,
        bandpass=bandpass,
    )
    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    n_channels, n_timepoints = Y.shape
    n_spatial = ST.shape[1]
    n_freq = X_freq_full.shape[1]

    seed_metrics = None
    if 0 <= seed_channel_index < n_channels:
        def _seed_mean(M):
            return float((np.sum(M[:, seed_channel_index]) - 1) / (M.shape[0] - 1))
        seed_metrics = {
            "data": _seed_mean(save_dict["correlation_data"]),
            "fit": _seed_mean(save_dict["correlation_fit"]),
            "residual": _seed_mean(save_dict["correlation_residual"]),
        }

    n_freq_all_int = int(n_freq)
    n_freq_used = n_fourier if n_fourier is not None else n_freq_all_int
    n_temporal_dof = 2 * n_freq_used - 1
    if n_freq_used == n_freq_all_int and n_timepoints % 2 == 0:
        n_temporal_dof -= 1
    n_temporal_dof = min(n_temporal_dof, int(n_timepoints))
    n_params = int(n_spatial) * n_temporal_dof
    n_obs = int(n_channels) * int(n_timepoints)
    dof = n_obs - n_params

    reduced_chi2 = None
    if noise_variance is not None and dof > 0:
        sigma2 = np.maximum(np.array(noise_variance), 1e-30)
        chi2 = float(np.sum(residuals ** 2 / sigma2[:, None]))
        reduced_chi2 = chi2 / dof

    typer.echo(f"Fit complete.")
    typer.echo(f"  Channels: {n_channels}")
    typer.echo(f"  Timepoints: {n_timepoints}")
    typer.echo(f"  Spatial basis functions: {n_spatial}")
    typer.echo(f"  Frequency bins: {n_freq}")
    typer.echo(f"  Residual RMS (mean over channels): {residual_rms:.6f}")
    if reduced_chi2 is not None:
        typer.echo(f"  Reduced χ² (dof={dof}): {reduced_chi2:.4f}")
    elif noise_variance is None:
        typer.echo(f"  Reduced χ²: (requires --estimate-noise)")
    else:
        typer.echo(f"  Reduced χ²: (dof={dof}, non-positive — not enough data)")
    if seed_metrics is not None:
        typer.echo(f"  Seed channel index: {seed_channel_index}")
        typer.echo(f"  Mean seed-vs-rest correlation: data={seed_metrics['data']:.4f}, fit={seed_metrics['fit']:.4f}, residual={seed_metrics['residual']:.4f}")
    else:
        typer.echo(f"  Seed channel index {seed_channel_index} out of range (0..{n_channels - 1}); skipping seed-correlation summary.")
    if noise_variance is not None:
        typer.echo(f"  IRLS iterations: {n_iter}")
        typer.echo(f"  Noise variance: min={float(noise_variance.min()):.6f}, max={float(noise_variance.max()):.6f}, median={float(jnp.median(noise_variance)):.6f}")
    typer.echo(f"  Results saved to: {output}")

    if plots:
        try:
            _run_plots(output)
        except Exception as e:
            typer.echo(f"Warning: plotting stage failed: {e}")

    return dict(
        residual_rms=residual_rms,
        n_channels=int(n_channels),
        n_spatial=int(n_spatial),
        n_freq=int(n_freq),
        n_iter=int(n_iter),
        noise_variance_median=float(jnp.median(noise_variance)) if noise_variance is not None else None,
        seed_channel_index=int(seed_channel_index),
        seed_mean_corr=seed_metrics,
        reduced_chi2=reduced_chi2,
        dof=int(dof),
    )


def _run_plots(model_dir: Path, output: Optional[Path] = None):
    """Generate figures for a fitted model directory."""
    import numpy as np

    from fnirs.plotting import (
        plot_correlation_matrix,
        plot_data_vs_fit,
        plot_harmonics_timeseries,
        plot_residuals,
        plot_noise_variance,
        plot_noise_variance_history,
        plot_power_spectrum,
        plot_short_channel_pca,
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
    noise_variance_history = model_data.get("noise_variance_history", None)
    short_pca_basis = model_data.get("short_pca_basis", None)
    short_channels_raw = model_data.get("short_channels_raw", None)
    time_vec = model_data.get("time", None)

    with open(model_dir / "config.json") as f:
        config = json.load(f)
    if short_pca_basis is not None:
        # In-memory only: plot functions read this via _load_data_for_plot.
        config["_short_pca_basis"] = np.asarray(short_pca_basis)

    data_path = Path(config["data"]) if config.get("data") else None
    has_data = data_path is not None and data_path.exists()
    if data_path is not None and not has_data:
        typer.echo(f"Warning: data file from config not found at {data_path}; skipping data-dependent plots.")

    # Output directory
    fig_dir = output if output else model_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Harmonics timeseries
    plot_harmonics_timeseries(X_freq, terms, n_timepoints, fig_dir / "harmonics_timeseries.png")
    typer.echo(f"Saved harmonics_timeseries.png")

    # 2. Residuals + data-vs-fit (need original data)
    if has_data:
        plot_residuals(X_freq, ST, n_timepoints, data_path, config, fig_dir / "residuals.png")
        typer.echo(f"Saved residuals.png")

        plot_data_vs_fit(
            X_freq, ST, n_timepoints, data_path, config,
            fig_dir / "data_vs_fit.png",
            noise_variance=noise_variance,
        )
        typer.echo(f"Saved data_vs_fit.png")

        plot_correlation_matrix(
            X_freq, ST, n_timepoints, data_path, config,
            fig_dir / "correlation_matrix.png",
        )
        typer.echo(f"Saved correlation_matrix.png")

    # 3. Noise variance
    if noise_variance is not None:
        plot_noise_variance(noise_variance, fig_dir / "noise_variance.png")
        typer.echo(f"Saved noise_variance.png")
    if noise_variance_history is not None:
        plot_noise_variance_history(noise_variance_history, fig_dir / "noise_variance_history.png")
        typer.echo(f"Saved noise_variance_history.png")

    # Short-channel PCA diagnostic
    if short_pca_basis is not None and short_channels_raw is not None:
        t_axis = np.asarray(time_vec) if time_vec is not None else np.arange(short_channels_raw.shape[1])
        plot_short_channel_pca(
            np.asarray(short_channels_raw), np.asarray(short_pca_basis), t_axis,
            fig_dir / "short_channel_pca.png",
        )
        typer.echo(f"Saved short_channel_pca.png")

    # 4. Power spectrum
    if has_data:
        plot_power_spectrum(X_freq, ST, n_timepoints, data_path, config, fig_dir / "power_spectrum.png")
        typer.echo(f"Saved power_spectrum.png")

    # 5. Spatial snapshot — temporarily disabled, suspected incorrect
    # plot_spatial_snapshot(X_freq, terms, n_timepoints, config, fig_dir / "spatial_snapshot.png")
    # typer.echo(f"Saved spatial_snapshot.png")

    typer.echo(f"All figures saved to: {fig_dir}")


@app.command()
def plot(
    model_dir: Path = typer.Argument(..., help="Path to output from fnirs fit"),
    output: Optional[Path] = typer.Option(None, help="Directory for figures (default: model-dir/figures/)"),
):
    """Visualize a fitted model."""
    _run_plots(model_dir, output)


@app.command()
def interact(
    model_dir: Path = typer.Argument(..., help="Path to output from fnirs fit"),
):
    """Interactive head-montage GUI: click a channel to see its data + fit."""
    import matplotlib
    for candidate in ("MacOSX", "TkAgg", "QtAgg"):
        try:
            matplotlib.use(candidate)
            break
        except Exception:
            continue
    else:
        raise typer.Exit(code=1)

    from fnirs.interactive import run as run_interactive
    run_interactive(model_dir)


@app.command()
def explore(
    data: Path = typer.Argument(..., help="Path to .snirf file"),
    output: Optional[Path] = typer.Option(None, help="Output directory (default: <data>.explore/)"),
    chromophore: str = typer.Option("hbo", help="Chromophore for subset plots: hbo, hbr, or hbt"),
):
    """Produce exploratory plots for a SNIRF file."""
    if data.suffix != ".snirf":
        raise typer.BadParameter(f"explore expects a .snirf file, got: {data}")
    if output is None:
        output = data.with_suffix(data.suffix + ".explore")
    output.mkdir(parents=True, exist_ok=True)

    from fnirs.explore import explore as run_explore

    typer.echo(f"Exploring: {data}")
    typer.echo(f"Output dir: {output}")
    generated = run_explore(data, output, chromophore=chromophore)
    for p in generated:
        typer.echo(f"  Saved {p.name}")
    typer.echo(f"Done. {len(generated)} figures saved to {output}")


def _parse_csv(s: str, parser):
    """Split comma-separated string and apply parser to each item."""
    return [parser(x.strip()) for x in s.split(",") if x.strip() != ""]


def _parse_optional_int(s: str) -> Optional[int]:
    if s.lower() in ("none", "null", ""):
        return None
    return int(s)


def _parse_optional_str(s: str) -> Optional[str]:
    if s.lower() in ("none", "null", ""):
        return None
    return s


def _parse_bool(s: str) -> bool:
    return s.lower() in ("true", "1", "yes", "t", "y")


def _format_value(v) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


@app.command()
def gs(
    data: Path = typer.Argument(..., help="Path to .mat or .snirf file"),
    output: Path = typer.Argument(..., help="Parent output directory; one subdir per combination"),
    max_degree: str = typer.Option("5", help="Comma-separated max spherical harmonic degrees"),
    n_fourier: str = typer.Option("none", help="Comma-separated frequency bin counts (none = all)"),
    chromophore: str = typer.Option("hbo", help="Comma-separated chromophores"),
    temporal_kernel: str = typer.Option("none", help="Comma-separated temporal kernels: matern12 or none"),
    kernel_lengthscale: str = typer.Option("1.0", help="Comma-separated kernel lengthscales"),
    kernel_variance: str = typer.Option("1.0", help="Comma-separated kernel variances"),
    estimate_noise: str = typer.Option("false", help="Comma-separated bools for IRLS"),
    max_irls_iter: str = typer.Option("20", help="Comma-separated max IRLS iterations"),
    irls_tol: str = typer.Option("1e-4", help="Comma-separated IRLS tolerances"),
    include_short_channels: bool = typer.Option(False, "--include-short-channels"),
    seed: int = typer.Option(42),
    plots: bool = typer.Option(True, "--plots/--no-plots", help="Run per-combo plotting (default: on)."),
):
    """Grid search: sweep comma-separated parameter values; one fit per combination."""
    import itertools
    import numpy as np

    sweeps = {
        "max_degree": _parse_csv(max_degree, int),
        "n_fourier": _parse_csv(n_fourier, _parse_optional_int),
        "chromophore": _parse_csv(chromophore, str),
        "temporal_kernel": _parse_csv(temporal_kernel, _parse_optional_str),
        "kernel_lengthscale": _parse_csv(kernel_lengthscale, float),
        "kernel_variance": _parse_csv(kernel_variance, float),
        "estimate_noise": _parse_csv(estimate_noise, _parse_bool),
        "max_irls_iter": _parse_csv(max_irls_iter, int),
        "irls_tol": _parse_csv(irls_tol, float),
    }
    for k, vs in sweeps.items():
        if not vs:
            raise typer.BadParameter(f"--{k.replace('_','-')} parsed to empty list")

    keys = list(sweeps.keys())
    combos = list(itertools.product(*[sweeps[k] for k in keys]))
    varied_keys = [k for k in keys if len(sweeps[k]) > 1]

    typer.echo(f"Grid search: {len(combos)} combination(s) over {len(varied_keys)} varied parameter(s): {varied_keys or '(none)'}")

    output.mkdir(parents=True, exist_ok=True)

    summary = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        if varied_keys:
            tag = "_".join(f"{k}={_format_value(params[k])}" for k in varied_keys)
        else:
            tag = "run"
        run_dir = output / f"{i:03d}_{tag}"
        typer.echo(f"\n[{i+1}/{len(combos)}] {run_dir.name}")
        try:
            metrics = _run_fit(
                data=data, output=run_dir,
                include_short_channels=include_short_channels,
                seed=seed, plots=plots,
                **params,
            )
            summary.append(dict(index=i, dir=run_dir.name, params=params, metrics=metrics, ok=True))
        except Exception as e:
            typer.echo(f"  FAILED: {e}")
            summary.append(dict(index=i, dir=run_dir.name, params=params, error=str(e), ok=False))

    with open(output / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    typer.echo(f"\nWrote {output / 'summary.json'}")

    successes = [s for s in summary if s["ok"]]
    if successes and varied_keys:
        try:
            _plot_gs_comparison(output / "comparison.png", successes, varied_keys)
            typer.echo(f"Wrote {output / 'comparison.png'}")
        except Exception as e:
            typer.echo(f"Warning: comparison plot failed: {e}")


def _plot_gs_comparison(output_path: Path, successes: list, varied_keys: list):
    """Plot residual RMS across grid-search runs, labeled by varied parameters."""
    import matplotlib.pyplot as plt

    rms = [s["metrics"]["residual_rms"] for s in successes]
    labels = ["\n".join(f"{k}={_format_value(s['params'][k])}" for k in varied_keys) for s in successes]

    if len(varied_keys) == 1:
        k = varied_keys[0]
        xs = [s["params"][k] for s in successes]
        numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in xs)
        fig, ax = plt.subplots(figsize=(8, 5))
        if numeric:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            ax.plot([xs[i] for i in order], [rms[i] for i in order], marker="o")
            ax.set_xlabel(k)
        else:
            ax.bar(range(len(xs)), rms)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([_format_value(x) for x in xs], rotation=30, ha="right")
            ax.set_xlabel(k)
        ax.set_ylabel("Residual RMS (mean over channels)")
        ax.set_title(f"Grid search: residual RMS vs {k}")
        ax.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(successes)), 5))
        ax.bar(range(len(rms)), rms, color="steelblue")
        ax.set_xticks(range(len(rms)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Residual RMS (mean over channels)")
        ax.set_title(f"Grid search: residual RMS by combination ({len(successes)} runs)")
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    app()
