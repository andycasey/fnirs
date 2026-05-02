#!/usr/bin/env python3
"""Plotting functions for fnirs model visualization."""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_harmonics_timeseries(
    X_freq: np.ndarray,
    terms: List[Tuple[int, int]],
    n_timepoints: int,
    output_path: Path,
):
    """Plot amplitude of each spherical harmonic component over time.

    For each (l, m) pair, compute the time-domain signal via IRFFT.
    Plot grouped by degree l with different colors per m.
    """
    max_l = max(t[0] for t in terms)

    fig, axes = plt.subplots(max_l + 1, 1, figsize=(12, 3 * (max_l + 1)), sharex=True)
    if max_l == 0:
        axes = [axes]

    cmap = plt.cm.tab10

    for l_deg in range(max_l + 1):
        ax = axes[l_deg]
        m_values = [t[1] for t in terms if t[0] == l_deg]
        indices = [i for i, t in enumerate(terms) if t[0] == l_deg]

        for idx, (i, m) in enumerate(zip(indices, m_values)):
            signal = np.fft.irfft(X_freq[i, :], n=n_timepoints)
            color = cmap(idx % 10)
            ax.plot(signal, color=color, label=f"m={m}", alpha=0.8, linewidth=0.8)

        ax.set_ylabel(f"l={l_deg}")
        ax.legend(loc="upper right", fontsize=7, ncol=min(len(m_values), 5))
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (samples)")
    fig.suptitle("Spherical Harmonic Amplitudes over Time", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_data_for_plot(data_path: Path, config: dict, short_pca_basis: Optional[np.ndarray] = None):
    """Load (Y, time) honoring the chromophore filter recorded in config.

    If config["_short_pca_basis"] (k, n_t) is set or short_pca_basis is passed,
    regress those components out of Y to mirror what was done at fit time.
    """
    if short_pca_basis is None:
        short_pca_basis = config.get("_short_pca_basis", None)
    from fnirs.io import load_hemodynamic_data, load_snirf_data, load_lob_data, ChromophoreType

    chrom_str = config.get("chromophore", "hbo").lower()
    suffix = Path(data_path).suffix
    if suffix in (".snirf", ".lob"):
        nirs_data = load_snirf_data(str(data_path)) if suffix == ".snirf" else load_lob_data(str(data_path))
        label_map = {"hbo": "HbO", "hbr": "HbR", "hbt": "HbT"}
        target_label = label_map.get(chrom_str)
        if target_label is None:
            raise ValueError(f"Unknown chromophore in config: {chrom_str!r}")
        selected = nirs_data.get_channels_by_data_type_label(target_label)
        if not selected:
            available = sorted({ch.measurement_info.data_type_label for ch in nirs_data.channels})
            raise ValueError(
                f"No channels with data_type_label={target_label!r} in {data_path}. "
                f"Available: {available}"
            )
        if not config.get("include_short_channels", False):
            selected = [ch for ch in selected if not ch.is_short_separation]
        ch_indices = np.array([ch.channel_idx for ch in selected])
        Y = nirs_data.time_series[:, ch_indices].T
        time = np.asarray(nirs_data.time)
    else:
        hemo_data = load_hemodynamic_data(str(data_path))
        chrom_map = {"hbo": ChromophoreType.HbO, "hbr": ChromophoreType.HbR, "hbt": ChromophoreType.HbT}
        chrom = chrom_map[chrom_str]
        Y = hemo_data.get_concentration_matrix(chrom).T
        time = np.asarray(hemo_data.time)

    bp = config.get("bandpass")
    if bp:
        from fnirs.cli import _parse_bandpass, _apply_bandpass
        low_hz, high_hz = _parse_bandpass(bp)
        dt_plot = float(time[1] - time[0])
        Y = _apply_bandpass(Y, dt_plot, low_hz, high_hz)

    if short_pca_basis is not None and short_pca_basis.shape[1] == Y.shape[1]:
        Y_c = Y - Y.mean(axis=1, keepdims=True)
        beta = short_pca_basis @ Y_c.T  # (k, n_long)
        Y = Y - beta.T @ short_pca_basis

    if config.get("mav_scale", False):
        mav = np.mean(np.abs(Y), axis=1, keepdims=True)
        scale = np.where(mav > 0, mav, 1.0)
        Y = Y / scale

    return Y, time


def plot_residuals(
    X_freq: np.ndarray,
    ST: np.ndarray,
    n_timepoints: int,
    data_path: Path,
    config: dict,
    output_path: Path,
):
    """Plot per-channel residual RMS."""
    Y, _ = _load_data_for_plot(data_path, config)

    # Predict
    pred_freq = ST @ X_freq
    Y_hat = np.fft.irfft(pred_freq, n=n_timepoints, axis=1)

    # Residuals
    residuals = Y[:, :n_timepoints] - Y_hat
    rms = np.sqrt(np.mean(residuals ** 2, axis=1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(rms)), rms, color="steelblue", alpha=0.8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Residual RMS")
    ax.set_title("Per-Channel Residual RMS")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_data_vs_fit(
    X_freq: np.ndarray,
    ST: np.ndarray,
    n_timepoints: int,
    data_path: Path,
    config: dict,
    output_path: Path,
    noise_variance: Optional[np.ndarray] = None,
    max_channels: Optional[int] = None,
):
    """Plot per-channel data signal vs model fit, with optional ±σ noise band."""
    Y, time = _load_data_for_plot(data_path, config)

    pred_freq = ST @ X_freq
    Y_hat = np.fft.irfft(pred_freq, n=n_timepoints, axis=1)

    Y = Y[:, :n_timepoints]
    time = time[:n_timepoints]

    n_channels = Y.shape[0]
    if max_channels is not None and n_channels > max_channels:
        channel_indices = np.linspace(0, n_channels - 1, max_channels, dtype=int)
    else:
        channel_indices = np.arange(n_channels)

    n_show = len(channel_indices)
    n_cols = int(np.ceil(np.sqrt(n_show)))
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 1.8 * n_rows),
        sharex=True, squeeze=False,
    )

    sigma = np.sqrt(noise_variance) if noise_variance is not None else None
    mse_per_ch = np.mean((Y - Y_hat) ** 2, axis=1)
    var_per_ch = np.var(Y, axis=1)

    for ax_idx, ch in enumerate(channel_indices):
        ax = axes[ax_idx // n_cols, ax_idx % n_cols]
        ax.plot(time, Y[ch], color="black", linewidth=0.8, alpha=0.9, label="data", zorder=1)
        if sigma is not None:
            ax.fill_between(
                time,
                Y_hat[ch] - sigma[ch],
                Y_hat[ch] + sigma[ch],
                color="C3", alpha=0.2, linewidth=0, label="±σ", zorder=2,
            )
        ax.plot(time, Y_hat[ch], color="C3", linewidth=1.0, linestyle="--", label="fit", zorder=3)
        r2 = 1.0 - mse_per_ch[ch] / var_per_ch[ch] if var_per_ch[ch] > 0 else float("nan")
        ax.set_title(f"ch {ch}: MSE={mse_per_ch[ch]:.2e}, R²={r2:.2f}", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for ax_idx in range(n_show, n_rows * n_cols):
        axes[ax_idx // n_cols, ax_idx % n_cols].axis("off")

    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.supxlabel("Time", fontsize=10)
    fig.supylabel("Signal", fontsize=10)
    fig.suptitle("Data vs Model Fit", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_short_channel_pca(
    Y_short: np.ndarray,
    V_top: np.ndarray,
    time: np.ndarray,
    output_path: Path,
):
    """Three-panel diagnostic for short-channel PCA regression:
    raw short-channel signals, the top-k eigenvector time courses, and the
    short-channel residuals after removing those k components.
    """
    n_short = Y_short.shape[0]
    k = V_top.shape[0]

    Y_short_c = Y_short - Y_short.mean(axis=1, keepdims=True)
    beta = V_top @ Y_short_c.T  # (k, n_short)
    Y_short_resid = Y_short - beta.T @ V_top

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for c in range(n_short):
        axes[0].plot(time, Y_short[c], linewidth=0.7, alpha=0.8, label=f"sh{c}")
    axes[0].set_ylabel("Signal")
    axes[0].set_title(f"Short channel data ({n_short} channels)")
    axes[0].grid(True, alpha=0.3)
    if n_short <= 20:
        axes[0].legend(loc="upper right", fontsize=7, ncol=min(n_short, 5))

    cmap = plt.cm.viridis
    for j in range(k):
        axes[1].plot(time, V_top[j], color=cmap(j / max(k - 1, 1)),
                     linewidth=1.0, label=f"PC{j+1}")
    axes[1].set_ylabel("Component")
    axes[1].set_title(f"Top {k} PCA eigenvectors of centered short data")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8, ncol=min(k, 6))

    for c in range(n_short):
        axes[2].plot(time, Y_short_resid[c], linewidth=0.7, alpha=0.8)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Residual")
    axes[2].set_title(f"Short channel residuals after removing top-{k} components")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(
    X_freq: np.ndarray,
    ST: np.ndarray,
    n_timepoints: int,
    data_path: Path,
    config: dict,
    output_path: Path,
):
    """Plot channel-channel correlation matrices for data, model fit, and residual."""
    Y, _ = _load_data_for_plot(data_path, config)
    Y = Y[:, :n_timepoints]

    Y_hat = np.fft.irfft(ST @ X_freq, n=n_timepoints, axis=1)
    R = Y - Y_hat

    def _corr(M):
        Mc = M - M.mean(axis=1, keepdims=True)
        std = Mc.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        Mn = Mc / std
        return (Mn @ Mn.T) / Mn.shape[1]

    C_data = _corr(Y)
    C_fit = _corr(Y_hat)
    C_res = _corr(R)
    n = Y.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, C, title in zip(axes, [C_data, C_fit, C_res], ["Data", "Fit", "Residual"]):
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(f"{title} (n={n})")
        ax.set_xlabel("channel")
    axes[0].set_ylabel("channel")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, label="corr")
    fig.suptitle("Channel-Channel Correlation", fontsize=13)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_noise_variance_history(
    noise_variance_history: np.ndarray,
    output_path: Path,
):
    """Plot per-channel σ across IRLS iterations.

    noise_variance_history has shape (n_iter+1, n_channels). Index 0 is the
    OLS-init; subsequent rows are after each IRLS iteration.
    """
    sigma = np.sqrt(np.maximum(noise_variance_history, 0))
    n_steps, n_channels = sigma.shape
    iters = np.arange(n_steps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax = axes[0]
    cmap = plt.cm.viridis
    for c in range(n_channels):
        ax.plot(iters, sigma[:, c], color=cmap(c / max(n_channels - 1, 1)),
                linewidth=0.6, alpha=0.7)
    ax.set_xlabel("IRLS iteration (0 = OLS init)")
    ax.set_ylabel("σ (per channel)")
    ax.set_title("Per-channel σ across IRLS iterations (linear)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(iters)

    # Log scale — easier to see runaway
    ax = axes[1]
    sigma_pos = np.where(sigma > 0, sigma, np.nan)
    for c in range(n_channels):
        ax.plot(iters, sigma_pos[:, c], color=cmap(c / max(n_channels - 1, 1)),
                linewidth=0.6, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("IRLS iteration (0 = OLS init)")
    ax.set_ylabel("σ (per channel, log)")
    ax.set_title("Per-channel σ across IRLS iterations (log)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xticks(iters)

    fig.suptitle(f"IRLS noise σ trajectory ({n_channels} channels, {n_steps - 1} iterations)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_noise_variance(
    noise_variance: np.ndarray,
    output_path: Path,
):
    """Plot per-channel estimated noise variance (IRLS)."""
    n = len(noise_variance)
    fig, ax = plt.subplots(figsize=(max(10, 0.18 * n), 4))
    ax.bar(range(n), noise_variance, color="coral", alpha=0.8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Noise Variance")
    ax.set_title("Per-Channel Estimated Noise Variance (IRLS)")
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], rotation=90, fontsize=6)
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_power_spectrum(
    X_freq: np.ndarray,
    ST: np.ndarray,
    n_timepoints: int,
    data_path: Path,
    config: dict,
    output_path: Path,
):
    """Plot average power spectrum of data vs model prediction."""
    Y, _ = _load_data_for_plot(data_path, config)

    # Data power spectrum
    Y_freq_data = np.fft.rfft(Y, axis=1)
    data_power = np.mean(np.abs(Y_freq_data) ** 2, axis=0)

    # Model power spectrum
    pred_freq = ST @ X_freq
    model_power = np.mean(np.abs(pred_freq) ** 2, axis=0)

    freqs = np.fft.rfftfreq(Y.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(freqs, data_power, label="Data", alpha=0.7, linewidth=0.8)
    ax.semilogy(freqs[:len(model_power)], model_power, label="Model", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Frequency (cycles/sample)")
    ax.set_ylabel("Power")
    ax.set_title("Average Power Spectrum: Data vs Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_snapshot(
    X_freq: np.ndarray,
    terms: List[Tuple[int, int]],
    n_timepoints: int,
    config: dict,
    output_path: Path,
    n_snapshots: int = 4,
):
    """Plot spatial field on sphere at selected time points."""
    from scipy.special import sph_harm_y

    # Reconstruct time-domain coefficients for each harmonic
    # X_time[i, t] = amplitude of harmonic i at time t
    X_time = np.fft.irfft(X_freq, n=n_timepoints, axis=1)

    # Create grid of (theta, phi) points
    n_grid = 30
    theta_grid = np.linspace(0.1, np.pi - 0.1, n_grid)
    phi_grid = np.linspace(0, 2 * np.pi, n_grid)
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid)
    theta_flat = theta_mesh.flatten()
    phi_flat = phi_mesh.flatten()

    # Build basis at grid points
    n_spatial = len(terms)
    ST_grid = np.zeros((len(theta_flat), n_spatial))
    for i, (l, m) in enumerate(terms):
        Y_val = sph_harm_y(l, m, phi_flat, theta_flat)
        if m == 0:
            ST_grid[:, i] = Y_val.real
        elif m > 0:
            ST_grid[:, i] = np.sqrt(2) * (-1) ** m * Y_val.real
        else:
            ST_grid[:, i] = np.sqrt(2) * (-1) ** m * Y_val.imag

    # Select time points
    time_indices = np.linspace(0, n_timepoints - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=(4 * n_snapshots, 3.5),
                             subplot_kw={"projection": "mollweide"})
    if n_snapshots == 1:
        axes = [axes]

    for ax, t_idx in zip(axes, time_indices):
        coeffs = X_time[:, t_idx]
        field = ST_grid @ coeffs
        field_2d = field.reshape(theta_mesh.shape)

        # Mollweide projection expects lon in [-pi, pi], lat in [-pi/2, pi/2]
        lon = phi_mesh - np.pi
        lat = np.pi / 2 - theta_mesh

        vmax = np.max(np.abs(field_2d))
        im = ax.pcolormesh(lon, lat, field_2d, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(f"t={t_idx}", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08)
    fig.suptitle("Spatial Field Snapshots", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
