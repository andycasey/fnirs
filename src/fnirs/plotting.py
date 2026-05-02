#!/usr/bin/env python3
"""Plotting functions for fnirs model visualization."""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
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


def plot_residuals(
    X_freq: np.ndarray,
    ST: np.ndarray,
    n_timepoints: int,
    data_path: Path,
    config: dict,
    output_path: Path,
):
    """Plot per-channel residual RMS."""
    from fnirs.io import load_hemodynamic_data, load_snirf_data, ChromophoreType

    # Load original data
    if str(data_path).endswith(".snirf"):
        nirs_data = load_snirf_data(str(data_path))
        Y = nirs_data.time_series.T
    else:
        hemo_data = load_hemodynamic_data(str(data_path))
        chrom_map = {"hbo": ChromophoreType.HbO, "hbr": ChromophoreType.HbR, "hbt": ChromophoreType.HbT}
        chrom = chrom_map[config.get("chromophore", "hbo").lower()]
        Y = hemo_data.get_concentration_matrix(chrom).T

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


def plot_noise_variance(
    noise_variance: np.ndarray,
    output_path: Path,
):
    """Plot per-channel estimated noise variance (IRLS)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(noise_variance)), noise_variance, color="coral", alpha=0.8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Noise Variance")
    ax.set_title("Per-Channel Estimated Noise Variance (IRLS)")
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
    from fnirs.io import load_hemodynamic_data, load_snirf_data, ChromophoreType

    if str(data_path).endswith(".snirf"):
        nirs_data = load_snirf_data(str(data_path))
        Y = nirs_data.time_series.T
    else:
        hemo_data = load_hemodynamic_data(str(data_path))
        chrom_map = {"hbo": ChromophoreType.HbO, "hbr": ChromophoreType.HbR, "hbt": ChromophoreType.HbT}
        chrom = chrom_map[config.get("chromophore", "hbo").lower()]
        Y = hemo_data.get_concentration_matrix(chrom).T

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
