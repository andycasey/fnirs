#!/usr/bin/env python3
"""Exploratory plots for raw fNIRS SNIRF data."""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from fnirs.io import NIRSData


_LABEL_COLORS = {
    "RAW": "0.4",
    "dOD": "tab:purple",
    "HbO": "tab:red",
    "HbR": "tab:blue",
    "HbT": "tab:green",
}


def _select_channels_by_label(nirs: NIRSData, label: str):
    sel = nirs.get_channels_by_data_type_label(label)
    if not sel:
        available = sorted({ch.measurement_info.data_type_label for ch in nirs.channels})
        raise ValueError(
            f"No channels with data_type_label={label!r}. Available: {available}"
        )
    indices = np.array([ch.channel_idx for ch in sel])
    return sel, indices


def plot_channel_type_breakdown(nirs: NIRSData, output_path: Path):
    counts = defaultdict(lambda: defaultdict(int))
    for ch in nirs.channels:
        info = ch.measurement_info
        counts[info.data_type_label][float(info.wavelength)] += 1

    labels = sorted(counts.keys())
    wavelengths = sorted({wl for d in counts.values() for wl in d.keys()})

    x = np.arange(len(labels))
    width = 0.8 / max(len(wavelengths), 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, wl in enumerate(wavelengths):
        vals = [counts[lab].get(wl, 0) for lab in labels]
        ax.bar(x + i * width, vals, width, label=f"{wl:g} nm")
    ax.set_xticks(x + width * (len(wavelengths) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Channel count")
    ax.set_title("Channel-type breakdown by wavelength")
    ax.legend(title="Wavelength")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries_grid(
    nirs: NIRSData,
    label: str,
    output_path: Path,
    max_channels: int = 64,
):
    _, indices = _select_channels_by_label(nirs, label)
    if len(indices) > max_channels:
        indices = np.linspace(0, len(indices) - 1, max_channels, dtype=int).tolist()
        indices = np.array(indices)
    Y = nirs.time_series[:, indices]
    time = np.asarray(nirs.time)
    n = len(indices)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    color = _LABEL_COLORS.get(label, "0.3")

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols, 1.4 * n_rows),
        sharex=True, squeeze=False,
    )
    for ax_idx, ch_i in enumerate(indices):
        ax = axes[ax_idx // n_cols, ax_idx % n_cols]
        ax.plot(time, nirs.time_series[:, ch_i], color=color, linewidth=0.6)
        ax.set_title(f"ch {ch_i}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
    for ax_idx in range(n, n_rows * n_cols):
        axes[ax_idx // n_cols, ax_idx % n_cols].axis("off")
    fig.suptitle(f"{label} time series ({n} channels)", fontsize=12)
    fig.supxlabel("Time (s)", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_channel_summary_stats(nirs: NIRSData, output_path: Path):
    Y = nirs.time_series  # (n_t, n_ch)
    means = Y.mean(axis=0)
    stds = Y.std(axis=0)
    ranges = Y.max(axis=0) - Y.min(axis=0)
    labels = np.array([ch.measurement_info.data_type_label for ch in nirs.channels])
    unique_labels = sorted(set(labels.tolist()))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for label in unique_labels:
        mask = labels == label
        idx = np.where(mask)[0]
        color = _LABEL_COLORS.get(label, None)
        axes[0].scatter(idx, means[mask], s=8, label=label, color=color, alpha=0.7)
        axes[1].scatter(idx, stds[mask], s=8, label=label, color=color, alpha=0.7)
        axes[2].scatter(idx, ranges[mask], s=8, label=label, color=color, alpha=0.7)
    for ax, name in zip(axes, ["mean", "std", "range"]):
        ax.set_yscale("symlog", linthresh=1e-4)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=len(unique_labels))
    axes[-1].set_xlabel("Channel index")
    fig.suptitle("Per-channel summary statistics", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_power_spectrum_by_type(nirs: NIRSData, output_path: Path):
    fs = float(nirs.sampling_frequency)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = sorted({ch.measurement_info.data_type_label for ch in nirs.channels})
    for label in labels:
        idx = np.array([ch.channel_idx for ch in nirs.get_channels_by_data_type_label(label)])
        if len(idx) == 0:
            continue
        Y = nirs.time_series[:, idx]
        # Per-channel normalized PSD to allow comparison across different unit scales
        Y_norm = (Y - Y.mean(axis=0, keepdims=True))
        std = Y_norm.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        Y_norm = Y_norm / std
        freqs = np.fft.rfftfreq(Y.shape[0], d=1.0 / fs)
        spec = np.mean(np.abs(np.fft.rfft(Y_norm, axis=0)) ** 2, axis=1)
        color = _LABEL_COLORS.get(label, None)
        ax.semilogy(freqs, spec, label=label, color=color, linewidth=0.9, alpha=0.9)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (normalized)")
    ax.set_title("Average power spectrum by channel type")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_probe_layout(nirs: NIRSData, output_path: Path):
    src = nirs.probe.source_positions_2d
    det = nirs.probe.detector_positions_2d
    mids = nirs.get_spatial_coordinates_2d()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(src[:, 0], src[:, 1], color="tab:red", s=60, marker="o",
               label=f"Sources (n={len(src)})", zorder=3)
    ax.scatter(det[:, 0], det[:, 1], color="tab:blue", s=60, marker="s",
               label=f"Detectors (n={len(det)})", zorder=3)
    ax.scatter(mids[:, 0], mids[:, 1], color="0.3", s=4, alpha=0.4,
               label=f"Channel midpoints (n={len(mids)})", zorder=2)

    # Draw lines source->detector for each unique pair
    seen_pairs = set()
    for ch in nirs.channels:
        pair = (ch.measurement_info.source_index, ch.measurement_info.detector_index)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        s = ch.source_pos_2d
        d = ch.detector_pos_2d
        ax.plot([s[0], d[0]], [s[1], d[1]], color="0.7", linewidth=0.4, alpha=0.5, zorder=1)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Source-detector probe layout (2D)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distance_histogram(nirs: NIRSData, output_path: Path):
    distances = np.array([ch.distance for ch in nirs.channels if ch.distance is not None])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(distances, bins=40, color="steelblue", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.axvline(15, color="tab:orange", linestyle="--", linewidth=1.0, label="15 mm (short-sep cutoff)")
    ax.set_xlabel("Source-detector distance")
    ax.set_ylabel("Count")
    ax.set_title("Source-detector distance distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stim_overlay(nirs: NIRSData, label: str, output_path: Path):
    if not nirs.stimulus:
        return False
    sel, indices = _select_channels_by_label(nirs, label)
    # Pick the channel with largest std as the "representative"
    Y_sel = nirs.time_series[:, indices]
    rep = int(np.argmax(Y_sel.std(axis=0)))
    rep_idx = int(indices[rep])
    time = np.asarray(nirs.time)
    color = _LABEL_COLORS.get(label, "0.3")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax_ts, ax_raster = axes
    ax_ts.plot(time, nirs.time_series[:, rep_idx], color=color, linewidth=0.8,
               label=f"{label} ch {rep_idx}")

    cmap = plt.cm.tab10
    for i, stim in enumerate(nirs.stimulus):
        c = cmap(i % 10)
        for j, onset in enumerate(stim.onsets):
            duration = stim.durations[j] if len(stim.durations) > j else 0.0
            ax_ts.axvspan(onset, onset + max(duration, 0.0), color=c, alpha=0.2,
                          label=stim.name if j == 0 else None)
            ax_raster.vlines(onset, i - 0.4, i + 0.4, color=c, linewidth=1.2)
        ax_raster.text(time[0], i, stim.name, fontsize=8, va="center", ha="right")

    ax_ts.set_ylabel("Signal")
    ax_ts.legend(loc="upper right", fontsize=8)
    ax_ts.grid(True, alpha=0.3)
    ax_raster.set_yticks(range(len(nirs.stimulus)))
    ax_raster.set_yticklabels([s.name for s in nirs.stimulus])
    ax_raster.set_xlabel("Time (s)")
    ax_raster.set_title("Stimulus events", fontsize=9)
    ax_raster.grid(True, alpha=0.3, axis="x")
    fig.suptitle(f"Stimulus timing over a representative {label} channel", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_correlation_heatmap(nirs: NIRSData, label: str, output_path: Path):
    _, indices = _select_channels_by_label(nirs, label)
    Y = nirs.time_series[:, indices]
    # zero-mean / unit-std along time
    Yc = Y - Y.mean(axis=0, keepdims=True)
    std = Yc.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Yn = Yc / std
    n = Yn.shape[1]
    if n < 2:
        return False
    C = (Yn.T @ Yn) / Yn.shape[0]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(f"{label} channel-channel correlation (n={n})")
    ax.set_xlabel(f"{label} channel idx (within label)")
    ax.set_ylabel(f"{label} channel idx (within label)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="corr")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def explore(data_path: Path, output_dir: Path, chromophore: str = "hbo"):
    """Generate a directory of exploratory plots for the SNIRF file."""
    from fnirs.io import load_snirf_data

    nirs = load_snirf_data(str(data_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = {"hbo": "HbO", "hbr": "HbR", "hbt": "HbT"}
    target = label_map.get(chromophore.lower())
    if target is None:
        raise ValueError(f"Unknown chromophore {chromophore!r}")

    generated = []

    p = output_dir / "channel_type_breakdown.png"
    plot_channel_type_breakdown(nirs, p)
    generated.append(p)

    p = output_dir / f"timeseries_grid_{target}.png"
    plot_timeseries_grid(nirs, target, p)
    generated.append(p)

    p = output_dir / "channel_summary_stats.png"
    plot_channel_summary_stats(nirs, p)
    generated.append(p)

    p = output_dir / "power_spectrum_by_type.png"
    plot_power_spectrum_by_type(nirs, p)
    generated.append(p)

    p = output_dir / "probe_layout.png"
    plot_probe_layout(nirs, p)
    generated.append(p)

    p = output_dir / "distance_histogram.png"
    plot_distance_histogram(nirs, p)
    generated.append(p)

    p = output_dir / f"stim_overlay_{target}.png"
    if plot_stim_overlay(nirs, target, p):
        generated.append(p)

    p = output_dir / f"correlation_{target}.png"
    if plot_correlation_heatmap(nirs, target, p):
        generated.append(p)

    return generated
