"""Interactive head-montage GUI for inspecting per-channel fits."""

import json
from pathlib import Path
from typing import List

import numpy as np


def _load_channel_positions(data_path: Path, config: dict):
    """Return list of (midpoint_2d, source_pos_2d, detector_pos_2d) in the same
    order as the rows of Y returned by `_load_data_for_plot`.
    """
    from fnirs.io import load_snirf_data, load_lob_data

    suffix = data_path.suffix
    if suffix not in (".snirf", ".lob"):
        raise ValueError(f"interact requires a .snirf or .lob source; got {suffix}")

    nirs_data = load_snirf_data(str(data_path)) if suffix == ".snirf" else load_lob_data(str(data_path))
    label_map = {"hbo": "HbO", "hbr": "HbR", "hbt": "HbT"}
    target_label = label_map.get(config.get("chromophore", "hbo").lower())
    selected = nirs_data.get_channels_by_data_type_label(target_label)
    if not config.get("include_short_channels", False):
        selected = [ch for ch in selected if not ch.is_short_separation]
    return selected


def run(model_dir: Path) -> None:
    """Launch the interactive montage GUI for a fitted model directory."""
    import matplotlib.pyplot as plt

    # Load model artifacts.
    model_data = np.load(model_dir / "model.npz")
    X_freq = model_data["X_freq_real"] + 1j * model_data["X_freq_imag"]
    ST = np.asarray(model_data["ST"])
    n_timepoints = int(model_data["n_timepoints"])
    short_pca_basis = model_data.get("short_pca_basis", None)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    data_path = Path(config["data"]) if config.get("data") else None
    if data_path is None or not data_path.exists():
        raise SystemExit(f"Error: data file from config not found: {data_path}")

    # Defer plotting import until after backend is set by the CLI shim.
    from fnirs.plotting import _load_data_for_plot

    if short_pca_basis is not None:
        config["_short_pca_basis"] = np.asarray(short_pca_basis)

    Y, time = _load_data_for_plot(data_path, config)
    Y = Y[:, :n_timepoints]
    time = np.asarray(time[:n_timepoints])

    Y_hat = np.fft.irfft(ST @ X_freq, n=n_timepoints, axis=1)

    selected_channels = _load_channel_positions(data_path, config)
    if len(selected_channels) != Y.shape[0]:
        raise SystemExit(
            f"Error: channel count mismatch ({len(selected_channels)} positions vs {Y.shape[0]} rows of Y)."
        )

    midpoints = np.array([ch.midpoint_2d for ch in selected_channels])
    source_pos = np.array([ch.source_pos_2d for ch in selected_channels])
    det_pos = np.array([ch.detector_pos_2d for ch in selected_channels])

    # Unique sources / detectors (positions repeat across channels).
    src_unique = np.unique(source_pos, axis=0)
    det_unique = np.unique(det_pos, axis=0)

    # Head outline: circle around the centroid covering all positions.
    all_pts = np.vstack([midpoints, src_unique, det_unique])
    center = all_pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(all_pts - center, axis=1))) * 1.15

    mse = np.mean((Y - Y_hat) ** 2, axis=1)
    var = np.var(Y, axis=1)

    fig, (ax_mont, ax_ts) = plt.subplots(1, 2, figsize=(13, 6.5),
                                         gridspec_kw={"width_ratios": [1, 1.3]})

    # --- Montage panel ---
    ax_mont.set_aspect("equal")
    ax_mont.set_title(f"Montage ({len(midpoints)} channels)")
    ax_mont.set_xticks([])
    ax_mont.set_yticks([])
    for spine in ax_mont.spines.values():
        spine.set_visible(False)

    # Head outline + nose tick.
    theta = np.linspace(0, 2 * np.pi, 256)
    ax_mont.plot(center[0] + radius * np.cos(theta),
                 center[1] + radius * np.sin(theta),
                 color="black", linewidth=1.2)
    nose_w = radius * 0.08
    nose_h = radius * 0.10
    ax_mont.plot(
        [center[0] - nose_w, center[0], center[0] + nose_w],
        [center[1] + radius, center[1] + radius + nose_h, center[1] + radius],
        color="black", linewidth=1.2,
    )

    # Source-detector channel lines.
    for s, d in zip(source_pos, det_pos):
        ax_mont.plot([s[0], d[0]], [s[1], d[1]], color="0.7", linewidth=0.8, zorder=1)

    ax_mont.scatter(src_unique[:, 0], src_unique[:, 1],
                    s=55, c="red", edgecolors="black", linewidths=0.4, zorder=3, label="source")
    ax_mont.scatter(det_unique[:, 0], det_unique[:, 1],
                    s=55, c="royalblue", edgecolors="black", linewidths=0.4, zorder=3, label="detector")

    # Channel midpoints.
    ch_scatter = ax_mont.scatter(
        midpoints[:, 0], midpoints[:, 1],
        s=70, facecolors="white", edgecolors="black", linewidths=0.8,
        zorder=4,
    )

    # Channel index labels.
    label_offset = radius * 0.018
    for i, (mx, my) in enumerate(midpoints):
        ax_mont.text(mx + label_offset, my + label_offset, str(i),
                     fontsize=6, color="black", zorder=6,
                     ha="left", va="bottom")

    # Highlight ring for the currently hovered channel.
    sel_marker = ax_mont.scatter(
        [midpoints[0, 0]], [midpoints[0, 1]],
        s=180, facecolors="none", edgecolors="orange", linewidths=2.2, zorder=5,
    )

    ax_mont.legend(loc="lower right", fontsize=8, frameon=False)
    pad = radius * 0.15
    ax_mont.set_xlim(center[0] - radius - pad, center[0] + radius + pad)
    ax_mont.set_ylim(center[1] - radius - pad, center[1] + radius + pad)

    # --- Time-series panel ---
    (line_data,) = ax_ts.plot(time, Y[0], color="black", linewidth=0.9, label="data")
    (line_fit,) = ax_ts.plot(time, Y_hat[0], color="red", linestyle="--", linewidth=1.0, label="fit")
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("Signal")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc="upper right", fontsize=9)

    def _update(ch_idx: int):
        line_data.set_ydata(Y[ch_idx])
        line_fit.set_ydata(Y_hat[ch_idx])
        ax_ts.relim()
        ax_ts.autoscale_view()
        r2 = 1.0 - mse[ch_idx] / var[ch_idx] if var[ch_idx] > 0 else float("nan")
        ax_ts.set_title(f"ch {ch_idx}: MSE={mse[ch_idx]:.2e}, R²={r2:.2f}")
        sel_marker.set_offsets(midpoints[ch_idx:ch_idx + 1])
        fig.canvas.draw_idle()

    # Hover detection radius: ~3% of the head radius (in data coords).
    hover_radius = radius * 0.05
    last_idx = [0]

    def _on_motion(event):
        if event.inaxes is not ax_mont or event.xdata is None:
            return
        dists = np.hypot(midpoints[:, 0] - event.xdata, midpoints[:, 1] - event.ydata)
        nearest = int(np.argmin(dists))
        if dists[nearest] > hover_radius:
            return
        if nearest == last_idx[0]:
            return
        last_idx[0] = nearest
        _update(nearest)

    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    _update(0)
    fig.suptitle(f"fnirs interact: {model_dir}", fontsize=11)
    plt.tight_layout()

    print("Hover over a channel to view its time series. Close the window to exit.")
    plt.show()
