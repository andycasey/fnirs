"""Interactive head-montage GUI for inspecting per-channel fits."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_channel_geometry(data_path: Path, kept_indices: np.ndarray):
    """Return (midpoints, source_pos, det_pos, src_idx, det_idx), each (n_kept,...) in row order of Y."""
    suffix = data_path.suffix
    if suffix == ".snirf":
        from fnirs.io import load_snirf_data

        nirs_data = load_snirf_data(str(data_path))
        channels = [nirs_data.channels[int(i)] for i in kept_indices]
        midpoints = np.array([ch.midpoint_2d for ch in channels])
        source_pos = np.array([ch.source_pos_2d for ch in channels])
        det_pos = np.array([ch.detector_pos_2d for ch in channels])
        src_idx = np.array([ch.measurement_info.source_index for ch in channels], dtype=int)
        det_idx = np.array([ch.measurement_info.detector_index for ch in channels], dtype=int)
        return midpoints, source_pos, det_pos, src_idx, det_idx

    if suffix in (".mat",):
        from fnirs.io import load_hemodynamic_data

        hemo_data = load_hemodynamic_data(str(data_path))
        channels = [hemo_data.channels[int(i)] for i in kept_indices]
        midpoints = np.array([np.asarray(ch.midpoint)[:2] for ch in channels])
        source_pos = np.array([np.asarray(ch.source_pos)[:2] for ch in channels])
        det_pos = np.array([np.asarray(ch.detector_pos)[:2] for ch in channels])
        src_idx = np.array([ch.source_idx for ch in channels], dtype=int)
        det_idx = np.array([ch.detector_idx for ch in channels], dtype=int)
        return midpoints, source_pos, det_pos, src_idx, det_idx

    raise ValueError(f"interact: unsupported data suffix {suffix!r} (expected .snirf or .mat)")


def run(model_dir: Path) -> None:
    """Launch the interactive montage GUI for a fitted model directory."""
    import matplotlib.pyplot as plt

    model_data = np.load(model_dir / "model.npz")
    Y = np.asarray(model_data["Y"])
    posterior_mean = np.asarray(model_data["posterior_mean"])
    noise_var = np.asarray(model_data["noise_var"])
    dt = float(model_data["dt"])
    n_timepoints = int(model_data["n_timepoints"])
    kept_indices = (
        np.asarray(model_data["kept_channel_indices"], dtype=int)
        if "kept_channel_indices" in model_data.files
        else np.arange(Y.shape[0], dtype=int)
    )
    channel_labels = (
        np.asarray(model_data["channel_labels"]).astype(str)
        if "channel_labels" in model_data.files
        else np.array([str(int(c)) for c in kept_indices])
    )

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    data_path = Path(config["data"]) if config.get("data") else None
    if data_path is None or not data_path.exists():
        raise SystemExit(f"Error: data file from config not found: {data_path}")

    midpoints, source_pos, det_pos, src_idx_per_ch, det_idx_per_ch = _load_channel_geometry(data_path, kept_indices)
    if midpoints.shape[0] != Y.shape[0]:
        raise SystemExit(
            f"Error: channel count mismatch ({midpoints.shape[0]} positions vs {Y.shape[0]} rows of Y)."
        )

    time = np.arange(n_timepoints) * dt
    sigma = np.sqrt(noise_var)
    residuals = Y - posterior_mean
    mse = np.mean(residuals**2, axis=1)
    var = np.var(Y, axis=1)

    stim_onsets = (
        np.asarray(model_data["stim_onsets"], dtype=float)
        if "stim_onsets" in model_data.files else np.zeros(0)
    )
    stim_durations = (
        np.asarray(model_data["stim_durations"], dtype=float)
        if "stim_durations" in model_data.files else np.zeros(0)
    )
    if stim_onsets.size != stim_durations.size:
        stim_onsets = np.zeros(0)
        stim_durations = np.zeros(0)

    src_map: dict[int, np.ndarray] = {}
    for idx, pos in zip(src_idx_per_ch, source_pos):
        src_map.setdefault(int(idx), pos)
    det_map: dict[int, np.ndarray] = {}
    for idx, pos in zip(det_idx_per_ch, det_pos):
        det_map.setdefault(int(idx), pos)
    src_idx_unique = np.array(sorted(src_map))
    det_idx_unique = np.array(sorted(det_map))
    src_unique = np.array([src_map[i] for i in src_idx_unique])
    det_unique = np.array([det_map[i] for i in det_idx_unique])
    all_pts = np.vstack([midpoints, src_unique, det_unique])
    center = all_pts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(all_pts - center, axis=1))) * 1.15

    fig, (ax_mont, ax_ts) = plt.subplots(
        1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1, 1.3]}
    )

    # --- Montage panel ---
    ax_mont.set_aspect("equal")
    ax_mont.set_title(f"Montage ({len(midpoints)} channels)")
    ax_mont.set_xticks([])
    ax_mont.set_yticks([])
    for spine in ax_mont.spines.values():
        spine.set_visible(False)

    theta = np.linspace(0, 2 * np.pi, 256)
    ax_mont.plot(
        center[0] + radius * np.cos(theta),
        center[1] + radius * np.sin(theta),
        color="black",
        linewidth=1.2,
    )
    nose_w = radius * 0.08
    nose_h = radius * 0.10
    ax_mont.plot(
        [center[0] - nose_w, center[0], center[0] + nose_w],
        [center[1] + radius, center[1] + radius + nose_h, center[1] + radius],
        color="black",
        linewidth=1.2,
    )

    for s, d in zip(source_pos, det_pos):
        ax_mont.plot([s[0], d[0]], [s[1], d[1]], color="0.7", linewidth=0.8, zorder=1)

    ax_mont.scatter(
        src_unique[:, 0], src_unique[:, 1],
        s=140, c="red", edgecolors="black", linewidths=0.4, zorder=3, label="source",
    )
    for i, (sx, sy) in zip(src_idx_unique, src_unique):
        ax_mont.text(
            sx, sy, str(int(i)),
            fontsize=6, color="white", fontweight="bold",
            ha="center", va="center", zorder=3.5,
        )
    ax_mont.scatter(
        det_unique[:, 0], det_unique[:, 1],
        s=140, c="royalblue", edgecolors="black", linewidths=0.4, zorder=3, label="detector",
    )
    for i, (dx, dy) in zip(det_idx_unique, det_unique):
        ax_mont.text(
            dx, dy, str(int(i)),
            fontsize=6, color="white", fontweight="bold",
            ha="center", va="center", zorder=3.5,
        )
    ax_mont.scatter(
        midpoints[:, 0], midpoints[:, 1],
        s=70, facecolors="white", edgecolors="black", linewidths=0.8, zorder=4,
    )

    label_offset = radius * 0.018
    for i, (mx, my) in enumerate(midpoints):
        ax_mont.text(
            mx + label_offset, my + label_offset, str(channel_labels[i]),
            fontsize=6, color="black", zorder=6, ha="left", va="bottom",
        )

    sel_marker = ax_mont.scatter(
        [midpoints[0, 0]], [midpoints[0, 1]],
        s=180, facecolors="none", edgecolors="orange", linewidths=2.2, zorder=5,
    )

    ax_mont.legend(loc="lower right", fontsize=8, frameon=False)
    pad = radius * 0.15
    ax_mont.set_xlim(center[0] - radius - pad, center[0] + radius + pad)
    ax_mont.set_ylim(center[1] - radius - pad, center[1] + radius + pad)

    # --- Time-series panel ---
    for o, d in zip(stim_onsets, stim_durations):
        ax_ts.axvspan(o, o + d, color="0.6", alpha=0.25, linewidth=0, zorder=0)
    band = ax_ts.fill_between(
        time, posterior_mean[0] - sigma[0], posterior_mean[0] + sigma[0],
        color="crimson", alpha=0.18, linewidth=0,
    )
    (line_data,) = ax_ts.plot(time, Y[0], color="steelblue", linewidth=0.7, alpha=0.7, label="Y")
    (line_fit,) = ax_ts.plot(time, posterior_mean[0], color="crimson", linewidth=1.0, label=r"$E[z\mid Y]$")
    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("Signal")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(loc="upper right", fontsize=9)

    state = {"band": band, "last_idx": 0}

    def _update(ch_idx: int):
        line_data.set_ydata(Y[ch_idx])
        line_fit.set_ydata(posterior_mean[ch_idx])
        state["band"].remove()
        state["band"] = ax_ts.fill_between(
            time,
            posterior_mean[ch_idx] - sigma[ch_idx],
            posterior_mean[ch_idx] + sigma[ch_idx],
            color="crimson", alpha=0.18, linewidth=0,
        )
        ax_ts.relim()
        ax_ts.autoscale_view()
        r2 = 1.0 - mse[ch_idx] / var[ch_idx] if var[ch_idx] > 0 else float("nan")
        ax_ts.set_title(
            f"{channel_labels[ch_idx]}  (row {ch_idx}):  "
            f"MSE={mse[ch_idx]:.2e}  R²={r2:.2f}  σ={sigma[ch_idx]:.2e}"
        )
        sel_marker.set_offsets(midpoints[ch_idx:ch_idx + 1])
        fig.canvas.draw_idle()

    hover_radius = radius * 0.05

    def _on_motion(event):
        if event.inaxes is not ax_mont or event.xdata is None:
            return
        dists = np.hypot(midpoints[:, 0] - event.xdata, midpoints[:, 1] - event.ydata)
        nearest = int(np.argmin(dists))
        if dists[nearest] > hover_radius or nearest == state["last_idx"]:
            return
        state["last_idx"] = nearest
        _update(nearest)

    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    _update(0)
    fig.suptitle(f"fnirs interact: {model_dir}", fontsize=11)
    plt.tight_layout()

    print("Hover over a channel to view its time series. Close the window to exit.")
    plt.show()
