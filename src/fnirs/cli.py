#!/usr/bin/env python3
"""Command-line interface for the fnirs Whittle GP model."""

import json
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="fNIRS spatio-temporal GP modelling tools.")


def _place_chunks_in_slots(T, n_chunks, chunk_size, rng):
    """Place n_chunks non-overlapping chunks of length chunk_size in [0, T) by
    splitting [0, T) into n_chunks equal slots and randomising chunk start within each."""
    slot_size = T // max(n_chunks, 1)
    out = []
    for c in range(n_chunks):
        slot_start = c * slot_size
        slot_end = min((c + 1) * slot_size, T)
        max_start = slot_end - chunk_size
        start = int(rng.integers(slot_start, max_start + 1)) if max_start > slot_start else slot_start
        out.append((start, start + chunk_size))
    return out


def _build_validation_mask(N, T, val_fraction, mode, chunk_size, rng):
    """Return (mask of shape (N,T), description string).

    mode = 'independent'  : each channel has its own random chunks (default).
    mode = 'synchronous'  : all channels share the same chunks.
    mode = 'disjoint'     : each channel masked in its own non-overlapping time slot.
    """
    import numpy as np

    if mode == "disjoint":
        if val_fraction > 1.0 / N - 1e-9:
            raise ValueError(
                f"validation-fraction={val_fraction:.4f} exceeds 1/N={1/N:.4f} (N={N}); "
                f"disjoint mode requires non-overlapping val regions across channels."
            )
        slot_size = T // N
        chunk_len = int(round(val_fraction * T))
        chunk_len = min(chunk_len, slot_size)
        mask = np.zeros((N, T), dtype=bool)
        slot_assignment = rng.permutation(N)
        for i in range(N):
            slot_idx = int(slot_assignment[i])
            slot_start = slot_idx * slot_size
            slot_end = min((slot_idx + 1) * slot_size, T)
            max_start = slot_end - chunk_len
            start = int(rng.integers(slot_start, max_start + 1)) if max_start > slot_start else slot_start
            mask[i, start:start + chunk_len] = True
        desc = f"disjoint mode: 1 chunk of {chunk_len} timepoints per channel ({100*chunk_len/T:.1f}%), no time overlap across channels"
        return mask, desc

    n_chunks = max(1, int(round(val_fraction * T / chunk_size)))
    if T // n_chunks < chunk_size:
        chunk_size = int(round(val_fraction * T))
        n_chunks = 1

    if mode == "synchronous":
        single = np.zeros(T, dtype=bool)
        for s, e in _place_chunks_in_slots(T, n_chunks, chunk_size, rng):
            single[s:e] = True
        mask = np.broadcast_to(single, (N, T)).copy()
        held = int(single.sum())
        desc = f"synchronous mode: {n_chunks} chunks of {chunk_size} timepoints, identical across all channels ({100*held/T:.1f}%)"
        return mask, desc

    # independent: each channel rolls its own.
    mask = np.zeros((N, T), dtype=bool)
    for i in range(N):
        for s, e in _place_chunks_in_slots(T, n_chunks, chunk_size, rng):
            mask[i, s:e] = True
    held = int(mask[0].sum())
    desc = f"independent mode: {n_chunks} chunks of {chunk_size} timepoints per channel ({100*held/T:.1f}%), independent random offsets per channel"
    return mask, desc


def _canonical_hrf_kernel(fs: float, length_s: float = 32.0,
                           peak: float = 6.0, undershoot: float = 16.0,
                           ratio: float = 6.0) -> "np.ndarray":
    """Discrete-time SPM-style double-gamma canonical HRF kernel.

    h(t) = Γ-pdf(t; α=peak) − Γ-pdf(t; α=undershoot) / ratio  (scale=1, in seconds).
    Truncated to `length_s` seconds and area-normalised.
    """
    import numpy as np
    from scipy.stats import gamma

    n = max(1, int(round(length_s * fs)))
    t = np.arange(n) / fs
    h = gamma.pdf(t, peak, scale=1.0) - gamma.pdf(t, undershoot, scale=1.0) / ratio
    s = h.sum()
    return h / s if s != 0 else h


def _build_glm_design_matrix(
    n_t: int,
    dt: float,
    stim_onsets,
    stim_durations,
    short_signals=None,
    *,
    regress_stim: bool = True,
    regress_short_channels: bool = True,
    regress_drift: bool = True,
    drift_order: int = 1,
    hrf_peak: float = 6.0,
    hrf_undershoot: float = 16.0,
    short_pca_components: int = 0,
):
    """Build the GLM design matrix `X` (n_t × n_reg) and a list of regressor names.

    Columns (in order, depending on flags):
        intercept | drift_p1 ... drift_pK | stim_HRF (one per stim group, if any)
        | short_<i> (one per short-channel input row, OR top-k PCs if
          short_pca_components > 0)

    `short_signals` is (n_short_inputs, n_t), already preprocessed (e.g. HbO+HbR
    of short-separation channels). When `short_pca_components > 0`, those rows
    are PCA-decomposed in time and the top-k orthonormal temporal components
    (right singular vectors) are used as regressors instead of the raw rows —
    this matches the Mesquita rsFC-fNIRS course's `PhysiologyRegression_GLM`
    convention and avoids design-matrix multicollinearity from highly-correlated
    short channels.
    """
    import numpy as np

    fs = 1.0 / dt
    cols = []
    names = []

    # Intercept (always present so the residual is mean-zero by construction).
    cols.append(np.ones(n_t))
    names.append("intercept")

    if regress_drift:
        t_norm = np.linspace(-1.0, 1.0, n_t)
        for p in range(1, max(1, int(drift_order)) + 1):
            cols.append(t_norm ** p)
            names.append(f"drift_p{p}")

    if regress_stim and len(stim_onsets) > 0:
        # Build a single boxcar covering all stim events, convolve with the HRF.
        boxcar = np.zeros(n_t)
        for onset, dur in zip(stim_onsets, stim_durations):
            i_start = max(0, int(round(float(onset) * fs)))
            i_end = min(n_t, int(round((float(onset) + float(dur)) * fs)))
            if i_end > i_start:
                boxcar[i_start:i_end] = 1.0
        kernel = _canonical_hrf_kernel(fs, length_s=32.0, peak=hrf_peak, undershoot=hrf_undershoot)
        stim_reg = np.convolve(boxcar, kernel, mode="full")[:n_t]
        cols.append(stim_reg)
        names.append("stim_HRF")

    if regress_short_channels and short_signals is not None:
        short_signals = np.asarray(short_signals, dtype=np.float64)
        if short_signals.shape[1] != n_t:
            raise ValueError(f"short_signals second axis must equal n_t={n_t}, got {short_signals.shape}")
        short_centered = short_signals - short_signals.mean(axis=1, keepdims=True)
        if short_pca_components and short_pca_components > 0:
            # Top-k right singular vectors (orthonormal in time) of the short-channel block.
            _, _, Vt = np.linalg.svd(short_centered, full_matrices=False)
            k = int(min(short_pca_components, Vt.shape[0]))
            for i in range(k):
                cols.append(Vt[i])
                names.append(f"short_pc{i}")
        else:
            for i in range(short_centered.shape[0]):
                cols.append(short_centered[i])
                names.append(f"short_{i}")

    X = np.stack(cols, axis=-1)
    return X, names


def _yule_walker_prewhiten(Y, max_order: int):
    """AR(p) pre-whitening per channel via Yule-Walker, with order chosen by BIC.

    For each row of `Y` (shape (N, T)):
        1. Fit AR(p) for p = 0..max_order (Yule-Walker; numpy linalg).
        2. Pick p* minimising BIC = T·log(rss/T) + p·log(T).
        3. Apply the AR(p*) whitening filter: w_t = y_t − Σ_{k=1..p*} a_k · y_{t-k}.
    Returns (Y_white, ar_orders) where Y_white has the same shape as Y (the
    first p* samples per channel are kept as-is — they aren't whitened) and
    ar_orders is shape (N,).

    Reference: Mesquita rsFC-fNIRS course `RemoveAutocorrelation_dc`. The
    cited justification is Santosa & Huppert / Barker on FDR control for
    fNIRS RSFC.
    """
    import numpy as np

    Y = np.asarray(Y, dtype=np.float64)
    N, T = Y.shape
    Y_white = Y.copy()
    orders = np.zeros(N, dtype=np.int64)
    log_T = np.log(T)
    for i in range(N):
        y = Y[i] - Y[i].mean()
        rss_per_p = np.empty(max_order + 1)
        coeffs_per_p: list = [None] * (max_order + 1)
        var_y = float(np.var(y))
        rss_per_p[0] = T * var_y if var_y > 0 else 1e-300
        coeffs_per_p[0] = np.zeros(0)
        if max_order >= 1:
            # Sample autocorrelations r_k for k = 0..max_order.
            r = np.array([np.dot(y[k:], y[:T - k]) / T for k in range(max_order + 1)])
            for p in range(1, max_order + 1):
                R = np.array([[r[abs(j - k)] for k in range(p)] for j in range(p)])
                try:
                    a = np.linalg.solve(R, r[1:p + 1])
                except np.linalg.LinAlgError:
                    a = np.zeros(p)
                coeffs_per_p[p] = a
                rss = T * (r[0] - np.dot(a, r[1:p + 1]))
                rss_per_p[p] = max(rss, 1e-300)
        bic = T * np.log(rss_per_p / T) + np.arange(max_order + 1) * log_T
        p_star = int(np.argmin(bic))
        orders[i] = p_star
        if p_star > 0:
            a = coeffs_per_p[p_star]
            yw = y.copy()
            # y_t = a_1 y_{t-1} + ... + a_p y_{t-p} + ε_t
            # Whitened: yw[t] = y[t] − a · [y[t-1], y[t-2], …, y[t-p]]
            for t in range(p_star, T):
                lags = y[t - p_star:t][::-1]
                yw[t] = y[t] - float(np.dot(a, lags))
            Y_white[i] = yw + Y[i].mean()
    return Y_white, orders


def _glm_regress(Y, X):
    """Per-row OLS regression of Y on X, subtract the fit. Returns (Y_clean, betas).

    Y: (n_channels, n_t).  X: (n_t, n_reg).  betas: (n_reg, n_channels).
    """
    import numpy as np

    XTX = X.T @ X
    XTY = X.T @ Y.T
    betas = np.linalg.solve(XTX + 1e-10 * np.eye(XTX.shape[0]), XTY)  # (n_reg, n_ch)
    Y_clean = Y - (X @ betas).T
    return Y_clean, betas


def _short_channel_pca_regress(Y_long, Y_short, n_components: int):
    """Regress top-k PCA temporal components of Y_short out of Y_long.

    Both inputs are (n_channels, n_t). Returns (Y_long_clean, V_top, betas, sv) where
    V_top is (k, n_t) of orthonormal time-domain components, betas is (k, n_long) of
    per-long-channel regression coefficients, sv is (k,) singular values of the
    short-channel time series. Per-channel mean is removed before SVD and before the
    least-squares projection.
    """
    import numpy as np

    Y_short_c = Y_short - Y_short.mean(axis=1, keepdims=True)
    _, S, Vt = np.linalg.svd(Y_short_c, full_matrices=False)
    k = min(int(n_components), Vt.shape[0])
    V_top = Vt[:k]
    sv = S[:k]
    Y_long_c = Y_long - Y_long.mean(axis=1, keepdims=True)
    betas = V_top @ Y_long_c.T  # (k, n_long); orthonormality of V_top makes this OLS.
    Y_long_clean = Y_long - betas.T @ V_top
    return Y_long_clean, V_top, betas, sv


def _global_pca_regress(Y, n_components: int):
    """Decompose the data as Y = W H + residual + noise, with H the top-k right
    singular vectors of Y (orthonormal in time) and W the OLS weights.

    Returns (Y_residual, H, W, sv) where H is (k, n_t), W is (n_channels, k),
    and sv is the singular values (k,). Per-channel mean is removed before SVD
    and the projection.
    """
    import numpy as np

    Y_c = Y - Y.mean(axis=1, keepdims=True)
    _, S, Vt = np.linalg.svd(Y_c, full_matrices=False)
    k = min(int(n_components), Vt.shape[0])
    H = Vt[:k]               # (k, n_t)
    sv = S[:k]
    W = Y_c @ H.T            # (n_channels, k); OLS by orthonormality of H.
    Y_residual = Y - W @ H
    return Y_residual, H, W, sv


def _snirf_optode_labels(snirf_path: str) -> tuple[list[str] | None, list[str] | None]:
    """Return (source_labels, detector_labels) from a SNIRF file's probe group, or (None, None)."""
    import h5py

    def _decode(arr):
        return [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in arr]

    try:
        with h5py.File(snirf_path, "r") as f:
            probe = f["nirs/probe"]
            src = _decode(probe["sourceLabels"][:]) if "sourceLabels" in probe else None
            det = _decode(probe["detectorLabels"][:]) if "detectorLabels" in probe else None
        return src, det
    except (OSError, KeyError):
        return None, None


@app.command()
def fit(
    data: Path = typer.Argument(..., help="Path to .mat or .snirf file"),
    output: Path = typer.Argument(..., help="Output directory for results"),
    chromophore: str = typer.Option("hbo", help="Chromophore: hbo, hbr, or hbt"),
    init_length_scale: float = typer.Option(30.0, help="Initial Matérn-3/2 length scale (samples)"),
    n_iter: int = typer.Option(10000, help="Max LBFGS iterations"),
    seed: int = typer.Option(0, help="Seed for parameter initialisation"),
    verbose: bool = typer.Option(True, help="Print per-iteration progress"),
    plots: bool = typer.Option(True, "--plots/--no-plots", help="Generate diagnostic plots after fitting"),
    include_short_channels: bool = typer.Option(
        False,
        "--include-short-channels",
        help="Include short-separation channels (default: excluded; they measure superficial physiology, not cortex).",
    ),
    regress_short_channels: bool = typer.Option(
        True,
        "--regress-short-channels/--no-regress-short-channels",
        help="GLM-regress the short-separation channel time series (HbO + HbR) out of the long channels before fitting (= GLMSSRC: superficial physiology removal).",
    ),
    short_channel_pca_components: int = typer.Option(
        0, "--short-channel-pca-components",
        help="Decompose the short-channel block via SVD and use only the top-K orthonormal temporal components as GLM regressors (Mesquita-course convention). 0 = use each short channel directly.",
    ),
    regress_stim: bool = typer.Option(
        True,
        "--regress-stim/--no-regress-stim",
        help="GLM-regress the HRF-convolved stim boxcar (using SNIRF stim events) out of the long channels — leaves the 'background connectivity' residual.",
    ),
    regress_drift: bool = typer.Option(
        True,
        "--regress-drift/--no-regress-drift",
        help="Include polynomial drift regressors of order --drift-order (default 1, i.e. linear).",
    ),
    drift_order: int = typer.Option(
        1, "--drift-order",
        help="Polynomial drift order: 1=linear, 2=quadratic, etc. Only used when --regress-drift is on.",
    ),
    hrf_peak: float = typer.Option(
        6.0, "--hrf-peak",
        help="Canonical HRF peak time (seconds). Default 6.",
    ),
    hrf_undershoot: float = typer.Option(
        16.0, "--hrf-undershoot",
        help="Canonical HRF undershoot time (seconds). Default 16.",
    ),
    post_glm_pca_components: int = typer.Option(
        0, "--post-glm-pca-components",
        help="After the explicit GLM, optionally run PCA on the GLM-residual long-channel data and subtract the top K modes (CompCor-style data-driven backup). 0 = off (default).",
    ),
    prewhiten: bool = typer.Option(
        False, "--prewhiten/--no-prewhiten",
        help="Yule-Walker AR(p) pre-whitening per channel before computing the empirical FC (Pearson) — gives correctly-sized p-values / FDR for the FC outputs (Mesquita-course convention). Order chosen by BIC up to --prewhiten-max-order. Affects only fc_pearson_* outputs; the GP-fitted Σ is unaffected.",
    ),
    prewhiten_max_order: int = typer.Option(
        20, "--prewhiten-max-order",
        help="Maximum AR order to consider for Yule-Walker pre-whitening (BIC-selected up to this).",
    ),
    log_sigma_min: Optional[float] = typer.Option(
        2.6,
        "--log-sigma-min",
        help="Lower bound on log σ (uniform prior). σ is the per-channel noise std. Default: 2.6.",
    ),
    log_sigma_max: Optional[float] = typer.Option(
        4.0,
        "--log-sigma-max",
        help="Upper bound on log σ (uniform prior). Default: 4.",
    ),
    min_length_scale: Optional[float] = typer.Option(
        None,
        "--min-length-scale",
        help="Lower bound on the Matérn-3/2 length scale (samples). Default: unbounded.",
    ),
    max_length_scale: Optional[float] = typer.Option(
        60.0,
        "--max-length-scale",
        help="Upper bound on the Matérn-3/2 length scale (samples). Default: 60.",
    ),
    rank: Optional[int] = typer.Option(
        4,
        "--rank",
        help="Rank r of the channel covariance Σ = L Lᵀ + diag(d), with L ∈ ℝ^{N×r}. Default: 4.",
    ),
    seed_channel_index: Optional[int] = typer.Option(
        6,
        "--seed-channel-index",
        help="Channel row to use as a seed for a quick mean-correlation summary printed at the end of the fit.",
    ),
    seed_k_neighbors: int = typer.Option(
        2,
        "--seed-k-neighbors",
        help="When --seed-channel-index is set, also print the max correlation among the K physically closest channels (using 3D channel midpoints).",
    ),
    validation_fraction: float = typer.Option(
        0.1,
        "--validation-fraction",
        help="Total fraction of timepoints to hold out per channel for validation. 0 = disabled.",
    ),
    validation_chunk_size: int = typer.Option(
        30,
        "--validation-chunk-size",
        help="Length (samples) of each held-out chunk; the val fraction is split across multiple non-overlapping chunks of this size per channel. Pick this to be O(length scale). Ignored in 'disjoint' mode (1 chunk per channel sized by --validation-fraction).",
    ),
    validation_mode: str = typer.Option(
        "independent",
        "--validation-mode",
        help="independent = per-channel random chunks (default; tests full model). synchronous = same chunks across all channels (tests temporal kernel only). disjoint = non-overlapping across channels, ≤1 channel masked at any time (matrix-completion CV; tests connectivity).",
    ),
    two_pass: bool = typer.Option(
        True,
        "--two-pass/--no-two-pass",
        help="After the first fit, set per-channel σ_i to the validation residual RMS (the irreducible noise estimate) and refit with σ_i FIXED. Identifies σ from data and pins Σ_ii = Var(Y_i) − σ_i² so the off-diagonal Σ is the only free part. Requires --validation-fraction > 0.",
    ),
    nu: Optional[float] = typer.Option(
        None,
        "--nu",
        help="Degrees of freedom for a Student-t Whittle likelihood (per-frequency multivariate t). Smaller = more robust to outlier frequencies. Typical values: 4–10. None = Gaussian (default).",
    ),
):
    """Fit the Whittle spatio-temporal GP to fNIRS data."""
    import numpy as np

    from fnirs.io import load_hemodynamic_data, load_snirf_data, ChromophoreType
    from fnirs.whittle import fit as whittle_fit

    data_path = str(data)
    if data.suffix == ".snirf":
        nirs_data = load_snirf_data(data_path)
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
        short_channels_in_label = [ch for ch in selected if ch.is_short_separation]
        n_short = len(short_channels_in_label)
        if not include_short_channels:
            selected = [ch for ch in selected if not ch.is_short_separation]
            if not selected:
                raise typer.BadParameter(
                    f"All {target_label} channels are short-separation; pass --include-short-channels to fit them."
                )
        ch_indices = np.array([ch.channel_idx for ch in selected])
        print(ch_indices)
        src_labels, det_labels = _snirf_optode_labels(data_path)

        def _src_name(i: int) -> str:
            return src_labels[i - 1] if src_labels is not None and 1 <= i <= len(src_labels) else f"S{i}"

        def _det_name(i: int) -> str:
            return det_labels[i - 1] if det_labels is not None and 1 <= i <= len(det_labels) else f"D{i}"

        ch_labels = np.array(
            [f"{_src_name(ch.measurement_info.source_index)}-{_det_name(ch.measurement_info.detector_index)}"
             for ch in selected]
        )
        n_total = len(nirs_data.channels)
        short_msg = (
            f", excluded {n_short} short" if (n_short and not include_short_channels)
            else (f", including {n_short} short" if include_short_channels and n_short else "")
        )
        typer.echo(f"Selected {len(ch_indices)} {target_label} channels (of {n_total} total{short_msg})")
        Y = nirs_data.time_series[:, ch_indices].T
        positions_3d_all = nirs_data.get_spatial_coordinates_3d()
        if positions_3d_all is None:
            positions_2d_all = nirs_data.get_spatial_coordinates_2d()
            positions_3d_all = np.column_stack([positions_2d_all, np.zeros(len(positions_2d_all))])
        positions_3d = positions_3d_all[ch_indices]
        t = np.asarray(nirs_data.time)

        if nirs_data.stimulus:
            stim_onsets = np.concatenate([np.asarray(s.onsets, dtype=np.float64) for s in nirs_data.stimulus])
            stim_durations = np.concatenate([np.asarray(s.durations, dtype=np.float64) for s in nirs_data.stimulus])
        else:
            stim_onsets = np.zeros(0, dtype=np.float64)
            stim_durations = np.zeros(0, dtype=np.float64)

        # Short-channel time series (same chromophore as Y), for GLM regression.
        if short_channels_in_label and not include_short_channels:
            _short_idx = np.array([ch.channel_idx for ch in short_channels_in_label])
            short_signals_for_glm = nirs_data.time_series[:, _short_idx].T
        else:
            short_signals_for_glm = None
    else:
        hemo_data = load_hemodynamic_data(data_path)
        chrom_map = {"hbo": ChromophoreType.HbO, "hbr": ChromophoreType.HbR, "hbt": ChromophoreType.HbT}
        chrom = chrom_map[chromophore.lower()]
        Y_full = hemo_data.get_concentration_matrix(chrom).T  # (n_channels, n_timepoints)
        positions_3d_all = hemo_data.get_spatial_coordinates_3d()
        n_total = Y_full.shape[0]
        is_short = np.array([bool(ch.is_short_separation) for ch in hemo_data.channels])
        n_short = int(is_short.sum())
        if include_short_channels:
            ch_indices = np.arange(n_total)
        else:
            ch_indices = np.flatnonzero(~is_short)
            if ch_indices.size == 0:
                raise typer.BadParameter(
                    "All channels are short-separation; pass --include-short-channels to fit them."
                )
        Y = Y_full[ch_indices]
        positions_3d = positions_3d_all[ch_indices] if positions_3d_all is not None else None
        ch_labels = np.array([f"S{hemo_data.channels[i].source_idx}-D{hemo_data.channels[i].detector_idx}" for i in ch_indices])

        if not include_short_channels and is_short.any():
            short_signals_for_glm = Y_full[np.flatnonzero(is_short)]
        else:
            short_signals_for_glm = None
        short_msg = (
            f", excluded {n_short} short" if (n_short and not include_short_channels)
            else (f", including {n_short} short" if include_short_channels and n_short else "")
        )
        typer.echo(f"Selected {len(ch_indices)} {chromophore.upper()} channels (of {n_total} total{short_msg})")
        t = np.asarray(hemo_data.time)
        stim_onsets = np.zeros(0, dtype=np.float64)
        stim_durations = np.zeros(0, dtype=np.float64)

    Y = np.asarray(Y, dtype=np.float64)
    dt = float(t[1] - t[0]) if len(t) >= 2 else 1.0

    # GLM regression of nuisance signals (drift, HRF×stim, short-channel HbO/HbR).
    glm_X = None
    glm_betas = None
    glm_names: list = []
    if regress_drift or regress_stim or regress_short_channels:
        Xmat, glm_names = _build_glm_design_matrix(
            Y.shape[1], dt, stim_onsets, stim_durations,
            short_signals=short_signals_for_glm,
            regress_stim=regress_stim,
            regress_short_channels=regress_short_channels,
            regress_drift=regress_drift,
            drift_order=drift_order,
            hrf_peak=hrf_peak,
            hrf_undershoot=hrf_undershoot,
            short_pca_components=short_channel_pca_components,
        )
        # Always include the intercept (1 col); skip GLM only if that's the only column.
        if Xmat.shape[1] > 1:
            Y, glm_betas = _glm_regress(Y, Xmat)
            glm_X = Xmat
            n_short_used = sum(1 for n in glm_names if n.startswith("short_"))
            n_drift_used = sum(1 for n in glm_names if n.startswith("drift_"))
            stim_present = any(n == "stim_HRF" for n in glm_names)
            typer.echo(
                f"GLM regression: {len(glm_names)} regressors "
                f"(drift_order={n_drift_used}, stim_HRF={stim_present}, n_short={n_short_used})."
            )

    # Optional CompCor-style data-driven PCA on the GLM-residual (off by default).
    glm_residual_pca_basis = None
    glm_residual_pca_weights = None
    glm_residual_pca_sv = None
    if post_glm_pca_components > 0:
        Y, glm_residual_pca_basis, glm_residual_pca_weights, glm_residual_pca_sv = _global_pca_regress(
            Y, post_glm_pca_components
        )
        _, S_all, _ = np.linalg.svd(Y - Y.mean(axis=1, keepdims=True), full_matrices=False)
        total_var = float(np.sum(glm_residual_pca_sv ** 2) + np.sum(S_all ** 2))
        ve = float(np.sum(glm_residual_pca_sv ** 2) / total_var) if total_var > 0 else 0.0
        typer.echo(
            f"Post-GLM PCA: removed top-{glm_residual_pca_basis.shape[0]} residual common modes "
            f"({100 * ve:.1f}% of pre-removal variance)."
        )

    Y_original = Y.copy()  # stays untouched; npz "Y" field saves this (post-regression).
    val_mask = None
    if validation_fraction > 0:
        if not 0 < validation_fraction < 1:
            raise typer.BadParameter("validation-fraction must be in (0, 1)")
        rng_val = np.random.default_rng(seed)
        try:
            val_mask, desc = _build_validation_mask(
                Y.shape[0], Y.shape[1], validation_fraction,
                validation_mode, int(validation_chunk_size), rng_val,
            )
        except ValueError as e:
            raise typer.BadParameter(str(e))
        for i in range(Y.shape[0]):
            ch_mean = float(Y[i, ~val_mask[i]].mean())
            Y[i, val_mask[i]] = ch_mean
        typer.echo(f"Validation: {desc}.")

    log_ell_min = float(np.log(min_length_scale)) if min_length_scale is not None else None
    log_ell_max = float(np.log(max_length_scale)) if max_length_scale is not None else None

    if two_pass and val_mask is None:
        raise typer.BadParameter("--two-pass requires --validation-fraction > 0")

    res = whittle_fit(
        Y,
        rank=rank,
        init_length_scale=init_length_scale,
        n_iter=n_iter,
        verbose=verbose,
        seed=seed,
        log_sigma_min=log_sigma_min,
        log_sigma_max=log_sigma_max,
        log_ell_min=log_ell_min,
        log_ell_max=log_ell_max,
        nu=nu,
    )

    res_pass1 = None
    if two_pass:
        # Per-channel out-of-sample residual std (= empirical noise estimate).
        z1 = res["posterior_mean"]
        N_ch = Y.shape[0]
        sigma_emp = np.zeros(N_ch)
        for i in range(N_ch):
            sigma_emp[i] = float(np.sqrt(np.mean((Y_original[i, val_mask[i]] - z1[i, val_mask[i]]) ** 2)))
        # Clip so σ_i² < Var(Y_i) (otherwise Σ_ii ≤ 0).
        ch_var = Y_original.var(axis=1)
        sigma_emp = np.minimum(sigma_emp, 0.99 * np.sqrt(ch_var))
        sigma_emp = np.maximum(sigma_emp, 1e-6)
        fixed_log_sigma2 = 2.0 * np.log(sigma_emp)
        typer.echo(
            f"Two-pass: per-channel σ from val residuals — median={float(np.median(sigma_emp)):.3f}, "
            f"range=[{float(sigma_emp.min()):.3f}, {float(sigma_emp.max()):.3f}]. Refitting with σ_i fixed."
        )
        res_pass1 = res
        res = whittle_fit(
            Y,
            rank=rank,
            init_length_scale=res_pass1["length_scale"],
            n_iter=n_iter,
            verbose=verbose,
            seed=seed,
            log_ell_min=log_ell_min,
            log_ell_max=log_ell_max,
            fixed_log_sigma2=fixed_log_sigma2,
            nu=nu,
        )

    output.mkdir(parents=True, exist_ok=True)

    save_dict = dict(
        sigma=res["sigma"],
        correlation=res["correlation"],
        noise_var=res["noise_var"],
        length_scale=np.float64(res["length_scale"]),
        losses=res["losses"],
        posterior_mean=res["posterior_mean"],
        L=res["L"],
        d=res["d"],
        rank=np.int64(res["rank"]),
        Y=Y,
        positions_3d=np.asarray(positions_3d, dtype=np.float64) if positions_3d is not None else np.zeros((Y.shape[0], 3)),
        kept_channel_indices=np.asarray(ch_indices, dtype=np.int64),
        channel_labels=np.asarray(ch_labels),
        dt=np.float64(dt),
        n_timepoints=np.int64(Y.shape[1]),
        stim_onsets=stim_onsets,
        stim_durations=stim_durations,
    )
    if glm_X is not None:
        save_dict["glm_X"] = np.asarray(glm_X, dtype=np.float64)
        save_dict["glm_betas"] = np.asarray(glm_betas, dtype=np.float64)
        save_dict["glm_regressor_names"] = np.asarray(glm_names)
    if glm_residual_pca_basis is not None:
        save_dict["glm_residual_pca_basis"] = np.asarray(glm_residual_pca_basis, dtype=np.float64)
        save_dict["glm_residual_pca_weights"] = np.asarray(glm_residual_pca_weights, dtype=np.float64)
        save_dict["glm_residual_pca_singular_values"] = np.asarray(glm_residual_pca_sv, dtype=np.float64)

    # Functional connectivity:
    #   fc_pearson_data  : Pearson r on (post-GLM-regression) channel data Y —
    #                      what fNIRS papers compute and report.
    #   fc_pearson_denoised: Pearson r on E[z|Y] — the GP's denoised latent;
    #                      the principled "model FC" for our pipeline.
    # If --prewhiten, AR(p)-prewhiten each channel before correlating so the
    # resulting r-values have correctly-sized null distributions.
    Y_for_fc = Y_original
    Z_for_fc = res["posterior_mean"]
    ar_orders = None
    if prewhiten:
        Y_for_fc, ar_orders = _yule_walker_prewhiten(Y_original, max_order=int(prewhiten_max_order))
        Z_for_fc, _ = _yule_walker_prewhiten(np.asarray(res["posterior_mean"]), max_order=int(prewhiten_max_order))
    fc_pearson_data = np.corrcoef(Y_for_fc)
    fc_pearson_data = np.where(np.isfinite(fc_pearson_data), fc_pearson_data, 0.0)
    fc_pearson_denoised = np.corrcoef(Z_for_fc)
    fc_pearson_denoised = np.where(np.isfinite(fc_pearson_denoised), fc_pearson_denoised, 0.0)
    save_dict["fc_pearson_data"] = fc_pearson_data
    save_dict["fc_pearson_denoised"] = fc_pearson_denoised
    save_dict["fc_fisher_z_denoised"] = np.arctanh(np.clip(fc_pearson_denoised, -0.999999, 0.999999))
    if ar_orders is not None:
        save_dict["prewhiten_ar_orders"] = ar_orders

    if res_pass1 is not None:
        save_dict["sigma_pass1"] = res_pass1["sigma"]
        save_dict["correlation_pass1"] = res_pass1["correlation"]
        save_dict["noise_var_pass1"] = res_pass1["noise_var"]
        save_dict["length_scale_pass1"] = np.float64(res_pass1["length_scale"])

    config = dict(
        data=str(data),
        chromophore=chromophore,
        init_length_scale=init_length_scale,
        n_iter=n_iter,
        seed=seed,
        include_short_channels=include_short_channels,
        regress_short_channels=regress_short_channels,
        short_channel_pca_components=short_channel_pca_components,
        regress_stim=regress_stim,
        regress_drift=regress_drift,
        drift_order=drift_order,
        hrf_peak=hrf_peak,
        hrf_undershoot=hrf_undershoot,
        post_glm_pca_components=post_glm_pca_components,
        prewhiten=prewhiten,
        prewhiten_max_order=prewhiten_max_order,
        log_sigma_min=log_sigma_min,
        log_sigma_max=log_sigma_max,
        min_length_scale=min_length_scale,
        max_length_scale=max_length_scale,
        rank=rank,
        seed_channel_index=seed_channel_index,
        validation_fraction=validation_fraction,
        validation_chunk_size=validation_chunk_size,
        validation_mode=validation_mode,
        two_pass=two_pass,
        nu=nu,
    )
    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    n_channels, n_timepoints = Y.shape

    # Reduced χ² diagnostics: residual / σ, per channel and overall.
    residuals = Y - res["posterior_mean"]
    sigma_i = np.sqrt(res["noise_var"])
    chi2_per_ch = np.sum((residuals / sigma_i[:, None]) ** 2, axis=1)
    chi2_red_per_ch = chi2_per_ch / float(n_timepoints)
    # Σ = LLᵀ + diag(d): L has N·r entries but is identifiable up to an r×r
    # orthogonal rotation, so subtract r(r-1)/2 from the L count. Plus N for d,
    # N for log_sigma2, 1 for log_ell.
    r_rank = int(res["rank"])
    n_free_params = (n_channels * r_rank - r_rank * (r_rank - 1) // 2) + 2 * n_channels + 1
    dof_total = max(n_channels * n_timepoints - n_free_params, 1)
    chi2_red_total = float(np.sum(chi2_per_ch) / dof_total)
    save_dict["chi2_red_per_channel"] = chi2_red_per_ch
    save_dict["chi2_red_total"] = np.float64(chi2_red_total)

    val_metrics = None
    if val_mask is not None:
        # Posterior mean was computed on the imputed Y; held-out residual is Y_original − ẑ
        # restricted to the per-channel masked timepoints.
        z_full = res["posterior_mean"]
        chi2_val_per_ch = np.zeros(n_channels)
        for i in range(n_channels):
            r_i = Y_original[i, val_mask[i]] - z_full[i, val_mask[i]]
            chi2_val_per_ch[i] = np.sum(r_i ** 2 / res["noise_var"][i]) / val_mask[i].sum()
        n_val_total = int(val_mask.sum())
        chi2_val_total = float(
            np.sum([
                np.sum((Y_original[i, val_mask[i]] - z_full[i, val_mask[i]]) ** 2 / res["noise_var"][i])
                for i in range(n_channels)
            ]) / n_val_total
        )
        val_metrics = {
            "chi2_red_per_channel": chi2_val_per_ch,
            "chi2_red_total": chi2_val_total,
            "n_val_timepoints": n_val_total,
        }
        save_dict["Y"] = Y_original
        save_dict["val_mask"] = val_mask
        save_dict["chi2_red_val_per_channel"] = chi2_val_per_ch
        save_dict["chi2_red_val_total"] = np.float64(chi2_val_total)

    np.savez(output / "model.npz", **save_dict)

    # Also save the model-predicted correlation matrix as a standalone .npy
    # file alongside model.npz so other tools can load it without parsing npz.
    np.save(output / "correlation_model.npy", np.asarray(res["correlation"]))
    np.save(output / "correlation_pearson_denoised.npy", fc_pearson_denoised)

    typer.echo("Fit complete.")
    typer.echo(f"  Channels:        {n_channels}")
    typer.echo(f"  Timepoints:      {n_timepoints}")
    typer.echo(f"  dt:              {dt:.4f} s")
    typer.echo(f"  Length scale:    {res['length_scale']:.2f} samples ({res['length_scale'] * dt:.2f} s)")
    typer.echo(f"  LBFGS iters:     {res['n_iter']}  (converged={res['converged']})")
    if not res['converged']:
        typer.echo(f"  scipy exit:      status={res.get('scipy_status', '?')}  fevals={res.get('scipy_nfev', '?')}")
        typer.echo(f"                   {res.get('scipy_message', '?')}")
    typer.echo(f"  Noise std range: {float(np.sqrt(res['noise_var']).min()):.4f} .. {float(np.sqrt(res['noise_var']).max()):.4f}")
    if len(res['losses']) > 0:
        typer.echo(f"  Final -loglik:   {float(res['losses'][-1]):.3f}")
    else:
        typer.echo(f"  Final -loglik:   <no callback fired — fit may have terminated immediately>")
    typer.echo(f"  Rank:            {res['rank']} (of {n_channels})")
    typer.echo(
        f"  Reduced χ² (train):  total={chi2_red_total:.4f}    "
        f"per-channel median={float(np.median(chi2_red_per_ch)):.4f}    "
        f"range=[{float(chi2_red_per_ch.min()):.4f}, {float(chi2_red_per_ch.max()):.4f}]"
    )
    if val_metrics is not None:
        per_ch_val = val_metrics["chi2_red_per_channel"]
        typer.echo(
            f"  Reduced χ² (val):    total={val_metrics['chi2_red_total']:.4f}    "
            f"per-channel median={float(np.median(per_ch_val)):.4f}    "
            f"range=[{float(per_ch_val.min()):.4f}, {float(per_ch_val.max()):.4f}]"
        )
        typer.echo(f"                       on {val_metrics['n_val_timepoints']} held-out (channel × timepoint) cells")

    # Functional connectivity on the GP's denoised latent E[z|Y] — model output.
    iu = np.triu_indices(n_channels, k=1)
    fc_d = fc_pearson_denoised[iu]
    fc_y = fc_pearson_data[iu]
    typer.echo(
        f"  FC (Pearson r on E[z|Y], denoised): mean |r|={float(np.mean(np.abs(fc_d))):.4f}    "
        f"median |r|={float(np.median(np.abs(fc_d))):.4f}    "
        f"max |r|={float(np.max(np.abs(fc_d))):.4f}    "
        f"signed mean r={float(np.mean(fc_d)):+.4f}"
    )
    typer.echo(
        f"  FC (Pearson r on raw Y, comparison): mean |r|={float(np.mean(np.abs(fc_y))):.4f}    "
        f"median |r|={float(np.median(np.abs(fc_y))):.4f}    "
        f"max |r|={float(np.max(np.abs(fc_y))):.4f}"
    )

    if seed_channel_index is not None:
        if 0 <= seed_channel_index < n_channels:
            seed_label = str(ch_labels[seed_channel_index])
            row_data = np.corrcoef(Y)[seed_channel_index]
            row_model = res['correlation'][seed_channel_index]
            row_resid = np.corrcoef(Y - res['posterior_mean'])[seed_channel_index]

            mask = np.arange(n_channels) != seed_channel_index
            seed_off = res['correlation'][seed_channel_index, mask]
            mean_r = float(seed_off.mean())
            mean_abs_r = float(np.abs(seed_off).mean())
            max_r = float(seed_off.max())
            min_r = float(seed_off.min())
            typer.echo(f"  Seed channel:    row {seed_channel_index} ({seed_label})")
            typer.echo(f"    mean r over off-diag (FC score): {mean_r:+.4f}    mean |r|: {mean_abs_r:.4f}")
            typer.echo(f"    max r: {max_r:+.4f}    min r: {min_r:+.4f}")

            # K physically closest channels (3D midpoints).
            seed_pos = positions_3d[seed_channel_index]
            distances = np.linalg.norm(positions_3d - seed_pos[None, :], axis=1)
            distances[seed_channel_index] = np.inf  # exclude self
            k = max(1, min(int(seed_k_neighbors), n_channels - 1))
            nearest = np.argsort(distances)[:k]
            nearest_labels = [str(ch_labels[i]) for i in nearest]
            nearest_dist = distances[nearest]
            data_max = float(np.max(np.abs(row_data[nearest])))
            model_max = float(np.max(np.abs(row_model[nearest])))
            resid_max = float(np.max(np.abs(row_resid[nearest])))
            typer.echo(
                f"    K={k} closest channels (dist mm: {', '.join(f'{d:.1f}' for d in nearest_dist)}): "
                f"{', '.join(nearest_labels)}"
            )
            typer.echo(
                f"    max |corr| over K closest:  data={data_max:.4f}    model={model_max:.4f}    resid={resid_max:.4f}"
            )

        else:
            typer.echo(f"  Seed channel index {seed_channel_index} out of range (0..{n_channels - 1}); skipping.")

    typer.echo(f"  Saved to:        {output}")

    if plots:
        _generate_plots(output, output / "figures")


def _generate_plots(model_dir: Path, fig_dir: Path) -> None:
    import numpy as np

    from fnirs.plotting import (
        plot_connectivity,
        plot_correlation,
        plot_noise_std,
        plot_loss_curve,
        plot_channel_traces,
        plot_latent_draws,
        plot_residuals,
        plot_power_spectrum,
    )

    model_data = np.load(model_dir / "model.npz")
    sigma = model_data["sigma"]
    correlation = model_data["correlation"]
    noise_var = model_data["noise_var"]
    losses = model_data["losses"]
    posterior_mean = model_data["posterior_mean"]
    Y = model_data["Y"]
    dt = float(model_data["dt"])
    if "channel_labels" in model_data.files:
        channel_labels = model_data["channel_labels"]
    elif "kept_channel_indices" in model_data.files:
        channel_labels = model_data["kept_channel_indices"]
    else:
        channel_labels = None
    stim_onsets = model_data["stim_onsets"] if "stim_onsets" in model_data.files else None
    stim_durations = model_data["stim_durations"] if "stim_durations" in model_data.files else None
    chi2_red_per_channel = model_data["chi2_red_per_channel"] if "chi2_red_per_channel" in model_data.files else None
    val_mask = model_data["val_mask"] if "val_mask" in model_data.files else None

    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_connectivity(sigma, fig_dir / "connectivity.png")
    typer.echo("Saved connectivity.png")

    plot_correlation(correlation, fig_dir / "correlation.png")
    typer.echo("Saved correlation.png")

    plot_noise_std(noise_var, fig_dir / "noise_std.png", channel_labels=channel_labels)
    typer.echo("Saved noise_std.png")

    plot_loss_curve(losses, fig_dir / "loss_curve.png")
    typer.echo("Saved loss_curve.png")

    plot_channel_traces(
        Y, posterior_mean, noise_var, dt, fig_dir / "channel_traces.png",
        channel_labels=channel_labels,
        stim_onsets=stim_onsets, stim_durations=stim_durations,
        sigma_matrix=sigma, length_scale=float(model_data["length_scale"]),
        chi2_red_per_channel=chi2_red_per_channel,
        val_mask=val_mask,
    )
    typer.echo("Saved channel_traces.png")

    plot_latent_draws(
        Y, posterior_mean, sigma, noise_var, float(model_data["length_scale"]), dt,
        fig_dir / "latent_draws.png",
        channel_labels=channel_labels,
        stim_onsets=stim_onsets, stim_durations=stim_durations,
        chi2_red_per_channel=chi2_red_per_channel,
        val_mask=val_mask,
    )
    typer.echo("Saved latent_draws.png")

    plot_residuals(
        Y, posterior_mean, fig_dir / "residuals.png",
        channel_labels=channel_labels,
        val_mask=val_mask,
        noise_std=np.sqrt(noise_var),
    )
    typer.echo("Saved residuals.png")

    plot_power_spectrum(Y, posterior_mean, dt, fig_dir / "power_spectrum.png")
    typer.echo("Saved power_spectrum.png")

    typer.echo(f"All figures saved to: {fig_dir}")


@app.command()
def plot(
    model_dir: Path = typer.Argument(..., help="Path to output from fnirs fit"),
    output: Optional[Path] = typer.Option(None, help="Directory for figures (default: model-dir/figures/)"),
):
    """Visualise a fitted Whittle GP model."""
    fig_dir = output if output else model_dir / "figures"
    _generate_plots(model_dir, fig_dir)


@app.command()
def preprocess(
    data: Path = typer.Argument(..., help="Path to a raw .snirf, .lob, or .nirs file"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output .snirf path. Default for .snirf input: overwrite the input. Required for .lob and .nirs input.",
    ),
    tddr: bool = typer.Option(True, "--tddr/--no-tddr", help="Apply TDDR motion correction (Fishburn 2019)."),
    bandpass: bool = typer.Option(True, "--bandpass/--no-bandpass", help="Apply Butterworth bandpass filter."),
    wavelet: bool = typer.Option(True, "--wavelet/--no-wavelet", help="Apply wavelet MAD-thresholding spike removal."),
    hampel: bool = typer.Option(True, "--hampel/--no-hampel", help="Apply Hampel time-domain outlier filter (sample-wise spike removal)."),
    bandpass_low_hz: float = typer.Option(0.009, "--bandpass-low-hz", help="High-pass cutoff (Hz) of the bandpass."),
    bandpass_high_hz: float = typer.Option(0.08, "--bandpass-high-hz", help="Low-pass cutoff (Hz) of the bandpass."),
    wavelet_iqr: float = typer.Option(1.5, "--wavelet-iqr", help="MAD-σ multiplier for wavelet spike thresholding (effective k = 2 × this)."),
    hampel_window: int = typer.Option(7, "--hampel-window", help="Hampel filter half-window size (samples)."),
    hampel_k: float = typer.Option(4.0, "--hampel-k", help="Hampel rejection threshold in robust σ units (k=4 ≈ 99.99%-conservative)."),
    ppf_w1: float = typer.Option(6.0, "--ppf-w1", help="Partial pathlength factor at wavelength 1."),
    ppf_w2: float = typer.Option(6.0, "--ppf-w2", help="Partial pathlength factor at wavelength 2."),
    edge_trim_samples: int = typer.Option(
        0, "--edge-trim-samples",
        help="Drop this many samples from each end after bandpass/MBLL — Homer-style filtfilt-edge cleanup. 0 = no trim.",
    ),
    input_label: Optional[str] = typer.Option(
        None, "--input-label",
        help="Filter input to channels with this data_type_label. If omitted, auto-detect: pick the label whose channels span all wavelength indices.",
    ),
):
    """Preprocess a raw fNIRS file (.snirf or .lob): TDDR + bandpass + wavelet
    despike on intensity, then convert to optical density and HbO/HbR
    concentrations via Modified Beer-Lambert. Writes the result as SNIRF.
    """
    import numpy as np

    from fnirs.io import load_snirf_data, load_lob_data, load_nirs_data, save_concentration_snirf
    from fnirs.preprocess import (
        preprocess_optical_density, intensity_to_od, od_to_concentration,
    )

    suffix = data.suffix.lower()
    if suffix == ".snirf":
        if output is None:
            output = data
        nirs_data = load_snirf_data(str(data))
    elif suffix == ".lob":
        if output is None:
            raise typer.BadParameter(
                "--output is required for .lob input (we don't write back to .lob format)."
            )
        nirs_data = load_lob_data(str(data))
    elif suffix == ".nirs":
        if output is None:
            raise typer.BadParameter(
                "--output is required for .nirs input (we don't write back to .nirs format)."
            )
        nirs_data = load_nirs_data(str(data))
    else:
        raise typer.BadParameter(f"expected .snirf, .lob, or .nirs input; got {data.suffix}")

    n_wavelengths = len(np.asarray(nirs_data.probe.wavelengths))
    if input_label is None:
        # Auto-detect: pick the label whose channels span every wavelength index.
        from collections import defaultdict
        by_label = defaultdict(list)
        for ch in nirs_data.channels:
            by_label[ch.measurement_info.data_type_label].append(ch)
        expected = set(range(1, n_wavelengths + 1))
        candidates = []
        for label, chs in by_label.items():
            wav_set = {ch.measurement_info.wavelength_index for ch in chs}
            if expected.issubset(wav_set):
                candidates.append((label, len(chs)))
        if not candidates:
            avail = sorted(by_label.keys())
            raise typer.BadParameter(
                f"Auto-detect failed: no data_type_label has channels at all wavelength indices "
                f"{sorted(expected)}. Available labels: {avail}. Pass --input-label explicitly."
            )
        # Most channels wins; tie-break on label string for determinism.
        candidates.sort(key=lambda x: (-x[1], x[0]))
        input_label = candidates[0][0]
        typer.echo(f"Auto-detected input label: {input_label!r} (from {sorted(by_label.keys())}).")

    raw_channels = [ch for ch in nirs_data.channels if ch.measurement_info.data_type_label == input_label]
    if not raw_channels:
        avail = sorted({ch.measurement_info.data_type_label for ch in nirs_data.channels})
        raise typer.BadParameter(
            f"No channels with data_type_label={input_label!r}. Available: {avail}"
        )
    raw_idx = np.array([ch.channel_idx for ch in raw_channels])
    typer.echo(f"Filtering to {len(raw_idx)} {input_label!r} channels (out of {len(nirs_data.channels)} total).")

    intensity_t_x_c = nirs_data.time_series[:, raw_idx]   # (n_t, n_raw_ch)
    n_t, n_ch = intensity_t_x_c.shape
    time = np.asarray(nirs_data.time)
    if len(time) >= 2:
        fs = float(1.0 / np.median(np.diff(time)))
    else:
        fs = 1.0
    typer.echo(f"Loaded {data.name}: {n_ch} channels, {n_t} timepoints, fs ≈ {fs:.3f} Hz.")

    # Per-channel preprocessing in OD space (standard Homer / MNE-NIRS ordering).
    intensity = intensity_t_x_c.T.astype(np.float64)
    od_raw = intensity_to_od(intensity)
    typer.echo(
        f"  TDDR={tddr}  hampel={hampel} (w={hampel_window} k={hampel_k})  "
        f"wavelet={wavelet} (k_robσ={2*wavelet_iqr:.1f})  "
        f"bandpass={bandpass} [{bandpass_low_hz}, {bandpass_high_hz}] Hz"
    )
    od = preprocess_optical_density(
        od_raw, fs,
        apply_tddr=tddr,
        apply_bandpass=bandpass,
        apply_wavelet=wavelet,
        apply_hampel=hampel,
        bandpass_low_hz=bandpass_low_hz,
        bandpass_high_hz=bandpass_high_hz,
        wavelet_iqr_threshold=wavelet_iqr,
        hampel_window=hampel_window,
        hampel_k=hampel_k,
    )

    # Modified Beer-Lambert → HbO, HbR per source-detector pair (filtered set).
    src_idx = np.array([ch.measurement_info.source_index for ch in raw_channels])
    det_idx = np.array([ch.measurement_info.detector_index for ch in raw_channels])
    wav_idx = np.array([ch.measurement_info.wavelength_index for ch in raw_channels])
    distances = np.array([ch.distance for ch in raw_channels])
    spatial_unit = (nirs_data.metadata.get("LengthUnit", "mm") if nirs_data.metadata else "mm")
    if isinstance(spatial_unit, (bytes, np.bytes_)):
        spatial_unit = spatial_unit.decode("utf-8")
    spatial_unit = str(spatial_unit).lower().strip()

    hbo, hbr, pair_keys, pair_distance = od_to_concentration(
        od, src_idx, det_idx, wav_idx,
        wavelengths=np.asarray(nirs_data.probe.wavelengths),
        distances=distances,
        ppf=(ppf_w1, ppf_w2),
        spatial_unit=spatial_unit if spatial_unit in ("mm", "cm") else "mm",
    )
    # Standard fNIRS reporting unit is micromolar (μM = 10⁻⁶ M).
    hbo *= 1e6
    hbr *= 1e6
    typer.echo(f"  Computed HbO/HbR for {len(pair_keys)} source-detector pairs (units: μM).")

    # Pack output: HbO channels, then HbR channels (each pair contributes 2).
    n_pairs = len(pair_keys)
    out_time_series = np.zeros((n_t, 2 * n_pairs), dtype=np.float64)
    measurement_list = []
    # SNIRF spec: for processed concentration data (dataType=2), wavelengthIndex
    # is not meaningful per-wavelength; conventionally set to 0.
    for p_idx, (s, d) in enumerate(pair_keys):
        out_time_series[:, p_idx] = hbo[p_idx]
        measurement_list.append({
            "sourceIndex": s, "detectorIndex": d, "wavelengthIndex": 0,
            "dataType": 2, "dataTypeIndex": 1, "dataTypeLabel": "HbO",
            "dataUnit": "uM",
        })
    for p_idx, (s, d) in enumerate(pair_keys):
        out_time_series[:, n_pairs + p_idx] = hbr[p_idx]
        measurement_list.append({
            "sourceIndex": s, "detectorIndex": d, "wavelengthIndex": 0,
            "dataType": 2, "dataTypeIndex": 2, "dataTypeLabel": "HbR",
            "dataUnit": "uM",
        })

    if edge_trim_samples and edge_trim_samples > 0:
        et = int(edge_trim_samples)
        if 2 * et >= n_t:
            raise typer.BadParameter(f"edge-trim-samples={et} too large for n_t={n_t}")
        out_time_series = out_time_series[et:n_t - et]
        # Trim the template's time vector to match.
        nirs_data.time = np.asarray(nirs_data.time)[et:n_t - et]
        # Re-base stim event onsets to the new time origin and drop any that
        # fall outside the trimmed window.
        new_t0 = float(np.asarray(nirs_data.time)[0])
        new_t_end = float(np.asarray(nirs_data.time)[-1])
        if nirs_data.stimulus:
            kept = []
            for s in nirs_data.stimulus:
                data = np.asarray(s.data, dtype=np.float64).copy()
                if data.size == 0:
                    continue
                onsets = data[:, 0]
                durations = data[:, 1] if data.shape[1] > 1 else np.zeros(len(data))
                ends = onsets + durations
                in_window = (ends >= new_t0) & (onsets <= new_t_end)
                if not in_window.any():
                    continue
                data = data[in_window]
                data[:, 0] = data[:, 0] - new_t0  # re-zero the time axis
                from fnirs.io import StimInfo
                kept.append(StimInfo(name=s.name, data=data))
            nirs_data.stimulus = kept if kept else None
        # Re-zero the time axis so it starts at 0 (matching the trimmed series).
        nirs_data.time = nirs_data.time - new_t0
        n_t = out_time_series.shape[0]
        typer.echo(f"  Edge-trim: dropped {et} samples from each end → {n_t} timepoints retained.")

    save_concentration_snirf(
        output, template=nirs_data, time_series=out_time_series,
        measurement_list=measurement_list,
        metadata_extra={
            "preprocessing": (
                f"fnirs preprocess: tddr={tddr} bandpass={bandpass}({bandpass_low_hz},{bandpass_high_hz}) "
                f"wavelet={wavelet}(IQR={wavelet_iqr}) ppf={ppf_w1},{ppf_w2} "
                f"edge_trim={edge_trim_samples}"
            ),
        },
    )
    typer.echo(f"Saved {output}: {2 * n_pairs} channels (HbO + HbR), {n_t} timepoints.")


@app.command()
def montage(
    data: Path = typer.Argument(..., help="Path to a .snirf or .lob file with HbO/HbR channels"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output PNG path. Default: <input_stem>_montage_<chromophore>_<metric>.png alongside the input.",
    ),
    chromophore: str = typer.Option("hbo", "--chromophore", help="hbo, hbr, or hbt — which channels to summarise."),
    metric: str = typer.Option(
        "std", "--metric",
        help="Per-channel scalar to plot: std, var, rms, max, peak-to-peak.",
    ),
    include_short_channels: bool = typer.Option(
        False, "--include-short-channels",
        help="Include short-separation channels (default: excluded).",
    ),
    log_scale: bool = typer.Option(False, "--log-scale", help="Plot log10(metric)."),
    cmap: str = typer.Option("viridis", "--cmap", help="Matplotlib colormap name."),
):
    """Topographic montage: per-channel signal-summary scalar drawn as colour at
    each channel's midpoint, overlaid on a head outline with sources (red) and
    detectors (blue) labelled. Useful for visualising per-channel signal
    properties to compare against external plots.
    """
    import numpy as np

    from fnirs.io import load_snirf_data, load_lob_data
    from fnirs.plotting import plot_montage_metric

    suffix = data.suffix.lower()
    if suffix == ".snirf":
        nd = load_snirf_data(str(data))
    elif suffix == ".lob":
        nd = load_lob_data(str(data))
    else:
        raise typer.BadParameter(f"expected .snirf or .lob input; got {data.suffix}")

    label_map = {"hbo": "HbO", "hbr": "HbR", "hbt": "HbT"}
    target_label = label_map.get(chromophore.lower())
    if target_label is None:
        raise typer.BadParameter(f"chromophore must be hbo/hbr/hbt; got {chromophore!r}")

    selected = [ch for ch in nd.channels if ch.measurement_info.data_type_label == target_label]
    if not selected:
        avail = sorted({ch.measurement_info.data_type_label for ch in nd.channels})
        raise typer.BadParameter(
            f"No channels labelled {target_label!r}. Available: {avail}"
        )
    if not include_short_channels:
        selected = [ch for ch in selected if not ch.is_short_separation]
        if not selected:
            raise typer.BadParameter(
                f"All {target_label} channels are short-separation; pass --include-short-channels."
            )

    ch_idx = np.array([ch.channel_idx for ch in selected])
    Y = nd.time_series[:, ch_idx].T.astype(np.float64)  # (n_ch, n_t)

    metric_funcs = {
        "std": lambda Y: np.std(Y, axis=1),
        "var": lambda Y: np.var(Y, axis=1),
        "rms": lambda Y: np.sqrt(np.mean(Y ** 2, axis=1)),
        "max": lambda Y: np.max(np.abs(Y), axis=1),
        "peak-to-peak": lambda Y: np.ptp(Y, axis=1),
    }
    if metric not in metric_funcs:
        raise typer.BadParameter(f"metric must be one of {sorted(metric_funcs)}; got {metric!r}")
    values = metric_funcs[metric](Y)

    midpoints = np.array([ch.midpoint_2d for ch in selected])
    source_pos = np.array([ch.source_pos_2d for ch in selected])
    det_pos = np.array([ch.detector_pos_2d for ch in selected])
    src_indices = np.array([ch.measurement_info.source_index for ch in selected])
    det_indices = np.array([ch.measurement_info.detector_index for ch in selected])
    channel_labels = np.array([
        f"S{ch.measurement_info.source_index}-D{ch.measurement_info.detector_index}"
        for ch in selected
    ])

    if output is None:
        output = data.with_name(f"{data.stem}_montage_{chromophore.lower()}_{metric}.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    plot_montage_metric(
        midpoints, source_pos, det_pos, src_indices, det_indices, values,
        channel_labels, output,
        metric_name=f"{target_label} {metric}",
        title=f"{data.name} — {target_label} {metric}  ({len(selected)} channels)",
        cmap=cmap, log_scale=log_scale,
    )
    typer.echo(f"Saved {output}")
    typer.echo(f"  metric={metric!r}  chromophore={target_label}  channels={len(selected)}  "
               f"value range: [{float(values.min()):.3e}, {float(values.max()):.3e}]")


@app.command()
def interact(
    model_dir: Path = typer.Argument(..., help="Path to output from fnirs fit"),
):
    """Hover over a channel on the montage to see its raw trace, posterior mean, and ±σ band."""
    from fnirs.interactive import run as run_interactive

    run_interactive(model_dir)


if __name__ == "__main__":
    app()
