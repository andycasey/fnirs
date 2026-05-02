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
    n_iter: int = typer.Option(100, help="Max LBFGS iterations"),
    seed: int = typer.Option(0, help="Seed for parameter initialisation"),
    verbose: bool = typer.Option(True, help="Print per-iteration progress"),
    plots: bool = typer.Option(True, "--plots/--no-plots", help="Generate diagnostic plots after fitting"),
    include_short_channels: bool = typer.Option(
        False,
        "--include-short-channels",
        help="Include short-separation channels (default: excluded; they measure superficial physiology, not cortex).",
    ),
    short_pca_components: int = typer.Option(
        0,
        "--short-pca-components",
        help="Run PCA on the short-separation channels (top-k temporal components) and regress them out of the long channels before fitting. 0 = disabled.",
    ),
    global_pca_components: int = typer.Option(
        0,
        "--global-pca-components",
        help="After short-channel regression, decompose the long-channel data as Y = W H + GP + noise with H = top-k right singular vectors of Y, fit W in closed form, and subtract W H before the GP. 0 = disabled.",
    ),
    log_sigma_min: Optional[float] = typer.Option(
        -4.0,
        "--log-sigma-min",
        help="Lower bound on log σ (uniform prior). σ is the per-channel noise std. Default: -4.",
    ),
    log_sigma_max: Optional[float] = typer.Option(
        1.0,
        "--log-sigma-max",
        help="Upper bound on log σ (uniform prior). Default: 1.",
    ),
    min_length_scale: Optional[float] = typer.Option(
        None,
        "--min-length-scale",
        help="Lower bound on the Matérn-3/2 length scale (samples). Default: unbounded.",
    ),
    max_length_scale: Optional[float] = typer.Option(
        None,
        "--max-length-scale",
        help="Upper bound on the Matérn-3/2 length scale (samples). Default: unbounded.",
    ),
    rank: Optional[int] = typer.Option(
        None,
        "--rank",
        help="Rank r of the channel covariance Σ = L Lᵀ + diag(d), with L ∈ ℝ^{N×r}. Default: full rank N.",
    ),
    seed_channel_index: Optional[int] = typer.Option(
        None,
        "--seed-channel-index",
        help="Channel row to use as a seed for a quick mean-correlation summary printed at the end of the fit.",
    ),
    validation_fraction: float = typer.Option(
        0.0,
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

        short_pca_basis = None
        short_pca_betas = None
        short_pca_sv = None
        if short_pca_components > 0:
            if include_short_channels:
                typer.echo("Warning: --short-pca-components ignored when --include-short-channels is set.")
            elif not short_channels_in_label:
                typer.echo(f"Warning: --short-pca-components={short_pca_components} requested but no short {target_label} channels available.")
            else:
                short_idx = np.array([ch.channel_idx for ch in short_channels_in_label])
                Y_short = nirs_data.time_series[:, short_idx].T
                Y, short_pca_basis, short_pca_betas, short_pca_sv = _short_channel_pca_regress(
                    Y, Y_short, short_pca_components
                )
                _, S_all, _ = np.linalg.svd(Y_short - Y_short.mean(axis=1, keepdims=True), full_matrices=False)
                var_explained = float(np.sum(short_pca_sv ** 2) / np.sum(S_all ** 2)) if S_all.size else 0.0
                typer.echo(
                    f"Regressed top-{short_pca_basis.shape[0]} PCA components of {len(short_idx)} short channels "
                    f"out of {Y.shape[0]} long channels (var explained in shorts: {100 * var_explained:.1f}%)."
                )

        global_pca_basis = None
        global_pca_weights = None
        global_pca_sv = None
        if global_pca_components > 0:
            Y, global_pca_basis, global_pca_weights, global_pca_sv = _global_pca_regress(
                Y, global_pca_components
            )
            _, S_all, _ = np.linalg.svd(Y - Y.mean(axis=1, keepdims=True), full_matrices=False)
            total_var = float(np.sum(global_pca_sv ** 2) + np.sum(S_all ** 2))
            ve = float(np.sum(global_pca_sv ** 2) / total_var) if total_var > 0 else 0.0
            typer.echo(
                f"Regressed top-{global_pca_basis.shape[0]} global PCA components out of "
                f"{Y.shape[0]} channels (var explained pre-regression: {100 * ve:.1f}%)."
            )
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

        short_pca_basis = None
        short_pca_betas = None
        short_pca_sv = None
        if short_pca_components > 0:
            short_idx = np.flatnonzero(is_short)
            if include_short_channels:
                typer.echo("Warning: --short-pca-components ignored when --include-short-channels is set.")
            elif short_idx.size == 0:
                typer.echo(f"Warning: --short-pca-components={short_pca_components} requested but no short channels available.")
            else:
                Y_short = Y_full[short_idx]
                Y, short_pca_basis, short_pca_betas, short_pca_sv = _short_channel_pca_regress(
                    Y, Y_short, short_pca_components
                )
                _, S_all, _ = np.linalg.svd(Y_short - Y_short.mean(axis=1, keepdims=True), full_matrices=False)
                var_explained = float(np.sum(short_pca_sv ** 2) / np.sum(S_all ** 2)) if S_all.size else 0.0
                typer.echo(
                    f"Regressed top-{short_pca_basis.shape[0]} PCA components of {len(short_idx)} short channels "
                    f"out of {Y.shape[0]} long channels (var explained in shorts: {100 * var_explained:.1f}%)."
                )

        global_pca_basis = None
        global_pca_weights = None
        global_pca_sv = None
        if global_pca_components > 0:
            Y, global_pca_basis, global_pca_weights, global_pca_sv = _global_pca_regress(
                Y, global_pca_components
            )
            _, S_all, _ = np.linalg.svd(Y - Y.mean(axis=1, keepdims=True), full_matrices=False)
            total_var = float(np.sum(global_pca_sv ** 2) + np.sum(S_all ** 2))
            ve = float(np.sum(global_pca_sv ** 2) / total_var) if total_var > 0 else 0.0
            typer.echo(
                f"Regressed top-{global_pca_basis.shape[0]} global PCA components out of "
                f"{Y.shape[0]} channels (var explained pre-regression: {100 * ve:.1f}%)."
            )
        short_msg = (
            f", excluded {n_short} short" if (n_short and not include_short_channels)
            else (f", including {n_short} short" if include_short_channels and n_short else "")
        )
        typer.echo(f"Selected {len(ch_indices)} {chromophore.upper()} channels (of {n_total} total{short_msg})")
        t = np.asarray(hemo_data.time)
        stim_onsets = np.zeros(0, dtype=np.float64)
        stim_durations = np.zeros(0, dtype=np.float64)

    Y = np.asarray(Y, dtype=np.float64)
    Y_original = Y.copy()  # stays untouched; npz "Y" field saves this.
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
    dt = float(t[1] - t[0]) if len(t) >= 2 else 1.0

    log_ell_min = float(np.log(min_length_scale)) if min_length_scale is not None else None
    log_ell_max = float(np.log(max_length_scale)) if max_length_scale is not None else None

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
    if short_pca_basis is not None:
        save_dict["short_pca_basis"] = np.asarray(short_pca_basis, dtype=np.float64)
        save_dict["short_pca_betas"] = np.asarray(short_pca_betas, dtype=np.float64)
        save_dict["short_pca_singular_values"] = np.asarray(short_pca_sv, dtype=np.float64)
    if global_pca_basis is not None:
        save_dict["global_pca_basis"] = np.asarray(global_pca_basis, dtype=np.float64)
        save_dict["global_pca_weights"] = np.asarray(global_pca_weights, dtype=np.float64)
        save_dict["global_pca_singular_values"] = np.asarray(global_pca_sv, dtype=np.float64)

    config = dict(
        data=str(data),
        chromophore=chromophore,
        init_length_scale=init_length_scale,
        n_iter=n_iter,
        seed=seed,
        include_short_channels=include_short_channels,
        short_pca_components=short_pca_components,
        global_pca_components=global_pca_components,
        log_sigma_min=log_sigma_min,
        log_sigma_max=log_sigma_max,
        min_length_scale=min_length_scale,
        max_length_scale=max_length_scale,
        rank=rank,
        seed_channel_index=seed_channel_index,
        validation_fraction=validation_fraction,
        validation_chunk_size=validation_chunk_size,
        validation_mode=validation_mode,
    )
    with open(output / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    n_channels, n_timepoints = Y.shape

    # Reduced χ² diagnostics: residual / σ, per channel and overall.
    residuals = Y - res["posterior_mean"]
    sigma_i = np.sqrt(res["noise_var"])
    chi2_per_ch = np.sum((residuals / sigma_i[:, None]) ** 2, axis=1)
    chi2_red_per_ch = chi2_per_ch / float(n_timepoints)
    n_free_params = n_channels * res["rank"] + 2 * n_channels + 1  # L + log_d + log_sigma2 + log_ell
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

    if seed_channel_index is not None:
        if 0 <= seed_channel_index < n_channels:
            seed_label = str(ch_labels[seed_channel_index])
            row_data = np.corrcoef(Y)[seed_channel_index]
            row_model = res['correlation'][seed_channel_index]
            row_resid = np.corrcoef(Y - res['posterior_mean'])[seed_channel_index]

            functional_connectivity_score = (np.sum(res['correlation'][seed_channel_index]) - 1.0) / (n_channels - 1)
            mean_correlation = (0.5 * (res['correlation'][seed_channel_index].sum() - n_channels))/(n_channels - 1)**2

            mask = np.arange(n_channels) != seed_channel_index
            typer.echo(f"  Seed channel:    row {seed_channel_index} ({seed_label})")
            typer.echo(f"    functional connectivity score: {functional_connectivity_score:.4f}")
            typer.echo(f"    mean correlation: {mean_correlation:.4f}")
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
def interact(
    model_dir: Path = typer.Argument(..., help="Path to output from fnirs fit"),
):
    """Hover over a channel on the montage to see its raw trace, posterior mean, and ±σ band."""
    from fnirs.interactive import run as run_interactive

    run_interactive(model_dir)


if __name__ == "__main__":
    app()
