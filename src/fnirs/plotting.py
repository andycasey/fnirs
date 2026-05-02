"""Plotting functions for the Whittle GP model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _contiguous_true_runs(mask_row: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end) sample indices for each contiguous True run."""
    mask = np.asarray(mask_row).astype(bool)
    if not mask.any():
        return []
    diffs = np.diff(mask.astype(int))
    starts = list(np.where(diffs == 1)[0] + 1)
    ends = list(np.where(diffs == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    return list(zip(starts, ends))


def _posterior_predictive_samples(
    sigma: np.ndarray,
    noise_var: np.ndarray,
    length_scale: float,
    Y: np.ndarray,
    n_samples: int,
    seed: int = 0,
    add_observation_noise: bool = True,
) -> np.ndarray:
    """Draw `n_samples` from p(y_new | Y) (or p(z | Y) if add_observation_noise=False)
    under the Whittle separable GP model.

    Math (per Whittle frequency bin k and signal eigenmode j):
        prior:        z̃_kj ~ CN(0, T · S_k · λ_j)
        likelihood:   W_kj | z̃_kj ~ CN(z̃_kj, T)
        posterior:    z̃_kj | W_kj ~ CN(η_kj · W_kj, η_kj · T)
    where η_kj = S_k λ_j / (S_k λ_j + 1) and λ_j are the eigenvalues of
    M = D^{-½} Σ D^{-½}, D = diag(σ²). Real bins (k=0, k=T/2) get N instead of CN.
    Transform back via D^½ U then irfft to reach the time domain. If
    `add_observation_noise`, additionally add ε ~ N(0, diag(σ²)) per channel.

    Returns array of shape (n_samples, N, T).
    """
    rng = np.random.default_rng(seed)
    N, T = Y.shape

    sqrt_d = np.sqrt(noise_var)
    inv_sqrt_d = 1.0 / sqrt_d
    M = inv_sqrt_d[:, None] * sigma * inv_sqrt_d[None, :]
    M = 0.5 * (M + M.T) + 1e-8 * np.eye(N)
    lam, U = np.linalg.eigh(M)

    mean = Y.mean(axis=-1, keepdims=True)
    Yk = np.fft.rfft(Y - mean, axis=-1)
    F = Yk.shape[-1]

    freqs = np.fft.rfftfreq(T, d=1.0)
    omega = 2 * np.pi * freqs
    lam_kernel = np.sqrt(3.0) / float(length_scale)
    psd = 4.0 * lam_kernel**3 / (lam_kernel**2 + omega**2) ** 2  # (F,)

    Wk = U.T @ (inv_sqrt_d[:, None] * Yk)         # (N, F)
    scaled_lam = psd[:, None] * lam[None, :]       # (F, N)
    eta = scaled_lam / (scaled_lam + 1.0)          # (F, N), in [0, 1)
    post_mean_eig = eta.T * Wk                     # (N, F)
    post_std_eig = np.sqrt(eta.T * T)              # (N, F)

    is_real = np.zeros(F, dtype=bool)
    is_real[0] = True
    if T % 2 == 0:
        is_real[-1] = True

    samples = np.empty((n_samples, N, T), dtype=np.float64)
    for s in range(n_samples):
        re = rng.standard_normal((N, F))
        im = rng.standard_normal((N, F))
        # CN(0, σ²) ⇒ each of re, im ~ N(0, σ²/2).
        noise_eig = (post_std_eig / np.sqrt(2)) * (re + 1j * im)
        for k_real in np.where(is_real)[0]:
            noise_eig[:, k_real] = post_std_eig[:, k_real] * rng.standard_normal(N)
        Z_eig = post_mean_eig + noise_eig
        Z_freq = sqrt_d[:, None] * (U @ Z_eig)
        z_t = np.fft.irfft(Z_freq, n=T, axis=-1)
        if add_observation_noise:
            z_t = z_t + sqrt_d[:, None] * rng.standard_normal((N, T))
        samples[s] = z_t + mean

    return samples


def plot_connectivity(sigma: np.ndarray, output_path: Path):
    vmax = float(np.max(np.abs(sigma)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sigma, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    ax.set_title(r"Channel covariance $\Sigma$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation(correlation: np.ndarray, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(correlation, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    ax.set_title("Channel correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_noise_std(
    noise_var: np.ndarray,
    output_path: Path,
    channel_labels: np.ndarray | None = None,
):
    noise_std = np.sqrt(np.asarray(noise_var))
    n = len(noise_std)
    width = max(10.0, 0.18 * n)
    fig, ax = plt.subplots(figsize=(width, 4))
    ax.bar(range(n), noise_std, color="coral", alpha=0.8)
    labels = [str(c) for c in channel_labels] if channel_labels is not None else [str(i) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Noise std")
    ax.set_title("Per-channel noise standard deviation")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(losses: np.ndarray, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, marker="o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Negative log-likelihood")
    ax.set_title("Optimisation trace")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_channel_traces(
    Y: np.ndarray,
    posterior_mean: np.ndarray,
    noise_var: np.ndarray,
    dt: float,
    output_path: Path,
    channel_labels: np.ndarray | None = None,
    n_cols: int = 4,
    stim_onsets: np.ndarray | None = None,
    stim_durations: np.ndarray | None = None,
    sigma_matrix: np.ndarray | None = None,
    length_scale: float | None = None,
    n_posterior_samples: int = 20,
    posterior_seed: int = 0,
    chi2_red_per_channel: np.ndarray | None = None,
    val_mask: np.ndarray | None = None,
):
    """Grid of small per-channel panels: data + posterior predictive draws.

    If `sigma_matrix` and `length_scale` are provided, draws `n_posterior_samples`
    samples from p(y_new | Y) and overlays each as a faint red line. Otherwise
    falls back to a ±σ band around the posterior mean.
    """
    n_channels = Y.shape[0]
    t = np.arange(Y.shape[1]) * dt
    noise_std = np.sqrt(np.asarray(noise_var))

    onsets = np.asarray(stim_onsets, dtype=float).ravel() if stim_onsets is not None else np.zeros(0)
    durations = np.asarray(stim_durations, dtype=float).ravel() if stim_durations is not None else np.zeros(0)
    if onsets.size != durations.size:
        onsets = np.zeros(0)
        durations = np.zeros(0)

    # If a validation mask is provided, the model was fit on imputed Y (channel
    # mean at masked positions). Sampling must use the same imputed Y so that
    # the Wiener filter — and hence the saved posterior_mean — stay consistent.
    Y_for_sampler = np.asarray(Y).copy()
    if val_mask is not None:
        vm = np.asarray(val_mask, dtype=bool)
        for i in range(Y_for_sampler.shape[0]):
            train_idx = ~vm[i]
            if train_idx.any():
                Y_for_sampler[i, vm[i]] = Y_for_sampler[i, train_idx].mean()

    samples: np.ndarray | None = None
    if sigma_matrix is not None and length_scale is not None and n_posterior_samples > 0:
        samples = _posterior_predictive_samples(
            np.asarray(sigma_matrix), np.asarray(noise_var), float(length_scale),
            Y_for_sampler, int(n_posterior_samples), seed=int(posterior_seed),
            add_observation_noise=True,
        )

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_channels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, max(1.6, 0.9 * n_rows)),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for i in range(n_channels):
        ax = axes_flat[i]
        for o, d in zip(onsets, durations):
            ax.axvspan(o, o + d, color="0.6", alpha=0.25, linewidth=0, zorder=0)
        if val_mask is not None:
            for s_idx, e_idx in _contiguous_true_runs(val_mask[i]):
                ax.axvspan(s_idx * dt, e_idx * dt, color="gold", alpha=0.25, linewidth=0, zorder=0)
        z = posterior_mean[i]
        if samples is not None:
            for s in range(samples.shape[0]):
                ax.plot(t, samples[s, i], color="red", alpha=0.12, linewidth=0.4, zorder=1)
        else:
            ax.fill_between(t, z - noise_std[i], z + noise_std[i], color="red", alpha=0.18, linewidth=0, zorder=1)
        ax.plot(t, z, color="red", linestyle="--", linewidth=0.8, label=r"$E[z\mid Y]$", zorder=2)
        ax.plot(t, Y[i], color="black", alpha=0.95, linewidth=0.5, label="Y", zorder=3)
        label = str(channel_labels[i]) if channel_labels is not None else f"ch {i}"
        if chi2_red_per_channel is not None:
            label = f"{label}   χ²ᵣ={float(chi2_red_per_channel[i]):.3f}"
        ax.set_title(label, fontsize=8, pad=2)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=6)

    for j in range(n_channels, n_rows * n_cols):
        axes_flat[j].set_visible(False)

    # Single legend on the first axis; xlabels only on the bottom-most visible panel of each column.
    axes_flat[0].legend(loc="upper right", fontsize=6, ncol=2)
    for col in range(n_cols):
        last_idx_in_col = max((i for i in range(col, n_channels, n_cols)), default=None)
        if last_idx_in_col is not None:
            axes[last_idx_in_col // n_cols, col].set_xlabel("Time (s)", fontsize=8)

    title = (
        r"Data (black) vs posterior predictive draws (red, low alpha) and $E[z\mid Y]$ (dashed)"
        if samples is not None
        else r"Raw vs Wiener-filtered channel traces (band: $E[z\mid Y]\pm\sigma_i$)"
    )
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_latent_draws(
    Y: np.ndarray,
    posterior_mean: np.ndarray,
    sigma_matrix: np.ndarray,
    noise_var: np.ndarray,
    length_scale: float,
    dt: float,
    output_path: Path,
    channel_labels: np.ndarray | None = None,
    n_cols: int = 4,
    stim_onsets: np.ndarray | None = None,
    stim_durations: np.ndarray | None = None,
    n_samples: int = 30,
    seed: int = 0,
    chi2_red_per_channel: np.ndarray | None = None,
    val_mask: np.ndarray | None = None,
):
    """Latent posterior draws p(z | Y) with the observed data on top.

    Each blue line is one sample of the latent signal z conditioned on the
    observations; their spread shows the posterior uncertainty in the GP. The
    black line is Y. No observation noise is added to the draws (use
    plot_channel_traces for that).
    """
    n_channels = Y.shape[0]
    t = np.arange(Y.shape[1]) * dt

    onsets = np.asarray(stim_onsets, dtype=float).ravel() if stim_onsets is not None else np.zeros(0)
    durations = np.asarray(stim_durations, dtype=float).ravel() if stim_durations is not None else np.zeros(0)
    if onsets.size != durations.size:
        onsets = np.zeros(0)
        durations = np.zeros(0)

    # Reconstruct imputed Y if a mask is given, so draws line up with the
    # saved posterior_mean (the model never saw the masked timepoints).
    Y_for_sampler = np.asarray(Y).copy()
    if val_mask is not None:
        vm = np.asarray(val_mask, dtype=bool)
        for i in range(Y_for_sampler.shape[0]):
            train_idx = ~vm[i]
            if train_idx.any():
                Y_for_sampler[i, vm[i]] = Y_for_sampler[i, train_idx].mean()

    samples = _posterior_predictive_samples(
        np.asarray(sigma_matrix), np.asarray(noise_var), float(length_scale),
        Y_for_sampler, int(n_samples), seed=int(seed),
        add_observation_noise=False,
    )

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_channels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, max(1.6, 0.9 * n_rows)),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for i in range(n_channels):
        ax = axes_flat[i]
        for o, d in zip(onsets, durations):
            ax.axvspan(o, o + d, color="0.6", alpha=0.25, linewidth=0, zorder=0)
        if val_mask is not None:
            for s_idx, e_idx in _contiguous_true_runs(val_mask[i]):
                ax.axvspan(s_idx * dt, e_idx * dt, color="gold", alpha=0.25, linewidth=0, zorder=0)
        for s in range(samples.shape[0]):
            ax.plot(t, samples[s, i], color="steelblue", alpha=0.12, linewidth=0.4, zorder=1)
        ax.plot(t, posterior_mean[i], color="steelblue", linewidth=0.8, alpha=0.9,
                label=r"$E[z\mid Y]$", zorder=2)
        ax.plot(t, Y[i], color="black", alpha=0.95, linewidth=0.5, label="Y", zorder=3)
        label = str(channel_labels[i]) if channel_labels is not None else f"ch {i}"
        if chi2_red_per_channel is not None:
            label = f"{label}   χ²ᵣ={float(chi2_red_per_channel[i]):.3f}"
        ax.set_title(label, fontsize=8, pad=2)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=6)

    for j in range(n_channels, n_rows * n_cols):
        axes_flat[j].set_visible(False)

    axes_flat[0].legend(loc="upper right", fontsize=6, ncol=2)
    for col in range(n_cols):
        last_idx_in_col = max((i for i in range(col, n_channels, n_cols)), default=None)
        if last_idx_in_col is not None:
            axes[last_idx_in_col // n_cols, col].set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(r"Latent posterior draws $z \sim p(z\mid Y)$ (blue) with data (black)", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    Y: np.ndarray,
    posterior_mean: np.ndarray,
    output_path: Path,
    channel_labels: np.ndarray | None = None,
    val_mask: np.ndarray | None = None,
    noise_std: np.ndarray | None = None,
):
    """Per-channel RMS of Y − E[z|Y]. If val_mask is given, plot train and val
    residuals as paired bars; otherwise plot a single bar per channel."""
    residuals = Y - posterior_mean
    n = Y.shape[0]
    width = max(10.0, 0.22 * n)
    fig, ax = plt.subplots(figsize=(width, 4.5))
    labels = [str(c) for c in channel_labels] if channel_labels is not None else [str(i) for i in range(n)]
    x = np.arange(n)

    if val_mask is not None:
        vm = np.asarray(val_mask, dtype=bool)
        train_rms = np.array([
            np.sqrt(np.mean(residuals[i, ~vm[i]] ** 2)) if (~vm[i]).any() else 0.0
            for i in range(n)
        ])
        val_rms = np.array([
            np.sqrt(np.mean(residuals[i, vm[i]] ** 2)) if vm[i].any() else 0.0
            for i in range(n)
        ])
        bar_w = 0.4
        ax.bar(x - bar_w/2, train_rms, bar_w, color="steelblue", alpha=0.85, label="train (in-sample)")
        ax.bar(x + bar_w/2, val_rms, bar_w, color="goldenrod", alpha=0.85, label="val (held-out)")
    else:
        rms = np.sqrt(np.mean(residuals ** 2, axis=1))
        ax.bar(x, rms, 0.8, color="steelblue", alpha=0.85, label="residual")

    if noise_std is not None:
        sigma_med = float(np.median(np.asarray(noise_std)))
        ax.axhline(sigma_med, color="black", linestyle="--", linewidth=0.8,
                   label=f"σ (median) = {sigma_med:.2f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Residual RMS")
    ax.set_title("Per-channel residual RMS  (Y − E[z|Y])")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_power_spectrum(
    Y: np.ndarray,
    posterior_mean: np.ndarray,
    dt: float,
    output_path: Path,
):
    T = Y.shape[1]
    freqs = np.fft.rfftfreq(T, d=dt)
    Y_pow = np.mean(np.abs(np.fft.rfft(Y, axis=1)) ** 2, axis=0)
    Z_pow = np.mean(np.abs(np.fft.rfft(posterior_mean, axis=1)) ** 2, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(freqs[1:], Y_pow[1:], label="Data", alpha=0.7, linewidth=0.9)
    ax.loglog(freqs[1:], Z_pow[1:], label="E[z|Y]", alpha=0.9, linewidth=0.9)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title("Average power spectrum: data vs latent")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
