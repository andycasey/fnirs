"""Synthetic-data recovery test for the Whittle GP."""
from __future__ import annotations

import numpy as np

from fnirs.whittle import fit


def _generate_synthetic(
    N: int = 8,
    T: int = 1024,
    ell: float = 50.0,
    noise_std_mean: float = 0.3,
    seed: int = 0,
):
    """Sample from the model: low-rank channel covariance + heterogeneous noise."""
    rng = np.random.default_rng(seed)

    rank = 4
    F_load = rng.standard_normal((N, rank))
    Sigma_true = F_load @ F_load.T + 0.5 * np.eye(N)

    noise_std = noise_std_mean * np.exp(0.4 * rng.standard_normal(N))

    freqs = np.fft.rfftfreq(T, d=1.0)
    omega = 2 * np.pi * freqs
    lam = np.sqrt(3.0) / ell
    psd = 4 * lam**3 / (lam**2 + omega**2) ** 2

    Lz = np.linalg.cholesky(Sigma_true)
    Z_fft = np.zeros((N, len(freqs)), dtype=complex)
    for k in range(len(freqs)):
        scale = np.sqrt(T * psd[k])
        is_real_k = (k == 0) or (T % 2 == 0 and k == len(freqs) - 1)
        if is_real_k:
            Z_fft[:, k] = scale * (Lz @ rng.standard_normal(N))
        else:
            re = rng.standard_normal(N)
            im = rng.standard_normal(N)
            Z_fft[:, k] = scale / np.sqrt(2) * (Lz @ (re + 1j * im))
    z = np.fft.irfft(Z_fft, n=T, axis=-1)

    noise = noise_std[:, None] * rng.standard_normal((N, T))
    y = z + noise
    return y, Sigma_true, noise_std


def test_synthetic_recovery():
    Y, Sigma_true, noise_std_true = _generate_synthetic(
        N=8, T=1024, ell=40.0, noise_std_mean=0.3, seed=0
    )
    res = fit(Y, init_length_scale=20.0, n_iter=80, verbose=False)

    # Length scale: within 30% of truth.
    rel_err_ell = abs(res["length_scale"] - 40.0) / 40.0
    assert rel_err_ell < 0.3, f"length scale off: got {res['length_scale']:.2f}, expected ~40"

    # Off-diagonal correlation: Pearson with truth >= 0.85.
    d_true = np.sqrt(np.diag(Sigma_true))
    corr_true = Sigma_true / (d_true[:, None] * d_true[None, :])
    iu = np.triu_indices(Y.shape[0], k=1)
    pearson = np.corrcoef(res["correlation"][iu], corr_true[iu])[0, 1]
    assert pearson >= 0.85, f"correlation Pearson too low: {pearson:.3f}"

    # Noise: within ~50% on the median (synthetic stochasticity is forgiving).
    noise_std_est = np.sqrt(res["noise_var"])
    median_rel_err = np.median(np.abs(noise_std_est - noise_std_true) / noise_std_true)
    assert median_rel_err < 0.5, f"noise std off: median rel err {median_rel_err:.2f}"


def test_log_sigma_bounds_clip():
    """Set a tight upper bound on log σ and check it's respected."""
    Y, _, noise_std_true = _generate_synthetic(N=6, T=512, ell=30.0, seed=2)
    # The synthetic noise std is ~0.3; force log σ ≤ log(0.05) so the fit must hit the cap.
    log_sigma_max = float(np.log(0.05))
    res = fit(Y, init_length_scale=20.0, n_iter=40, verbose=False,
              log_sigma_max=log_sigma_max)
    fitted_log_sigma = 0.5 * np.log(res["noise_var"])
    # Allow a tiny numerical slack above the cap.
    assert (fitted_log_sigma <= log_sigma_max + 1e-6).all(), (
        f"log σ exceeded cap: max={fitted_log_sigma.max():.4f}, cap={log_sigma_max:.4f}"
    )


def test_log_ell_bounds_clip():
    """Set a tight upper bound on log ell and check the fit respects it."""
    Y, _, _ = _generate_synthetic(N=6, T=512, ell=80.0, seed=3)
    # Synthetic ell is 80; cap log ell at log(20) so the fit must hit the cap.
    log_ell_max = float(np.log(20.0))
    res = fit(Y, init_length_scale=50.0, n_iter=40, verbose=False,
              log_ell_max=log_ell_max)
    assert np.log(res["length_scale"]) <= log_ell_max + 1e-6, (
        f"length scale exceeded cap: got {res['length_scale']:.4f}, cap=exp({log_ell_max:.4f})={np.exp(log_ell_max):.4f}"
    )


def test_shapes_and_keys():
    Y, _, _ = _generate_synthetic(N=5, T=256, seed=1)
    res = fit(Y, init_length_scale=20.0, n_iter=20, verbose=False)
    assert res["sigma"].shape == (5, 5)
    assert res["correlation"].shape == (5, 5)
    assert res["noise_var"].shape == (5,)
    assert res["posterior_mean"].shape == Y.shape
    assert isinstance(res["length_scale"], float)
    # Diagonal of correlation == 1.
    np.testing.assert_allclose(np.diag(res["correlation"]), 1.0, atol=1e-8)
