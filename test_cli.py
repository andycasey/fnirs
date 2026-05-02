"""Tests for the fnirs CLI and plotting functions."""

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from fnirs.cli import app
from fnirs.plotting import (
    plot_connectivity,
    plot_correlation,
    plot_noise_std,
    plot_loss_curve,
    plot_channel_traces,
    plot_residuals,
    plot_power_spectrum,
)

runner = CliRunner()


def test_build_validation_mask_modes():
    """Each mode produces a mask with the expected structural property."""
    from fnirs.cli import _build_validation_mask
    rng = np.random.default_rng(42)
    N, T = 10, 1000
    f = 0.1

    mask_indep, _ = _build_validation_mask(N, T, f, "independent", 30, rng)
    assert mask_indep.shape == (N, T)
    # different rows have different per-row indices
    assert not all((mask_indep[0] == mask_indep[i]).all() for i in range(1, N))
    # each row has ~10% masked
    assert all(0.05 * T < mask_indep[i].sum() < 0.15 * T for i in range(N))

    mask_sync, _ = _build_validation_mask(N, T, f, "synchronous", 30, rng)
    assert mask_sync.shape == (N, T)
    # all rows identical
    for i in range(1, N):
        assert (mask_sync[0] == mask_sync[i]).all()

    mask_disj, _ = _build_validation_mask(N, T, 0.05, "disjoint", 30, rng)  # 0.05 < 1/N=0.1
    assert mask_disj.shape == (N, T)
    # at any time at most one channel is masked
    assert (mask_disj.sum(axis=0) <= 1).all()
    # every channel has some mask
    assert (mask_disj.sum(axis=1) > 0).all()

    # disjoint with too-large fraction errors out
    try:
        _build_validation_mask(N, T, 0.2, "disjoint", 30, rng)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_global_pca_regress_factorisation():
    """Y = W H + residual: after global PCA regression with k components,
    the OLS reconstruction WH should equal projection onto top-k right singular
    vectors of Y, and the residual should be orthogonal to those vectors."""
    from fnirs.cli import _global_pca_regress

    rng = np.random.default_rng(1)
    N, T = 12, 600
    raw = rng.standard_normal((3, T))
    raw -= raw.mean(axis=1, keepdims=True)
    Q, _ = np.linalg.qr(raw.T)
    H_true = Q.T  # (3, T) orthonormal

    a = rng.standard_normal((N, 3)) * 2.0
    cortex = rng.standard_normal((N, T)) * 0.3
    Y = a @ H_true + cortex

    Y_res, H, W, sv = _global_pca_regress(Y, n_components=3)
    assert H.shape == (3, T) and W.shape == (N, 3) and sv.shape == (3,)

    # Y - WH should be orthogonal in time to each row of H (after centering).
    Y_res_c = Y_res - Y_res.mean(axis=1, keepdims=True)
    proj = Y_res_c @ H.T
    assert np.max(np.abs(proj)) < 1e-8, f"residual not orthogonal to H: max proj {np.max(np.abs(proj)):.2e}"

    # H rows are orthonormal.
    assert np.allclose(H @ H.T, np.eye(3), atol=1e-10)


def test_short_channel_pca_regress_removes_known_signal():
    """Inject a known rank-2 common signal into both short and long channels;
    after regression with k=2 the long channels should retain only the unique
    cortex component."""
    from fnirs.cli import _short_channel_pca_regress

    rng = np.random.default_rng(0)
    n_short, n_long, T = 5, 12, 600
    # Two orthonormal common temporal components.
    raw = rng.standard_normal((2, T))
    raw -= raw.mean(axis=1, keepdims=True)
    Q, _ = np.linalg.qr(raw.T)
    common = Q.T  # (2, T), orthonormal rows

    a_short = rng.standard_normal((n_short, 2)) * 2.0
    a_long = rng.standard_normal((n_long, 2)) * 2.0
    cortex = rng.standard_normal((n_long, T)) * 0.3  # unique long-channel signal

    Y_short = a_short @ common + 0.05 * rng.standard_normal((n_short, T))
    Y_long = a_long @ common + cortex

    Y_clean, V, betas, sv = _short_channel_pca_regress(Y_long, Y_short, n_components=2)

    assert V.shape == (2, T)
    assert betas.shape == (2, n_long)
    assert sv.shape == (2,)

    # Fit minus cortex should be ~0 modulo the small noise in Y_short.
    residual_vs_cortex = Y_clean - cortex
    cortex_demeaned = cortex - cortex.mean(axis=1, keepdims=True)
    explained = 1.0 - np.var(residual_vs_cortex) / np.var(cortex_demeaned)
    assert explained > 0.95, f"common-signal removal too weak: explained={explained:.3f}"


def test_main_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fit" in result.output
    assert "plot" in result.output


def test_fit_help():
    result = runner.invoke(app, ["fit", "--help"])
    assert result.exit_code == 0
    assert "DATA" in result.output
    assert "OUTPUT" in result.output
    assert "--init-length-scale" in result.output
    assert "--n-iter" in result.output
    assert "--no-plots" in result.output


def test_plot_help():
    result = runner.invoke(app, ["plot", "--help"])
    assert result.exit_code == 0
    assert "MODEL_DIR" in result.output


def _make_synthetic_model(tmp_path: Path):
    """Create a synthetic model.npz matching the Whittle GP save schema."""
    rng = np.random.default_rng(0)
    N, T = 6, 256

    A = rng.standard_normal((N, N))
    sigma = A @ A.T + 0.5 * np.eye(N)
    d = np.sqrt(np.diag(sigma))
    correlation = sigma / (d[:, None] * d[None, :])
    noise_var = rng.uniform(0.01, 0.1, size=N)
    losses = np.linspace(100.0, 10.0, 30) + rng.standard_normal(30) * 0.1
    Y = rng.standard_normal((N, T))
    posterior_mean = 0.8 * Y + 0.05 * rng.standard_normal((N, T))
    positions_3d = rng.standard_normal((N, 3))
    dt = 0.1

    np.savez(
        tmp_path / "model.npz",
        sigma=sigma,
        correlation=correlation,
        noise_var=noise_var,
        length_scale=np.float64(25.0),
        losses=losses,
        posterior_mean=posterior_mean,
        Y=Y,
        positions_3d=positions_3d,
        dt=np.float64(dt),
        n_timepoints=np.int64(T),
    )

    config = dict(
        data="synthetic",
        chromophore="hbo",
        init_length_scale=20.0,
        n_iter=50,
        seed=0,
    )
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    return sigma, correlation, noise_var, losses, Y, posterior_mean, dt


def test_plot_connectivity(tmp_path):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    sigma = A @ A.T
    out = tmp_path / "connectivity.png"
    plot_connectivity(sigma, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_correlation(tmp_path):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    sigma = A @ A.T
    d = np.sqrt(np.diag(sigma))
    correlation = sigma / (d[:, None] * d[None, :])
    out = tmp_path / "correlation.png"
    plot_correlation(correlation, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_noise_std(tmp_path):
    noise = np.random.default_rng(0).uniform(0.01, 0.1, size=10)
    out = tmp_path / "noise.png"
    plot_noise_std(noise, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_loss_curve(tmp_path):
    losses = np.linspace(100.0, 10.0, 50)
    out = tmp_path / "loss.png"
    plot_loss_curve(losses, out)
    assert out.exists() and out.stat().st_size > 0


def test_posterior_predictive_samples_match_posterior_mean():
    """Average of many posterior predictive draws (latent only, no obs noise)
    should converge to the closed-form posterior mean."""
    from fnirs.plotting import _posterior_predictive_samples
    from fnirs.whittle import fit as whittle_fit

    rng = np.random.default_rng(7)
    N, T = 5, 256
    L = rng.standard_normal((N, 2))
    Sigma = L @ L.T + 0.5 * np.eye(N)
    ell = 30.0
    freqs = np.fft.rfftfreq(T, d=1.0)
    omega = 2 * np.pi * freqs
    lam = np.sqrt(3.0) / ell
    psd = 4.0 * lam**3 / (lam**2 + omega**2) ** 2
    Lz = np.linalg.cholesky(Sigma)
    Z_fft = np.zeros((N, len(freqs)), dtype=complex)
    for k in range(len(freqs)):
        scale = np.sqrt(T * psd[k])
        is_real_k = (k == 0) or (T % 2 == 0 and k == len(freqs) - 1)
        if is_real_k:
            Z_fft[:, k] = scale * (Lz @ rng.standard_normal(N))
        else:
            Z_fft[:, k] = scale / np.sqrt(2) * (Lz @ (rng.standard_normal(N) + 1j * rng.standard_normal(N)))
    z = np.fft.irfft(Z_fft, n=T, axis=-1)
    Y = z + 0.3 * rng.standard_normal((N, T))

    res = whittle_fit(Y, init_length_scale=20.0, n_iter=40, verbose=False)
    samples = _posterior_predictive_samples(
        res["sigma"], res["noise_var"], res["length_scale"],
        Y, n_samples=500, seed=0, add_observation_noise=False,
    )
    sample_mean = samples.mean(axis=0)
    err = np.max(np.abs(sample_mean - res["posterior_mean"]))
    rel = err / np.abs(res["posterior_mean"]).max()
    assert rel < 0.05, f"posterior-mean mismatch: rel err {rel:.4f}, max abs {err:.4e}"


def test_plot_channel_traces(tmp_path):
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((4, 200))
    z = 0.7 * Y + 0.1 * rng.standard_normal((4, 200))
    noise_var = rng.uniform(0.01, 0.1, size=4)
    out = tmp_path / "traces.png"
    plot_channel_traces(Y, z, noise_var, dt=0.1, output_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_residuals(tmp_path):
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((6, 200))
    z = 0.7 * Y
    out = tmp_path / "residuals.png"
    plot_residuals(Y, z, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_power_spectrum(tmp_path):
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((6, 200))
    z = 0.7 * Y
    out = tmp_path / "spectrum.png"
    plot_power_spectrum(Y, z, dt=0.1, output_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_subcommand_with_synthetic(tmp_path):
    """Test the plot subcommand end-to-end against a synthetic model.npz."""
    _make_synthetic_model(tmp_path)
    result = runner.invoke(app, ["plot", str(tmp_path)])
    assert result.exit_code == 0, result.output
    fig_dir = tmp_path / "figures"
    for name in (
        "connectivity.png",
        "correlation.png",
        "noise_std.png",
        "loss_curve.png",
        "channel_traces.png",
        "residuals.png",
        "power_spectrum.png",
    ):
        assert (fig_dir / name).exists(), f"missing {name}"
        assert (fig_dir / name).stat().st_size > 0
