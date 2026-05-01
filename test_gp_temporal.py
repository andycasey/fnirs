"""Tests for GP temporal kernel (Matern-1/2) in the fnirs model."""

import numpy as np
import jax.numpy as jnp
import pytest
from scipy.special import sph_harm_y

from fnirs import fit, matern12_psd


def _make_synthetic_data(n_channels=20, n_samples=200, seed=42):
    """
    Create synthetic spatial-temporal data: low-freq signal + high-freq noise.

    Returns t, theta, phi, Y_clean, Y_noisy.
    """
    rng = np.random.default_rng(seed)

    # Spatial positions on the sphere
    theta = rng.uniform(0, np.pi, n_channels)
    phi = rng.uniform(0, 2 * np.pi, n_channels)

    # Time axis (evenly sampled, e.g., 10 Hz for 20 seconds)
    t = np.linspace(0, 20, n_samples, endpoint=False)

    # Smooth signal: sum of low-frequency sinusoids
    signal = (
        1.0 * np.sin(2 * np.pi * 0.1 * t)
        + 0.5 * np.cos(2 * np.pi * 0.3 * t)
        + 0.3 * np.sin(2 * np.pi * 0.5 * t)
    )

    # Each channel gets a scaled version (simple spatial pattern)
    spatial_weights = np.cos(theta)  # varies with polar angle
    Y_clean = np.outer(spatial_weights, signal)

    # Add high-frequency noise
    noise = 0.3 * rng.standard_normal((n_channels, n_samples))
    Y_noisy = Y_clean + noise

    return (
        jnp.array(t),
        jnp.array(theta),
        jnp.array(phi),
        jnp.array(Y_clean),
        jnp.array(Y_noisy),
    )


def _high_freq_power(signal, n_samples):
    """Compute fraction of power in upper half of frequency spectrum."""
    fft_vals = np.fft.rfft(signal, axis=-1)
    power = np.abs(fft_vals) ** 2
    n_freq = power.shape[-1]
    midpoint = n_freq // 2
    high = np.sum(power[..., midpoint:])
    total = np.sum(power)
    return high / total if total > 0 else 0.0


class TestMatern12PSD:
    """Tests for the matern12_psd helper."""

    def test_dc_value(self):
        """At f=0, PSD should be 2 * variance * lengthscale."""
        freqs = jnp.array([0.0])
        result = matern12_psd(freqs, lengthscale=2.0, variance=3.0)
        expected = 2 * 3.0 * 2.0  # 12.0
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_known_values(self):
        """Check PSD at known frequencies against analytic formula."""
        freqs = jnp.array([0.0, 0.5, 1.0, 2.0])
        ell = 1.5
        sigma2 = 2.0
        result = matern12_psd(freqs, lengthscale=ell, variance=sigma2)
        for i, f in enumerate(freqs):
            omega = 2 * np.pi * float(f)
            expected = 2 * sigma2 * ell / (1 + (ell * omega) ** 2)
            np.testing.assert_allclose(float(result[i]), expected, rtol=1e-6)

    def test_monotonically_decreasing(self):
        """PSD should decrease with increasing frequency."""
        freqs = jnp.linspace(0, 10, 100)
        psd = matern12_psd(freqs, lengthscale=1.0, variance=1.0)
        diffs = jnp.diff(psd)
        assert jnp.all(diffs <= 0), "PSD should be non-increasing with frequency"

    def test_positive(self):
        """PSD should always be positive."""
        freqs = jnp.linspace(0, 100, 1000)
        psd = matern12_psd(freqs, lengthscale=0.5, variance=1.0)
        assert jnp.all(psd > 0)


class TestGPTemporalFit:
    """Tests for fit() with temporal_kernel='matern12'."""

    def test_none_kernel_unchanged(self):
        """temporal_kernel=None should give identical results to baseline."""
        t, theta, phi, _, Y = _make_synthetic_data()
        max_degree = 2
        n_fourier = 30

        XT1, *_ = fit(t, theta, phi, Y, max_degree, n_fourier)
        XT2, *_ = fit(t, theta, phi, Y, max_degree, n_fourier, temporal_kernel=None)

        np.testing.assert_allclose(XT1, XT2, atol=1e-5)

    def test_gp_fit_smoother(self):
        """GP fit should have less high-frequency residual than plain fit."""
        t, theta, phi, Y_clean, Y_noisy = _make_synthetic_data()
        max_degree = 2
        n_fourier = 80

        # Plain fit
        XT_plain, f_plain, *_ = fit(t, theta, phi, Y_noisy, max_degree, n_fourier)
        Y_hat_plain = f_plain(XT_plain)

        # GP fit with lengthscale that matches the signal (~few seconds)
        XT_gp, f_gp, *_ = fit(
            t, theta, phi, Y_noisy, max_degree, n_fourier,
            temporal_kernel="matern12",
            kernel_lengthscale=2.0,
            kernel_variance=1.0,
        )
        Y_hat_gp = f_gp(XT_gp)

        # Compute high-frequency power fraction of residuals
        resid_plain = np.array(Y_noisy - Y_hat_plain)
        resid_gp = np.array(Y_noisy - Y_hat_gp)

        # The GP fit absorbs less high-frequency noise, so the fit itself
        # should be smoother. Check high-freq power of the fit.
        hf_plain = _high_freq_power(np.array(Y_hat_plain), Y_noisy.shape[1])
        hf_gp = _high_freq_power(np.array(Y_hat_gp), Y_noisy.shape[1])

        assert hf_gp < hf_plain, (
            f"GP fit should have less high-freq power: GP={hf_gp:.4f} vs plain={hf_plain:.4f}"
        )

    def test_gp_closer_to_clean(self):
        """GP fit should be closer to the clean signal than plain fit when overfitting."""
        # Use more fourier components and more noise so plain fit overfits
        t, theta, phi, Y_clean, Y_noisy = _make_synthetic_data(
            n_channels=20, n_samples=200, seed=42
        )
        # Add extra noise to make the overfitting more pronounced
        rng = np.random.default_rng(99)
        Y_noisy = Y_noisy + 0.5 * jnp.array(rng.standard_normal(Y_noisy.shape))

        max_degree = 2
        n_fourier = 150  # many modes -> plain fit will overfit

        XT_plain, f_plain, *_ = fit(t, theta, phi, Y_noisy, max_degree, n_fourier)
        Y_hat_plain = f_plain(XT_plain)

        XT_gp, f_gp, *_ = fit(
            t, theta, phi, Y_noisy, max_degree, n_fourier,
            temporal_kernel="matern12",
            kernel_lengthscale=2.0,
            kernel_variance=10.0,
        )
        Y_hat_gp = f_gp(XT_gp)

        mse_plain = float(jnp.mean((Y_clean - Y_hat_plain) ** 2))
        mse_gp = float(jnp.mean((Y_clean - Y_hat_gp) ** 2))

        assert mse_gp < mse_plain, (
            f"GP fit should be closer to clean signal: GP MSE={mse_gp:.6f} vs plain MSE={mse_plain:.6f}"
        )

    def test_invalid_kernel_raises(self):
        """Unknown kernel name should raise ValueError."""
        t, theta, phi, _, Y = _make_synthetic_data()
        with pytest.raises(ValueError, match="Unknown temporal kernel"):
            fit(t, theta, phi, Y, 2, 30, temporal_kernel="rbf")
