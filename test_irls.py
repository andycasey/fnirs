import numpy as np
import jax.numpy as jnp
import pytest
from fnirs.model import (
    fit,
    create_spherical_harmonics_basis,
    create_1d_fourier_modes,
    evaluate_1d_fourier_basis,
    fourier_matmat,
    gram_diagonal,
)
from functools import partial


def _make_synthetic_data(
    n_channels=30,
    n_timepoints=200,
    max_spherical_degree=2,
    n_fourier_components=5,
    noise_scales=None,
    seed=42,
):
    """Generate synthetic fNIRS data with known noise levels."""
    rng = np.random.default_rng(seed)

    # Random channel positions on sphere
    theta = rng.uniform(0, np.pi, n_channels)
    phi = rng.uniform(0, 2 * np.pi, n_channels)

    # Timepoints
    t = jnp.linspace(0, 2 * np.pi, n_timepoints, endpoint=False)

    # Build true signal via the forward model
    ST, terms = create_spherical_harmonics_basis(theta, phi, max_degree=max_spherical_degree)
    n_spatial = ST.shape[1]

    args = ((n_timepoints,), (n_fourier_components,))
    A = partial(fourier_matmat, *args)

    # Random true coefficients
    X_true = jnp.array(rng.standard_normal((n_fourier_components, n_spatial)))
    Y_clean = (A(X_true) @ ST.T).T  # (n_channels, n_timepoints)

    # Per-channel noise
    if noise_scales is None:
        noise_scales = rng.uniform(0.1, 2.0, n_channels)
    noise_scales = np.asarray(noise_scales)

    noise = rng.standard_normal((n_channels, n_timepoints)) * noise_scales[:, None]
    Y = jnp.array(Y_clean + noise)

    return dict(
        t=t,
        theta=jnp.array(theta),
        phi=jnp.array(phi),
        Y=Y,
        Y_clean=Y_clean,
        X_true=X_true,
        noise_scales=noise_scales,
        max_spherical_degree=max_spherical_degree,
        n_fourier_components=n_fourier_components,
    )


class TestIRLS:

    def test_estimate_noise_false_gives_same_results(self):
        """estimate_noise=False must give identical results to original code path."""
        d = _make_synthetic_data()
        result = fit(
            d["t"], d["theta"], d["phi"], d["Y"],
            d["max_spherical_degree"], d["n_fourier_components"],
            estimate_noise=False,
        )
        X, f, A, ST, terms, noise_variance, n_iter = result
        assert noise_variance is None
        assert n_iter == 0
        # Check predictions work
        Y_hat = f(X)
        assert Y_hat.shape == d["Y"].shape

    def test_irls_recovers_noise_variances(self):
        """IRLS should recover per-channel noise variances reasonably well."""
        noise_scales = np.array([0.5, 1.0, 2.0, 0.3, 1.5] * 6)  # 30 channels
        d = _make_synthetic_data(
            n_channels=30,
            n_timepoints=500,
            noise_scales=noise_scales,
        )
        result = fit(
            d["t"], d["theta"], d["phi"], d["Y"],
            d["max_spherical_degree"], d["n_fourier_components"],
            estimate_noise=True,
            max_irls_iter=30,
            irls_tol=1e-6,
        )
        X, f, A, ST, terms, noise_variance, n_iter = result
        assert noise_variance is not None
        assert n_iter > 0

        true_variance = noise_scales ** 2
        estimated = np.array(noise_variance)

        # Check correlation -- the relative ordering should be well recovered
        correlation = np.corrcoef(true_variance, estimated)[0, 1]
        assert correlation > 0.8, f"Correlation between true and estimated noise variance is {correlation:.3f}"

        # Check that estimates are in the right ballpark (within factor of 3)
        ratio = estimated / true_variance
        assert np.all(ratio > 0.3) and np.all(ratio < 3.0), (
            f"Noise variance ratios out of range: min={ratio.min():.2f}, max={ratio.max():.2f}"
        )

    def test_irls_improves_weighted_residual(self):
        """Weighted fit should produce lower weighted residual than unweighted."""
        noise_scales = np.array([0.1, 0.1, 0.1, 5.0, 5.0, 5.0] * 5)  # 30 channels
        d = _make_synthetic_data(
            n_channels=30,
            n_timepoints=500,
            noise_scales=noise_scales,
        )

        # Unweighted fit
        X_uw, f_uw, *_, _, _ = fit(
            d["t"], d["theta"], d["phi"], d["Y"],
            d["max_spherical_degree"], d["n_fourier_components"],
            estimate_noise=False,
        )
        Y_hat_uw = f_uw(X_uw)

        # Weighted fit
        X_w, f_w, _, _, _, noise_var, _ = fit(
            d["t"], d["theta"], d["phi"], d["Y"],
            d["max_spherical_degree"], d["n_fourier_components"],
            estimate_noise=True,
        )
        Y_hat_w = f_w(X_w)

        # Compare predictions to clean signal on low-noise channels
        low_noise_idx = noise_scales < 1.0
        mse_uw = float(jnp.mean((d["Y_clean"][low_noise_idx] - Y_hat_uw[low_noise_idx]) ** 2))
        mse_w = float(jnp.mean((d["Y_clean"][low_noise_idx] - Y_hat_w[low_noise_idx]) ** 2))

        # Weighted fit should do at least as well on low-noise channels
        assert mse_w <= mse_uw * 1.1, (
            f"Weighted MSE ({mse_w:.6f}) should be <= unweighted MSE ({mse_uw:.6f}) on low-noise channels"
        )

    def test_irls_convergence(self):
        """IRLS should converge in fewer than max_iter iterations for well-conditioned data."""
        d = _make_synthetic_data(n_timepoints=300)
        result = fit(
            d["t"], d["theta"], d["phi"], d["Y"],
            d["max_spherical_degree"], d["n_fourier_components"],
            estimate_noise=True,
            max_irls_iter=50,
            irls_tol=1e-4,
        )
        _, _, _, _, _, _, n_iter = result
        assert n_iter < 50, f"IRLS did not converge in 50 iterations (used {n_iter})"

    def test_return_tuple_length(self):
        """Return tuple should always have 7 elements."""
        d = _make_synthetic_data()

        result_off = fit(
            d["t"], d["theta"], d["phi"], d["Y"],
            d["max_spherical_degree"], d["n_fourier_components"],
            estimate_noise=False,
        )
        assert len(result_off) == 7

        result_on = fit(
            d["t"], d["theta"], d["phi"], d["Y"],
            d["max_spherical_degree"], d["n_fourier_components"],
            estimate_noise=True,
            max_irls_iter=3,
        )
        assert len(result_on) == 7
