"""
Joint multi-channel connectivity estimation via the Whittle likelihood.

Model
-----
    y_i(t) = z_i(t) + eps_i(t),      i = 1..N,  t = 0..T-1
    z(t) ~ GP(0, Sigma * k_t(.))     separable spatiotemporal GP
    eps_i(t) ~ N(0, sigma_i^2)       iid in time, independent across channels
    k_t = Matern-3/2 with length scale ell (samples), shared across channels

Sigma (the N x N channel covariance, the "connectivity") is treated as a free
parameter -- no spatial basis is imposed, so the estimator does not bake in
geometry-driven smoothness between nearby channels. The connectivity is what
the data demands, modulo the (very minor) effect of shared temporal smoothing.

For evenly-sampled data the temporal GP is diagonalised in the Fourier domain
(Whittle approximation: independent frequencies). The per-frequency covariance

    C_k  =  S_k * Sigma  +  diag(sigma^2)

shares the same generalised eigendecomposition for every k:

    M = diag(sigma^2)^{-1/2} Sigma diag(sigma^2)^{-1/2} = U Lambda U^T

so we eigendecompose once, project the FFT into the eigenbasis once, and
every per-frequency log-determinant and quadratic form is a vector op.
"""
from __future__ import annotations

from functools import partial

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.optimize


# ---------------------------------------------------------------------------
# Parameter packing
# ---------------------------------------------------------------------------
#
# Low-rank-plus-diagonal channel covariance:
#   Sigma = L L^T + diag(d),   L in R^{N x r},   d > 0
#
# Free parameters
# ---------------
#   L           : N x r matrix, unconstrained; controls the r shared signal modes.
#   log_d       : N, per-channel idiosyncratic signal variance d_i = exp(log_d_i).
#   log_sigma2  : N, per-channel white-noise log-variance.
#   log_ell     : scalar, log temporal length scale (samples).

def _flat_size(N: int, r: int) -> int:
    return N * r + N + N + 1


def pack(params: dict) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(params["L"]).ravel(),
            np.asarray(params["log_d"]).ravel(),
            np.asarray(params["log_sigma2"]).ravel(),
            np.asarray(params["log_ell"]).reshape(-1),
        ]
    ).astype(np.float64)


def unpack(flat: jnp.ndarray, N: int, r: int) -> dict:
    a = N * r
    b = a + N
    c = b + N
    return {
        "L": flat[:a].reshape(N, r),
        "log_d": flat[a:b],
        "log_sigma2": flat[b:c],
        "log_ell": flat[c],
    }


# ---------------------------------------------------------------------------
# Matern-3/2 power spectral density
# ---------------------------------------------------------------------------

def matern32_psd(omega: jnp.ndarray, ell: jnp.ndarray) -> jnp.ndarray:
    """PSD of unit-variance Matern-3/2 kernel.

        k(tau) = (1 + sqrt(3) |tau|/ell) exp(-sqrt(3) |tau|/ell)
        S(omega) = 4 lam^3 / (lam^2 + omega^2)^2,   lam = sqrt(3)/ell

    omega is angular frequency in rad/sample, ell is in samples.
    """
    lam = jnp.sqrt(3.0) / ell
    return 4.0 * lam**3 / (lam**2 + omega**2) ** 2


# ---------------------------------------------------------------------------
# Sigma from low-rank-plus-diagonal parameters
# ---------------------------------------------------------------------------

def sigma_from_params(params: dict) -> jnp.ndarray:
    L = params["L"]
    d = jnp.exp(params["log_d"])
    return L @ L.T + jnp.diag(d)


def _eig_decompose(params: dict):
    """Return (lam, U, sqrt_d, inv_sqrt_d) where M = U diag(lam) U^T and
    M = D^{-1/2} Sigma D^{-1/2}, D = diag(sigma^2)."""
    Sigma = sigma_from_params(params)
    sigma2 = jnp.exp(params["log_sigma2"])
    sqrt_d = jnp.sqrt(sigma2)
    inv_sqrt_d = 1.0 / sqrt_d
    M = inv_sqrt_d[:, None] * Sigma * inv_sqrt_d[None, :]
    M = 0.5 * (M + M.T)
    M = M + 1e-8 * jnp.eye(M.shape[0], dtype=M.dtype)
    lam, U = jnp.linalg.eigh(M)
    return lam, U, sqrt_d, inv_sqrt_d


# ---------------------------------------------------------------------------
# Likelihood and posterior
# ---------------------------------------------------------------------------

def neg_log_likelihood(params: dict, Y: jnp.ndarray) -> jnp.ndarray:
    """Whittle negative log-likelihood for Y of shape (N, T)."""
    N, T = Y.shape

    Yc = Y - Y.mean(axis=-1, keepdims=True)
    Yk = jnp.fft.rfft(Yc, axis=-1)
    F = Yk.shape[-1]

    freqs = jnp.fft.rfftfreq(T, d=1.0)
    omega = 2.0 * jnp.pi * freqs
    psd = matern32_psd(omega, jnp.exp(params["log_ell"]))

    lam, U, _, inv_sqrt_d = _eig_decompose(params)

    Wk = U.T.astype(Yk.dtype) @ (inv_sqrt_d[:, None].astype(Yk.dtype) * Yk)

    eig = psd[:, None] * lam[None, :] + 1.0  # (F, N), real positive

    log_d = params["log_sigma2"].sum()
    log_det_TC = N * jnp.log(T) + log_d + jnp.log(eig).sum(axis=-1)  # (F,)

    Wsq = (Wk.conj() * Wk).real  # (N, F)
    quad = (Wsq.T / eig).sum(axis=-1) / T  # (F,)

    is_real = jnp.zeros(F, dtype=bool).at[0].set(True)
    if T % 2 == 0:
        is_real = is_real.at[-1].set(True)

    ll_real = -0.5 * N * jnp.log(2 * jnp.pi) - 0.5 * log_det_TC - 0.5 * quad
    ll_complex = -N * jnp.log(jnp.pi) - log_det_TC - quad
    ll = jnp.where(is_real, ll_real, ll_complex)
    return -ll.sum()


def posterior_mean(params: dict, Y: jnp.ndarray) -> jnp.ndarray:
    """E[z | Y]: GP posterior mean of the latent signal, via FFT.

    In the eigenbasis the Wiener weight at frequency k and component j is
        eta_kj = S_k lam_j / (S_k lam_j + 1)  in [0, 1).
    """
    N, T = Y.shape
    mean = Y.mean(axis=-1, keepdims=True)
    Yk = jnp.fft.rfft(Y - mean, axis=-1)

    freqs = jnp.fft.rfftfreq(T, d=1.0)
    omega = 2.0 * jnp.pi * freqs
    psd = matern32_psd(omega, jnp.exp(params["log_ell"]))

    lam, U, sqrt_d, inv_sqrt_d = _eig_decompose(params)
    Wk = U.T.astype(Yk.dtype) @ (inv_sqrt_d[:, None].astype(Yk.dtype) * Yk)  # (N, F)

    scaled_lam = psd[:, None] * lam[None, :]  # (F, N)
    eta = scaled_lam / (scaled_lam + 1.0)  # (F, N)

    Zk = sqrt_d[:, None].astype(Yk.dtype) * (
        U.astype(Yk.dtype) @ (eta.T.astype(Yk.dtype) * Wk)
    )
    z = jnp.fft.irfft(Zk, n=T, axis=-1)
    return z + mean


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def correlation_from_params(params: dict) -> jnp.ndarray:
    S = sigma_from_params(params)
    d = jnp.sqrt(jnp.diagonal(S))
    return S / (d[:, None] * d[None, :])


def evaluate(params_flat: np.ndarray, Y: np.ndarray, N: int, r: int) -> dict:
    """Evaluate fitted params on new Y (e.g. held-out validation segment).

    Returns the Whittle negative log-likelihood, posterior mean of the latent,
    and per-channel reduced χ² (residual / σ).
    """
    Y_j = jnp.asarray(np.asarray(Y, dtype=np.float64))
    params = unpack(jnp.asarray(params_flat), N, r)
    nll = float(neg_log_likelihood(params, Y_j))
    z = posterior_mean(params, Y_j)
    z_np = np.asarray(z)
    sigma2 = np.asarray(jnp.exp(params["log_sigma2"]))
    res = np.asarray(Y) - z_np
    chi2_per_ch = np.sum(res**2 / sigma2[:, None], axis=1)
    chi2_red_per_ch = chi2_per_ch / Y.shape[1]
    return {
        "neg_log_likelihood": nll,
        "posterior_mean": z_np,
        "residuals": res,
        "chi2_red_per_channel": chi2_red_per_ch,
        "chi2_red_total": float(np.sum(chi2_per_ch) / (Y.shape[0] * Y.shape[1])),
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _init_params(N: int, r: int, init_length_scale: float, init_noise_std: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # Init L so that L L^T contributes ~init_noise_std^2 along its r columns; this
    # spreads the eigenvalues of M = D^{-1/2} Sigma D^{-1/2} (otherwise M ≈ I and
    # the gradient through eigh blows up).  log_d small so the diagonal addition
    # doesn't dominate.
    s = init_noise_std / np.sqrt(max(r, 1))
    L0 = (s * rng.standard_normal((N, r))).astype(np.float64)
    print(f"L0 shape: {L0.shape} {L0.size}")
    log_d0 = np.full(N, float(np.log((0.1 * init_noise_std) ** 2)), dtype=np.float64)
    log_sigma2 = np.full(N, float(np.log(init_noise_std**2)), dtype=np.float64)
    log_ell = np.array(np.log(init_length_scale), dtype=np.float64)
    return {
        "L": jnp.asarray(L0),
        "log_d": jnp.asarray(log_d0),
        "log_sigma2": jnp.asarray(log_sigma2),
        "log_ell": jnp.asarray(log_ell),
    }


# ---------------------------------------------------------------------------
# Optimisation wrapper
# ---------------------------------------------------------------------------

def fit(
    Y: np.ndarray,
    *,
    rank: int | None = None,
    init_length_scale: float = 30.0,
    n_iter: int = 100,
    verbose: bool = True,
    seed: int = 0,
    log_sigma_min: float | None = None,
    log_sigma_max: float | None = None,
    log_ell_min: float | None = None,
    log_ell_max: float | None = None,
) -> dict:
    """Fit the Whittle GP to Y of shape (N, T) and return a summary dict.

    Sigma is parameterised as L L^T + diag(d) with L of shape (N, rank) and
    d > 0. rank=None defaults to N (full-rank).
    """
    Y_np = np.asarray(Y, dtype=np.float64)
    N, T = Y_np.shape
    r = int(rank) if rank is not None else N
    if r < 1 or r > N:
        raise ValueError(f"rank must be in [1, {N}], got {rank}")

    # std of first differences / sqrt(2) is a rough noise estimate, but it
    # collapses when the data has been pre-low-passed; floor it at 10 % of the
    # broadband per-channel std so the init isn't catastrophically below scale
    # (which gives huge gradients and a one-step "convergence" exit).
    first_diff = float(np.median(np.std(np.diff(Y_np, axis=-1), axis=-1) / np.sqrt(2)))
    broadband_floor = 0.1 * float(np.median(np.std(Y_np, axis=-1)))
    init_noise = max(first_diff, broadband_floor, 1e-3)
    # Keep the init at least BUFFER log units inside any active bound, so the
    # L-BFGS-B line search has room to make progress (a tight init at the bound
    # leaves the projected gradient nearly zero and the line search stalls).
    BUFFER = 0.5
    log_init_noise = np.log(init_noise)
    if log_sigma_min is not None and log_init_noise <= log_sigma_min + BUFFER:
        log_init_noise = log_sigma_min + BUFFER
    if log_sigma_max is not None and log_init_noise >= log_sigma_max - BUFFER:
        log_init_noise = log_sigma_max - BUFFER
    init_noise = float(np.exp(log_init_noise))

    log_init_ell = np.log(init_length_scale)
    if log_ell_min is not None and log_init_ell <= log_ell_min + BUFFER:
        log_init_ell = log_ell_min + BUFFER
    if log_ell_max is not None and log_init_ell >= log_ell_max - BUFFER:
        log_init_ell = log_ell_max - BUFFER
    init_length_scale = float(np.exp(log_init_ell))

    params0 = _init_params(N, r, init_length_scale, init_noise, seed=seed)
    Y_j = jnp.asarray(Y_np)

    @jax.jit
    def value_and_grad_flat(flat):
        params = unpack(flat, N, r)
        return jax.value_and_grad(neg_log_likelihood)(params, Y_j)

    losses: list[float] = []

    def loss_and_grad(flat: np.ndarray):
        val, grad = value_and_grad_flat(jnp.asarray(flat))
        flat_grad = np.concatenate(
            [
                np.asarray(grad["L"]).ravel(),
                np.asarray(grad["log_d"]).ravel(),
                np.asarray(grad["log_sigma2"]).ravel(),
                np.asarray(grad["log_ell"]).reshape(-1),
            ]
        ).astype(np.float64)
        return float(val), flat_grad

    def callback(xk: np.ndarray):
        val, _ = value_and_grad_flat(jnp.asarray(xk))
        losses.append(float(val))
        if verbose and (len(losses) % 5 == 1 or len(losses) == n_iter):
            params = unpack(jnp.asarray(xk), N, r)
            ell = float(jnp.exp(params["log_ell"]))
            print(f"iter {len(losses):3d}  -loglik = {float(val):.3f}  ell = {ell:.2f}")

    x0 = pack(params0)

    has_sigma_bounds = log_sigma_min is not None or log_sigma_max is not None
    has_ell_bounds = log_ell_min is not None or log_ell_max is not None
    bounds: list[tuple[float | None, float | None]] | None = None
    if has_sigma_bounds or has_ell_bounds:
        bounds = [(None, None)] * x0.size
        # Layout: [L (N*r), log_d (N), log_sigma2 (N), log_ell (1)].
        sigma2_start = N * r + N
        if has_sigma_bounds:
            lo = 2.0 * log_sigma_min if log_sigma_min is not None else None
            hi = 2.0 * log_sigma_max if log_sigma_max is not None else None
            for j in range(sigma2_start, sigma2_start + N):
                bounds[j] = (lo, hi)
        if has_ell_bounds:
            bounds[sigma2_start + N] = (log_ell_min, log_ell_max)

    # Relaxed line-search options: tighter tolerances kill us with "ABNORMAL:
    # line search failed" when the iterate is pinned against a bound, where
    # the projected gradient has tiny magnitude. Looser ftol/gtol + bigger
    # maxls give the line search more rope.
    base_options = {
        "ftol": 1e-9,
        "gtol": 1e-6,
        "maxcor": 20,
        "maxls": 50,
    }

    def _clip_to_bounds(x: np.ndarray) -> np.ndarray:
        if bounds is None:
            return x
        x = x.copy()
        for j, (lo, hi) in enumerate(bounds):
            if lo is not None:
                x[j] = max(x[j], lo + 1e-6)
            if hi is not None:
                x[j] = min(x[j], hi - 1e-6)
        return x

    perturb_rng = np.random.default_rng(seed + 1)
    x_current = x0.copy()
    iters_used = 0
    max_restarts = 3
    res = None
    for attempt in range(max_restarts + 1):
        remaining = max(1, n_iter - iters_used)
        res = scipy.optimize.minimize(
            loss_and_grad,
            x_current,
            jac=True,
            method="L-BFGS-B",
            callback=callback,
            bounds=bounds,
            options={**base_options, "maxiter": remaining},
        )
        iters_used = len(losses)
        if res.success or iters_used >= n_iter:
            break
        # status == 2 is L-BFGS-B's "ABNORMAL TERMINATION" (line search failed).
        if res.status != 2:
            break
        if verbose:
            print(f"  line search failed at iter {iters_used} ({str(res.message)[:80]}); "
                  f"perturbing and retrying ({attempt + 1}/{max_restarts})")
        x_current = _clip_to_bounds(res.x + 1e-3 * perturb_rng.standard_normal(res.x.shape))

    params_fit = unpack(jnp.asarray(res.x), N, r)
    sigma = sigma_from_params(params_fit)
    correlation = correlation_from_params(params_fit)
    noise_var = jnp.exp(params_fit["log_sigma2"])
    length_scale = float(jnp.exp(params_fit["log_ell"]))
    z_mean = posterior_mean(params_fit, Y_j)

    return {
        "sigma": np.asarray(sigma),
        "correlation": np.asarray(correlation),
        "noise_var": np.asarray(noise_var),
        "length_scale": length_scale,
        "losses": np.asarray(losses, dtype=np.float64),
        "posterior_mean": np.asarray(z_mean),
        "params_flat": np.asarray(res.x, dtype=np.float64),
        "L": np.asarray(params_fit["L"]),
        "d": np.asarray(jnp.exp(params_fit["log_d"])),
        "rank": r,
        "converged": bool(res.success),
        "n_iter": int(len(losses)),
        "scipy_status": int(res.status),
        "scipy_message": str(res.message),
        "scipy_nfev": int(res.nfev),
    }
