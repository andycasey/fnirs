import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.special import sph_harm_y
from typing import List, Optional


def matern12_psd(freqs: jnp.ndarray, lengthscale: float, variance: float) -> jnp.ndarray:
    """
    Matérn-1/2 (exponential kernel) power spectral density.

    The Matérn-1/2 kernel is k(τ) = σ² exp(-|τ|/ℓ), and its 1D PSD is:
        S(f) = 2σ²ℓ / (1 + (2πfℓ)²)

    Args:
        freqs: frequencies (in cycles per unit time), shape (n,)
        lengthscale: ℓ, kernel lengthscale in same units as time axis
        variance: σ², kernel variance

    Returns:
        PSD values, shape (n,)
    """
    omega = 2 * jnp.pi * freqs
    return 2 * variance * lengthscale / (1 + (lengthscale * omega) ** 2)


#@partial(jax.jit, static_argnames=("max_spherical_degree", "n_fourier_components"))
def fit(
    t: jnp.array,
    θ: jnp.array,
    ϕ: jnp.array,
    Y: jnp.ndarray,
    max_spherical_degree: int,
    n_fourier_components: int,
    temporal_kernel: Optional[str] = None,
    kernel_lengthscale: float = 1.0,
    kernel_variance: float = 1.0,
):
    assert Y.shape[1] == len(t), "Y must have shape (n_channels, n_samples)"
    assert Y.shape[0] == len(θ) == len(ϕ), "Y must have shape (n_channels, n_samples)"
    ST, terms = create_spherical_harmonics_basis(
        θ, ϕ, max_degree=max_spherical_degree
    )
    if len(terms) > len(θ):
        print(
            f"Warning: number of spherical harmonics basis functions ({len(terms)}) exceeds number of channels ({len(θ)})."
        )

    args = ((len(t), ), (n_fourier_components, ))
    A = partial(fourier_matmat, *args)
    AT = partial(fourier_rmatmat, *args)

    ATA = gram_diagonal(*args)
    lhs = ST.T @ ST
    rhs = ((AT(Y) @ ST) / ATA[:, None]).T

    if temporal_kernel is None:
        XT, *_ = jnp.linalg.lstsq(lhs, rhs, rcond=None)
    elif temporal_kernel == "matern12":
        # Compute frequencies for each Fourier mode.
        # Modes have integer frequency indices; the actual frequency in cycles
        # per time unit depends on the sampling: the basis spans [0, 2π) over
        # n_samples points, so mode index k completes k cycles in n_samples.
        # With sampling interval dt = (t[-1] - t[0]) / (n_samples - 1), the
        # total duration T ≈ n_samples * dt, giving f_k = k / T.
        n_samples = len(t)
        modes = create_1d_fourier_modes(n_samples, n_fourier_components)
        mode_indices = modes[:, 0]  # integer frequency indices

        dt = (t[-1] - t[0]) / (n_samples - 1)
        duration = n_samples * dt
        freqs = mode_indices / duration  # cycles per time unit

        psd = matern12_psd(freqs, kernel_lengthscale, kernel_variance)
        lambdas = 1.0 / psd  # regularization per Fourier mode

        n_spatial = lhs.shape[0]
        eye = jnp.eye(n_spatial)

        # Solve per Fourier mode: (lhs + λ_k I) x_k = rhs_k
        def solve_one(k):
            return jnp.linalg.solve(lhs + lambdas[k] * eye, rhs[:, k])

        XT = jax.vmap(solve_one)(jnp.arange(n_fourier_components)).T
    else:
        raise ValueError(f"Unknown temporal kernel: {temporal_kernel!r}")

    @jax.jit
    def f(X):
        return (A(X) @ ST.T).T
    return (XT.T, f, A, ST, terms)



def create_spherical_harmonics_basis(θ, ϕ, max_degree):
    """
    Create spherical harmonics basis functions for given theta and phi.
    """
    bases = []
    indices = []
    for n in range(max_degree + 1):
        for m in range(-n, n + 1):
            Y = sph_harm_y(n, m, ϕ, θ)
            if m == 0:
                bases.append(Y.real)
            elif m > 0:
                bases.append(np.sqrt(2) * (-1)**m * Y.real)
            else:
                bases.append(np.sqrt(2) * (-1)**m * Y.imag)
            indices.append((n, m))

    return np.column_stack(bases), indices



def create_1d_fourier_modes(n_samples: int, n_modes: int) -> jnp.ndarray:
    """
    Create the mode indices for 1D real Fourier basis.

    Returns array of shape (n_modes, 2) where each row is [frequency, type]
    type: 0 = constant, 1 = cosine, 2 = sine
    """
    modes = []

    # Constant term
    if n_modes > 0:
        modes.append([0, 0])  # freq=0, type=constant

    # Add cosine/sine pairs
    freq = 1
    while len(modes) < n_modes:
        if len(modes) < n_modes:
            modes.append([freq, 1])  # cosine
        if len(modes) < n_modes:
            modes.append([freq, 2])  # sine
        freq += 1

    return jnp.array(modes[:n_modes])



def evaluate_1d_fourier_basis(x: jnp.ndarray, modes: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate 1D Fourier basis functions at points x.

    Args:
        x: sampling points, shape (n_samples,)
        modes: mode specification, shape (n_modes, 2)

    Returns:
        basis matrix of shape (n_samples, n_modes)
    """
    n_samples = x.shape[0]
    n_modes = modes.shape[0]

    # Vectorized evaluation
    freqs = modes[:, 0]  # shape (n_modes,)
    types = modes[:, 1]  # shape (n_modes,)

    # Broadcast: x is (n_samples, 1), freqs is (1, n_modes)
    x_expanded = x[:, None]  # (n_samples, 1)
    freqs_expanded = freqs[None, :]  # (1, n_modes)

    # Compute all frequency-point combinations
    phase = freqs_expanded * x_expanded  # (n_samples, n_modes)

    # Apply the appropriate function based on type
    basis = jnp.where(
        types == 0,
        1.0,  # constant
        jnp.where(types == 1, jnp.cos(phase), jnp.sin(phase)),  # cosine
    )  # sine

    return basis


def fourier_matvec(
    samples_per_dim: List[int], modes_per_dim: List[int], x: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute A @ x where A is an N-dimensional Fourier design matrix.

    Uses the separable structure: the N-D transform is a sequence of 1-D transforms.

    Args:
        samples_per_dim: number of samples in each dimension
        modes_per_dim: number of Fourier modes in each dimension
        x: coefficient vector, shape (prod(modes_per_dim),)

    Returns:
        result vector, shape (prod(samples_per_dim),)
    """
    n_dims = len(samples_per_dim)

    # Reshape x to tensor form: (modes_0, modes_1, ..., modes_{n_dims-1})
    x_tensor = x.reshape(modes_per_dim)

    # Create sampling grids for each dimension
    coords = []
    for i, n_samples in enumerate(samples_per_dim):
        coord = jnp.linspace(0, 2 * jnp.pi, n_samples, endpoint=False)
        coords.append(coord)

    # Apply separable transform: transform along each dimension sequentially
    result = x_tensor

    for dim in range(n_dims):
        # Get modes for this dimension
        modes = create_1d_fourier_modes(samples_per_dim[dim], modes_per_dim[dim])

        # Create 1D basis matrix for this dimension
        basis_1d = evaluate_1d_fourier_basis(coords[dim], modes)  # (n_samples, n_modes)

        # Apply transformation along this dimension
        # We need to contract along the current dimension
        # Move the dimension to be transformed to the last axis
        result = jnp.moveaxis(result, dim, -1)

        # Reshape for matrix multiplication: (..., modes_dim) -> (..., samples_dim)
        original_shape = result.shape
        result_2d = result.reshape(-1, original_shape[-1])  # (batch, modes_dim)

        # Apply 1D transform: (batch, modes) @ (modes, samples)^T = (batch, samples)
        result_2d = result_2d @ basis_1d.T

        # Reshape back and move dimension back to original position
        new_shape = original_shape[:-1] + (samples_per_dim[dim],)
        result = result_2d.reshape(new_shape)
        result = jnp.moveaxis(result, -1, dim)

    # Flatten to vector
    return result.flatten()


def fourier_rmatvec(
    samples_per_dim: List[int], modes_per_dim: List[int], y: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute A.T @ y where A is an N-dimensional Fourier design matrix.

    Args:
        samples_per_dim: number of samples in each dimension
        modes_per_dim: number of Fourier modes in each dimension
        y: input vector, shape (prod(samples_per_dim),)

    Returns:
        result vector, shape (prod(modes_per_dim),)
    """
    n_dims = len(samples_per_dim)

    # Reshape y to tensor form: (samples_0, samples_1, ..., samples_{n_dims-1})
    y_tensor = y.reshape(samples_per_dim)

    # Create sampling grids for each dimension
    coords = []
    for i, n_samples in enumerate(samples_per_dim):
        coord = jnp.linspace(0, 2 * jnp.pi, n_samples, endpoint=False)
        coords.append(coord)

    # Apply adjoint separable transform
    result = y_tensor

    for dim in range(n_dims):
        # Get modes for this dimension
        modes = create_1d_fourier_modes(samples_per_dim[dim], modes_per_dim[dim])

        # Create 1D basis matrix for this dimension
        basis_1d = evaluate_1d_fourier_basis(coords[dim], modes)  # (n_samples, n_modes)

        # Apply adjoint transformation along this dimension
        # Move the dimension to be transformed to the last axis
        result = jnp.moveaxis(result, dim, -1)

        # Reshape for matrix multiplication: (..., samples_dim) -> (..., modes_dim)
        original_shape = result.shape
        result_2d = result.reshape(-1, original_shape[-1])  # (batch, samples_dim)

        # Apply 1D adjoint transform: (batch, samples) @ (samples, modes) = (batch, modes)
        result_2d = result_2d @ basis_1d

        # Reshape back and move dimension back to original position
        new_shape = original_shape[:-1] + (modes_per_dim[dim],)
        result = result_2d.reshape(new_shape)
        result = jnp.moveaxis(result, -1, dim)

    # Flatten to vector
    return result.flatten()

#fourier_matmat = jax.jit(jax.vmap(fourier_matvec, in_axes=(None, None, 1)), static_argnums=(0, 1))
#fourier_rmatmat = jax.jit(jax.vmap(fourier_rmatvec, in_axes=(None, None, 0)), static_argnums=(0, 1))

@partial(jax.jit, static_argnums=(0, 1))
def fourier_matmat(
    samples_per_dim: List[int], modes_per_dim: List[int], X: jnp.ndarray
) -> jnp.ndarray:
    return jax.vmap(fourier_matvec, in_axes=(None, None, 1))(samples_per_dim, modes_per_dim, X).T

@partial(jax.jit, static_argnums=(0, 1))
def fourier_rmatmat(
    samples_per_dim: List[int], modes_per_dim: List[int], Y: jnp.ndarray
) -> jnp.ndarray:
    return jax.vmap(fourier_rmatvec, in_axes=(None, None, 0))(samples_per_dim, modes_per_dim, Y).T


@partial(jax.jit, static_argnums=(0, 1))
def gram_diagonal(samples_per_dim: List[int], modes_per_dim: List[int]) -> jnp.ndarray:
    """
    Compute the diagonal of A.T @ A where A is an N-dimensional Fourier design matrix.

    Uses the fact that for separable bases, the Gram matrix diagonal is the
    Kronecker product of 1D Gram matrix diagonals.

    Args:
        samples_per_dim: number of samples in each dimension
        modes_per_dim: number of Fourier modes in each dimension

    Returns:
        diagonal vector, shape (prod(modes_per_dim),)
    """

    n_dims = len(samples_per_dim)

    # Compute 1D Gram matrix diagonals for each dimension
    gram_diagonals_1d = []

    for dim in range(n_dims):
        n_samples = samples_per_dim[dim]
        n_modes = modes_per_dim[dim]

        # Create coordinate array for this dimension
        coord = jnp.linspace(0, 2 * jnp.pi, n_samples, endpoint=False)

        # Get modes for this dimension
        modes = create_1d_fourier_modes(n_samples, n_modes)

        # Compute 1D basis matrix
        basis_1d = evaluate_1d_fourier_basis(coord, modes)  # (n_samples, n_modes)

        # Compute diagonal of 1D Gram matrix
        gram_diag_1d = jnp.sum(basis_1d * basis_1d, axis=0)  # (n_modes,)
        gram_diagonals_1d.append(gram_diag_1d)

    # The N-D Gram matrix diagonal is the Kronecker product of 1D diagonals
    # For diagonals, Kronecker product becomes outer products
    gram_diagonal_nd = gram_diagonals_1d[0]

    for dim in range(1, n_dims):
        # Compute outer product with next dimension
        gram_diagonal_nd = jnp.outer(gram_diagonal_nd, gram_diagonals_1d[dim])
        gram_diagonal_nd = gram_diagonal_nd.flatten()

    return gram_diagonal_nd
