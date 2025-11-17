
import numpy as np

def make_on_filter(filter_type, par):
    """
    Generates Orthonormal QMF Filter for Wavelet Transform.

    Args:
        filter_type (str): Type of filter (e.g., 'db2').
        par (int): Parameter related to the support and vanishing moments.

    Returns:
        np.ndarray: Quadrature mirror filter.
    """

    if filter_type == 'db2':
        # Hardcoded db2 filter coefficients (from MATLAB's db2 wavelet)
        if par == 2:
            f = np.array([
                -0.12940952255092145, 0.22414386804187116, 0.6884590394543206,
                0.3420201049631971, -0.04560110450706494, -0.010626360009500936
            ])
        else:
            raise ValueError("Only Par=2 is implemented for 'db2' wavelet.")
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")

    return f / np.linalg.norm(f)
