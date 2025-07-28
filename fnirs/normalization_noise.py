
import numpy as np
from scipy.signal import convolve
from statsmodels.robust.scale import mad

def normalization_noise(y, qmf):
    """
    Estimates the noise level and normalizes the signal.

    Args:
        y (np.ndarray): Signal to normalize.
        qmf (np.ndarray): Quadrature mirror filter.

    Returns:
        tuple: A tuple containing:
            - y_norm (np.ndarray): Normalized signal.
            - coeff (float): 1/sigma_estimated.
    """

    c = convolve(y, qmf, mode='full')
    # Circular convolution: pad c to match length of y
    c = c[ : len(y)]

    # Downsample by 2
    y_downsampled = c[::2]

    median_abs_dev = mad(y_downsampled)

    if median_abs_dev != 0:
        y_norm = (1 / 1.4826) * y / median_abs_dev
        coeff = 1 / (1.4826 * median_abs_dev)
    else:
        y_norm = y
        coeff = 1

    return y_norm, coeff
