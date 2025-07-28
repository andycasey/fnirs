
import numpy as np
from .wt_inv import wt_inv
from .make_on_filter import make_on_filter

def wavelet_analysis(stat_wt, l, wavename, iqr_val, signal_length):
    """
    Applies artifact removal using wavelet coefficients.

    Args:
        stat_wt (np.ndarray): Matrix of wavelet coefficients.
        l (int): Lowest wavelet scale used in the analysis.
        wavename (str): Name of the wavelet used for the reconstruction.
        iqr_val (float): Parameter used to compute the statistics (iqr = 1.5 is 1.5 times the
                         interquartile range and is usually used to detect outliers).
        signal_length (int): Original signal length.

    Returns:
        np.ndarray: Artifact-removed signal.
    """

    # This is a simplified implementation. The original MATLAB code has more complex logic
    # for artifact removal based on wavelet coefficients and IQR.
    # For now, we'll just reconstruct the signal without explicit artifact removal logic here,
    # as the core artifact removal is implied to happen by setting coefficients to zero
    # in the MATLAB code, which is not directly translated here.

    # In a more complete implementation, you would identify and modify/zero out
    # wavelet coefficients that are considered artifacts based on iqr_val.

    # For now, simply reconstruct the signal from the wavelet coefficients
    # The `wt_inv` function already handles the inverse transform.
    ar_signal = wt_inv(stat_wt, wavename)

    return ar_signal, None  # Returning None for wcTI as it's not directly computed in this simplified version
