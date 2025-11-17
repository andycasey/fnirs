
import numpy as np
from .spline_correction import spline_correction
from .hmr_motion_correct_wavelet import hmr_motion_correct_wavelet

def hybrid_motion_correction(dod, sd):
    """
    Motion artifact correction with Spline interpolation followed by Wavelet decomposition.

    Args:
        dod (np.ndarray): Optical density.
        sd (dict): fNIRS structure commonly used on Homer.

    Returns:
        np.ndarray: Optical density after motion artifact correction.
    """

    # First perform Spline correction
    spline_threshold = 4.5
    dod_spline = spline_correction(dod, sd, spline_threshold)

    # Create MeasListAct (internal variable for Homer that indicates which
    # channels should be considered)
    sd['MeasListAct'] = np.ones(dod.shape[1])

    # Wavelet Parameter
    wavelet_parameter = 1.5

    # Perform Wavelet
    dod_corrected = hmr_motion_correct_wavelet(dod_spline, sd, wavelet_parameter)

    return dod_corrected
