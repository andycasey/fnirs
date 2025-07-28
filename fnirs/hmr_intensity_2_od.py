
import numpy as np

def hmr_intensity_2_od(d):
    """
    Converts intensity (raw data) to optical density.

    Args:
        d (np.ndarray): Intensity data (#time points x #data channels).

    Returns:
        np.ndarray: The change in optical density.
    """
    assert d.shape[0] > d.shape[1], "I don't believe that you have more channels than time points"

    # convert to dod
    dm = np.mean(np.abs(d), axis=0)
    nt_pts = d.shape[0]
    dod = -np.log(np.abs(d) / (np.ones((nt_pts, 1)) * dm))

    if np.any(d <= 0):
        print('WARNING: Some data points in d are zero or negative.')

    return dod
