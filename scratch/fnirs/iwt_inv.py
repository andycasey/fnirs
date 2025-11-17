
import numpy as np
import pywt

def iwt_inv(stat_wt, wavename):
    """
    Perform a discrete wavelet inverse transform using the wavelet coefficients
    found in wp. It is shift invariant.

    Args:
        stat_wt (np.ndarray): matrix of wavelet coefficients (# of time points x # of levels+1).
        wavename (str): name of the wavelet used for the recontruction.

    Returns:
        np.ndarray: Reconstructed signal after the wavelet inverse transform.
    """

    n, d = stat_wt.shape
    d = d - 1

    wp = stat_wt
    pywt.dwtmode('per')

    approx = wp[:, 0].T  # approximation coefficients in the first column
    for d_ in range(d - 1, -1, -1):
        n_blocks = 2 ** d_
        l_blocks = n // n_blocks
        for b in range(2 ** d_):
            cd = wp[b * l_blocks:b * l_blocks + l_blocks // 2, d_ + 2].T
            cd_shift = wp[b * l_blocks + l_blocks // 2:b * l_blocks + l_blocks, d_ + 2].T
            ca = approx[b * l_blocks:b * l_blocks + l_blocks // 2]
            ca_shift = approx[b * l_blocks + l_blocks // 2:b * l_blocks + l_blocks]

            s1 = pywt.idwt(ca, cd, wavename)
            s_shift = pywt.idwt(ca_shift, cd_shift, wavename)
            s2 = np.roll(s_shift, -1)

            approx[b * l_blocks:b * l_blocks + l_blocks] = (s1 + s2) / 2

    return approx
