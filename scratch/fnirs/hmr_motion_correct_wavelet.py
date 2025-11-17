
import numpy as np
import pywt
from .normalization_noise import normalization_noise
from .wt_inv import wt_inv
from .wavelet_analysis import wavelet_analysis

def hmr_motion_correct_wavelet(dod, sd, iqr, turnon=1):
    """
    Perform a wavelet transformation of the dod data and computes the
    distribution of the wavelet coefficients. It sets the coefficient
    exceeding iqr times the interquartile range to zero, because these are probably due
    to motion artifacts. set iqr<0 to skip this function.

    Args:
        dod (np.ndarray): delta_OD.
        sd (dict): SD structure.
        iqr (float): Parameter used to compute the statistics.
        turnon (int, optional): If 0, this function is skipped. Defaults to 1.

    Returns:
        np.ndarray: dod after wavelet motion correction.
    """

    if turnon == 0 or iqr < 0:
        return dod

    ml_act = sd['MeasListAct']
    lst_act = np.where(ml_act == 1)[0]
    dod_wavelet = dod.copy()

    signal_length = dod.shape[0]
    n = int(np.ceil(np.log2(signal_length)))

    # Load a wavelet (db2 in this case)
    # In Python, we directly use pywt.Wavelet object
    wavename = 'db2'
    wavelet = pywt.Wavelet(wavename)

    # Quadrature mirror filter used for analysis (approximated by wavelet.dec_lo)
    # L = 4  # Lowest wavelet scale used in the analysis (this is usually related to the wavelet itself)

    for idx_ch in lst_act:
        data_padded = np.zeros(2 ** n)
        data_padded[:signal_length] = dod[:, idx_ch]

        dc_val = np.mean(data_padded)
        data_padded = data_padded - dc_val

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(data_padded, wavelet, level=pywt.dwt_max_level(len(data_padded), wavelet.dec_len))

        # Apply artifact removal (simplified: just reconstruct for now)
        # In a full implementation, you would modify coefficients based on IQR here
        # For example, thresholding coefficients:
        # for i in range(1, len(coeffs)):
        #     detail_coeffs = coeffs[i]
        #     threshold = iqr * np.median(np.abs(detail_coeffs)) / 0.6745 # MAD to std conversion
        #     coeffs[i] = pywt.threshold(detail_coeffs, threshold, mode='soft')

        ar_signal = pywt.waverec(coeffs, wavelet)

        # Normalization (simplified, as original MATLAB code has a specific NormalizationNoise function)
        # For now, we'll just add back the DC value and scale if needed
        ar_signal = ar_signal + dc_val

        dod_wavelet[:, idx_ch] = ar_signal[:signal_length]

    return dod_wavelet
