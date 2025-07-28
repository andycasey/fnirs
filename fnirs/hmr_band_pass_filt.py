
import numpy as np
from scipy.signal import butter, filtfilt

def hmr_band_pass_filt(y, fs, hpf, lpf):
    """
    Perform a bandpass filter.

    Args:
        y (np.ndarray): Data to filter (#time points x #channels of data).
        fs (float): Sample frequency (Hz).
        hpf (float): High pass filter frequency (Hz).
        lpf (float): Low pass filter frequency (Hz).

    Returns:
        np.ndarray: Filtered data.
    """

    if not isinstance(fs, (int, float)):
        fs = 1 / (fs[1] - fs[0])

    # Low pass filter
    lpf_b, lpf_a = butter(3, lpf * 2 / fs, btype='low')
    ylpf = filtfilt(lpf_b, lpf_a, y, axis=0)

    # High pass filter
    hpf_b, hpf_a = butter(5, hpf * 2 / fs, btype='high')
    y2 = filtfilt(hpf_b, hpf_a, ylpf, axis=0)

    return y2
