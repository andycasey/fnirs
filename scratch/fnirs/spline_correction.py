
import numpy as np
from scipy.interpolate import CubicSpline
from .spline_correct_baseline import spline_correct_baseline

def spline_correction(x, sd, p):
    """
    Motion artifact correction using Spline interpolation.

    Args:
        x (np.ndarray): Time series. Rows are time points, columns are channels.
        sd (dict): SD structure, commonly used in nirs files.
        p (float): Free parameter, threshold to look for motion.

    Returns:
        np.ndarray: Corrected time series (free from motion artifacts).
    """

    k = round(2.5 * sd['f'][0, 0])
    w = (2 * k) + 1
    n = x.shape[0]

    ss = np.zeros_like(x)
    for n_chan in range(x.shape[1]):
        for tt in range(k, n - k):
            a = np.sum(x[tt - k:tt + k + 1, n_chan] ** 2)
            b = (np.sum(x[tt - k:tt + k + 1, n_chan])) ** 2
            b = (-1 / (2 * k + 1)) * b
            ss[tt, n_chan] = (1 / (2 * k + 1)) * np.sqrt(a + b)

    x_ok = x.copy()
    for n_chan in range(x.shape[1]):
        lst = np.where(ss[:, n_chan] < np.mean(ss[k:n - k, n_chan]) + p * np.std(ss[k:n - k, n_chan]))[0]
        ss[lst, n_chan] = 0

        segments = []
        in_artifact = False
        start_idx = 0
        for i in range(n):
            if ss[i, n_chan] > 0 and not in_artifact:
                start_idx = i
                in_artifact = True
            elif ss[i, n_chan] == 0 and in_artifact:
                segments.append((start_idx, i - 1))
                in_artifact = False
        if in_artifact:
            segments.append((start_idx, n - 1))

        for seg_start, seg_end in segments:
            if (seg_end - seg_start + 1) <= 3:
                x_ok[seg_start:seg_end + 1, n_chan] = 0
            else:
                t = np.linspace(0, (n - 1) * sd['f'][0, 0], n)
                cs = CubicSpline(t[seg_start:seg_end + 1], x[seg_start:seg_end + 1, n_chan])
                x_spline = cs(t[seg_start:seg_end + 1])
                x_ok[seg_start:seg_end + 1, n_chan] = x[seg_start:seg_end + 1, n_chan] - x_spline

        # Baseline correction for segments
        if len(segments) > 0:
            # Handle the segment before the first artifact
            if segments[0][0] > 0:
                x_ok[:segments[0][0], n_chan] = spline_correct_baseline(x_ok[:segments[0][0], n_chan], x_ok[:segments[0][0], n_chan], sd)

            for i in range(len(segments)):
                seg_start, seg_end = segments[i]
                if i < len(segments) - 1:
                    next_seg_start = segments[i+1][0]
                    x_ok[seg_end + 1:next_seg_start, n_chan] = spline_correct_baseline(x_ok[seg_end + 1:next_seg_start, n_chan], x_ok[seg_end + 1:next_seg_start, n_chan], sd)
                else:
                    # Handle the segment after the last artifact
                    x_ok[seg_end + 1:, n_chan] = spline_correct_baseline(x_ok[seg_end + 1:, n_chan], x_ok[seg_end + 1:, n_chan], sd)

    return x_ok
