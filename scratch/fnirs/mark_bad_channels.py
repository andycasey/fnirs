
import numpy as np
from .utils import detrend, en_prune_channels

def mark_bad_channels(d, sd, snr_threshold=8):
    """
    Find channels with a low signal-to-noise ratio based on average and
    and amplitude of the signals.

    Args:
        d (np.ndarray): Raw light intensity measurements.
        sd (dict): Structure common in Homer.
        snr_threshold (int, optional): Threshold for deciding between bad and good channels. Defaults to 8.

    Returns:
        list: List with low SNR channels.
    """

    # Remove long drifts from the data that can be misleading when computing
    # the quality of the channel
    baseline = np.mean(d, axis=0)
    d = detrend(d) + baseline

    SD = en_prune_channels(
        d.T, 
        sd, 
        np.ones((d.shape[0], 1)),
        (-10, 1e7),
        snr_threshold,
        (0, 100),
        0
    )

    raise a
    # Calculate SNR for each channel
    signal_mean = np.mean(d, axis=0)
    signal_std = np.std(d, axis=0)
    snr = np.abs(signal_mean / signal_std)

    # Find bad channels
    bad_channels = np.where(snr < snr_threshold)[0].tolist()

    return bad_channels
