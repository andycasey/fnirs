
import numpy as np

def compute_correlation_coefficient(dc, bad_channels):
    """
    Compute Pearson Correlation for all hemoglobin.

    Args:
        dc (np.ndarray): Hemoglobin concentration - Time pnts x channel x hemoglobin.
        bad_channels (list): List of channels with low SNR.

    Returns:
        np.ndarray: Correlation coefficient matrix.
    """

    # List of Short Channels
    ss_list = np.array([8, 29, 52, 66, 75, 92, 112, 125])
    ss_list -= 1 # Convert to zero-based index 

    # Exclude channels from Correlation Matrix
    exclude_channels = np.unique(np.concatenate((ss_list, bad_channels))).astype(int)

    # Compute for HbO, HbR, and HbT
    correlation_coefficient = np.zeros((dc.shape[1], dc.shape[1], 3))
    for hb in range(3):
        correlation_coefficient[:, :, hb] = np.corrcoef(dc[:, :, hb], rowvar=False)

    # Assign "Exclude channels" as zeros
    correlation_coefficient[exclude_channels, :, :] = 0
    correlation_coefficient[:, exclude_channels, :] = 0

    return correlation_coefficient
