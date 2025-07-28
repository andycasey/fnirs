
import numpy as np
from sklearn.decomposition import PCA

def perform_pca_regression(dc, sd, n_sv, bad_channels):
    """
    Perform PCA regression to remove systemic physiology.

    Args:
        dc (np.ndarray): Hemoglobin concentration changes.
        sd (dict): standard fNIRS SD structure.
        n_sv (list): Number of components to be removed.
        bad_channels (list): List of channels with low SNR.

    Returns:
        np.ndarray: PCA-filtered hemoglobin concentration changes.
    """

    # List of Short Channels
    ss_list = [8, 29, 52, 66, 75, 92, 112, 125]

    # Exclude channels from Correlation Matrix
    exclude_channels = np.unique(np.concatenate((ss_list, bad_channels)))

    # Create MeasListAct
    sd['MeasListAct'] = np.ones(dc.shape[1])
    sd['MeasListAct'][exclude_channels] = 0

    # Perform PCA considering the whole time-series
    t_inc = np.ones(dc.shape[0])

    # Perform PCA
    dc_pca = np.zeros_like(dc)
    for hb in range(2):
        pca = PCA(n_components=n_sv[hb])
        dc_hb = dc[:, sd['MeasListAct'] == 1, hb]
        pca.fit(dc_hb)
        dc_pca[:, sd['MeasListAct'] == 1, hb] = dc_hb - pca.inverse_transform(pca.transform(dc_hb))

    dc_pca[:, :, 2] = dc_pca[:, :, 0] + dc_pca[:, :, 1]

    return dc_pca
