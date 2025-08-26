
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from typing import Optional, Sequence




def regress_physiological_signals(
    hemo_data,
    short_channel_indices: Optional[Sequence[int]] = (7, 28, 51, 65, 74, 91, 111, 124),
    remove_n_eigenvectors: Optional[int] = 2,
):
    """
    Regress physiological signals from fNIRS data using short-channel PCA and 
    linear regression.
    """

    short_channel_indices = np.array(short_channel_indices)

    # Do PCA on the short channels
    is_short_channel = np.zeros(len(hemo_data.channels), dtype=bool)
    is_short_channel[short_channel_indices] = True

    scaler = StandardScaler()
    y_reference = scaler.fit_transform(np.vstack([
        hemo_data.get_hbo_data()[:, is_short_channel].T,
        hemo_data.get_hbr_data()[:, is_short_channel].T
    ]))

    pca = PCA()
    pca.fit(y_reference)

    y_data = scaler.transform(hemo_data.get_hbo_data()[:, ~is_short_channel].T)
    target_scores = pca.transform(y_data)

    target_scores_modified = target_scores.copy()
    target_scores_modified[:, :remove_n_eigenvectors] = 0

    target_reconstructed = pca.inverse_transform(target_scores_modified)
    target_cleaned = scaler.inverse_transform(target_reconstructed)

    # Fit linear model given physiological data
    regression_model = LinearRegression()
    regression_model.fit(hemo_data.physiology_data, target_cleaned.T)
    target_cleaned -= regression_model.predict(hemo_data.physiology_data).T

    return target_cleaned
