
import numpy as np
import statsmodels.api as sm
from .adjust_temporal_shift import adjust_temporal_shift

def physiology_regression_glm(dc, sd, ss_list_good, additional_regressors):
    """
    This function performs physiological Regression based
    on Short-channel and with additional Regressors of physiology.

    Args:
        dc (np.ndarray): Hemoglobin concentration - Time pnts x Channels x Hemoglobin.
        sd (dict): standard fNIRS SD structure.
        ss_list_good (list): List with good short channels.
        additional_regressors (np.ndarray): Physiological data to be regressed in
            addition to the short channels.

    Returns:
        tuple: A tuple containing:
            - filtered_dc (np.ndarray): The filtered concentration data after all regression.
            - stats (dict): A dictionary containing the regression statistics.
    """

    filtered_dc = np.zeros_like(dc)
    stats = {}

    for hb in range(2):
        for n_chan in range(dc.shape[1]):
            y = dc[:, n_chan, hb]
            x = np.array([]).reshape(len(y), 0)

            if ss_list_good:
                x_short = np.concatenate([dc[:, ss_list_good, 0], dc[:, ss_list_good, 1]], axis=1)
                # PCA for removing collinearity
                from sklearn.decomposition import PCA
                pca = PCA()
                x_short_pca = pca.fit_transform(x_short)
                x = np.concatenate([x, x_short_pca], axis=1)

            if additional_regressors.size > 0:
                max_lag = round(20 * sd['f'][0, 0])
                y, additional_regressors_s, shift_ad, coor_max = adjust_temporal_shift(y, additional_regressors, max_lag)

                if x.size > 0:
                    x = x[max_lag:-max_lag, :]

                x = np.concatenate([x, additional_regressors_s], axis=1)

            rlm_model = sm.RLM(y, x, M=sm.robust.norms.TukeyBiweight())
            rlm_results = rlm_model.fit()

            filtered_dc[:, n_chan, hb] = rlm_results.resid

            stats_dummy = {
                'resid': rlm_results.resid,
                'params': rlm_results.params,
                'bse': rlm_results.bse,
                'pvalues': rlm_results.pvalues,
            }

            if additional_regressors.size > 0:
                stats_dummy['shift_AD'] = shift_ad / sd['f'][0, 0]
                stats_dummy['coor_AD'] = coor_max

            if n_chan not in stats:
                stats[n_chan] = {}
            stats[n_chan][hb] = stats_dummy

    filtered_dc[:, :, 2] = filtered_dc[:, :, 0] + filtered_dc[:, :, 1]

    return filtered_dc, stats
