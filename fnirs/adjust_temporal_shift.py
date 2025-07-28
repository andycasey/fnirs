
import numpy as np

def adjust_temporal_shift(y, x, max_lag):
    """
    Adjusts the temporal shift of the regressors to maximize the correlation with the fNIRS signal.

    Args:
        y (np.ndarray): Signal to be filtered (regressed out).
        x (np.ndarray): Regressors.
        max_lag (int): Maximum allowed lag.

    Returns:
        tuple: A tuple containing:
            - y_new (np.ndarray): The new y vector.
            - x_new (np.ndarray): The shifted data with proper size.
            - shift (np.ndarray): The optimal shifts.
            - coor_max (np.ndarray): The maximum correlation values.
    """

    x_new = np.zeros_like(x)
    shift = np.zeros(x.shape[1])
    coor_max = np.zeros(x.shape[1])

    for n_add in range(x.shape[1]):
        corr_values = np.correlate(y, x[:, n_add], mode='full')
        lags = np.arange(-len(y) + 1, len(y))
        corr_values = corr_values[(lags >= -max_lag) & (lags <= max_lag)]
        lags = lags[(lags >= -max_lag) & (lags <= max_lag)]

        max_coor_value = np.max(np.abs(corr_values))
        index_lag_corr = np.argmax(np.abs(corr_values))

        shift[n_add] = lags[index_lag_corr]
        coor_max[n_add] = corr_values[index_lag_corr]

        x_new[:, n_add] = np.roll(x[:, n_add], int(shift[n_add]))

    y_new = y.copy()
    x_new = x_new[max_lag:-max_lag, :]
    y_new = y_new[max_lag:-max_lag]

    return y_new, x_new, shift, coor_max
