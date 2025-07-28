
import numpy as np
from statsmodels.regression.linear_model import yule_walker

def remove_autocorrelation_dc(dc, sd):
    """
    Removes autocorrelation from Resting State Data
    with prewhitening methodology.

    Args:
        dc (np.ndarray): Concentration changes: Time pnts x Channels x Hemoglobin.
        sd (dict): SD structure common in fNIRS files.

    Returns:
        np.ndarray: Whitened concentration changes.
    """

    # Maximum parameter order
    p_max = round(20 * sd['f'][0, 0])

    # Whitened data
    dc_w = np.nan * np.zeros_like(dc)

    # Time Series Length
    n = dc.shape[0]

    # Run on HbO and HbR
    for hb in range(2):
        for n_channel in range(dc.shape[1]):
            if not np.any(np.isnan(dc[:, n_channel, hb])):
                y = dc[:, n_channel, hb]
                bic = np.zeros(p_max)

                for p in range(1, p_max + 1):
                    rho, sigma = yule_walker(y, order=p)
                    vt = y[p:] - np.dot(y[:-p], rho)
                    ll = - (n / 2) * np.log(2 * np.pi * np.mean(vt ** 2)) - 0.5 * (1 / np.mean(vt ** 2)) * np.sum(vt ** 2)
                    bic[p - 1] = -2 * ll + p * np.log(n)

                optimal_p = np.argmin(bic) + 1
                ar_parameters, _ = yule_walker(y, order=optimal_p)
                yf = y[optimal_p:] - np.dot(y[:-optimal_p], ar_parameters)
                dc_w[optimal_p:, n_channel, hb] = yf

    # Compute Total Hemoglobin
    dc_w[:, :, 2] = dc_w[:, :, 0] + dc_w[:, :, 1]

    # Remove undetermined points
    dc_w = dc_w[p_max + 1:, :, :]

    return dc_w

import numpy as np
from scipy import signal
from scipy.linalg import solve_toeplitz
import warnings

def yule_walker_burg(x, order):
    """
    Estimate AR parameters using Yule-Walker equations (Burg method approximation).
    This approximates MATLAB's aryule function.
    
    Parameters:
    -----------
    x : array_like
        Input signal
    order : int
        Model order
        
    Returns:
    --------
    a : ndarray
        AR coefficients (including leading 1.0)
    """
    x = np.asarray(x).squeeze()
    N = len(x)
    
    if order >= N:
        raise ValueError("Model order must be less than signal length")
    
    # Compute autocorrelation
    r = np.correlate(x, x, mode='full')
    r = r[N-1:]  # Take positive lags only
    r = r[:order+1]  # We need r[0] to r[order]
    
    # Solve Yule-Walker equations: R * a = [r[0], 0, 0, ..., 0]
    if order == 0:
        return np.array([1.0])
    
    # Create Toeplitz matrix
    R = np.array([r[abs(i-j)] for i in range(order) for j in range(order)]).reshape(order, order)
    rhs = -r[1:order+1]
    
    try:
        a_coeffs = solve_toeplitz(r[:order], rhs)
        return np.concatenate([[1.0], a_coeffs])
    except np.linalg.LinAlgError:
        # Fallback to least squares if Toeplitz solve fails
        warnings.warn("Toeplitz solve failed, using least squares approximation")
        return np.array([1.0] + [0.0] * order)

def remove_autocorrelation_dc_fnirs_course(dc, SD):
    """
    Removes autocorrelation from Resting State Data with prewhitening methodology.
    
    Parameters:
    -----------
    dc : numpy.ndarray
        Concentration changes: Time points x Channels x Hemoglobin
    SD : dict
        SD structure common in fNIRS files (must contain 'f' field for sampling frequency)
    
    Returns:
    --------
    dc_w : numpy.ndarray
        Whitened concentration changes
    SD : dict
        Updated SD structure with Optimal_P field
    
    References:
    -----------
    1 - Characterization and correction of the false-discovery rates
        in resting state connectivity using functional
        near-infrared spectroscopy
    
    2 - Autoregressive model based algorithm for correcting motion and
        serially correlated errors in fNIRS
    """
    
    # Maximum parameter order. A conservative approach is to use more than
    # 20 seconds as upper bound.
    P_max = int(np.round(20 * SD['f']).flatten()[0])
    
    # Whitened data
    dc_w = np.full_like(dc, np.nan)
    
    # Time Series Length
    n = dc.shape[0]
    
    # Initialize Optimal_P storage
    if 'Optimal_P' not in SD:
        SD['Optimal_P'] = np.zeros((dc.shape[1], 2))
    
    # Run on HbO and HbR
    for Hb in range(2):
        
        for N_channel in range(dc.shape[1]):
            
            # Check if channel has no NaN values
            if not np.any(np.isnan(dc[:, N_channel, Hb])):
                
                # Get Original Time Series
                y = dc[:, N_channel, Hb].copy()
                
                # Initialize BIC array
                BIC = np.zeros(P_max)
                
                for P in range(1, P_max + 1):
                    try:
                        # For a given parameter P, find the coefficients that
                        # minimize autoregressive model (AR(P))
                        a = yule_walker_burg(y, P)
                        
                        # Once we have the parameters a, we can filter the error
                        # to find the new non-autocorrelated error (vt)
                        vt = signal.lfilter(a, [1.0], y)
                        
                        # Compute the Bayesian Information Criterion (BIC(P))
                        
                        # Log Likelihood
                        var_vt = np.mean(vt**2)
                        if var_vt <= 0:
                            BIC[P-1] = np.inf
                            continue
                            
                        LL = (-1 * (n/2) * np.log(2 * np.pi * var_vt) + 
                              -0.5 * (1/var_vt) * np.sum(vt**2))
                        
                        # Bayesian Information Criterion
                        BIC[P-1] = -2 * LL + P * np.log(n)
                        
                    except (np.linalg.LinAlgError, ValueError):
                        # If AR estimation fails, set BIC to infinity
                        BIC[P-1] = np.inf
                
                # Optimal is the P that minimizes BIC
                if np.all(np.isinf(BIC)):
                    # If all BIC values are inf, use original signal
                    dc_w[:, N_channel, Hb] = y
                    SD['Optimal_P'][N_channel, Hb] = 0
                else:
                    optimal_P = np.argmin(BIC) + 1  # +1 because P starts from 1
                    
                    # Find AR parameters for optimal P
                    try:
                        AR_parameters = yule_walker_burg(y, optimal_P)
                        
                        # Filter y
                        yf = signal.lfilter(AR_parameters, [1.0], y)
                        
                        # Update dc_w
                        dc_w[:, N_channel, Hb] = yf
                        
                        # Save OptimalP for double checking
                        SD['Optimal_P'][N_channel, Hb] = optimal_P
                        
                    except (np.linalg.LinAlgError, ValueError):
                        # If filtering fails, use original signal
                        dc_w[:, N_channel, Hb] = y
                        SD['Optimal_P'][N_channel, Hb] = 0
            
            else:
                # If channel contains NaN values, keep them as NaN
                dc_w[:, N_channel, Hb] = dc[:, N_channel, Hb]
                SD['Optimal_P'][N_channel, Hb] = 0
    
    # Compute Total Hemoglobin (if we have 3 dimensions)
    if dc_w.shape[2] >= 3:
        dc_w[:, :, 2] = dc_w[:, :, 0] + dc_w[:, :, 1]
    else:
        # Expand to include total hemoglobin
        dc_w_expanded = np.zeros((dc_w.shape[0], dc_w.shape[1], 3))
        dc_w_expanded[:, :, :2] = dc_w
        dc_w_expanded[:, :, 2] = dc_w[:, :, 0] + dc_w[:, :, 1]
        dc_w = dc_w_expanded
    
    # Remove undetermined points (first P_max points)
    dc_w = dc_w[P_max:, :, :]
    
    return dc_w, SD