import numpy as np
def detrend(data, deg=1):
    """
    Remove a polynomial trend from some data.

    This mimics MATLAB's `detrend` function.
    """
    data = np.atleast_2d(data)
    assert data.ndim <= 2, "Data must be 1D or 2D"

    A = np.vander(np.arange(data.shape[0]), deg + 1, increasing=True)
    detrended_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        X, *_ = np.linalg.lstsq(A, data[:, i], rcond=None)
        detrended_data[:, i] = data[:, i] - A @ X
    return detrended_data if data.ndim > 1 else detrended_data[:, 0]


def en_prune_channels(d, SD, t_inc, d_range, snr_thresh, sd_range, reset_flag):
    """
    Prune channels from the measurement list if their signal is too weak, too
    strong, or their standard deviation is too great. This function
    updates SD['MeasListAct'] based on whether data 'd' meets these conditions
    as specified by 'd_range' and 'snr_thresh'.
    
    Parameters:
    -----------
    d : numpy.ndarray
        Data (nTpts x nChannels)
    SD : dict
        Data structure describing the probe
    t_inc : numpy.ndarray
        Time inclusion array (1 for include, 0 for exclude)
    d_range : list or tuple
        If mean(d) < d_range[0] or > d_range[1] then it is excluded as an
        active channel
    snr_thresh : float
        If mean(d)/std(d) < snr_thresh then it is excluded as an
        active channel
    sd_range : list or tuple
        Will prune channels with a source-detector separation <
        sd_range[0] or > sd_range[1]
    reset_flag : int
        Reset previously pruned channels (automatic and manual)
    
    Returns:
    --------
    SD : dict
        Updated data structure describing the probe
    """
        
    # Find time points to include
    lst_inc = np.where(t_inc == 1)[0]
    d = d[lst_inc, :]
    
    # Initialize MeasListAct if it doesn't exist or if reset is requested
    if 'MeasListAct' not in SD or reset_flag == 1:
        SD['MeasListAct'] = np.ones(SD['MeasList'].shape[0], dtype=int)
    
    # Calculate mean and standard deviation
    d_mean = np.mean(d, axis=0)
    d_std = np.std(d, axis=0, ddof=0)  # ddof=0 for population std like MATLAB
    
    # Get number of wavelengths
    n_lambda = len(SD['Lambda'])
    
    # Find first wavelength measurements
    lst1 = np.where(SD['MeasList'][:, 3] == 1)[0]  # 0-based indexing
    
    # Initialize channel list
    chan_list = np.zeros((len(lst1), n_lambda), dtype=int)
    
    for ii in range(n_lambda):
        lst = []
        rho_sd = []
        
        for jj in range(len(lst1)):
            # Find matching measurements for current wavelength
            matching_idx = np.where(
                (SD['MeasList'][:, 0] == SD['MeasList'][lst1[jj], 0]) &
                (SD['MeasList'][:, 1] == SD['MeasList'][lst1[jj], 1]) &
                (SD['MeasList'][:, 3] == ii + 1)  # MATLAB uses 1-based indexing
            )[0]
            
            if len(matching_idx) > 0:
                lst.append(matching_idx[0])
            else:
                lst.append(-1)  # Invalid index
            
            # Calculate source-detector distance
            src_idx = SD['MeasList'][lst1[jj], 0] - 1  # Convert to 0-based
            det_idx = SD['MeasList'][lst1[jj], 1] - 1  # Convert to 0-based
            
            rho_sd.append(np.linalg.norm(
                SD['SrcPos'][src_idx, :] - SD['DetPos'][det_idx, :]
            ))
        
        # Convert to numpy arrays
        lst = np.array(lst)
        rho_sd = np.array(rho_sd)
        
        # Find valid indices (not -1)
        valid_idx = lst != -1
        
        # Apply criteria for channel selection
        if np.any(valid_idx):
            valid_lst = lst[valid_idx]
            valid_rho = rho_sd[valid_idx]
            valid_jj = np.where(valid_idx)[0]
            
            # Check all conditions
            condition = (
                (d_mean[valid_lst] > d_range[0]) &
                (d_mean[valid_lst] < d_range[1]) &
                ((d_mean[valid_lst] / d_std[valid_lst]) > snr_thresh) &
                (valid_rho >= sd_range[0]) &
                (valid_rho <= sd_range[1])
            )
            
            # Update channel list
            chan_list[valid_jj[condition], ii] = 1
    
    # Take minimum across wavelengths
    chan_list = np.min(chan_list, axis=1)
    
    # Update SD.MeasListAct
    SD['MeasListActAuto'] = np.ones(SD['MeasList'].shape[0], dtype=int)
    pruned_channels = np.where(chan_list == 0)[0]
    
    for idx in pruned_channels:
        # Find corresponding channels in full measurement list
        src = SD['MeasList'][lst1[idx], 0]
        det = SD['MeasList'][lst1[idx], 1]
        
        # Find all measurements for this source-detector pair
        matching_channels = np.where(
            (SD['MeasList'][:, 0] == src) &
            (SD['MeasList'][:, 1] == det)
        )[0]
        
        SD['MeasListActAuto'][matching_channels] = 0
        SD['MeasListAct'][matching_channels] = 0
    
    return SD