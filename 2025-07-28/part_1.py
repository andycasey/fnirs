import numpy as np
from scipy.io import loadmat
from fnirs.mark_bad_channels import mark_bad_channels
from fnirs.hmr_intensity_2_od import hmr_intensity_2_od
from fnirs.hmr_od_2_conc import hmr_od_2_conc
from fnirs.hmr_band_pass_filt import hmr_band_pass_filt
from fnirs.remove_autocorrelation_dc import remove_autocorrelation_dc, remove_autocorrelation_dc_fnirs_course
from fnirs.compute_correlation_coefficient import compute_correlation_coefficient
from fnirs.plot_seed_based_sphere_style import plot_seed_based_sphere_style
from fnirs.plot_correlation_matrices import plot_correlation_matrix
from fnirs.hybrid_motion_correction import hybrid_motion_correction

# Load data from one participant
data = loadmat('rsFC-fnirs-course/Data_for_Part_I.mat')['data']

# Get Light Intensity, SD, and additional physiological measurements
d = data['d'][0, 0]
sd = data['SD'][0, 0][0,0]
# Convert SD to a dictionary
sd = { k: sd[k] for k in sd.dtype.names }

phys_data = np.concatenate([
    data['Phys'][0, 0]['MAP_d'][0, 0],
    data['Phys'][0, 0]['HR_d'][0, 0],
    data['Phys'][0, 0]['CapData'][0, 0]
], axis=1)

# List of short channels for the used probe
ss_list = [8, 29, 52, 66, 75, 92, 112, 125]

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
    if "MeasListAct" not in SD or reset_flag == 1:    
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
        np.ones((d.shape[1], 1)),
        (-10, 1e7),
        snr_threshold,
        (0, 100),
        0
    )
    # TODO: en_prune_channels is wrong, it is marking everything as a bad channel
    return np.where(SD["MeasListActAuto"] == 0)[0]



# fNIRS measurements made every 3.9 Hz
t_seconds = np.arange(d.shape[0]) / 3.9  # 


# Find channels with low SNR
bad_channels = mark_bad_channels(d, sd)

# Compute Optical Density
dod = hmr_intensity_2_od(d)

"""
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from tinygp import kernels, GaussianProcess
import flax.linen as nn
from flax.linen.initializers import zeros

import optax



class GPModule(nn.Module):
    @nn.compact
    def __call__(self, x, y, t):
        mean = self.param("mean", zeros, ())
        log_jitter = self.param("log_jitter", zeros, ())
        log_sigma = self.param("log_sigma", zeros, ())
        log_tau = self.param("log_tau", zeros, ())
        kernel = (
            jnp.exp(2 * log_sigma)
        *   kernels.ExpSquared(2.5 * sd["f"][0,0] + jnp.exp(log_tau))
        )

        gp = GaussianProcess(kernel, x, diag=jnp.exp(log_jitter), mean=mean)
        log_prob, gp_cond = gp.condition(y, t)
        return -log_prob, gp_cond.loc
    



y = dod[:, 101]
t = true_t = np.arange(dod.shape[0])

def loss(params):
    return model.apply(params, t, y, true_t)[0]


model = GPModule()
params = model.init(jax.random.PRNGKey(0), t, y, true_t)
tx = optax.sgd(learning_rate=1e-4)
opt_state = tx.init(params)
loss_grad_fn = jax.jit(jax.value_and_grad(loss))

losses = []
for i in range(50):
    loss_val, grads = loss_grad_fn(params)
    print(i, loss_val)
    losses.append(loss_val)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

def build_gp(params):
    kernel = (
        jnp.exp(2 * params["log_sigma"])
    *   kernels.ExpSquared(2.5 * sd["f"][0,0] + jnp.exp(params["log_tau"]))
    )

    return GaussianProcess(kernel, true_t, diag=jnp.exp(params["log_jitter"]), mean=params["mean"])

gp = build_gp(params["params"])

fig, ax = plt.subplots()
ax.scatter(true_t, y, c="#666666", s=2)
ax.scatter(np.arange(dod.shape[0]), dod[:, 101], c="k", s=1)

T = np.arange(dod.shape[0])
#pred = model.apply(params, T, y, true_t)[1]
ax.plot(T, gp.predict(y, T), c="tab:blue")

"""

#dod2 = hybrid_motion_correction(dod, sd, spline_threshold=1.5)


# Let's load in the data for part 2




"""

fig, (ax_raw, ax_optical_density, ax_residual) = plt.subplots(3, 1, figsize=(10, 3))
plot_before_motion_correction, = ax_optical_density.plot(dod[:, 0], c='k')
plot_after_motion_correction, = ax_optical_density.plot(dod[:, 0], c="tab:blue",  ls=":")
plot_raw, = ax_raw.plot(dod[:, 0], c="tab:orange")
diff, = ax_residual.plot(dod2[:, 0] - dod[:, 0], c="#666666")
ax_raw.set_title('Raw Light Intensity')
ax_optical_density.set_title('Optical Density')

for i in range(0, dod.shape[1]):
    print(i)
    plot_before_motion_correction.set_ydata(dod[:, i])
    plot_after_motion_correction.set_ydata(dod2[:, i])
    plot_raw.set_ydata(d[:, i])
    # re-scale plot_raw y-axis
    ax_raw.set_ylim(np.min(d[:, i]) - 0.05 * np.ptp(d[:, i]), np.max(d[:, i]) + 0.05 * np.ptp(d[:, i]))

    this_data = np.hstack([dod[:, i], dod2[:, i]])
    offset = 0.05 * np.ptp(this_data)
    ax_optical_density.set_ylim(np.min(this_data) - offset, np.max(this_data) + offset)

    this_diff = dod2[:, i] - dod[:, i]
    diff.set_ydata(this_diff)
    ylim = np.abs(this_diff).max() + 0.1
    ax_residual.set_ylim(-ylim, ylim)
    fig.tight_layout()
    fig.savefig(f"figures/motion/motion_{i:02d}.png")
"""


# Compute Hemoglobin Concentration changes
dc = hmr_od_2_conc(dod, sd, [6, 6, 6, 6])

"""
si, ei = (200, -199)

fig, axes = plt.subplots(3, 1, figsize=(10, 3))
plot_hbo, = axes[0].plot(dc[si:ei, 0, 0], c="tab:red", lw=2)
plot_hbr, = axes[1].plot(dc[si:ei, 1, 0], c="tab:blue", lw=2)
plot_hbt, = axes[2].plot(dc[si:ei, 2, 0], c="tab:green", lw=2)
plot_comp_hbo, = axes[0].plot(dc[si:ei, 0, 0], c="k", lw=1)
plot_comp_hbr, = axes[1].plot(dc[si:ei, 1, 0], c="k", lw=1)
plot_comp_hbt, = axes[2].plot(dc[si:ei, 2, 0], c="k", lw=1)

for i in range(dc.shape[-1]):
    print(i)
    plot_hbo.set_ydata(dc[si:ei, 0, i])
    plot_hbr.set_ydata(dc[si:ei, 1, i])
    plot_hbt.set_ydata(dc[si:ei, 2, i])
    plot_comp_hbo.set_ydata(part2["dc"][:, i, 0])
    plot_comp_hbr.set_ydata(part2["dc"][:, i, 1])
    plot_comp_hbt.set_ydata(part2["dc"][:, i, 2])
    fig.tight_layout()
    fig.savefig(f"figures/dc/concentration_{i:02d}.png")
"""


# Permute dc
dc = np.transpose(dc, (0, 2, 1))

# Band-Pass Filter Hemoglobin concentrations
dc = hmr_band_pass_filt(dc, sd['f'][0, 0], 0.009, 0.08)

# Band-Pass Filter Additional Physiological Measurements
phys_data = hmr_band_pass_filt(phys_data, sd['f'][0, 0], 0.009, 0.08)

# Remove border effects
dc = dc[200:-199, :, :]
phys_data = phys_data[200:-199, :]



# Save Data
np.savez('Data_for_Part_II.npz', dc=dc, sd=sd, phys_data=phys_data, ss_list=ss_list, bad_channels=bad_channels)

# Remove Autocorrelation
pw_dc, sd = remove_autocorrelation_dc_fnirs_course(dc, sd)


"""
part2 = loadmat('rsFC-fnirs-course/Data_for_Part_II.mat')
si, ei = (200, -199)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(10, 3))
plot_hbo, = axes[0].plot(dc[:, 0, 0], c="tab:red", lw=2)
plot_hbr, = axes[1].plot(dc[:, 0, 1], c="tab:blue", lw=2)
plot_hbt, = axes[2].plot(dc[:, 0, 2], c="tab:green", lw=2)
plot_comp_hbo, = axes[0].plot(part2["dc"][:, 0, 0], c="k", lw=1)
plot_comp_hbr, = axes[1].plot(part2["dc"][:, 0, 1], c="k", lw=1)
plot_comp_hbt, = axes[2].plot(part2["dc"][:, 0, 2], c="k", lw=1)

for i in range(dc.shape[1]):
    print(i)
    plot_hbo.set_ydata(dc[:, i, 0])
    plot_hbr.set_ydata(dc[:, i, 1])
    plot_hbt.set_ydata(dc[:, i, 2])
    plot_comp_hbo.set_ydata(part2["dc"][:, i, 0])
    plot_comp_hbr.set_ydata(part2["dc"][:, i, 1])
    plot_comp_hbt.set_ydata(part2["dc"][:, i, 2])
    fig.tight_layout()
    fig.savefig(f"figures/dc/concentration_{i:02d}.png")
"""


#pw_dc = remove_autocorrelation_dc(dc, sd)

# for now just load the bandchannels from part 2 data
# I did that, and there are no bad channels
bad_channels = []

# Compute Pearson Correlation Coefficient
corr_matrix = compute_correlation_coefficient(pw_dc, bad_channels)

# Plot Sensory Motor Network
fig1 = plot_correlation_matrix(corr_matrix, "", bad_channels)
fig2 = plot_seed_based_sphere_style(corr_matrix[:, :, 2], bad_channels, [-1, 1])