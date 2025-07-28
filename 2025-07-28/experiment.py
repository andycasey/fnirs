
import numpy as np
from scipy.io import loadmat
from fnirs.hmr_intensity_2_od import hmr_intensity_2_od
from astropy.timeseries import LombScargle
from astropy import units as u
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from fnirs.hmr_intensity_2_od import hmr_intensity_2_od
from fnirs.hmr_od_2_conc import hmr_od_2_conc
from fnirs.hmr_band_pass_filt import hmr_band_pass_filt
from fnirs.remove_autocorrelation_dc import remove_autocorrelation_dc, remove_autocorrelation_dc_fnirs_course


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



ss_list = np.array([8, 29, 52, 66, 75, 92, 112, 125]) - 1


dod = hmr_intensity_2_od(d)
dc, rhos = hmr_od_2_conc(dod, sd, [6, 6, 6, 6])

raise a

# some ones worth looking at:
# 128 has sharp jump in orange
# Permute dc
dc = np.transpose(dc, (0, 2, 1))

# Band-Pass Filter Hemoglobin concentrations
filtered_dc = hmr_band_pass_filt(dc, sd['f'][0, 0], 0.009, 0.08)


for i in range(dc.shape[1]):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(dc[:, i, 0], c="tab:blue")
    axes[1].plot(dc[:, i, 1], c="tab:orange")
    axes[0].plot(filtered_dc[:, i, 0], c="k")
    axes[1].plot(filtered_dc[:, i, 1], c="k")
    axes[0].set_title(f'Hemoglobin {i}')


raise a





# Band-Pass Filter Additional Physiological Measurements
phys_data = hmr_band_pass_filt(phys_data, sd['f'][0, 0], 0.009, 0.08)

A = np.hstack([dc[:, ss_list, 1], dc[:, ss_list, 0]])
pca = PCA(n_components=2)
moo = pca.fit_transform(A)

fig, ax = plt.subplots()
for i in range(moo.shape[1]):
    ax.plot(moo[:, i] / np.abs(moo[:, i]).max() + i)

X = np.zeros((A.shape[1], dc.shape[1]))
for i in range(dc.shape[1]):
    X[:, i] = np.linalg.lstsq(A, dc[:, i, 0])[0]


N = 10    
fig, axes = plt.subplots(N)
for i, ax in enumerate(axes):
    y = dc[:, i, 0]
    y2 = dc[:, i, 0] - A @ X[:, i]

    ax.plot(y / np.ptp(y), c='k')
    ax.plot(y2 / np.ptp(y2), c="tab:red")


# Some thoughts on the data generating process:
# - Each channel has its own noise characteristics
# - All channels are affected by the same physiological signals, but to different extents
# - The physiological signals can lag or lead the hemodynamic response
# - The noise is not independent across channels, but rather correlated due to shared physiological signals

# Converting from dod -> dc involves computing the extinction in each wavelength,
# the distance between source and detector position, and partial pathlength factor.

# It makes sense to do time-domain filtering on `dc`, but I could also see a reason
# to do it on `dod` before converting to `dc`.