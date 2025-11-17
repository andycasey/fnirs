import numpy as np
from scipy.io import loadmat
from fnirs.hmr_intensity_2_od import hmr_intensity_2_od
from astropy.timeseries import LombScargle
from astropy import units as u
import matplotlib.pyplot as plt

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



dod = hmr_intensity_2_od(d)
t = np.arange(dod.shape[0]) / sd["f"][0, 0]

minimum_heart_rate = 40 / (60 * u.s)
maximum_heart_rate = 160 / (60 * u.s)

frequency, power = LombScargle(t, dod[:, 101]).autopower(
    minimum_frequency=minimum_heart_rate.value,
    maximum_frequency=maximum_heart_rate.value
)

indices = np.random.choice(np.arange(516), 32, replace=False)

fig, axes = plt.subplots(2, 1, sharex=True)
for i in indices:
    si = 1407
    ls = LombScargle(t[:si], dod[:si, i])
    ls2 = LombScargle(t[si:], dod[si:, i])
    axes[0].plot(frequency, ls.power(frequency), label=f'Index {i}')
    axes[1].plot(frequency, ls2.power(frequency), label=f'Index {i}')


ss_list = np.array([8, 29, 52, 66, 75, 92, 112, 125]) - 1

fig, ax = plt.subplots()
for i, j in enumerate(ss_list):
    ax.plot(t, dod[:, j])
