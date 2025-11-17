import numpy as np

from scipy.special import sph_harm_y
from spherical_projection import project_fnirs_to_sphere, create_spherical_harmonics_basis, cartesian_to_spherical
from load_fnirs_part_ii import load_hemodynamic_data

# Question: Better to use hbo concentration, or hbr concentration?


hemo_data = load_hemodynamic_data("../rsFC-fnirs-course/Data_for_Part_II.mat")


# Do PCA on the short channels
ss_list = np.array([8, 29, 52, 66, 75, 92, 112, 125]) - 1 # Convert to 0-based indexing

is_short_channel = np.zeros(len(hemo_data.channels), dtype=bool)
is_short_channel[ss_list] = True

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

y_reference = scaler.fit_transform(np.vstack([
    hemo_data.get_hbo_data()[:, is_short_channel].T,
    hemo_data.get_hbr_data()[:, is_short_channel].T
]))

pca = PCA()
pca.fit(y_reference)

y_data = scaler.transform(hemo_data.get_hbo_data()[:, ~is_short_channel].T)
target_scores = pca.transform(y_data)

n_pca_remove = 2
target_scores_modified = target_scores.copy()
target_scores_modified[:, :n_pca_remove] = 0

target_reconstructed = pca.inverse_transform(target_scores_modified)
target_cleaned = scaler.inverse_transform(target_reconstructed)

# Fit linear model given physiological data
from sklearn.linear_model import LinearRegression

regression_model = LinearRegression()
regression_model.fit(hemo_data.physiology_data, target_cleaned.T)
target_cleaned -= regression_model.predict(hemo_data.physiology_data).T

coords_3d = hemo_data.get_spatial_coordinates_3d()[~is_short_channel]

sphere_result = project_fnirs_to_sphere(coords_3d, fit_method='least_squares')
θ, ϕ = sphere_result['theta'], sphere_result['phi']


# Do a temporal basis as well, where we will use real-valued fourier basis.

# first, create a fourier design matrix for real-valued signals
def create_fourier_basis(t, n_terms):
    """
    Create a real-valued Fourier basis matrix.
    
    Parameters:
        t (np.ndarray): Time vector.
        n_terms (int): Number of Fourier terms.
        
    Returns:
        np.ndarray: Fourier basis matrix.
    """
    n = len(t)
    basis = np.ones((n, 2 * n_terms + 1))
    for i in range(1, n_terms):
        basis[:, 2*i + 1] = np.cos(2 * np.pi * i * t)
        basis[:, 2*i + 2] = np.sin(2 * np.pi * i * t)
    return basis


def create_spherical_harmonics_basis(θ, ϕ, max_degree):
    """
    Create spherical harmonics basis functions for given theta and phi.
    """
    bases = []
    indices = []
    for n in range(max_degree + 1):
        for m in range(-n, n + 1):
            Y = sph_harm_y(n, m, ϕ, θ)
            if m == 0:
                bases.append(Y.real)
            elif m > 0:
                bases.append(np.sqrt(2) * (-1)**m * Y.real)
            else:
                bases.append(np.sqrt(2) * (-1)**m * Y.imag)
            indices.append((n, m))

    return np.column_stack(bases), indices


max_spherical_degree = 4 
max_fourier_terms = 64
# sph=6 / fourier=64 with (1000, None) is expensive!

spherical_basis, terms = create_spherical_harmonics_basis(θ, ϕ, max_degree=max_spherical_degree)

# ignore the m=0, l=0 term
spherical_basis = spherical_basis[:, 1:]

n_channels = target_cleaned.shape[0]

si, ei = (0, None)
time = hemo_data.time[si:ei]
# Create the Fourier basis for the time vector
fourier_basis = create_fourier_basis(time / time[-1], n_terms=max_fourier_terms)

A_spatial_temporal = np.kron(fourier_basis, spherical_basis)

A = A_spatial_temporal

Y = target_cleaned.T[si:ei].flatten()
scale = 1e6

Y *= scale

assert A.shape[0] == Y.size

X, *extras = np.linalg.lstsq(A, Y, rcond=None)

Y_predicted = A @ X

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(Y, Y_predicted, s=1, alpha=0.1)
limits = np.hstack([ax.get_xlim(), ax.get_ylim()])
limits = np.min(limits), np.max(limits)
ax.plot(limits, limits, color='k', ls='--')
ax.set_xlim(limits)
ax.set_ylim(limits)

print(f"R^2: {np.corrcoef(Y, Y_predicted)[0, 1]**2:.3f}")


Y_predicted = Y_predicted.reshape(time.size, -1)
Y = Y.reshape(time.size, -1)

np.random.seed(42)
indices = np.random.choice(Y.shape[1], size=10, replace=False)
fig, ax = plt.subplots(figsize=(10, 10))
for i, index in enumerate(indices):
    ax.plot(time, Y[:, index]/10 + i, c='k', label='Data' if i == 0 else None)
    ax.plot(time, Y_predicted[:, index]/10 + i, c='tab:red', label="Model" if i == 0 else None)
    ax.text(time[-1], i, f"Channel {index + 1}", va='center', ha='left')
ax.set_xlim(0, time[-1] * 1.15)
ax.set_xlabel('Time (s)')
ax.set_ylabel("Channel signals (scaled and offset)")
ax.legend()
fig.savefig("20250811_channel_residuals_example.png", dpi=300, bbox_inches='tight')



abs_residuals = np.abs(Y - Y_predicted)

# Plot sum of absolute residuals on the sphere
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(121)
ax.plot(time, abs_residuals.sum(axis=1) / abs_residuals.shape[1], c='k')
ax.set_xlabel('Time (s)')
ax.set_ylabel(f'Mean of absolute residuals per channel ({1/scale:.1e})')


ax = fig.add_subplot(122, projection='3d')
scat = ax.scatter(
    coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
    c=abs_residuals.sum(axis=0) / abs_residuals.shape[0],
    cmap='viridis', 
    s=50
)
plt.colorbar(scat, ax=ax, label=f'Mean of absolute residuals per second ({1/scale:.1e})')
for i in range(4):
    fig.tight_layout()
fig.savefig("20250811_spatial_and_temporal_residuals.png", dpi=300, bbox_inches='tight')



fig, axes = plt.subplots(3)

xyz = hemo_data.get_spatial_coordinates_3d()

from skull_mesh_refiner import AdvancedSkullMeshRefinement

refiner = AdvancedSkullMeshRefinement(xyz)
fine_points, mesh = refiner.method_open3d_poisson(depth=2)


r, theta, phi = cartesian_to_spherical(fine_points, center=sphere_result["sphere_center"])

uniform_spherical_basis, terms = create_spherical_harmonics_basis(theta, phi, max_degree=max_spherical_degree)
uniform_spherical_basis = uniform_spherical_basis[:, 1:]  # ignore the m=0, l=0 term

Ai = np.kron(fourier_basis, uniform_spherical_basis)
Yi = Ai @ X
Yi = Yi.reshape((time.size, -1))

from matplotlib import cm

cmap = cm.RdBu
# 19, 17
seed = 17
    
np.random.seed(seed)
colors = ("tab:blue", "tab:red", "tab:purple", "tab:orange")
indices = np.random.choice(len(hemo_data.channels), size=4, replace=False)

fig = plt.figure(figsize=(15, 5))
ax_spec = fig.add_subplot(1, 3, (1, 2))

ax = fig.add_subplot(133, projection='3d')
collection = ax.scatter(
    fine_points[:, 0], fine_points[:, 1], fine_points[:, 2],
    c=Yi[0], cmap="viridis", vmin=-2.5, vmax=2.5
)
#ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='k', s=1, alpha=0.5)
cbar = plt.colorbar(collection, ax=ax)



lines = []
data_lines = []
for i, index in enumerate(indices):
    lines.append(ax_spec.plot(time, np.nan * np.ones_like(time), c=colors[i]))
    data_lines.append(ax_spec.plot(time, Y[:, index] / np.max(np.abs(Y[:, index])) + i, c="k", drawstyle="steps-mid"))
    ax.scatter(xyz[index, 0], xyz[index, 1], xyz[index, 2],
            c=colors[i], s=50, marker="s")


from matplotlib.animation import FuncAnimation
def animate(t):
    """Update function for the animation."""
    collection.set_array(Yi[t])
    ax.set_title(f"Time: {time[t]:.2f} s")
    for i, line in enumerate(lines):
        index = indices[i]
        d = Y_predicted[:, index] / np.max(np.abs(Y_predicted[:, index])) + i
        d[int(t):] = np.nan
        line[0].set_ydata(d)
        line[0].set_xdata(time)
    ax_spec.set_xlim(0, time[t])
    
    return collection,    

ax.set_xlabel("x [cm]")
ax.set_ylabel("y [cm]")
ax.set_zlabel("z [cm]")
ax_spec.set_xlabel("Time (s)")
ax_spec.set_ylabel("Scaled signal")
ax_spec.set_ylim(-1, len(indices))
interval = np.ptp(hemo_data.time) / hemo_data.time.size

fig.tight_layout()

ani = FuncAnimation(fig, animate, frames=hemo_data.time.size, interval=interval)
ani.save('model.mp4', writer='ffmpeg')
