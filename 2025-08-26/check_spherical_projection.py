from spherical_projection import project_fnirs_to_sphere, create_spherical_harmonics_basis
from load_fnirs_part_ii import load_hemodynamic_data


hemo_data = load_hemodynamic_data("rsFC-fnirs-course/Data_for_Part_II.mat")

coords_3d = hemo_data.get_spatial_coordinates_3d()
sphere_result = project_fnirs_to_sphere(coords_3d, fit_method='least_squares')
θ, ϕ = sphere_result['theta'], sphere_result['phi']


Y = hemo_data.get_hbo_data()

max_degree = 4

import numpy as np
from scipy.special import sph_harm
def spherical_harmonics_basis(θ, ϕ, max_degree):
    """
    Create spherical harmonics basis functions for given theta and phi.
    """
    bases = []
    indices = []
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            Y = sph_harm(abs(m), l, ϕ, θ)
            if m == 0:
                bases.append(Y.real)
            elif m > 0:
                bases.append(np.sqrt(2) * (-1)**m * Y.real)
            else:
                bases.append(np.sqrt(2) * (-1)**m * Y.imag)
            indices.append((l, m))

    return np.column_stack(bases), indices



#basis = create_spherical_harmonics_basis(θ, ϕ, max_degree)
basis, terms = spherical_harmonics_basis(θ, ϕ, max_degree)

# get uniform sampling on a sphere
import numpy as np
n = 30
θ_uniform = np.linspace(0, np.pi, n)
ϕ_uniform = np.linspace(0, 2 * np.pi, n)
θ_uniform, ϕ_uniform = map(np.ravel, np.meshgrid(θ_uniform, ϕ_uniform))
uniform_basis, indices = spherical_harmonics_basis(θ_uniform, ϕ_uniform, max_degree)

                                                 
import matplotlib.pyplot as plt
fig, axes = plt.subplots(max_degree + 1, 2 * max_degree + 1, figsize=(8, 6))
for k in range(uniform_basis.shape[1]):
    i, j = indices[k]
    ax = axes[i, j + max_degree]
    ax.scatter(θ_uniform, ϕ_uniform, c=uniform_basis[:, k], cmap='RdBu_r', s=1)
    ax.set_title(f'Y_{indices[k][0]}^{indices[k][1]}')

