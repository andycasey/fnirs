"""
fNIRS data analysis package for spatial-temporal modeling of hemodynamic responses.
"""

import jax
jax.config.update("jax_enable_x64", True)

from .io import (
    load_snirf_data,
    load_hemodynamic_data,
    load_lob_data,
    NIRSData,
    HemodynamicData,
    FNIRSChannel,
    HemoglobinChannel,
    ProbeInfo,
    FNIRSProbeConfig,
    StimInfo,
    DataType,
    ChromophoreType,
)

from .model import (
    fit,
    matern12_psd,
    create_spherical_harmonics_basis,
    create_1d_fourier_modes,
    evaluate_1d_fourier_basis,
    fourier_matmat,
    fourier_rmatmat,
    gram_diagonal,
)

from .spherical_projection import (
    project_fnirs_to_sphere,
    project_to_sphere,
    cartesian_to_spherical,
    spherical_to_cartesian,
    fit_sphere_to_head,
    visualize_spherical_projection,
)

# Optional mesh refinement (requires open3d)
try:
    from .skull_mesh_refiner import AdvancedSkullMeshRefinement
    _MESH_AVAILABLE = True
except ImportError:
    AdvancedSkullMeshRefinement = None
    _MESH_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    # IO
    "load_snirf_data",
    "load_hemodynamic_data",
    "load_lob_data",
    "NIRSData",
    "HemodynamicData",
    "FNIRSChannel",
    "HemoglobinChannel",
    "ProbeInfo",
    "FNIRSProbeConfig",
    "StimInfo",
    "DataType",
    "ChromophoreType",
    # Model
    "fit",
    "create_spherical_harmonics_basis",
    "create_1d_fourier_modes",
    "evaluate_1d_fourier_basis",
    "fourier_matmat",
    "fourier_rmatmat",
    "gram_diagonal",
    # Spherical projection
    "project_fnirs_to_sphere",
    "project_to_sphere",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "fit_sphere_to_head",
    "visualize_spherical_projection",
    # Mesh refinement
    "AdvancedSkullMeshRefinement",
]
