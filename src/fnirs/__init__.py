"""
fNIRS data analysis: spatio-temporal Whittle GP for connectivity estimation.
"""

from .io import (
    load_snirf_data,
    load_hemodynamic_data,
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

from .whittle import (
    fit,
    neg_log_likelihood,
    posterior_mean,
    matern32_psd,
    sigma_from_params,
    correlation_from_params,
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
    "neg_log_likelihood",
    "posterior_mean",
    "matern32_psd",
    "sigma_from_params",
    "correlation_from_params",
    # Mesh refinement
    "AdvancedSkullMeshRefinement",
]
