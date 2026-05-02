"""
fNIRS data analysis: spatio-temporal Whittle GP for connectivity estimation.
"""

import jax
jax.config.update("jax_enable_x64", True)

from .io import (
    load_snirf_data,
    load_lob_data,
    load_nirs_data,
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

from .whittle import (
    fit,
    neg_log_likelihood,
    posterior_mean,
    matern32_psd,
    sigma_from_params,
    correlation_from_params,
)

__version__ = "0.1.0"

__all__ = [
    # IO
    "load_snirf_data",
    "load_lob_data",
    "load_nirs_data",
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
    "neg_log_likelihood",
    "posterior_mean",
    "matern32_psd",
    "sigma_from_params",
    "correlation_from_params",
]
