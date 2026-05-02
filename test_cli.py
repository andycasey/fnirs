"""Tests for the fnirs CLI and plotting functions."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fnirs.cli import app
from fnirs.plotting import (
    plot_harmonics_timeseries,
    plot_noise_variance,
    plot_spatial_snapshot,
)

runner = CliRunner()


def test_main_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fit" in result.output
    assert "plot" in result.output
    assert "explore" in result.output
    assert "gs" in result.output


def test_gs_help():
    result = runner.invoke(app, ["gs", "--help"])
    assert result.exit_code == 0
    assert "DATA" in result.output
    assert "OUTPUT" in result.output
    assert "--max-irls-iter" in result.output
    assert "--kernel-lengthscale" in result.output


def test_fit_help():
    result = runner.invoke(app, ["fit", "--help"])
    assert result.exit_code == 0
    assert "DATA" in result.output
    assert "OUTPUT" in result.output
    assert "--max-degree" in result.output
    assert "--estimate-noise" in result.output
    assert "--no-plots" in result.output


def test_plot_help():
    result = runner.invoke(app, ["plot", "--help"])
    assert result.exit_code == 0
    assert "MODEL_DIR" in result.output
    assert "--data" not in result.output


def test_interact_help():
    result = runner.invoke(app, ["interact", "--help"])
    assert result.exit_code == 0
    assert "MODEL_DIR" in result.output


def test_explore_help():
    result = runner.invoke(app, ["explore", "--help"])
    assert result.exit_code == 0
    assert "DATA" in result.output
    assert "--output" in result.output
    assert "--chromophore" in result.output


def _make_synthetic_model(tmp_path: Path, with_noise: bool = False):
    """Create a synthetic model output for testing."""
    n_spatial = 4  # l=0 (1) + l=1 (3)
    n_freq = 33
    n_timepoints = 64
    rng = np.random.default_rng(0)

    X_freq = rng.standard_normal((n_spatial, n_freq)) + 1j * rng.standard_normal((n_spatial, n_freq))
    ST = rng.standard_normal((10, n_spatial))
    terms_l = np.array([0, 1, 1, 1])
    terms_m = np.array([0, -1, 0, 1])

    save_dict = dict(
        X_freq_real=X_freq.real,
        X_freq_imag=X_freq.imag,
        ST=ST,
        n_timepoints=n_timepoints,
        terms_l=terms_l,
        terms_m=terms_m,
    )
    if with_noise:
        save_dict["noise_variance"] = rng.uniform(0.01, 0.1, size=10)

    np.savez(tmp_path / "model.npz", **save_dict)

    config = dict(
        data="synthetic",
        max_degree=1,
        n_fourier=None,
        chromophore="hbo",
        temporal_kernel=None,
        kernel_lengthscale=1.0,
        kernel_variance=1.0,
        estimate_noise=with_noise,
        max_irls_iter=20,
        irls_tol=1e-4,
        seed=42,
    )
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    return X_freq, ST, terms_l, terms_m, n_timepoints


def test_plot_harmonics_timeseries(tmp_path):
    X_freq, ST, terms_l, terms_m, n_timepoints = _make_synthetic_model(tmp_path)
    terms = list(zip(terms_l.tolist(), terms_m.tolist()))
    out = tmp_path / "harmonics.png"
    plot_harmonics_timeseries(X_freq, terms, n_timepoints, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_noise_variance(tmp_path):
    noise = np.random.uniform(0.01, 0.1, size=10)
    out = tmp_path / "noise.png"
    plot_noise_variance(noise, out)
    assert out.exists()


def test_plot_spatial_snapshot(tmp_path):
    X_freq, ST, terms_l, terms_m, n_timepoints = _make_synthetic_model(tmp_path)
    terms = list(zip(terms_l.tolist(), terms_m.tolist()))
    config = {"max_degree": 1}
    out = tmp_path / "spatial.png"
    plot_spatial_snapshot(X_freq, terms, n_timepoints, config, out, n_snapshots=2)
    assert out.exists()


def test_plot_subcommand_with_synthetic(tmp_path):
    """Test the plot subcommand end-to-end with synthetic data."""
    _make_synthetic_model(tmp_path, with_noise=True)
    result = runner.invoke(app, ["plot", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "figures" / "harmonics_timeseries.png").exists()
    assert (tmp_path / "figures" / "noise_variance.png").exists()
    # spatial_snapshot temporarily disabled in _run_plots
    assert not (tmp_path / "figures" / "spatial_snapshot.png").exists()


_LOB_SESSION1 = Path(__file__).parent / "data" / "Session1.lob"


@pytest.mark.skipif(not _LOB_SESSION1.exists(), reason="Session1.lob not present")
def test_load_lob_data_session1():
    """Load .lob file and verify SNIRF-style API contract."""
    from fnirs.io import load_lob_data, NIRSData

    nd = load_lob_data(_LOB_SESSION1)
    assert isinstance(nd, NIRSData)
    assert nd.time_series.shape[0] == nd.time.shape[0]
    assert nd.time_series.shape[1] == len(nd.channels)
    hbo = nd.get_channels_by_data_type_label("HbO")
    hbr = nd.get_channels_by_data_type_label("HbR")
    assert len(hbo) > 0 and len(hbr) > 0
    assert len(hbo) == len(hbr)  # one of each per source-detector pair
    coords3d = nd.get_spatial_coordinates_3d()
    assert coords3d is not None and coords3d.shape == (len(nd.channels), 3)
    coords2d = nd.get_spatial_coordinates_2d()
    assert coords2d.shape == (len(nd.channels), 2)
    # Some channels should be flagged short-separation (rig has 8mm pairs)
    assert any(ch.is_short_separation for ch in nd.channels)


@pytest.mark.skipif(not _LOB_SESSION1.exists(), reason="Session1.lob not present")
def test_fit_lob_smoke(tmp_path):
    """`fnirs fit` runs end-to-end on a .lob file."""
    out = tmp_path / "lob_fit"
    result = runner.invoke(
        app,
        ["fit", str(_LOB_SESSION1), str(out), "--no-plots", "--max-degree", "2"],
    )
    assert result.exit_code == 0, result.output
    assert "Selected" in result.output and "HbO" in result.output
    assert (out / "model.npz").exists()
    assert (out / "config.json").exists()
