"""Preprocessing for raw fNIRS intensity data.

Pipeline:
    intensity ──▶ TDDR motion correction (optional)
              ──▶ bandpass filter (optional)
              ──▶ wavelet spike removal (optional)
              ──▶ optical density (mandatory)
              ──▶ HbO / HbR via Modified Beer-Lambert Law (mandatory)
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import butter, filtfilt

from ._extinctions import get_extinctions


# ---------------------------------------------------------------------------
# Per-channel motion / despiking / filtering
# ---------------------------------------------------------------------------

def tddr(signal: np.ndarray, fs: float, max_iter: int = 50, tune: float = 4.685) -> np.ndarray:
    """Temporal Derivative Distribution Repair (Fishburn et al., NeuroImage 2019).

    Removes step-like motion artifacts by IRLS-suppressing outlier temporal
    derivatives in the slow-trend component of the signal, then re-integrating.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("tddr expects a 1D signal")

    signal_mean = float(signal.mean())
    s = signal - signal_mean

    # Slow component: lowpass at 0.5 Hz.
    nyq = 0.5 * fs
    fc = min(0.5, 0.99 * nyq)
    b, a = butter(3, fc / nyq, btype="lowpass")
    s_low = filtfilt(b, a, s)
    s_high = s - s_low

    # First-order derivative; prepend keeps length identical and starts at 0.
    deriv = np.diff(s_low, prepend=s_low[0])

    # Tukey biweight IRLS for the typical (non-motion) derivative scale.
    eps = 1e-12
    mu = 0.0
    for _ in range(max_iter):
        sigma = 1.4826 * np.median(np.abs(deriv - np.median(deriv))) + eps
        r = (deriv - mu) / (tune * sigma)
        w = np.where(np.abs(r) < 1, (1 - r ** 2) ** 2, 0.0)
        new_mu = (w * deriv).sum() / (w.sum() + eps)
        if abs(new_mu - mu) < eps * (abs(mu) + 1):
            mu = new_mu
            break
        mu = new_mu

    # Final weights — outlier derivatives get zeroed → motion stays in place.
    sigma = 1.4826 * np.median(np.abs(deriv - np.median(deriv))) + eps
    r = (deriv - mu) / (tune * sigma)
    w = np.where(np.abs(r) < 1, (1 - r ** 2) ** 2, 0.0)
    deriv_clean = w * deriv

    # Integrate and recombine.
    s_low_clean = np.cumsum(deriv_clean) + s_low[0]
    return s_low_clean + s_high + signal_mean


def bandpass_filter(
    signal: np.ndarray, fs: float, low_hz: float | None, high_hz: float | None,
    order: int = 3,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass on a (time,) or (channels, time) signal."""
    signal = np.asarray(signal, dtype=np.float64)
    nyq = 0.5 * fs
    if low_hz is not None and high_hz is not None:
        b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    elif low_hz is not None:
        b, a = butter(order, low_hz / nyq, btype="high")
    elif high_hz is not None:
        b, a = butter(order, high_hz / nyq, btype="low")
    else:
        return signal
    axis = -1 if signal.ndim > 1 else 0
    return filtfilt(b, a, signal, axis=axis)


def wavelet_despike(
    signal: np.ndarray, iqr_threshold: float = 1.5, wavelet: str = "sym5",
) -> np.ndarray:
    """Wavelet-based spike removal using MAD-based thresholding on detail
    coefficients. More robust than the IQR-rule version because the spikes
    themselves don't bias the threshold estimate. ``iqr_threshold`` is kept as
    the parameter name for backwards compatibility, but its effective rule is
    now: zero any detail coefficient with |coeff − median| > k · 1.4826 · MAD
    where k = ``iqr_threshold`` × 2 (so k=3 at the default 1.5 — Tukey's
    "extreme-outlier" rule mapped to robust σ).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("wavelet_despike expects a 1D signal")
    n = len(signal)
    pad_n = int(2 ** np.ceil(np.log2(max(n, 2))))
    padded = np.zeros(pad_n)
    padded[:n] = signal - signal.mean()
    dc = float(signal.mean())

    wav = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(pad_n, wav.dec_len)
    coeffs = pywt.wavedec(padded, wav, level=max_level)

    k = float(iqr_threshold) * 2.0  # interpret as #(robust σ)
    thresholded = [coeffs[0]]
    for d in coeffs[1:]:
        med = np.median(d)
        mad = np.median(np.abs(d - med)) * 1.4826
        if mad <= 0:
            thresholded.append(d)
            continue
        thresholded.append(np.where(np.abs(d - med) > k * mad, 0.0, d))

    rec = pywt.waverec(thresholded, wav)
    return rec[:n] + dc


def hampel_filter(
    signal: np.ndarray, window_size: int = 7, k: float = 4.0,
) -> np.ndarray:
    """Hampel outlier filter: at each sample, replace |y_t − median(window)| > k·MAD
    with the local median. Simple, aggressive sample-wise spike removal.

    `window_size` is the half-width of the window in samples (full width = 2w+1).
    `k` is in robust-σ units (k=3 ⇒ ~ 99 % under Gaussian); 4 is a sensible default
    for fNIRS where occasional larger transients shouldn't blow up the threshold.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("hampel_filter expects a 1D signal")
    n = len(signal)
    out = signal.copy()
    w = int(window_size)
    if w < 1:
        return out
    # Sliding-window median + MAD using stride tricks.
    pad = np.pad(signal, (w, w), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1)
    med = np.median(windows, axis=-1)
    mad = np.median(np.abs(windows - med[:, None]), axis=-1) * 1.4826
    over = (mad > 0) & (np.abs(signal - med) > k * mad)
    out[over] = med[over]
    return out


# ---------------------------------------------------------------------------
# Intensity → OD → concentration
# ---------------------------------------------------------------------------

def intensity_to_od(intensity: np.ndarray) -> np.ndarray:
    """Optical density: OD = −log(|I| / mean|I|), per channel.

    `intensity` shape: (n_channels, n_timepoints) or (n_timepoints, n_channels) —
    the function operates on the LAST axis (per-row mean across time).
    """
    intensity = np.asarray(intensity, dtype=np.float64)
    abs_i = np.abs(intensity)
    mean_per_ch = abs_i.mean(axis=-1, keepdims=True)
    if (mean_per_ch == 0).any():
        raise ValueError("Some channels have zero mean absolute intensity; cannot compute OD.")
    return -np.log(abs_i / mean_per_ch)


def od_to_concentration(
    od: np.ndarray,
    source_indices: np.ndarray,
    detector_indices: np.ndarray,
    wavelength_indices: np.ndarray,
    wavelengths: np.ndarray,
    distances: np.ndarray,
    ppf: tuple[float, float] = (6.0, 6.0),
    spatial_unit: str = "mm",
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Modified Beer-Lambert Law: convert per-channel ΔOD to ΔHbO and ΔHbR.

    Each unique (source, detector) pair is expected to have data at exactly
    `len(wavelengths)` wavelengths (= 2 in the standard setup). Returns:
        hbo : (n_pairs, n_t) ΔHbO
        hbr : (n_pairs, n_t) ΔHbR
        pair_index : list of (source_idx, detector_idx) — 1-based, len n_pairs
        pair_distance : (n_pairs,) source-detector distances
    """
    od = np.asarray(od, dtype=np.float64)
    if od.shape[0] != len(source_indices):
        raise ValueError(f"od has {od.shape[0]} channels, indices have {len(source_indices)}")
    if len(wavelengths) != 2:
        raise ValueError(f"od_to_concentration expects 2 wavelengths, got {len(wavelengths)}")
    if len(ppf) != len(wavelengths):
        raise ValueError(f"ppf must match number of wavelengths ({len(wavelengths)}); got {len(ppf)}")
    n_t = od.shape[1]

    # Extinction coefficients for HbO / HbR at the two wavelengths.
    e_full = get_extinctions(np.asarray(wavelengths, dtype=np.float64))[:, :2]
    if spatial_unit == "mm":
        e_full = e_full / 10.0  # /cm → /mm
    elif spatial_unit != "cm":
        raise ValueError(f"unsupported spatial_unit {spatial_unit!r} (expected 'mm' or 'cm')")
    einv = np.linalg.solve(e_full.T @ e_full, np.eye(e_full.shape[1])) @ e_full.T  # (2, n_wav)

    # Group channels by (source, detector). For each pair, stack ΔOD across
    # wavelengths in the order expected by the inverse extinction matrix.
    pair_to_channel: dict[tuple[int, int], dict[int, int]] = {}
    for ch in range(len(source_indices)):
        key = (int(source_indices[ch]), int(detector_indices[ch]))
        wi = int(wavelength_indices[ch])
        pair_to_channel.setdefault(key, {})[wi] = ch
    pair_keys = sorted(pair_to_channel.keys())

    # Map source-detector index -> channel distance.
    distances = np.asarray(distances, dtype=np.float64)
    pair_distance = np.zeros(len(pair_keys))
    for p_idx, (s, d) in enumerate(pair_keys):
        any_ch = next(iter(pair_to_channel[(s, d)].values()))
        pair_distance[p_idx] = distances[any_ch]

    # Solve MBLL per pair.
    hbo = np.zeros((len(pair_keys), n_t))
    hbr = np.zeros((len(pair_keys), n_t))
    for p_idx, key in enumerate(pair_keys):
        per_wav = pair_to_channel[key]
        if set(per_wav.keys()) != set(range(1, len(wavelengths) + 1)):
            raise ValueError(
                f"pair (S{key[0]}, D{key[1]}) is missing wavelengths; "
                f"got {sorted(per_wav.keys())}, expected {list(range(1, len(wavelengths) + 1))}"
            )
        rho = pair_distance[p_idx]
        # Stack ΔOD by wavelength index (1-based → 0-based row), normalise by rho * ppf.
        od_stack = np.stack(
            [od[per_wav[wi]] / (rho * ppf[wi - 1]) for wi in range(1, len(wavelengths) + 1)],
            axis=0,
        )  # (n_wav, n_t)
        conc = einv @ od_stack  # (2, n_t) — rows: HbO, HbR
        hbo[p_idx] = conc[0]
        hbr[p_idx] = conc[1]

    return hbo, hbr, pair_keys, pair_distance


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def preprocess_optical_density(
    od: np.ndarray,
    fs: float,
    *,
    apply_tddr: bool = True,
    apply_bandpass: bool = True,
    apply_wavelet: bool = True,
    apply_hampel: bool = True,
    bandpass_low_hz: float = 0.009,
    bandpass_high_hz: float = 0.08,
    wavelet_iqr_threshold: float = 1.5,
    wavelet: str = "sym5",
    hampel_window: int = 7,
    hampel_k: float = 4.0,
) -> np.ndarray:
    """Per-channel OD preprocessing: TDDR → wavelet despike → bandpass.

    Operates on optical density (after intensity_to_od). This is the standard
    Homer/MNE-NIRS ordering: convert to OD first so the channel mean is well-
    defined (any subsequent bandpass that removes DC doesn't break the
    normalisation), then motion-correct, despike, and filter in OD space.

    `od` shape: (n_channels, n_timepoints). Returns the same shape.
    """
    od = np.asarray(od, dtype=np.float64)
    if od.ndim != 2:
        raise ValueError("od must be 2D (n_channels, n_timepoints)")
    out = od.copy()

    if apply_tddr:
        for i in range(out.shape[0]):
            out[i] = tddr(out[i], fs)

    if apply_hampel:
        for i in range(out.shape[0]):
            out[i] = hampel_filter(out[i], window_size=hampel_window, k=hampel_k)

    if apply_wavelet:
        for i in range(out.shape[0]):
            out[i] = wavelet_despike(out[i], iqr_threshold=wavelet_iqr_threshold, wavelet=wavelet)

    if apply_bandpass:
        out = bandpass_filter(out, fs, bandpass_low_hz, bandpass_high_hz)

    return out


def preprocess_intensity(
    intensity: np.ndarray,
    fs: float,
    **kwargs,
) -> np.ndarray:
    """Deprecated: preprocess raw intensity directly. Use intensity_to_od + preprocess_optical_density instead."""
    od = intensity_to_od(intensity)
    return preprocess_optical_density(od, fs, **kwargs)
