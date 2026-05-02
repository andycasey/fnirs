#!/usr/bin/env python3
"""
SNIRF format loader for fNIRS data.

This module provides functionality to load fNIRS data from SNIRF (Shared Near Infrared 
Spectroscopy Format) files. It includes dataclasses compatible with both SNIRF and MATLAB 
data formats for unified processing.

SNIRF Specification: https://github.com/fNIRS/snirf/blob/v1.1/snirf_specification.md
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union, Any
from enum import Enum
import numpy as np
import h5py
from pathlib import Path


class DataType(Enum):
    """SNIRF data types for measurements."""
    RAW = 99999           # Raw light intensity (A.U.)
    FLUORESCENCE = 50     # Fluorescence (A.U.)
    DOD = 1               # Change in optical density
    CONC_HBO = 2         # Oxygenated hemoglobin concentration (μM)
    CONC_HBR = 3         # Deoxygenated hemoglobin concentration (μM)
    CONC_HBT = 4         # Total hemoglobin concentration (μM)


class ChromophoreType(Enum):
    """Enumeration for hemoglobin chromophore types (for compatibility)."""
    HbO = 0  # Oxygenated hemoglobin
    HbR = 1  # Deoxygenated hemoglobin 
    HbT = 2  # Total hemoglobin


@dataclass
class MeasurementInfo:
    """Information about a single measurement channel."""
    source_index: int                    # 1-based source index
    detector_index: int                  # 1-based detector index  
    wavelength_index: int                # 1-based wavelength index
    data_type: int                       # SNIRF dataType value
    data_type_index: int                 # Index within data type
    data_type_label: str                 # Human-readable data type
    wavelength: float                    # Wavelength (nm)
    data_unit: Optional[str] = None      # Data unit (optional)
    module_index: Optional[int] = None   # Module index (optional)


@dataclass 
class ProbeInfo:
    """Probe configuration information."""
    source_positions_2d: np.ndarray      # Source positions in 2D (n_sources x 2)
    detector_positions_2d: np.ndarray    # Detector positions in 2D (n_detectors x 2)
    source_positions_3d: Optional[np.ndarray] = None  # 3D positions (n_sources x 3)
    detector_positions_3d: Optional[np.ndarray] = None  # 3D positions (n_detectors x 3)
    wavelengths: Optional[np.ndarray] = None     # Wavelengths (nm)
    frequency: Optional[float] = None            # Sampling frequency (Hz)
    time_unit: str = "s"                         # Time unit
    length_unit: str = "mm"                      # Length unit for positions
    coordinate_system: Optional[str] = None     # Coordinate system description
    landmark_positions_2d: Optional[np.ndarray] = None  # Landmark positions 2D
    landmark_positions_3d: Optional[np.ndarray] = None  # Landmark positions 3D
    landmark_labels: Optional[List[str]] = None  # Landmark labels


@dataclass
class StimInfo:
    """Stimulus/event information."""
    name: str                    # Stimulus name
    data: np.ndarray            # Stimulus timing data (n_events x 3: onset, duration, amplitude)
    
    @property
    def onsets(self) -> np.ndarray:
        """Event onset times."""
        return self.data[:, 0]
    
    @property
    def durations(self) -> np.ndarray:
        """Event durations."""  
        return self.data[:, 1] if self.data.shape[1] > 1 else np.zeros(len(self.data))
    
    @property
    def amplitudes(self) -> np.ndarray:
        """Event amplitudes."""
        return self.data[:, 2] if self.data.shape[1] > 2 else np.ones(len(self.data))


@dataclass
class FNIRSChannel:
    """Represents a single fNIRS measurement channel."""
    channel_idx: int                     # Channel index (0-based)
    measurement_info: MeasurementInfo    # Measurement metadata
    source_pos_2d: np.ndarray           # 2D source position
    detector_pos_2d: np.ndarray         # 2D detector position  
    source_pos_3d: Optional[np.ndarray] = None  # 3D source position
    detector_pos_3d: Optional[np.ndarray] = None  # 3D detector position
    distance: Optional[float] = None     # Source-detector separation
    midpoint_2d: Optional[np.ndarray] = None   # Channel midpoint 2D
    midpoint_3d: Optional[np.ndarray] = None   # Channel midpoint 3D
    
    def __post_init__(self):
        """Calculate derived properties."""
        # Calculate 2D midpoint
        self.midpoint_2d = (self.source_pos_2d + self.detector_pos_2d) / 2
        
        # Calculate 3D midpoint and distance if 3D positions available
        if self.source_pos_3d is not None and self.detector_pos_3d is not None:
            self.midpoint_3d = (self.source_pos_3d + self.detector_pos_3d) / 2
            self.distance = np.linalg.norm(self.source_pos_3d - self.detector_pos_3d)
        elif self.distance is None:
            # Fallback to 2D distance calculation
            self.distance = np.linalg.norm(self.source_pos_2d - self.detector_pos_2d)
    
    @property
    def is_short_separation(self, threshold: float = 15.0) -> bool:
        """Check if channel is short separation (default threshold: 15mm)."""
        return self.distance < threshold if self.distance else False


@dataclass
class NIRSData:
    """Main dataclass for SNIRF fNIRS data."""
    
    # Core data
    time_series: np.ndarray              # Time series data (n_timepoints x n_channels)
    time: np.ndarray                     # Time vector (seconds)
    channels: List[FNIRSChannel]        # Channel information
    probe: ProbeInfo                     # Probe configuration
    
    # Metadata
    format_version: str                  # SNIRF format version
    metadata: Dict[str, Any] = field(default_factory=dict)  # Metadata tags
    
    # Optional data
    aux_data: Optional[np.ndarray] = None      # Auxiliary data
    stimulus: Optional[List[StimInfo]] = None   # Stimulus information
    
    def __post_init__(self):
        """Validate data consistency."""
        if self.time_series.shape[0] != len(self.time):
            raise ValueError("Time series and time vector must have same length")
        if self.time_series.shape[1] != len(self.channels):
            raise ValueError("Time series columns must match number of channels")
    
    @property
    def sampling_frequency(self) -> float:
        """Calculate sampling frequency from time vector."""
        if len(self.time) < 2:
            return self.probe.frequency if self.probe.frequency else 1.0
        dt = np.mean(np.diff(self.time))
        return 1.0 / dt
    
    @property
    def n_timepoints(self) -> int:
        """Number of time points."""
        return len(self.time)
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return len(self.channels)
    
    def get_channels_by_wavelength(self, wavelength: float) -> List[FNIRSChannel]:
        """Get channels for specific wavelength."""
        return [ch for ch in self.channels 
                if ch.measurement_info.wavelength == wavelength]
    
    def get_channels_by_data_type(self, data_type: Union[DataType, int]) -> List[FNIRSChannel]:
        """Get channels for specific data type."""
        if isinstance(data_type, DataType):
            data_type = data_type.value
        return [ch for ch in self.channels 
                if ch.measurement_info.data_type == data_type]
    
    def get_channels_by_data_type_label(self, label: str) -> List[FNIRSChannel]:
        """Get channels for specific data type label."""
        return [ch for ch in self.channels 
                if ch.measurement_info.data_type_label.lower() == label.lower()]
    
    def get_raw_intensity_channels(self) -> List[FNIRSChannel]:
        """Get raw intensity measurement channels."""
        return self.get_channels_by_data_type(DataType.RAW)
    
    def get_concentration_channels(self) -> List[FNIRSChannel]:
        """Get concentration measurement channels."""
        conc_types = [DataType.CONC_HBO, DataType.CONC_HBR, DataType.CONC_HBT]
        conc_channels = []
        for dtype in conc_types:
            conc_channels.extend(self.get_channels_by_data_type(dtype))
        return conc_channels
    
    def get_spatial_coordinates_2d(self) -> np.ndarray:
        """Get 2D channel midpoint coordinates."""
        return np.array([ch.midpoint_2d for ch in self.channels])
    
    def get_spatial_coordinates_3d(self) -> Optional[np.ndarray]:
        """Get 3D channel midpoint coordinates if available."""
        if all(ch.midpoint_3d is not None for ch in self.channels):
            return np.array([ch.midpoint_3d for ch in self.channels])
        return None
    
    def get_source_detector_pairs(self) -> List[Tuple[int, int]]:
        """Get unique source-detector pairs."""
        pairs = set()
        for ch in self.channels:
            pairs.add((ch.measurement_info.source_index, ch.measurement_info.detector_index))
        return sorted(list(pairs))
    
    def get_wavelengths(self) -> np.ndarray:
        """Get unique wavelengths."""
        wavelengths = set(ch.measurement_info.wavelength for ch in self.channels)
        return np.array(sorted(list(wavelengths)))
    
    def create_spatial_mesh(self, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Create regular spatial mesh for interpolation."""
        coords = self.get_spatial_coordinates_2d()
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        x_grid = np.linspace(x_min, x_max, resolution)
        y_grid = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        return X, Y


def _decode_string_array(data: Any) -> str:
    """Decode HDF5 string data to Python string."""
    if isinstance(data, (bytes, np.bytes_)):
        return data.decode('utf-8')
    elif isinstance(data, np.ndarray):
        if data.dtype.kind in ['S', 'U']:  # String or Unicode
            if data.size == 1:
                item = data.item()
                if isinstance(item, bytes):
                    return item.decode('utf-8')
                return str(item)
            else:
                return str(data[0]) if len(data) > 0 else ""
    return str(data)


def _safe_decode_array(data: Any) -> Union[str, np.ndarray, float, int]:
    """Safely decode various HDF5 data types."""
    if hasattr(data, 'shape') and data.shape == ():
        # Scalar dataset
        val = data[()]
        if isinstance(val, (np.ndarray, list)):
            if len(val) > 0:
                val = val[0]
        if isinstance(val, bytes):
            return val.decode('utf-8')
        return val
    elif hasattr(data, 'shape') and len(data.shape) == 1 and data.shape[0] == 1:
        # Single element array
        val = data[0]
        if isinstance(val, bytes):
            return val.decode('utf-8')
        return val
    elif isinstance(data, (bytes, np.bytes_)):
        return data.decode('utf-8')
    else:
        return data


def load_snirf_data(snirf_file_path: Union[str, Path]) -> NIRSData:
    """
    Load fNIRS data from SNIRF format file.
    
    Parameters
    ----------
    snirf_file_path : str or Path
        Path to the SNIRF (.snirf) file
        
    Returns
    -------
    NIRSData
        Loaded SNIRF data structured for analysis
    """
    snirf_path = Path(snirf_file_path)
    if not snirf_path.exists():
        raise FileNotFoundError(f"SNIRF file not found: {snirf_path}")
    
    with h5py.File(snirf_path, 'r') as f:
        # Read format version
        format_version = _safe_decode_array(f['formatVersion'])
        if isinstance(format_version, np.ndarray):
            format_version = format_version[0] if len(format_version) > 0 else "unknown"
        format_version = str(format_version).strip('[]\'\"')
        
        # Access the nirs group (assumes single nirs group: nirs)
        nirs = f['nirs']
        
        # Load metadata
        metadata = {}
        if 'metaDataTags' in nirs:
            meta_group = nirs['metaDataTags']
            for key in meta_group.keys():
                try:
                    metadata[key] = _safe_decode_array(meta_group[key])
                except Exception as e:
                    print(f"Warning: Could not decode metadata key '{key}': {e}")
                    metadata[key] = None
        
        # Load probe information
        probe_group = nirs['probe']
        
        source_pos_2d = probe_group['sourcePos2D'][:]
        detector_pos_2d = probe_group['detectorPos2D'][:]
        wavelengths = probe_group['wavelengths'][:]
        
        # Optional 3D positions
        source_pos_3d = probe_group['sourcePos3D'][:] if 'sourcePos3D' in probe_group else None
        detector_pos_3d = probe_group['detectorPos3D'][:] if 'detectorPos3D' in probe_group else None
        
        # Extract units and other metadata
        length_unit = metadata.get('LengthUnit', 'mm')
        if isinstance(length_unit, (bytes, np.bytes_)):
            length_unit = length_unit.decode('utf-8')
        
        time_unit = metadata.get('TimeUnit', 's')
        if isinstance(time_unit, (bytes, np.bytes_)):
            time_unit = time_unit.decode('utf-8')
        
        probe = ProbeInfo(
            source_positions_2d=source_pos_2d,
            detector_positions_2d=detector_pos_2d,
            source_positions_3d=source_pos_3d,
            detector_positions_3d=detector_pos_3d,
            wavelengths=wavelengths,
            length_unit=length_unit,
            time_unit=time_unit
        )
        
        # Load data (assuming single data block: data1)
        data_group = nirs['data1']
        
        time_series = data_group['dataTimeSeries'][:]
        time = data_group['time'][:]
        
        # Parse measurement list
        ml_keys = [key for key in data_group.keys() if key.startswith('measurementList')]
        ml_keys.sort(key=lambda x: int(x.replace('measurementList', '')))
        
        channels = []
        for i, ml_key in enumerate(ml_keys):
            ml_group = data_group[ml_key]
            
            # SNIRF spec is loose about whether scalar fields are stored as
            # 1-element arrays or true scalar datasets; handle both.
            def _scalar(dset):
                arr = dset[()] if dset.shape == () else dset[0]
                return arr.item() if hasattr(arr, "item") else arr

            source_idx = int(_scalar(ml_group['sourceIndex']))
            detector_idx = int(_scalar(ml_group['detectorIndex']))
            wavelength_idx = int(_scalar(ml_group['wavelengthIndex']))
            data_type = int(_scalar(ml_group['dataType']))
            data_type_idx = int(_scalar(ml_group['dataTypeIndex']))
            data_type_label = _safe_decode_array(ml_group['dataTypeLabel'])

            # Get wavelength value (use first wavelength as fallback for non-CW data types).
            if 1 <= wavelength_idx <= len(wavelengths):
                wavelength = wavelengths[wavelength_idx - 1]
            else:
                wavelength = float("nan")

            # Optional fields
            data_unit = _safe_decode_array(ml_group['dataUnit']) if 'dataUnit' in ml_group else None
            module_idx = int(_scalar(ml_group['moduleIndex'])) if 'moduleIndex' in ml_group else None
            
            measurement_info = MeasurementInfo(
                source_index=source_idx,
                detector_index=detector_idx,
                wavelength_index=wavelength_idx,
                data_type=data_type,
                data_type_index=data_type_idx,
                data_type_label=data_type_label,
                wavelength=wavelength,
                data_unit=data_unit,
                module_index=module_idx
            )
            
            # Get source/detector positions (convert to 0-based indexing)
            src_pos_2d = source_pos_2d[source_idx - 1]
            det_pos_2d = detector_pos_2d[detector_idx - 1]
            src_pos_3d = source_pos_3d[source_idx - 1] if source_pos_3d is not None else None
            det_pos_3d = detector_pos_3d[detector_idx - 1] if detector_pos_3d is not None else None
            
            channel = FNIRSChannel(
                channel_idx=i,
                measurement_info=measurement_info,
                source_pos_2d=src_pos_2d,
                detector_pos_2d=det_pos_2d,
                source_pos_3d=src_pos_3d,
                detector_pos_3d=det_pos_3d
            )
            channels.append(channel)
        
        # Load stimulus information (optional)
        stimulus_info = []
        stim_keys = [key for key in nirs.keys() if key.startswith('stim')]
        for stim_key in stim_keys:
            stim_group = nirs[stim_key]
            stim_name = _safe_decode_array(stim_group['name']) if 'name' in stim_group else stim_key
            stim_data = stim_group['data'][:]
            
            stim_info = StimInfo(name=stim_name, data=stim_data)
            stimulus_info.append(stim_info)
        
        # Create main data object
        nirs_data = NIRSData(
            time_series=time_series,
            time=time,
            channels=channels,
            probe=probe,
            format_version=format_version,
            metadata=metadata,
            stimulus=stimulus_info if stimulus_info else None
        )
        
        return nirs_data


def _read_lob_mcos_arr(lob_path: Union[str, Path]) -> np.ndarray:
    """Extract the cw_nirs MCOS property cell array from a .lob (MATLAB v5)
    file. The MCOS bytes appear inside __function_workspace__; we replay
    them through the v5 reader."""
    import io as _io
    from scipy.io import loadmat
    from scipy.io.matlab._mio5 import MatFile5Reader

    raw = loadmat(str(lob_path), squeeze_me=False, struct_as_record=True)
    if "__function_workspace__" not in raw:
        raise ValueError(f"{lob_path}: not an MCOS-style .lob file")
    ws = raw["__function_workspace__"].tobytes()
    fake = b"\x00" * 124 + b"\x00\x00IM" + ws
    reader = MatFile5Reader(_io.BytesIO(fake))
    reader.byte_order = "<"
    reader.initialize_read()
    reader.mat_stream.seek(128 + 8)
    hdr, _ = reader.read_var_header()
    top = reader.read_var_array(hdr, process=True)
    return top["MCOS"][0, 0][0]["arr"]


def load_lob_data(lob_file_path: Union[str, Path]) -> NIRSData:
    """Load a .lob (cw_nirs) file as NIRSData with RAW intensity channels.

    Each (source, detector, wavelength) triple becomes one channel with
    data_type_label = 'RAW'. No motion correction or MBLL is applied; that's
    the preprocess pipeline's job.
    """
    arr = _read_lob_mcos_arr(lob_file_path)
    t = np.asarray(arr[2, 0]).flatten().astype(float)
    d = np.asarray(arr[3, 0]).astype(float)               # (T, n_raw_channels)
    sd = arr[6, 0][0, 0]
    meas_list = np.asarray(sd["MeasList"]).astype(int)    # (n_raw, 4): src, det, gain, wav_idx
    wavelengths = np.asarray(sd["Lambda"]).flatten().astype(float)
    src_pos_3d = np.asarray(sd["SrcPos"]).astype(float)
    det_pos_3d = np.asarray(sd["DetPos"]).astype(float)
    spatial_unit = "mm"
    if "SpatialUnit" in sd.dtype.names:
        try:
            spatial_unit = str(np.asarray(sd["SpatialUnit"]).flatten()[0])
        except Exception:
            pass

    src_pos_2d = src_pos_3d[:, :2]
    det_pos_2d = det_pos_3d[:, :2]

    channels: List[FNIRSChannel] = []
    for raw_idx, row in enumerate(meas_list):
        src, det, _gain, wav = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        wav_value = float(wavelengths[wav - 1]) if 1 <= wav <= len(wavelengths) else float("nan")
        mi = MeasurementInfo(
            source_index=src,
            detector_index=det,
            wavelength_index=wav,
            data_type=DataType.RAW.value,
            data_type_index=1,
            data_type_label="RAW",
            wavelength=wav_value,
        )
        ch = FNIRSChannel(
            channel_idx=raw_idx,
            measurement_info=mi,
            source_pos_2d=src_pos_2d[src - 1],
            detector_pos_2d=det_pos_2d[det - 1],
            source_pos_3d=src_pos_3d[src - 1],
            detector_pos_3d=det_pos_3d[det - 1],
        )
        channels.append(ch)

    fs = 1.0 / float(np.mean(np.diff(t))) if len(t) > 1 else None
    probe = ProbeInfo(
        source_positions_2d=src_pos_2d,
        detector_positions_2d=det_pos_2d,
        source_positions_3d=src_pos_3d,
        detector_positions_3d=det_pos_3d,
        wavelengths=wavelengths,
        frequency=fs,
        length_unit=spatial_unit,
    )

    return NIRSData(
        time_series=d,
        time=t,
        channels=channels,
        probe=probe,
        format_version="lob/cw_nirs",
        metadata={"source_format": "lob", "LengthUnit": spatial_unit},
    )


def save_concentration_snirf(
    output_path: Union[str, Path],
    template: NIRSData,
    time_series: np.ndarray,
    measurement_list: list[dict],
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a SNIRF file with concentration data, copying probe + stim from a
    template NIRSData and replacing the data block with the supplied
    `time_series` and per-channel `measurement_list`.

    `time_series` shape: (n_timepoints, n_channels).
    Each entry in `measurement_list` is a dict with keys:
        sourceIndex, detectorIndex, wavelengthIndex, dataType, dataTypeIndex,
        dataTypeLabel, [dataUnit].
    """
    output_path = Path(output_path)
    if time_series.shape[0] != len(template.time):
        raise ValueError(
            f"time_series rows ({time_series.shape[0]}) must equal template.time length "
            f"({len(template.time)})"
        )
    if time_series.shape[1] != len(measurement_list):
        raise ValueError(
            f"time_series cols ({time_series.shape[1]}) must equal measurement_list length "
            f"({len(measurement_list)})"
        )

    with h5py.File(output_path, "w") as f:
        f.create_dataset("formatVersion", data=np.bytes_(template.format_version or "1.0"))
        nirs = f.create_group("nirs")

        meta = nirs.create_group("metaDataTags")
        merged_meta: Dict[str, Any] = dict(template.metadata or {})
        if metadata_extra:
            merged_meta.update(metadata_extra)
        for k, v in merged_meta.items():
            try:
                if isinstance(v, str):
                    meta.create_dataset(k, data=np.bytes_(v))
                elif isinstance(v, (bytes, np.bytes_)):
                    meta.create_dataset(k, data=v)
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    meta.create_dataset(k, data=v)
                elif isinstance(v, np.ndarray):
                    meta.create_dataset(k, data=v)
                else:
                    meta.create_dataset(k, data=np.bytes_(str(v)))
            except Exception:
                pass  # Skip metadata entries we can't serialize.

        probe = nirs.create_group("probe")
        probe.create_dataset("sourcePos2D", data=template.probe.source_positions_2d)
        probe.create_dataset("detectorPos2D", data=template.probe.detector_positions_2d)
        if template.probe.source_positions_3d is not None:
            probe.create_dataset("sourcePos3D", data=template.probe.source_positions_3d)
        if template.probe.detector_positions_3d is not None:
            probe.create_dataset("detectorPos3D", data=template.probe.detector_positions_3d)
        if template.probe.wavelengths is not None:
            probe.create_dataset("wavelengths", data=np.asarray(template.probe.wavelengths))

        data1 = nirs.create_group("data1")
        data1.create_dataset("dataTimeSeries", data=np.asarray(time_series, dtype=np.float64))
        data1.create_dataset("time", data=np.asarray(template.time, dtype=np.float64))
        for i, ml in enumerate(measurement_list, start=1):
            ml_g = data1.create_group(f"measurementList{i}")
            ml_g.create_dataset("sourceIndex", data=np.array([int(ml["sourceIndex"])]))
            ml_g.create_dataset("detectorIndex", data=np.array([int(ml["detectorIndex"])]))
            ml_g.create_dataset("wavelengthIndex", data=np.array([int(ml["wavelengthIndex"])]))
            ml_g.create_dataset("dataType", data=np.array([int(ml["dataType"])]))
            ml_g.create_dataset("dataTypeIndex", data=np.array([int(ml["dataTypeIndex"])]))
            ml_g.create_dataset("dataTypeLabel", data=np.bytes_(str(ml["dataTypeLabel"])))
            if "dataUnit" in ml and ml["dataUnit"] is not None:
                ml_g.create_dataset("dataUnit", data=np.bytes_(str(ml["dataUnit"])))

        if template.stimulus:
            for i, stim in enumerate(template.stimulus, start=1):
                stim_g = nirs.create_group(f"stim{i}")
                stim_g.create_dataset("name", data=np.bytes_(stim.name))
                stim_g.create_dataset("data", data=np.asarray(stim.data))


# ========================================================================
# MATLAB Data Format Support for Hemodynamic Concentration Data
# ========================================================================

@dataclass
class HemoglobinChannel:
    """Represents a single fNIRS channel with hemoglobin concentration data."""
    channel_idx: int
    source_idx: int
    detector_idx: int
    source_pos: np.ndarray  # 2D or 3D position
    detector_pos: np.ndarray  # 2D or 3D position
    distance: float  # source-detector separation
    midpoint: np.ndarray  # channel midpoint position for spatial modeling
    is_short_separation: bool

    # Hemoglobin concentration time series [timepoints x chromophores]
    hbo_conc: np.ndarray  # Oxygenated Hb concentration changes
    hbr_conc: np.ndarray  # Deoxygenated Hb concentration changes
    hbt_conc: np.ndarray  # Total Hb concentration changes


@dataclass
class FNIRSProbeConfig:
    """Configuration of the fNIRS probe for MATLAB data."""
    n_sources: int
    n_detectors: int
    wavelengths: np.ndarray
    sampling_freq: float
    source_positions: np.ndarray
    detector_positions: np.ndarray
    source_positions_3d: Optional[np.ndarray] = None
    detector_positions_3d: Optional[np.ndarray] = None
    spatial_unit: str = "cm"
    short_separation_indices: Optional[np.ndarray] = None


@dataclass
class HemodynamicData:
    """Main dataclass for hemodynamic fNIRS data suitable for spatial-temporal modeling."""

    probe: FNIRSProbeConfig
    channels: List[HemoglobinChannel]
    time: np.ndarray

    # Raw concentration data [timepoints x channels x chromophores]
    concentration_data: np.ndarray

    # Auxiliary physiological data
    physiology_data: Optional[np.ndarray] = None
    bad_channels: Optional[np.ndarray] = None

    def get_spatial_coordinates(self) -> np.ndarray:
        """Get array of channel midpoint coordinates for spatial modeling (2D projection)."""
        return np.array([ch.midpoint for ch in self.channels])

    def get_spatial_coordinates_3d(self) -> np.ndarray:
        """Get array of channel midpoint coordinates in true 3D space (accounts for head curvature)."""
        if self.probe.source_positions_3d is not None and self.probe.detector_positions_3d is not None:
            coords_3d = []
            for ch in self.channels:
                src_3d = self.probe.source_positions_3d[ch.source_idx - 1]
                det_3d = self.probe.detector_positions_3d[ch.detector_idx - 1]
                midpoint_3d = (src_3d + det_3d) / 2
                coords_3d.append(midpoint_3d)
            return np.array(coords_3d)
        else:
            return self.get_spatial_coordinates()  # Fallback to 2D

    def get_concentration_matrix(self, chromophore: ChromophoreType) -> np.ndarray:
        """Get concentration data matrix for specified chromophore."""
        return self.concentration_data[:, :, chromophore.value]

    def get_hbo_data(self) -> np.ndarray:
        """Get oxygenated hemoglobin concentration matrix [time x channels]."""
        return self.concentration_data[:, :, ChromophoreType.HbO.value]

    def get_hbr_data(self) -> np.ndarray:
        """Get deoxygenated hemoglobin concentration matrix [time x channels]."""
        return self.concentration_data[:, :, ChromophoreType.HbR.value]

    def get_hbt_data(self) -> np.ndarray:
        """Get total hemoglobin concentration matrix [time x channels]."""
        return self.concentration_data[:, :, ChromophoreType.HbT.value]

    def get_long_separation_channels(self, threshold: float = 1.5) -> List[HemoglobinChannel]:
        """Get channels with source-detector distance above threshold (in probe spatial units)."""
        return [ch for ch in self.channels if ch.distance >= threshold]

    def get_short_separation_channels(self, threshold: float = 1.5) -> List[HemoglobinChannel]:
        """Get channels with source-detector distance below threshold (in probe spatial units)."""
        return [ch for ch in self.channels if ch.distance < threshold]

    def get_channels_near_position(self, position: np.ndarray, radius: float) -> List[HemoglobinChannel]:
        """Get channels within specified radius of a given position."""
        distances = np.linalg.norm(self.get_spatial_coordinates() - position, axis=1)
        nearby_indices = np.where(distances <= radius)[0]
        return [self.channels[i] for i in nearby_indices]

    def create_spatial_mesh(self, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Create regular spatial mesh for interpolation and visualization."""
        coords = self.get_spatial_coordinates()
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        x_grid = np.linspace(x_min, x_max, resolution)
        y_grid = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        return X, Y

    def get_temporal_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get temporal statistics for each chromophore."""
        stats = {}
        for chrom in ChromophoreType:
            data = self.get_concentration_matrix(chrom)
            stats[chrom.name] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data),
                'temporal_std': np.mean(np.std(data, axis=0)),  # avg temporal variability
                'spatial_std': np.mean(np.std(data, axis=1))    # avg spatial variability
            }
        return stats


def load_hemodynamic_data(matlab_file_path: str) -> HemodynamicData:
    """
    Load hemodynamic fNIRS data from MATLAB file.

    Parameters
    ----------
    matlab_file_path : str
        Path to the Data_for_Part_II.mat file

    Returns
    -------
    HemodynamicData
        Loaded hemodynamic data structured for spatial-temporal modeling
    """
    from scipy.io import loadmat

    # Load MATLAB data
    mat_data = loadmat(matlab_file_path)

    # Extract main data arrays
    dc = mat_data['dc']  # Shape: (timepoints, channels, chromophores)
    SD = mat_data['SD'][0, 0]

    # Extract optional arrays
    phys_data = mat_data.get('Phys_data')
    bad_channels = mat_data.get('BadChannels')
    short_sep_list = mat_data.get('SSlist')

    # Parse probe configuration
    n_sources = int(SD['nSrcs'].item())
    n_detectors = int(SD['nDets'].item())
    wavelengths = SD['Lambda'].flatten()
    sampling_freq = float(SD['f'].item())
    source_positions = SD['SrcPos']
    detector_positions = SD['DetPos']
    meas_list = SD['MeasList']

    # Optional 3D positions and spatial unit
    source_positions_3d = SD['SrcPos_3d'] if 'SrcPos_3d' in SD.dtype.names else None
    detector_positions_3d = SD['DetPos_3d'] if 'DetPos_3d' in SD.dtype.names else None

    # Use 3D positions for calculations if available (accounts for head curvature)
    src_pos_for_calc = source_positions_3d if source_positions_3d is not None else source_positions
    det_pos_for_calc = detector_positions_3d if detector_positions_3d is not None else detector_positions

    spatial_unit = "cm"
    if 'SpatialUnit' in SD.dtype.names:
        spatial_unit_data = SD['SpatialUnit']
        if spatial_unit_data.size > 0:
            spatial_unit = str(spatial_unit_data[0])

    # Extract short separation channel indices
    short_sep_indices = None
    if short_sep_list is not None:
        short_sep_indices = short_sep_list.flatten() - 1  # Convert to 0-based indexing

    # Create probe configuration
    probe = FNIRSProbeConfig(
        n_sources=n_sources,
        n_detectors=n_detectors,
        wavelengths=wavelengths,
        sampling_freq=sampling_freq,
        source_positions=source_positions,
        detector_positions=detector_positions,
        source_positions_3d=source_positions_3d,
        detector_positions_3d=detector_positions_3d,
        spatial_unit=spatial_unit,
        short_separation_indices=short_sep_indices
    )

    # Generate time vector (assuming regular sampling)
    n_timepoints = dc.shape[0]
    time = np.arange(n_timepoints) / sampling_freq

    # Create channel objects
    channels = []
    n_channels = dc.shape[1]

    # Map processed channels back to measurement list
    for ch_idx in range(n_channels):
        if ch_idx < len(meas_list):
            meas = meas_list[ch_idx]
            source_idx = int(meas[0])
            detector_idx = int(meas[1])
        else:
            # Fallback: estimate from channel index
            source_idx = (ch_idx // n_detectors) + 1
            detector_idx = (ch_idx % n_detectors) + 1

        # Get positions (convert to 0-based indexing)
        src_pos = source_positions[source_idx - 1]
        det_pos = detector_positions[detector_idx - 1]

        # Use 3D positions for distance calculation if available (more accurate)
        src_pos_3d = src_pos_for_calc[source_idx - 1]
        det_pos_3d = det_pos_for_calc[detector_idx - 1]

        # Calculate distance using 3D positions, midpoint using 2D for spatial modeling
        distance = np.linalg.norm(src_pos_3d - det_pos_3d)
        midpoint = (src_pos + det_pos) / 2  # Use 2D for spatial interpolation

        # Check if short separation
        is_short_sep = False
        if short_sep_indices is not None:
            is_short_sep = ch_idx in short_sep_indices

        # Extract hemoglobin time series for this channel
        hbo_conc = dc[:, ch_idx, ChromophoreType.HbO.value]
        hbr_conc = dc[:, ch_idx, ChromophoreType.HbR.value]
        hbt_conc = dc[:, ch_idx, ChromophoreType.HbT.value]

        channel = HemoglobinChannel(
            channel_idx=ch_idx,
            source_idx=source_idx,
            detector_idx=detector_idx,
            source_pos=src_pos,
            detector_pos=det_pos,
            distance=distance,
            midpoint=midpoint,
            is_short_separation=is_short_sep,
            hbo_conc=hbo_conc,
            hbr_conc=hbr_conc,
            hbt_conc=hbt_conc
        )
        channels.append(channel)

    # Create main data object
    hemodynamic_data = HemodynamicData(
        probe=probe,
        channels=channels,
        time=time,
        concentration_data=dc,
        physiology_data=phys_data,
        bad_channels=bad_channels
    )

    return hemodynamic_data
