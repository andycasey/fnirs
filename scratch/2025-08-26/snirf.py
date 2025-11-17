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
            
            # Extract measurement info
            source_idx = int(ml_group['sourceIndex'][0])
            detector_idx = int(ml_group['detectorIndex'][0])
            wavelength_idx = int(ml_group['wavelengthIndex'][0])
            data_type = int(ml_group['dataType'][0])
            data_type_idx = int(ml_group['dataTypeIndex'][0])
            data_type_label = _safe_decode_array(ml_group['dataTypeLabel'])
            
            # Get wavelength value
            wavelength = wavelengths[wavelength_idx - 1]  # Convert to 0-based
            
            # Optional fields
            data_unit = _safe_decode_array(ml_group['dataUnit']) if 'dataUnit' in ml_group else None
            module_idx = int(ml_group['moduleIndex'][0]) if 'moduleIndex' in ml_group else None
            
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
