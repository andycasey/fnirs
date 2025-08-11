#!/usr/bin/env python3
"""
Script to load fNIRS data from MATLAB file into Python dataclass structure.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.io import loadmat


@dataclass
class Channel:
    """Represents a single fNIRS measurement channel."""
    source_idx: int
    detector_idx: int
    wavelength_idx: int
    wavelength_nm: float
    source_pos: np.ndarray  # 3D position
    detector_pos: np.ndarray  # 3D position
    distance: float  # source-detector distance
    data: np.ndarray  # time series data
    

@dataclass
class FNIRSProbe:
    """Represents the fNIRS probe configuration."""
    n_sources: int
    n_detectors: int
    wavelengths: np.ndarray  # wavelengths in nm
    sampling_freq: float
    source_positions: np.ndarray  # (n_sources, 3) positions
    detector_positions: np.ndarray  # (n_detectors, 3) positions
    source_positions_3d: Optional[np.ndarray] = None
    detector_positions_3d: Optional[np.ndarray] = None
    spatial_unit: str = "unknown"
    measurement_list: Optional[np.ndarray] = None  # raw measurement list


@dataclass
class FNIRSData:
    """Main dataclass for fNIRS dataset."""
    probe: FNIRSProbe
    channels: list[Channel]
    time: np.ndarray
    raw_data: np.ndarray  # (timepoints, channels)
    stimulus: Optional[np.ndarray] = None
    auxiliary: Optional[np.ndarray] = None
    physiology: Optional[dict] = None
    
    def get_channels_by_wavelength(self, wavelength_nm: float) -> list[Channel]:
        """Get all channels for a specific wavelength."""
        return [ch for ch in self.channels if ch.wavelength_nm == wavelength_nm]
    
    def get_channels_by_source_detector(self, source_idx: int, detector_idx: int) -> list[Channel]:
        """Get all channels for a specific source-detector pair."""
        return [ch for ch in self.channels 
                if ch.source_idx == source_idx and ch.detector_idx == detector_idx]
    
    def get_channel_distances(self) -> np.ndarray:
        """Get array of all channel distances."""
        return np.array([ch.distance for ch in self.channels])
    
    def get_short_separation_channels(self, threshold: float = 1.5) -> list[Channel]:
        """Get channels with source-detector distance below threshold (in probe spatial units)."""
        return [ch for ch in self.channels if ch.distance < threshold]
    
    def get_long_separation_channels(self, threshold: float = 1.5) -> list[Channel]:
        """Get channels with source-detector distance above threshold (in probe spatial units)."""
        return [ch for ch in self.channels if ch.distance >= threshold]


def load_fnirs_data(matlab_file_path: str) -> FNIRSData:
    """
    Load fNIRS data from MATLAB file into Python dataclass structure.
    
    Parameters
    ----------
    matlab_file_path : str
        Path to the MATLAB .mat file containing fNIRS data
        
    Returns
    -------
    FNIRSData
        Loaded fNIRS data in structured format
    """
    # Load MATLAB data
    mat_data = loadmat(matlab_file_path)
    
    # Extract the main data structure
    if 'data' in mat_data:
        data_struct = mat_data['data'][0, 0]
    else:
        raise ValueError("Expected 'data' field not found in MATLAB file")
    
    # Extract basic arrays
    time = data_struct['t'].flatten()
    raw_data = data_struct['d']
    stimulus = data_struct['s'].flatten() if 's' in data_struct.dtype.names else None
    auxiliary = data_struct['aux'] if 'aux' in data_struct.dtype.names else None
    
    # Extract probe configuration
    SD = data_struct['SD'][0, 0]
    
    n_sources = int(SD['nSrcs'].item())
    n_detectors = int(SD['nDets'].item())
    wavelengths = SD['Lambda'].flatten()
    sampling_freq = float(SD['f'].item())
    source_positions = SD['SrcPos']
    detector_positions = SD['DetPos']
    measurement_list = SD['MeasList']
    
    # Optional 3D positions
    source_positions_3d = SD['SrcPos_3d'] if 'SrcPos_3d' in SD.dtype.names else None
    detector_positions_3d = SD['DetPos_3d'] if 'DetPos_3d' in SD.dtype.names else None
    
    # Spatial unit
    spatial_unit = "unknown"
    if 'SpatialUnit' in SD.dtype.names:
        spatial_unit_data = SD['SpatialUnit']
        if hasattr(spatial_unit_data, 'shape') and spatial_unit_data.size > 0:
            if spatial_unit_data.dtype.kind in ['U', 'S']:  # string types
                spatial_unit = str(spatial_unit_data.item())
            else:
                # Handle array of strings
                spatial_unit = str(spatial_unit_data[0]) if spatial_unit_data.size > 0 else "unknown"
    
    # Create probe object
    probe = FNIRSProbe(
        n_sources=n_sources,
        n_detectors=n_detectors,
        wavelengths=wavelengths,
        sampling_freq=sampling_freq,
        source_positions=source_positions,
        detector_positions=detector_positions,
        source_positions_3d=source_positions_3d,
        detector_positions_3d=detector_positions_3d,
        spatial_unit=spatial_unit,
        measurement_list=measurement_list
    )
    
    # Create individual channel objects
    channels = []
    for i, meas in enumerate(measurement_list):
        source_idx = int(meas[0])  # 1-based in MATLAB
        detector_idx = int(meas[1])  # 1-based in MATLAB
        wavelength_idx = int(meas[3])  # 1-based in MATLAB
        
        # Convert to 0-based indexing for array access
        src_pos = source_positions[source_idx - 1]
        det_pos = detector_positions[detector_idx - 1]
        wavelength_nm = wavelengths[wavelength_idx - 1]
        
        # Calculate distance
        distance = np.linalg.norm(src_pos - det_pos)
        
        # Extract channel data
        channel_data = raw_data[:, i]
        
        channel = Channel(
            source_idx=source_idx,
            detector_idx=detector_idx,
            wavelength_idx=wavelength_idx,
            wavelength_nm=wavelength_nm,
            source_pos=src_pos,
            detector_pos=det_pos,
            distance=distance,
            data=channel_data
        )
        channels.append(channel)
    
    # Extract physiology data if present
    physiology = None
    if 'Phys' in data_struct.dtype.names:
        phys_struct = data_struct['Phys'][0, 0]
        if hasattr(phys_struct, 'dtype') and phys_struct.dtype.names:
            physiology = {}
            for field_name in phys_struct.dtype.names:
                physiology[field_name] = phys_struct[field_name]
    
    # Create main data object
    fnirs_data = FNIRSData(
        probe=probe,
        channels=channels,
        time=time,
        raw_data=raw_data,
        stimulus=stimulus,
        auxiliary=auxiliary,
        physiology=physiology
    )
    
    return fnirs_data


def main():
    """Example usage of the fNIRS data loader."""
    # Load the data
    data_path = "rsFC-fnirs-course/Data_for_Part_I.mat"
    fnirs_data = load_fnirs_data(data_path)
    
    # Display basic information
    print("=== fNIRS Data Summary ===")
    print(f"Number of time points: {len(fnirs_data.time)}")
    print(f"Time range: {fnirs_data.time[0]:.2f} - {fnirs_data.time[-1]:.2f} seconds")
    print(f"Number of channels: {len(fnirs_data.channels)}")
    print(f"Number of sources: {fnirs_data.probe.n_sources}")
    print(f"Number of detectors: {fnirs_data.probe.n_detectors}")
    print(f"Wavelengths: {fnirs_data.probe.wavelengths} nm")
    print(f"Sampling frequency: {fnirs_data.probe.sampling_freq} Hz")
    print(f"Spatial unit: {fnirs_data.probe.spatial_unit}")
    
    print("\n=== Channel Analysis ===")
    distances = fnirs_data.get_channel_distances()
    print(f"Distance range: {distances.min():.2f} - {distances.max():.2f}")
    
    short_channels = fnirs_data.get_short_separation_channels()
    long_channels = fnirs_data.get_long_separation_channels()
    print(f"Short separation channels (<1.5 {fnirs_data.probe.spatial_unit}): {len(short_channels)}")
    print(f"Long separation channels (>=1.5 {fnirs_data.probe.spatial_unit}): {len(long_channels)}")
    
    # Show channels by wavelength
    for wavelength in fnirs_data.probe.wavelengths:
        channels_wl = fnirs_data.get_channels_by_wavelength(wavelength)
        print(f"Channels at {wavelength} nm: {len(channels_wl)}")
    
    print("\n=== Sample Channel Information ===")
    if fnirs_data.channels:
        ch = fnirs_data.channels[0]
        print(f"Channel 0:")
        print(f"  Source {ch.source_idx} -> Detector {ch.detector_idx}")
        print(f"  Wavelength: {ch.wavelength_nm} nm")
        print(f"  Distance: {ch.distance:.2f}")
        print(f"  Source position: {ch.source_pos}")
        print(f"  Detector position: {ch.detector_pos}")
        print(f"  Data range: {ch.data.min():.4f} - {ch.data.max():.4f}")
    
    print("\n=== Additional Data ===")
    print(f"Stimulus data: {'Yes' if fnirs_data.stimulus is not None else 'No'}")
    print(f"Auxiliary data: {'Yes' if fnirs_data.auxiliary is not None else 'No'}")
    print(f"Physiology data: {'Yes' if fnirs_data.physiology is not None else 'No'}")
    
    if fnirs_data.auxiliary is not None:
        print(f"Auxiliary data shape: {fnirs_data.auxiliary.shape}")


if __name__ == "__main__":
    main()