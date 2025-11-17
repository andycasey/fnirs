#!/usr/bin/env python3
"""
Script to load fNIRS Part II data containing hemoglobin concentration changes.
Focuses on spatial-temporal modeling of hemodynamic responses.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum
import numpy as np
from scipy.io import loadmat


class ChromophoreType(Enum):
    """Enumeration for hemoglobin chromophore types."""
    HbO = 0  # Oxygenated hemoglobin
    HbR = 1  # Deoxygenated hemoglobin 
    HbT = 2  # Total hemoglobin


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
    """Configuration of the fNIRS probe for Part II data."""
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
    Load hemodynamic fNIRS data from Part II MATLAB file.
    
    Parameters
    ----------
    matlab_file_path : str
        Path to the Data_for_Part_II.mat file
        
    Returns
    -------
    HemodynamicData
        Loaded hemodynamic data structured for spatial-temporal modeling
    """
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
    n_channels = dc.shape[1]  # Should be 129 for processed data
    
    # Map processed channels back to measurement list
    # For processed data, we may have fewer channels than the original measurement list
    for ch_idx in range(n_channels):
        # For Part II, channels appear to be the unique source-detector pairs
        # We need to map back to the measurement list to get source/detector info
        
        # Find the first measurement list entry for this processed channel
        # This assumes channels are ordered by first wavelength measurements
        if ch_idx < len(meas_list):
            meas = meas_list[ch_idx] 
            source_idx = int(meas[0])
            detector_idx = int(meas[1])
        else:
            # Fallback: estimate from channel index
            # This is approximate and may need adjustment based on actual data organization
            source_idx = (ch_idx // n_detectors) + 1
            detector_idx = (ch_idx % n_detectors) + 1
            
        # Get positions (convert to 0-based indexing)
        # Use 2D positions for midpoint calculation (for spatial modeling)
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


def main():
    """Example usage of the hemodynamic data loader."""
    # Load the data
    data_path = "../rsFC-fnirs-course/Data_for_Part_II.mat"
    hemo_data = load_hemodynamic_data(data_path)
    
    # Display basic information
    print("=== Hemodynamic fNIRS Data Summary ===")
    print(f"Number of time points: {len(hemo_data.time)}")
    print(f"Time range: {hemo_data.time[0]:.2f} - {hemo_data.time[-1]:.2f} seconds")
    print(f"Number of channels: {len(hemo_data.channels)}")
    print(f"Concentration data shape: {hemo_data.concentration_data.shape}")
    print(f"Sampling frequency: {hemo_data.probe.sampling_freq} Hz")
    print(f"Spatial unit: {hemo_data.probe.spatial_unit}")
    
    print("\n=== Spatial Analysis ===")
    coords = hemo_data.get_spatial_coordinates()
    print(f"Spatial coordinate range:")
    print(f"  X: {coords[:, 0].min():.3f} to {coords[:, 0].max():.3f} {hemo_data.probe.spatial_unit}")
    print(f"  Y: {coords[:, 1].min():.3f} to {coords[:, 1].max():.3f} {hemo_data.probe.spatial_unit}")
    
    # Channel distance analysis
    distances = np.array([ch.distance for ch in hemo_data.channels])
    long_channels = hemo_data.get_long_separation_channels()
    short_channels = hemo_data.get_short_separation_channels()
    
    print(f"\nDistance analysis:")
    print(f"  Range: {distances.min():.3f} - {distances.max():.3f} {hemo_data.probe.spatial_unit}")
    print(f"  Long separation channels (≥1.5 {hemo_data.probe.spatial_unit}): {len(long_channels)}")
    print(f"  Short separation channels (<1.5 {hemo_data.probe.spatial_unit}): {len(short_channels)}")
    
    print("\n=== Hemodynamic Signal Analysis ===")
    stats = hemo_data.get_temporal_statistics()
    for chrom_name, chrom_stats in stats.items():
        print(f"{chrom_name}:")
        print(f"  Range: {chrom_stats['min']:.2e} to {chrom_stats['max']:.2e}")
        print(f"  Mean ± Std: {chrom_stats['mean']:.2e} ± {chrom_stats['std']:.2e}")
        print(f"  Avg temporal variability: {chrom_stats['temporal_std']:.2e}")
        print(f"  Avg spatial variability: {chrom_stats['spatial_std']:.2e}")
    
    print("\n=== Sample Channel Information ===")
    if hemo_data.channels:
        ch = hemo_data.channels[0]
        print(f"Channel 0:")
        print(f"  Source {ch.source_idx} -> Detector {ch.detector_idx}")
        print(f"  Distance: {ch.distance:.3f} {hemo_data.probe.spatial_unit}")
        print(f"  Midpoint: {ch.midpoint}")
        print(f"  Short separation: {ch.is_short_separation}")
        print(f"  HbO range: {ch.hbo_conc.min():.2e} to {ch.hbo_conc.max():.2e}")
        print(f"  HbR range: {ch.hbr_conc.min():.2e} to {ch.hbr_conc.max():.2e}")
        
    print("\n=== Additional Data ===")
    print(f"Physiology data: {'Yes' if hemo_data.physiology_data is not None else 'No'}")
    if hemo_data.physiology_data is not None:
        print(f"  Shape: {hemo_data.physiology_data.shape}")
    print(f"Bad channels: {'Yes' if hemo_data.bad_channels is not None else 'No'}")
    if hemo_data.probe.short_separation_indices is not None:
        print(f"Short separation indices: {hemo_data.probe.short_separation_indices}")


if __name__ == "__main__":
    main()