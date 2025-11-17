# Visualization Scripts

This directory contains scripts for creating visualizations and demonstrations of the fNIRS spatial-temporal modeling approach.

## 1. Static Spherical Harmonics Grid

`visualize_spherical_harmonics_static.py` - Creates a static figure showing all spherical harmonics up to a maximum degree, arranged in a grid by (l, m).

### Usage

```bash
python scripts/visualize_spherical_harmonics_static.py [max_l] [output_file] [dpi]
```

**Parameters:**
- `max_l` (default: 3) - Maximum degree of spherical harmonics
- `output_file` (default: spherical_harmonics_static.png) - Output filename
- `dpi` (default: 150) - Resolution

**Examples:**

```bash
# Create figure with l=0,1,2,3 (16 spherical harmonics)
python scripts/visualize_spherical_harmonics_static.py 3 figures/harmonics.png 150

# High resolution version
python scripts/visualize_spherical_harmonics_static.py 3 figures/harmonics_hires.png 300
```

This creates a grid where:
- **Rows** represent degree (l)
- **Columns** represent order (m)
- Each sphere is colored by the plasma colormap showing the spatial pattern

## 2. Animated Spherical Harmonics

`visualize_spherical_harmonics_animation.py` - Creates an animated visualization showing how spherical harmonics combine with temporal basis functions to represent spatial-temporal data.

### What it shows

The animation demonstrates the core concept behind the spatial-temporal modeling:
- **Grid of spheres**: Each sphere shows a different spherical harmonic (Y_l^m) colored by the plasma colormap
- **Time series plot**: Shows the temporal coefficients for each spherical harmonic over time
  - Individual coefficients shown in color (thin lines)
  - **Sum (total signal) shown in black (thick line, linewidth=2)**
- **Animated opacity**: As time progresses, each sphere's opacity changes to reflect its contribution to the signal at that moment

This visualization helps understand how a spatial-temporal signal is decomposed into:
- **Spatial patterns** (spherical harmonics) that describe where on the head activity occurs
- **Temporal patterns** (coefficients over time) that describe when activity occurs

### Usage

```bash
python scripts/visualize_spherical_harmonics_animation.py [max_l] [duration] [output_file]
```

**Parameters:**
- `max_l` (default: 2) - Maximum degree of spherical harmonics to show
  - `max_l=1`: Shows 4 modes (l=0,1)
  - `max_l=2`: Shows 9 modes (l=0,1,2)
  - `max_l=3`: Shows 16 modes (l=0,1,2,3)
- `duration` (default: 10.0) - Duration of animation in seconds
- `output_file` (default: spherical_harmonics_animation.mp4) - Output filename

**Examples:**

```bash
# Create a 5-second animation with l=0,1,2 (9 spherical harmonics)
python scripts/visualize_spherical_harmonics_animation.py 2 5.0 figures/demo.mp4

# Create a 10-second animation with l=0,1,2,3 (16 spherical harmonics)
python scripts/visualize_spherical_harmonics_animation.py 3 10.0 figures/full_demo.mp4

# Quick test with just l=0,1 (4 spherical harmonics)
python scripts/visualize_spherical_harmonics_animation.py 1 3.0 figures/quick_test.mp4
```

### Output formats

The script will automatically create:
- **MP4 video** (if ffmpeg is installed) - Best quality, smaller file size
- **GIF animation** (fallback if ffmpeg not available) - Larger file size, works everywhere

### Installing ffmpeg (optional, for better quality)

For MP4 output instead of GIF:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Interpretation

When watching the animation:
- **Bright spheres** = High contribution at this time
- **Dim/transparent spheres** = Low contribution at this time
- **Colored patterns** = The spatial pattern of the basis function (red = positive, blue = negative in plasma colormap)
- **Colored time series** = How each spherical harmonic's coefficient changes over time
- **Black thick line** = The sum of all coefficients (total signal)
- **Red vertical line** = Current time position

The final signal (black line) is the weighted sum of all these spherical harmonics, where the weights change over time according to the temporal coefficients (colored lines).

### File size

Typical file sizes:
- GIF: ~2-4 MB per second (19 MB for 5 seconds)
- MP4: ~0.5-1 MB per second (much smaller)

For longer animations, use ffmpeg to create MP4 files.
