# kdc2tiff

Convert 1990s Kodak KDC raw files to high-quality 16-bit TIFF with accurate color correction.

Supports Kodak DC120 and DC50 digital cameras. Produces output that matches (as much as possible) the original Kodak software (Windows 3/95 conversion software) conversion, with full EXIF metadata preserved.

*PLEASE NOTE:* This project is largely LLM-generated using locally run LLMs and is deliberately "quick and dirty" to convert these files without having to use the ancient Kodak software and fixing a few issues on the way (like much better demosaicing and not doing the extreme sharpening the Kodak software applies).

## Features

- **16-bit output** — full 0–65535 precision per channel, no banding
- **Per-channel linear color correction** — calibrated from reference KDC/TIFF pairs
- **Flash-aware** — separate color params for flash and non-flash shots
- **Multi-camera support** — DC120 and DC50 with camera-specific processing
- **EXIF metadata** — Make, Model, ExposureTime, FNumber, ISO, FocalLength, etc.
- **Demosaic algorithms** — Menon2007, AHD, VNG, PPG (LMMSE/AMAZE with GPL packs)
- **7× oversampled resize** — bicubic upsample + Lanczos downsample for smooth results
- **Floyd-Steinberg dithering** — optional 8-bit output without banding
- **Batch processing** — convert entire directories in one command
- **Colored terminal output** — progress bar with resize support

## Installation

### Prerequisites

- **Python 3.10+** (tested with 3.14)
- **[exiftool](https://exiftool.org/)** — for writing EXIF metadata to TIFF
- **Homebrew** (recommended for macOS)

### macOS Setup

```bash
# 1. Install Homebrew if you haven't
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install exiftool
brew install exiftool

# 3. Clone the repository
git clone https://github.com/madrobby/kdc2tiff.git
cd kdc2tiff

# 4. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
```

### Linux

```bash
# Install exiftool (Ubuntu/Debian)
sudo apt-get install libimage-exiftool-perl

# Or (Fedora/RHEL)
sudo dnf install ImageMagick-exiftool

# Then follow steps 3–5 above
```

## Usage

### Basic Conversion

Convert a single KDC file to 16-bit TIFF:

```bash
python kdc2tiff.py photo.KDC
# Output: photo.tif (16-bit)
```

Convert with 8-bit output and dithering:

```bash
python kdc2tiff.py photo.KDC --bits 8
```

### Batch Conversion

Convert all KDC files in a directory:

```bash
python kdc2tiff.py ./kdc_folder/
# Output: ./kdc_folder/_converted/*.tif
```

Specify a custom output directory:

```bash
python kdc2tiff.py ./kdc_folder/ --output ./tiff_folder/
```

### Advanced Options

**Choose a demosaic algorithm:**

```bash
# Default: Menon2007 for DC120, AHD for DC50
python kdc2tiff.py photo.KDC --demosaic vng
python kdc2tiff.py photo.KDC --demosaic ahd
python kdc2tiff.py photo.KDC --demosaic ppg
```

Available algorithms: `menon2007`, `ahd`, `vng`, `ppg`, `lmmse`, `amaze`

Note: `lmmse` and `amaze` require GPL2/GPL3 demosaic packs (not included by default).

**Disable color correction:**

```bash
python kdc2tiff.py photo.KDC --no-color-correction
```

**Disable the percentile-based stretch:**

```bash
python kdc2tiff.py photo.KDC --no-stretch
```

**Overwrite existing files:**

```bash
python kdc2tiff.py ./kdc_folder/ --overwrite
```

**Verbose logging:**

```bash
python kdc2tiff.py photo.KDC -v
```

### Calibration

Rebuild the color correction parameters from reference KDC/TIFF pairs:

```bash
python kdc2tiff.py --calibrate photo1.KDC photo1.tif photo2.KDC photo2.tif
```

This generates `reference_lut.json` with per-channel linear gains/offsets.

## Output

### File Format

- **16-bit TIFF** (default) — maximum quality, no banding
- **8-bit TIFF** — with Floyd-Steinberg dithering (or without `--no-dither`)

### Dimensions

| Camera | Output Size | Notes |
|--------|-------------|-------|
| DC120  | 1301 × 976  | Aspect ratio corrected (pixel_aspect ≈ 1.53) |
| DC50   | 768 × 512   | Square pixels |

### Metadata

All EXIF tags from the KDC file are preserved and written to the TIFF:

- Make, Model
- DateTimeOriginal
- ExposureTime, FNumber, ISO
- FocalLength
- Flash, ExposureProgram
- WhiteBalance, LightSource

## Pipeline

1. **Decode** — rawpy reads the KDC file (16-bit Bayer data)
2. **Demosaic** — Menon2007 (DC120) or AHD/VNG/PPG (any camera)
3. **Noise reduction** — median filter + FBDD-like denoising (non-flash only)
4. **Resize** — 7× oversampled bicubic → Lanczos downsample
5. **Color correction** — per-channel linear transform + percentile stretch
6. **Write TIFF** — tifffile for image, exiftool for EXIF metadata

## Requirements

```
rawpy>=0.27.0
imageio>=2.31
tifffile>=2024.1
numpy>=1.24
Pillow>=10.0
tqdm>=4.65
scipy>=1.10
scikit-learn>=1.0
colour-demosaicing>=0.2.0
scikit-image>=0.21
```

## License

[MIT License](LICENSE) — Copyright (c) 2026 Thomas Fuchs

## Acknowledgments

- [rawpy](https://github.com/letmaik/rawpy) — LibRaw Python wrapper
- [colour-demosaicing](https://www.colour-science.org/) — Menon2007 demosaic algorithm
- [exiftool](https://exiftool.org/) — metadata writing
