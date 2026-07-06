# kdc2tiff

Convert 90's Kodak KDC raw files (from the DC50 and DC120 cameras) to high-quality 16-bit TIFF with the goal of maximizing image quality through the use of modern demosaicing, color correction and scaling (no "AI" scaling).

There's a color correction LUT included that was made by converting  KDC files with the original Kodak software (for Windows 3.11); you can also supply your own KDC/TIF pairs to match your specific camera better.

The project goal is to be faithful to the original software's conversion but without introducing artifacts such as aggressive sharpening—and provide a basis for further manual processing in other tools (such as Lightroom).

*PLEASE NOTE:* This project is largely LLM-generated using locally run LLMs and is deliberately "quick and dirty" to convert these files without having to use the ancient Kodak software and fixing a few issues on the way (like much better demosaicing and not doing the extreme sharpening the Kodak software applies).

## Features

- **Defaults chosen to maximize quality** while avoiding some of the issues the orignal Kodak software exhibits, such as oversharpening and overfiltering
- Outputs **16-bit TIFFs** or optionally **dithered 8-bit**
- **Per-channel linear color correction** — calibrated from reference KDC/TIFF pairs (that were made with the original Kodak software)
- **Flash-aware** — separate color params for flash and non-flash shots
- **Multi-camera support** — DC120 and DC50 with camera-specific processing
- **EXIF metadata** — Make, Model, ExposureTime, FNumber, ISO, FocalLength, etc.
- **Demosaic algorithms** — Menon2007, AHD, VNG, PPG (LMMSE/AMAZE with GPL packs)
- **2.5× oversampled resize** — sub-pixel shift fusion + OpenCV Lanczos4 upsample + configurable downscale (default: box, also lanczos/hamming/bilinear/bicubic/nearest)
- **Batch processing** — convert entire directories in one command

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

**Choose a resize algorithm:**

```bash
# Default: box with 2.5× oversampling
python kdc2tiff.py photo.KDC --resize lanczos
python kdc2tiff.py photo.KDC --resize hamming
python kdc2tiff.py photo.KDC --resize bilinear
python kdc2tiff.py photo.KDC --resize bicubic
python kdc2tiff.py photo.KDC --resize nearest
```

Available algorithms: `box`, `hamming`, `lanczos`, `bilinear`, `bicubic`, `nearest`

**Skip the 2.5× oversampling step (single-pass resize):**

```bash
python kdc2tiff.py photo.KDC --no-oversample
```

**Enable noise reduction (median filter + FBDD denoising):**

```bash
python kdc2tiff.py photo.KDC --noise-reduction
```

**Disable color correction:**

```bash
python kdc2tiff.py photo.KDC --no-color-correction
```

**Disable the percentile-based stretch:**

```bash
python kdc2tiff.py photo.KDC --no-stretch
```

**Use a custom params file:**

```bash
python kdc2tiff.py photo.KDC --params path/to/params.json
```

**8-bit output without dithering (may show banding):**

```bash
python kdc2tiff.py photo.KDC --bits 8 --no-dither
```

**Overwrite existing files:**

```bash
python kdc2tiff.py ./kdc_folder/ --overwrite
```

**Verbose logging:**

```bash
python kdc2tiff.py photo.KDC -v
```

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--calibrate KDC TIF [...]` | — | Rebuild `reference_lut.json` from reference pairs, then exit |
| `-o, --output PATH` | auto | Output file or directory |
| `--overwrite` | off | Reconvert even if output exists and is newer |
| `--no-color-correction` | off | Skip color correction (rawpy output only) |
| `--params PATH` | `reference_lut.json` | Path to color correction params JSON |
| `--bits {8,16}` | 16 | Output bit depth |
| `--no-dither` | off | Disable Floyd-Steinberg dithering (with `--bits 8`) |
| `--no-stretch` | off | Disable percentile-based highlight stretch |
| `--noise-reduction` | off | Enable median filter + FBDD noise reduction |
| `--resize {box,hamming,lanczos,bilinear,bicubic,nearest}` | box | Downscale algorithm (oversampled path) or single-pass algorithm |
| `--no-oversample` | off | Skip 2.5× oversampling (single-pass resize) |
| `--no-subpixel-fusion` | off | Disable sub-pixel shift fusion (4 shifted versions averaged before upscale) |
| `--demosaic {menon2007,ahd,vng,ppg,lmmse,amaze}` | camera-specific | Demosaic algorithm |
| `-v, --verbose` | off | Debug logging |

### Color Calibration

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
| DC120  | 1301×976*   | Aspect ratio corrected (pixel_aspect ≈1.53) |
| DC50   | 768×512     | Square pixels |

_*The original Kodak software outputs these slightly cropped as 1280x960_

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
3. **Noise reduction** — median filter + FBDD-like denoising (off by default, enable with `--noise-reduction`)
4. **Resize** — sub-pixel shift fusion + 2.5× OpenCV Lanczos4 upsample + configurable downscale algorithm (default: `box`)
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
