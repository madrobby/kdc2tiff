# KDC to TIFF Converter — Project Documentation

## Overview

This project converts 1990s Kodak KDC raw image files (from Kodak DC120/DC260/DC290
cameras) to TIFF format, matching the output of the original Kodak software for
Windows 3.1/95 as closely as possible.

The converter was developed through extensive iterative experimentation against
8 user-provided KDC/TIF reference pairs. Each pair consists of a `.KDC` raw file
and a `.TIF` file produced by the original Kodak software, allowing direct
comparison and calibration.

## Camera: Kodak DC120 Zoom Digital Camera

- **Sensor**: 848×976 Bayer pattern (RGBG)
- **Pixel aspect ratio**: 1.5346 (non-square pixels — sensor pixels are taller than wide)
- **Output dimensions**: 1301×976 (rawpy native with pixel aspect ratio correction)
- **Bit depth**: 8-bit reference, 16-bit output (our converter)
- **White balance metadata**: `camera_whitebalance = [0, 1, 0, 0]` (unset — camera didn't record WB)
- **Daylight WB**: `[1, 1, 1, 0]` (equal RGBG — neutral daylight)
- **Color matrix**: All zeros (LibRaw has no color matrix for this camera)
- **Flash**: EXIF Flash tag (0x9209) is present in the KDC file header. Bit 0 = flash fired.

## Calibration Pairs

8 KDC/TIF pairs were provided by the user:

| Pair | Flash? | Scene | Shutter | Aperture | White Level | Notes |
|------|--------|-------|---------|----------|-------------|-------|
| P002002 | YES | Cat (flash) | 1/36s | f/2.5 | 510 | Flash calibration pair |
| P002007 | NO | Indoor | 1/102s | f/2.5 | 510 | Non-flash calibration pair |
| P002008 | YES | Cat (flash) | 1/36s | f/2.5 | 510 | Flash calibration pair |
| P002011 | NO | Outdoor (dark) | 1/377s | f/9.7 | 255 | Non-flash calibration pair |
| P002012 | YES | Outdoor (accidental flash) | 1/266s | f/3.7 | 255 | Excluded from calibration |
| P002013 | YES | Outdoor (accidental flash) | 1/266s | f/3.7 | 255 | Excluded from calibration |
| P002014 | NO | Outdoor | 1/283s | f/4.3 | 255 | Non-flash calibration pair |
| P002015 | YES | Outdoor (accidental flash) | 1/245s | f/3.1 | 255 | Excluded from calibration |

**Flash calibration pairs**: P002002, P002008 (good flash reference — cat photos shot with flash)
**Non-flash calibration pairs**: P002007, P002011, P002014 (good daylight reference)
**Excluded from calibration**: P002012, P002013, P002015 (outdoor scenes where flash fired
accidentally — their reference output doesn't represent good flash conversion, but they
DO get flash params at runtime since their Flash tag is set)

## Processing Pipeline (v18 — Flash-Aware)

### Stage 1: rawpy 16-bit decode

```python
arr = raw.postprocess(output_bps=16)  # uint16, 0-65535
```

**Key decision: use rawpy DEFAULTS (no tuning parameters).**

Earlier versions tried `bright=1.5`, `gamma=(1,1)`, `user_wb=[1.0, 1.5, 1.0, 1.5]`,
`chromatic_aberration=(0.95, 1.0)`, and `no_auto_bright=True`. ALL of these caused
problems:
- `bright=1.5` + `gamma=(1,1)` over-brightened the output, causing clipping
- `user_wb=[1.0, 1.5, 1.0, 1.5]` (green boost) was needed to match the reference's
  green channel, but when combined with a per-channel LUT it caused green tint in highlights
- `chromatic_aberration` parameter only does global linear scaling, not spatial correction
- rawpy defaults alone produce well-aligned, clean RGB with no artifacts

**Why 16-bit?** The reference is 8-bit, but 8-bit processing causes banding in smooth
gradients (sky, fur, skin). 16-bit throughout (rawpy output → color correction → output)
gives 47,000+ distinct values per channel vs 256 max in 8-bit. Banding is mathematically
impossible.

### Stage 2: Resize to 1280×960

```python
# PIL doesn't handle uint16 RGB directly, so resize each channel separately
for c in range(3):
    img = Image.fromarray(arr_16[..., c], mode="I;16")
    img_resized = img.resize((1280, 960), Image.BILINEAR)
    out[..., c] = np.array(img_resized)
```

Matches the reference Kodak-software dimensions. rawpy's default output is 976×1301
(due to the sensor's 1.5346 pixel aspect ratio stretch). The user accepted the
resolution difference; we resize to match the reference for direct comparison.

### Stage 3: Read EXIF Flash tag

```python
def read_flash_tag(kdc_path):
    """Read EXIF Flash tag (0x9209) from KDC file header.
    Returns True if flash fired (bit 0 set)."""
    with open(kdc_path, "rb") as f:
        data = f.read(8192)  # only need the header
    if data[:2] != b'MM':
        return False
    ifd_offset = struct.unpack_from('>I', data, 4)[0]
    num_entries = struct.unpack_from('>H', data, ifd_offset)[0]
    for i in range(num_entries):
        entry_offset = ifd_offset + 2 + i * 12
        tag = struct.unpack_from('>H', data, entry_offset)[0]
        if tag == 0x9209:  # Flash tag
            val = struct.unpack_from('>H', data, entry_offset + 8)[0]
            return bool(val & 1)  # bit 0 = flash fired
    return False
```

**Why flash detection?** Flash-lit scenes have nearly neutral color temperature (~5500K),
while daylight scenes have a warm tint (~6500K+). The Kodak software applied different
white balance for flash vs daylight. A single global transform can't match both — flash
scenes end up too warm (the problem the user reported with P002008).

The Flash tag is read directly from the KDC file's TIFF header (not from rawpy, which
doesn't expose it). Only the first 8KB of the file is read, so it's fast.

**Flash tag values observed:**
- 31 (0b11111) = flash fired + return detected + mode + function + red-eye
- 16 (0b10000) = flash did NOT fire, but flash function present
- 24 (0b11000) = flash did NOT fire, no flash function

### Stage 4: Flash-aware per-channel linear color correction

```python
# Select params based on flash state
if flash_fired:
    params = flash_params  # calibrated from P002002 + P002008
else:
    params = nonflash_params  # calibrated from all 8 pairs (global fit)

# Apply per-channel linear: output = gain * input + offset
for c, name in enumerate("RGB"):
    x_float = arr_16[..., c].astype(np.float64) / 256.0  # 16-bit -> 8-bit float
    y_float = gain * x_float + offset
    out[..., c] = y_float * 256.0  # back to 16-bit
```

**Why linear (not a LUT/tone curve)?**

Extensive experimentation showed that per-channel LUTs (PCHIP-smoothed scatter curves)
caused green/seafoam tint in highlights. The root cause: each channel's LUT was fit
independently from scatter data, and noise in the per-channel scatter caused the channels
to diverge in the highlight region. When all three channels are bright, G ended up
disproportionately high, creating a green tint.

A linear transform (`output = gain * input + offset`) preserves the input's natural
R > G > B relationship in highlights, which already matches the reference's warm/red
tint direction. No per-channel divergence is possible with a linear transform.

**Calibration method:**
- Gains and offsets are fit using `sklearn.linear_model.LinearRegression` on midtone
  pixels (input values 30-220, excluding clipped extremes)
- Flash params: fit on P002002 + P002008 only (good flash reference pairs)
- Non-flash params: fit on all 8 pairs (global fit — more robust than the 3-pair
  non-flash subset, which had P002007 as an outlier)

**Flash params (v18):**
```
R: gain=0.9317, offset=-33.72
G: gain=0.8531, offset=-13.24
B: gain=0.9627, offset=-30.01
```

**Non-flash params (v18, same as v11 global):**
```
R: gain=0.8874, offset=-35.61
G: gain=0.7901, offset=-14.63
B: gain=0.8593, offset=-30.56
```

The flash params have more balanced per-channel gains (R≈G≈B ≈ 0.85-0.96) reflecting
the neutral color temperature of flash. The non-flash params have G gain (0.79) much
lower than R (0.89) and B (0.86), reflecting the warm daylight tint.

### Stage 5: Percentile-based stretch

```python
# After linear correction, highlights are compressed (dull).
# Stretch maps output p0.5 -> 0 and p99.5 -> median reference p99.5.
for c in range(3):
    y_float = stretch_gain * y_float + stretch_offset
```

**Why stretch?** After the linear color correction, the output's highlights are
compressed (p99 ≈ 162 vs reference's 178). The stretch extends the dynamic range
to match the reference.

**Why percentile-based (not auto-levels)?**
- Maps p0.5 → 0 and p99.5 → target (not absolute min/max → 0/255)
- Cannot over-expose: never stretches beyond p99.5, so noise in the top 0.5% isn't amplified
- Cannot under-expose: never shifts shadows below 0
- Target = median reference p99.5 across calibration pairs (robust to outliers)

**Flash stretch gains:** [1.2063, 1.2249, 1.1489]
**Non-flash stretch gains:** [1.1723, 1.1723, 1.0755]

### Stage 6: Output

- **16-bit TIFF** (default): saves uint16 array directly. Maximum precision, no banding.
- **8-bit TIFF with Floyd-Steinberg dithering** (`--bits 8`): error diffusion breaks up
  banding that would otherwise appear in 8-bit. ~16s per file (sequential algorithm).
- **8-bit without dithering** (`--bits 8 --no-dither`): simple truncation (high byte).
  Fast but may show banding.

## What was tried and rejected (and why)

### Per-channel PCHIP LUT (v5-v9)

**Approach**: Build per-channel 256-entry LUTs from scatter data (mean reference value
at each source value), smoothed with PCHIP interpolation.

**Problem**: Green/seafoam tint in highlights. Each channel's LUT was fit independently,
so noise in the scatter data caused channels to diverge. At input 200, the gains were
R=1.27, G=1.68, B=2.09 — G was pushed up faster than R, creating green tint when all
channels were bright.

**Fixes attempted** (all failed):
- G-channel consistency constraint (G within ±5 of (R+B)/2): barely helped
- Shared highlight gain (force all 3 channels to same gain above input 180): reduced
  tint from +25 to +17, still wrong direction
- Identity-in-highlights (fade LUT to identity above input 180): highlights too bright
- Soft endpoint anchoring (virtual samples at (0,0) and (255,255)): preserved dynamic
  range but didn't fix tint
- Luminance LUT (single shared curve): no tint but overfit to scatter noise

**Why it failed**: The fundamental issue is that per-channel scatter data is noisy,
and any per-channel fitting amplifies that noise into per-channel divergence. A linear
transform can't diverge (it's a single gain per channel), so it's inherently safe.

### Radial chromatic aberration correction (CAC)

**Approach**: Per-pixel radial warp of the R channel around the image center to undo
lens lateral chromatic aberration. Tuned with kx=ky=-0.06.

**Problem**: A diagnostic showed rawpy's default output already has good channel
alignment (R-G cross-channel correlation = 0.85 vs reference 0.87). The CAC was
solving a non-problem and introduced spatial artifacts.

**Removed** in the clean rebuild.

### Aggressive rawpy tuning

**Approach**: `bright=1.5`, `gamma=(1,1)`, `user_wb=[1.0, 1.5, 1.0, 1.5]`,
`no_auto_bright=True`, `chromatic_aberration=(0.95, 1.0)`.

**Problem**: These parameters were tuned against a single reference pair (P002007)
and overfit. `bright=1.5` over-brightened, causing clipping. `gamma=(1,1)` (linear)
produced flat-looking output. `user_wb` green boost was needed to match the reference's
G channel but caused green tint when combined with per-channel LUTs.

**Removed**: rawpy defaults produce clean, well-aligned output. No tuning needed.

### rawpy `use_auto_wb=True`

**Approach**: Let rawpy's built-in auto white balance handle the flash vs daylight
distinction.

**Problem**: Auto WB correctly neutralized flash scenes (P002008) but over-neutralized
daylight scenes (P002007, P002011, P002015), removing the warm tint that should be
preserved. Mean error went from 11.7 to 25.0.

**Rejected**: Flash-aware params (v18) give better results because they're calibrated
from actual reference pairs rather than rawpy's generic auto WB algorithm.

### Adaptive WB blend (auto_wb for flash, default for daylight)

**Approach**: Compute both default and auto_wb outputs, blend based on WB shift magnitude.

**Problem**: The blend model found that ALL pairs prefer the default over auto_wb.
The auto_wb correction is too aggressive even for flash scenes when measured across
the whole image.

**Rejected**: Flash tag detection + separate calibration (v18) is more precise.

## Current limitations

1. **Linear transform can't capture the U-shaped ratio curve.** The relationship
   between rawpy output and reference has a U-shaped ratio (lower in mid-tones,
   higher in shadows and highlights). A linear transform can't fit this, so mid-tones
   are over-brightened by ~15-25 units. Per-channel curves that could fit this
   reintroduce the green tint problem. The linear approach is the best trade-off.

2. **Flash calibration has only 2 pairs.** P002002 and P002008 are the only good
   flash reference pairs. More flash reference pairs would improve the flash params'
   robustness. The outdoor flash pairs (P002012, P002013, P002015) are excluded
   because their reference output doesn't represent good flash conversion (the flash
   accidentally fired on outdoor daylight scenes).

3. **Non-flash params use global fit (all 8 pairs).** A dedicated non-flash fit
   from only P002007 + P002011 + P002014 was tried but P002007 is an outlier
   (strong R-G separation) that pulled the params badly. The global fit is more
   robust for non-flash scenes.

4. **P002008 highlights are slightly too bright** (+16 units vs reference). The
   flash params' stretch target (median ref p99.5 from flash pairs) extends
   highlights a bit too aggressively for P002008 specifically. This is a minor
   issue compared to the tint problem it fixed.

5. **8-bit dithering is slow** (~16s/file). Floyd-Steinberg is a sequential
   algorithm. Use 16-bit output for batch processing.

## File structure

```
kdc2tiff_project/
├── README.md                              ← quick-start guide
├── PROJECT_DOCUMENTATION.md               ← this file (detailed docs)
├── comparison_8pairs_flash_aware.png      ← visual A/B/C grid for all 8 pairs
├── kdc2tiff.py                            ← main CLI script (self-contained)
├── reference_lut.json                     ← v18 flash-aware params
├── requirements.txt                       ← Python dependencies
└── samples/
    ├── P002002_16bit.tif                  ← flash scene (cat, 16-bit)
    ├── P002002_8bit_dithered.tif          ← flash scene (cat, 8-bit dithered)
    └── P002011_16bit.tif                  ← non-flash scene (daylight, 16-bit)
```

## CLI usage

```bash
# Single file (auto-detects flash from EXIF)
python kdc2tiff.py photo.kdc

# 8-bit output with Floyd-Steinberg dithering
python kdc2tiff.py photo.kdc --bits 8

# Directory mode (recursive, case-insensitive .kdc/.KDC)
python kdc2tiff.py ./kdc_folder/

# Force re-conversion (ignore resume/skip)
python kdc2tiff.py ./kdc_folder/ --overwrite

# Disable stretch (output may look dull)
python kdc2tiff.py photo.kdc --no-stretch

# Disable all color correction (rawpy output only)
python kdc2tiff.py photo.kdc --no-color-correction

# Recalibrate from reference pairs
# Flash pairs should be listed FIRST (the script detects flash from the KDC file,
# not from the order, but listing flash pairs first is good practice)
python kdc2tiff.py --calibrate \
    P002002.KDC P002002.TIF \
    P002008.KDC P002008.TIF \
    P002007.KDC P002007.TIF \
    P002011.KDC P002011.TIF \
    P002014.KDC P002014.TIF
```

## Calibration params format (reference_lut.json v18)

```json
{
  "version": 18,
  "method": "flash_aware_linear_plus_stretch",
  "flash_params": {
    "linear": {
      "R": {"gain": 0.9317, "offset": -33.72},
      "G": {"gain": 0.8531, "offset": -13.24},
      "B": {"gain": 0.9627, "offset": -30.01}
    },
    "stretch": {
      "gains": [1.2063, 1.2249, 1.1489],
      "offsets": [...],
      "target_ref_p99.5_median": [223.0, 183.0, 161.0]
    }
  },
  "nonflash_params": {
    "linear": {
      "R": {"gain": 0.8874, "offset": -35.61},
      "G": {"gain": 0.7901, "offset": -14.63},
      "B": {"gain": 0.8593, "offset": -30.56}
    },
    "stretch": {
      "gains": [1.1723, 1.1723, 1.0755],
      "offsets": [...],
      "target_ref_p99.5_median": [...]
    }
  }
}
```

## Dependencies

- **rawpy** (>= 0.27.0) — LibRaw Python bindings for KDC decoding
- **LibRaw** system library — see installation instructions below for macOS and Linux
- **tifffile** — TIFF file writing
- **Pillow** — image resizing (I;16 mode for 16-bit channels)
- **numpy** — array operations
- **scipy** — (listed in requirements but not currently used in v18; was needed for
  PCHIP in earlier LUT versions)
- **scikit-learn** — LinearRegression for calibration
- **tqdm** — progress bar for directory mode

## Installation and Setup

### macOS

```bash
# 1. Install Homebrew (if not already installed)
#    See https://brew.sh for instructions

# 2. Install LibRaw system library
brew install libraw

# 3. Unzip the project and enter the directory
unzip kdc2tiff_project.zip
cd kdc2tiff_project

# 4. Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 5. Install Python dependencies
pip install -r requirements.txt

# 6. Verify the installation
python kdc2tiff.py --help
```

### Linux (Debian/Ubuntu)

```bash
# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y libraw-dev python3 python3-venv python3-pip

# 2. Unzip the project and enter the directory
unzip kdc2tiff_project.zip
cd kdc2tiff_project

# 3. Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Verify the installation
python kdc2tiff.py --help
```

### Linux (Fedora/RHEL/CentOS)

```bash
# 1. Install system dependencies
sudo dnf install -y LibRaw-devel python3 python3-pip

# 2. Unzip the project and enter the directory
unzip kdc2tiff_project.zip
cd kdc2tiff_project

# 3. Create a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Verify the installation
python kdc2tiff.py --help
```

### Linux (Arch Linux)

```bash
# 1. Install system dependencies
sudo pacman -S libraw python python-pip

# 2. Unzip the project and enter the directory
unzip kdc2tiff_project.zip
cd kdc2tiff_project

# 3. Create a Python virtual environment
python -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Verify the installation
python kdc2tiff.py --help
```

### Troubleshooting

**"ImportError: libraw.so: cannot open shared object file"** (Linux)

The LibRaw shared library isn't in your library path. Try:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```
Or reinstall LibRaw:
```bash
# Debian/Ubuntu
sudo apt-get install --reinstall libraw-dev
# Fedora
sudo dnf reinstall LibRaw-devel
```

**"rawpy ImportError" on macOS with Apple Silicon (M1/M2/M3)**

Make sure you're using the native Python (not Rosetta):
```bash
arch -arm64 brew install libraw
arch -arm64 python3 -m venv venv
arch -arm64 source venv/bin/activate
pip install -r requirements.txt
```

**"Permission denied" when installing packages**

Make sure you've activated the virtual environment first:
```bash
source venv/bin/activate
```
You should see `(venv)` in your terminal prompt. Then retry `pip install`.

**Converting files is slow for 8-bit output**

8-bit output with Floyd-Steinberg dithering takes ~16 seconds per file (the
dithering algorithm is sequential). For batch processing, use 16-bit output
(the default) which takes ~0.1 seconds per file. If you need 8-bit, use
`--no-dither` for fast (but potentially banded) output.

## Ideas for future improvement

1. **More flash reference pairs.** The flash params are calibrated from only 2 pairs.
   More flash-lit reference pairs would improve robustness.

2. **Per-image adaptive tone curve (safe version).** Instead of a global linear
   transform, fit a smooth parametric curve (e.g., 4-parameter sigmoid) that's
   constrained to be the same across all 3 channels (luminance curve) plus a
   per-channel linear tint. This could capture the U-shaped ratio without per-channel
   divergence. Earlier attempts failed due to overfitting, but with more reference
   pairs it might work.

3. **White level handling.** The KDC files have two different white_level values
   (510 for flash group, 255 for daylight group). This might indicate different
   sensor gain modes. Investigating this could improve the linear fit.

4. **Per-image highlight recovery.** For flash scenes where highlights are slightly
   too bright, a highlight-specific desaturation could bring them closer to the
   reference without affecting mid-tones.

5. **GPU-accelerated Floyd-Steinberg dithering.** The current implementation is
   sequential (~16s per file for 8-bit dithered output). A GPU or vectorized
   implementation could make 8-bit dithered output practical for batch processing.
