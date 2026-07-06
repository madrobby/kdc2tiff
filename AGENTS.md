# AGENTS.md

This file provides guidance for agents working on this project.

## Project Overview

`kdc2tiff` is a Python CLI tool that converts 1990s Kodak KDC raw files to high-quality 16-bit TIFF format. It supports Kodak DC120 and DC50 digital cameras with accurate color correction and EXIF metadata preservation.

## Architecture

### Single Script

The entire tool is implemented in a single file: `scripts/kdc2tiff.py`

This is intentional — the tool is a focused utility with a clear pipeline:

1. Read KDC file (rawpy)
2. Demosaic Bayer pattern
3. Noise reduction
4. Resize with oversampling
5. Apply color correction
6. Write TIFF + EXIF (tifffile + exiftool)

### Key Components

- **`decode_kdc_16bit()`** — Stage 1: demosaic, noise reduction, metadata extraction
- **`resize_16bit_oversampled()`** — Stage 2: 7× bicubic → Lanczos resize
- **`apply_color_correction()`** — Stage 3: per-channel linear transform
- **`exiftool_write_tiff()`** — Stage 4: write image + metadata
- **`convert_one()`** — Orchestrates the full pipeline for one file
- **`main()`** — CLI argument parsing, batch processing loop

### Configuration

- **`scripts/reference_lut.json`** — Color correction parameters (gains/offsets per channel)
- **`scripts/requirements.txt`** — Python dependencies
- **`.gitignore`** — Excludes `.venv/`, `samples/`, `*.tif`, etc.

## Conventions

### Code Style

- Python 3.10+ type hints throughout
- Google-style docstrings
- 4-space indentation
- No trailing whitespace
- Unix line endings

### Naming

- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Classes: `PascalCase` (none currently)
- Private helpers: `_leading_underscore`

### Error Handling

- Log warnings for non-fatal issues (exiftool failures, missing params)
- Log errors with `exc_info=True` for debugging
- Return status strings from `convert_one()` (`"converted"`, `"skipped"`, `"failed: ..."`))

### CLI Design

- Use `argparse` with descriptive help text
- Support both file and directory input
- Provide `--overwrite`, `--verbose` flags
- Progress bar with `tqdm` (colored, `dynamic_ncols=True`)

## Testing

No formal test suite exists. Manual testing with sample KDC files:

```bash
# Test single file
python scripts/kdc2tiff.py samples/DC120/P002002.KDC -o /tmp/test.tif

# Test directory batch
python scripts/kdc2tiff.py samples/DC120/ -o /tmp/test_dir/

# Test with different demosaic
python scripts/kdc2tiff.py samples/DC120/P002002.KDC --demosaic vng

# Test 8-bit output
python scripts/kdc2tiff.py samples/DC120/P002002.KDC --bits 8
```

## Dependencies

### Core

- `rawpy` — LibRaw wrapper for reading KDC files
- `tifffile` — Writing 16-bit TIFF
- `Pillow` — Image resizing (bicubic/Lanczos)

### Image Processing

- `numpy` — Array operations
- `scipy` — Median/Gaussian filtering
- `colour-demosaicing` — Menon2007 demosaic algorithm
- `scikit-learn` — Linear regression for calibration

### UX

- `tqdm` — Progress bar
- `exiftool` (external) — EXIF metadata writing

## External Tools

### exiftool

Required for writing EXIF metadata to TIFF files. Install via:

- macOS: `brew install exiftool`
- Ubuntu/Debian: `sudo apt-get install libimage-exiftool-perl`

If exiftool is missing or fails, the script continues but skips metadata writing (logged as warning).

## Color Correction

The `reference_lut.json` file contains per-channel linear transform parameters:

```json
{
  "version": 20,
  "params": {
    "R": {"gain": 1.234, "offset": 56.78},
    "G": {"gain": 1.0, "offset": 0.0},
    "B": {"gain": 0.876, "offset": -12.34}
  },
  "stretch": {
    "gains": [1.1, 1.0, 0.95],
    "offsets": [0, 0, 0]
  }
}
```

Calibrate from reference pairs:

```bash
python scripts/kdc2tiff.py --calibrate a.KDC a.tif b.KDC b.tif
```

## Camera Support

### DC120

- Model string: `Kodak DC120 ZOOM Digital Camera`
- Output: 1301 × 976
- Pixel aspect: ~1.53 (needs correction)
- Default demosaic: Menon2007

### DC50

- Model string: `Kodak Digital Science DC50 Zoom Camera`
- Output: 768 × 512
- Pixel aspect: 1.0 (square pixels)
- Default demosaic: AHD (rawpy default)

## Git Workflow

- Main branch: `main`
- Remote: `main` → `git@github.com:madrobby/kdc2tiff.git`
- Commit messages: concise, imperative mood
- No force pushes to main

## Common Tasks

### Add a New Demosaic Algorithm

1. Add to `_RAWPY_DEMOSAIC` dict in `kdc2tiff.py`
2. Update `--demosaic` argparse choices
3. Test with sample file
4. Update README if needed

### Change Output Dimensions

1. Update `CAMERA_CONFIGS` dict
2. Adjust `resize_16bit_oversampled()` call in `convert_one()`
3. Update README output table
4. Test with sample file

### Modify Color Correction

1. Edit `reference_lut.json` (or regenerate with `--calibrate`)
2. Version number must match `LUT_VERSION` constant
3. Test visually with sample files

## Troubleshooting

### "Demosaic algorithm X requires GPL pack"

Some algorithms (LMMSE, AMAZE) need GPL2/GPL3 LibRaw packs. Install:

```bash
pip install --no-binary rawpy rawpy
```

Or use available algorithms: `ahd`, `vng`, `ppg`, `menon2007`.

### exiftool Not Found

Script continues without metadata. Install exiftool (see Installation section).

### Banding in 8-bit Output

Use `--bits 16` for full quality, or ensure `--no-dither` is NOT set for 8-bit output.

## Documentation

- `README.md` — User-facing documentation
- `LICENSE` — MIT License
- `PROJECT_DOCUMENTATION.md` — Technical details (pipeline, calibration methodology)
- `AGENTS.md` — This file (agent guidance)
