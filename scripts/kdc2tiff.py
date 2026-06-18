#!/usr/bin/env python3
"""
kdc2tiff.py — Convert 1990s Kodak KDC raw files to TIFF.

Clean 16-bit pipeline with per-channel linear color correction (no LUT):

  1. rawpy 16-bit output (output_bps=16) — full 0-65535 precision per channel
  2. Resize to 1280x960 (matches reference Kodak-software dimensions)
  3. Apply per-channel linear transform (output = gain * input + offset)
     calibrated from 8 KDC/TIF pairs using midtone pixels
  4. Save as 16-bit TIFF (default) or 8-bit TIFF with Floyd-Steinberg dithering

Why linear instead of a LUT?
  - Per-channel LUTs (even smooth PCHIP ones) introduce per-channel bias in
    highlights, causing green/seafoam tint in bright areas. The LUTs fit
    each channel independently, so noise in the scatter data causes the
    channels to diverge.
  - A linear transform preserves the input's natural R > G > B relationship
    in highlights (matching the reference's warm/red tint direction).
  - 16-bit precision eliminates banding without needing a complex LUT.

The linear gains/offsets (reference_lut.json, version 10) were calibrated
from 8 KDC/TIF pairs using midtone pixels (input 30-220) to avoid clipped
extremes.

Output formats:
  --bits 16 (default): 16-bit TIFF, max precision, no banding possible
  --bits 8:             8-bit TIFF with Floyd-Steinberg dithering (no banding)
  --bits 8 --no-dither: 8-bit TIFF, simple truncation (may show banding)

Usage:
    # Single file (default 16-bit output)
    python kdc2tiff.py photo.kdc
    python kdc2tiff.py photo.kdc --output /tmp/out.tif

    # 8-bit output with dithering
    python kdc2tiff.py photo.kdc --bits 8

    # Directory mode
    python kdc2tiff.py ./kdc_folder/
    python kdc2tiff.py ./kdc_folder/ --output ./tiff_folder/ --bits 8

    # Disable color correction (rawpy output only)
    python kdc2tiff.py photo.kdc --no-color-correction

    # Recalibrate from reference pairs
    python kdc2tiff.py --calibrate a.kdc a.tif [b.kdc b.tif ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rawpy
import tifffile
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIFF_DPI = (72, 72)
TIFF_PHOTOMETRIC = "rgb"
TIFF_COMPRESSION = None

KDC_EXTENSIONS = {".kdc"}
DEFAULT_LUT_PATH = Path(__file__).resolve().parent / "reference_lut.json"
LUT_VERSION = 20  # Multi-camera support
OVERSAMPLE_FACTOR = 7

# Camera-specific configurations
CAMERA_CONFIGS = {
    "DC120": {
        "model_string": "Kodak DC120 ZOOM Digital Camera",
        "output_width": 1301,
        "output_height": 976,
        "needs_aspect_ratio_correction": True,  # pixel_aspect=1.5346
    },
    "DC50": {
        "model_string": "Kodak Digital Science DC50 Zoom Camera",
        "output_width": 768,
        "output_height": 512,
        "needs_aspect_ratio_correction": False,  # pixel_aspect=1.0
    },
}

log = logging.getLogger("kdc2tiff")


# ---------------------------------------------------------------------------
# Color correction (per-channel linear transform)
# ---------------------------------------------------------------------------
def load_color_params(path: Optional[Path] = None) -> Optional[dict]:
    """Load color correction params from JSON. Supports v11 (single linear+stretch)
    and v18 (flash-aware: separate flash/nonflash params)."""
    if path is None:
        path = DEFAULT_LUT_PATH
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != LUT_VERSION:
        log.warning("Color params version %s does not match expected %s; using anyway",
                    data.get("version"), LUT_VERSION)
    return data


import struct as _struct

def read_tiff_tag(kdc_path: Path, tag_id: int) -> Optional[object]:
    """Read a specific TIFF/EXIF tag value from a KDC file header.
    Returns the value, or None if not found.
    """
    try:
        with open(kdc_path, "rb") as f:
            data = f.read(8192)
        if data[:2] != b'MM':
            return None
        ifd_offset = _struct.unpack_from('>I', data, 4)[0]
        if ifd_offset + 2 > len(data):
            return None
        num_entries = _struct.unpack_from('>H', data, ifd_offset)[0]
        for i in range(num_entries):
            entry_offset = ifd_offset + 2 + i * 12
            if entry_offset + 12 > len(data):
                break
            tag = _struct.unpack_from('>H', data, entry_offset)[0]
            if tag == tag_id:
                type_id = _struct.unpack_from('>H', data, entry_offset + 2)[0]
                count = _struct.unpack_from('>I', data, entry_offset + 4)[0]
                type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}
                vsize = type_sizes.get(type_id, 1) * count
                if vsize <= 4:
                    vbytes = data[entry_offset + 8:entry_offset + 12]
                else:
                    voff = _struct.unpack_from('>I', data, entry_offset + 8)[0]
                    vbytes = data[voff:voff + vsize] if voff + vsize <= len(data) else b''
                if type_id == 2:  # ASCII
                    return vbytes.decode('ascii', errors='replace').rstrip('\x00')
                elif type_id == 3:  # SHORT
                    return _struct.unpack_from('>H', vbytes, 0)[0]
                elif type_id == 4:  # LONG
                    return _struct.unpack_from('>I', vbytes, 0)[0]
        return None
    except Exception:
        return None

def read_flash_tag(kdc_path: Path) -> bool:
    """Read the EXIF Flash tag (0x9209) from a KDC file.
    Returns True if flash fired (bit 0 of the Flash tag is set).
    """
    val = read_tiff_tag(kdc_path, 0x9209)
    return bool(val and (val & 1))

def detect_camera(kdc_path: Path) -> str:
    """Detect camera model from KDC file's EXIF Model tag (0x0110).
    Returns camera key ('DC120', 'DC50') or 'unknown'.
    """
    model = read_tiff_tag(kdc_path, 0x0110)
    if not model:
        return "unknown"
    for key, config in CAMERA_CONFIGS.items():
        if config["model_string"] in model:
            return key
    log.warning("Unknown camera model: %s", model)
    return "unknown"


def get_effective_params(params: dict, flash_fired: bool, camera: str = "DC120") -> dict:
    """Extract the effective linear+stretch params for a given flash state and camera.
    
    For v20 (multi-camera): selects camera-specific flash/nonflash params.
    For v18/v19: uses flash/nonflash params (camera-agnostic).
    """
    if params.get("version") in (18, 19, 20):
        # v20: camera-specific params
        if params.get("version") == 20 and camera in params.get("cameras", {}):
            camera_params = params["cameras"][camera]
        else:
            # v18/v19: single camera params (treat as DC120)
            camera_params = params
        group = "flash_params" if flash_fired else "nonflash_params"
        return {
            "version": 18,
            "params": camera_params[group]["linear"],
            "stretch": camera_params[group]["stretch"],
        }
    else:
        return params


def apply_color_correction(arr_16: np.ndarray, params: dict, stretch: bool = True) -> np.ndarray:
    """Apply per-channel linear transform to a 16-bit RGB array.

    For each channel: output = gain * (input / 256) + offset, then * 256 back to 16-bit.
    This preserves sub-8-bit precision throughout.

    If stretch=True and the params include stretch_gains/stretch_offsets, a
    second linear transform is applied to extend the dynamic range:
      output_stretched = stretch_gain * output + stretch_offset
    This maps the output's p0.5 to 0 and p99.5 to the reference's median p99.5,
    eliminating the "dull" highlights without over-exposing.
    """
    out = np.zeros_like(arr_16, dtype=np.float64)
    for c, name in enumerate("RGB"):
        p = params["params"][name]
        gain = p["gain"]
        offset = p["offset"]
        # 16-bit input -> 8-bit float (preserve full precision)
        x_float = arr_16[..., c].astype(np.float64) / 256.0
        # Apply linear transform
        y_float = gain * x_float + offset
        # Apply stretch if enabled and params include it
        if stretch and "stretch" in params:
            sg = params["stretch"]["gains"][c]
            so = params["stretch"]["offsets"][c]
            y_float = sg * y_float + so
        # Back to 16-bit
        out[..., c] = y_float * 256.0
    return np.clip(out, 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# Floyd-Steinberg dithering (16-bit float -> 8-bit)
# ---------------------------------------------------------------------------
def floyd_steinberg_dither_channel(arr_float: np.ndarray, max_val: int = 255) -> np.ndarray:
    """Floyd-Steinberg dithering for a single 2D channel."""
    work = arr_float.copy().astype(np.float64)
    h, w = work.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            old = work[y, x]
            new = round(np.clip(old, 0, max_val))
            out[y, x] = new
            err = old - new
            if x + 1 < w:
                work[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x > 0:
                    work[y + 1, x - 1] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    work[y + 1, x + 1] += err * 1 / 16
    return out


def convert_16bit_to_8bit_dithered(arr_16: np.ndarray) -> np.ndarray:
    """Convert 16-bit RGB to 8-bit RGB with Floyd-Steinberg dithering."""
    arr_float = arr_16.astype(np.float64) / 256.0
    out_8 = np.zeros((arr_16.shape[0], arr_16.shape[1], 3), dtype=np.uint8)
    for c, name in enumerate("RGB"):
        log.debug("  Dithering %s channel...", name)
        out_8[..., c] = floyd_steinberg_dither_channel(arr_float[..., c])
    return out_8


def convert_16bit_to_8bit_simple(arr_16: np.ndarray) -> np.ndarray:
    """Convert 16-bit RGB to 8-bit RGB by taking the high byte (no dithering)."""
    return (arr_16 >> 8).astype(np.uint8)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate_from_pairs(kdc_paths: list[Path], ref_paths: list[Path], out_path: Path) -> Path:
    """Build per-channel linear color correction params from KDC/TIF pairs."""
    from sklearn.linear_model import LinearRegression

    if len(kdc_paths) != len(ref_paths):
        raise ValueError(f"KDC count ({len(kdc_paths)}) must match reference count ({len(ref_paths)})")
    if not kdc_paths:
        raise ValueError("No source/reference pairs provided")

    log.info("Loading %d pair(s) for calibration...", len(kdc_paths))
    all_data = []
    for kdc, ref in zip(kdc_paths, ref_paths):
        with rawpy.imread(str(kdc)) as raw:
            d_16 = raw.postprocess(output_bps=16)
        with Image.open(ref) as img:
            r = np.array(img)
        d_8 = (d_16 >> 8).astype(np.uint8)
        d_r = np.array(Image.fromarray(d_8).resize((r.shape[1], r.shape[0]), Image.BILINEAR))
        all_data.append((d_r, r))

    log.info("Fitting per-channel linear (output = gain * input + offset) using midtone pixels...")
    gains = {}
    offsets = {}
    for c, name in enumerate("RGB"):
        all_d = []
        all_r = []
        for d, r in all_data:
            mask = (d[..., c] >= 30) & (d[..., c] <= 220)
            all_d.append(d[..., c][mask].astype(np.float64))
            all_r.append(r[..., c][mask].astype(np.float64))
        all_d = np.concatenate(all_d).reshape(-1, 1)
        all_r = np.concatenate(all_r)
        if len(all_d) > 200000:
            idx = np.random.choice(len(all_d), 200000, replace=False)
            all_d = all_d[idx]
            all_r = all_r[idx]
        reg = LinearRegression().fit(all_d, all_r)
        gains[name] = reg.coef_[0]
        offsets[name] = reg.intercept_
        log.info("  %s: gain=%.4f, offset=%+.2f", name, gains[name], offsets[name])

    # Step 2: Compute percentile-based stretch parameters
    # After applying the linear color correction, the output's highlights are
    # compressed (dull). The stretch extends them to match the reference's
    # dynamic range without over-exposing.
    log.info("Computing percentile-based stretch parameters...")
    out_percentiles_low = [[] for _ in range(3)]   # per-channel lists of p0.5
    out_percentiles_high = [[] for _ in range(3)]  # per-channel lists of p99.5
    ref_percentiles_high = [[] for _ in range(3)]  # per-channel lists of ref p99.5
    for d_r, r in all_data:
        # Apply the linear color correction we just fit
        out_float = np.zeros_like(d_r, dtype=np.float64)
        for c in range(3):
            name = "RGB"[c]
            out_float[..., c] = np.clip(gains[name] * d_r[..., c].astype(np.float64) + offsets[name], 0, 255)
        for c in range(3):
            out_low, out_high = np.percentile(out_float[..., c], [0.5, 99.5])
            ref_high = np.percentile(r[..., c].astype(np.float64), 99.5)
            out_percentiles_low[c].append(out_low)
            out_percentiles_high[c].append(out_high)
            ref_percentiles_high[c].append(ref_high)

    # Compute average output percentiles (source for stretch mapping)
    avg_out_low = [np.mean(out_percentiles_low[c]) for c in range(3)]
    avg_out_high = [np.mean(out_percentiles_high[c]) for c in range(3)]
    # Use MEDIAN reference p99.5 as the target (robust to outliers)
    median_ref_high = [np.median(ref_percentiles_high[c]) for c in range(3)]

    # Per-channel stretch: out_stretched = gain * out + offset
    # Maps avg_out_low -> 0 and avg_out_high -> median_ref_high
    stretch_gains = []
    stretch_offsets = []
    for c in range(3):
        out_low = avg_out_low[c]
        out_high = avg_out_high[c]
        ref_high = median_ref_high[c]
        if out_high > out_low:
            gain = ref_high / (out_high - out_low)
        else:
            gain = 1.0
        offset = -out_low * gain
        stretch_gains.append(gain)
        stretch_offsets.append(offset)
        log.info("  Stretch %s: gain=%.4f, offset=%+.2f (maps %.1f->0, %.1f->%.1f)",
                 "RGB"[c], gain, offset, out_low, out_high, gain * out_high + offset)

    params = {
        "version": LUT_VERSION,
        "channels": ["R", "G", "B"],
        "method": "linear_fit_midtone_16bit_with_stretch",
        "params": {name: {"gain": gains[name], "offset": offsets[name]} for name in "RGB"},
        "stretch": {
            "gains": stretch_gains,
            "offsets": stretch_offsets,
            "source_out_p0.5": avg_out_low,
            "source_out_p99.5": avg_out_high,
            "target_ref_p99.5_median": median_ref_high,
            "description": (
                "Per-channel linear stretch applied after color correction. "
                "Maps output p0.5 -> 0 and p99.5 -> median reference p99.5. "
                "Eliminates 'dull' highlights without over-exposing."
            ),
        },
        "input_pipeline": "rawpy 16-bit default + resize to reference dimensions",
        "fit_range": "input 30-220 (midtone pixels only, avoids clipped extremes)",
        "output_range": "16-bit (0-65535), preserving sub-8-bit precision",
        "num_pairs": len(kdc_paths),
        "description": (
            "Per-channel linear transform (output = gain * input + offset) fit using midtone "
            "pixels, plus a percentile-based stretch that extends the output's dynamic range "
            "to match the reference. Applied in 16-bit float space. Preserves the input's "
            "natural R > G > B relationship in highlights, eliminating the green/seafoam tint "
            "that per-channel LUTs introduced, while the stretch eliminates dullness."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    log.info("Color params written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------
def resolve_input(input_path: Path, output_arg: Optional[Path]) -> tuple[str, list[Path], list[Path]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in KDC_EXTENSIONS:
            raise ValueError(f"Input file is not a .kdc file: {input_path}")
        if output_arg is None:
            out = input_path.with_suffix(".tif")
        elif output_arg.suffix.lower() == ".tif":
            out = output_arg
        elif output_arg.is_dir() or not output_arg.suffix:
            out = output_arg / f"{input_path.stem}.tif"
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out = output_arg
        out.parent.mkdir(parents=True, exist_ok=True)
        return "file", [input_path], [out]

    kdc_files = sorted(
        p for p in input_path.rglob("*") if p.suffix.lower() in KDC_EXTENSIONS and p.is_file()
    )
    if not kdc_files:
        raise FileNotFoundError(f"No .kdc files found under {input_path}")

    out_dir = output_arg if output_arg is not None else input_path / "_converted"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_files = []
    for kdc in kdc_files:
        rel = kdc.relative_to(input_path)
        out = out_dir / rel.with_suffix(".tif")
        out.parent.mkdir(parents=True, exist_ok=True)
        out_files.append(out)
    return "dir", kdc_files, out_files


def should_skip(kdc_path: Path, out_path: Path, overwrite: bool) -> bool:
    if overwrite:
        return False
    if not out_path.exists() or out_path.stat().st_size == 0:
        return False
    return out_path.stat().st_mtime >= kdc_path.stat().st_mtime


# ---------------------------------------------------------------------------
# 16-bit image resize with 7x oversampling
# ---------------------------------------------------------------------------
OVERSAMPLE_FACTOR = 7

def resize_16bit_oversampled(arr_16: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize a 16-bit RGB array with 7x oversampling for higher quality.

    Pipeline:
      1. Upscale each channel to 7x the target size (bicubic interpolation)
      2. Downscale back to the target size (Lanczos interpolation)

    This produces smoother results than a single-step resize, especially
    for the non-uniform aspect ratio correction (848→1301 wide, 976→976 tall).
    The oversampling gives the interpolator more data to work with, reducing
    aliasing and blocking artifacts.
    """
    oversample_w = target_w * OVERSAMPLE_FACTOR
    oversample_h = target_h * OVERSAMPLE_FACTOR

    out = np.zeros((target_h, target_w, 3), dtype=np.uint16)
    for c in range(3):
        img = Image.fromarray(arr_16[..., c], mode="I;16")
        # Step 1: upscale to 4x using bicubic (smooth upsampling)
        img_oversized = img.resize((oversample_w, oversample_h), Image.BICUBIC)
        # Step 2: downscale to target using Lanczos (sharp downsampling)
        img_final = img_oversized.resize((target_w, target_h), Image.LANCZOS)
        out[..., c] = np.array(img_final)
    return out


def resize_16bit(arr_16: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Simple resize without oversampling (fallback)."""
    out = np.zeros((target_h, target_w, 3), dtype=np.uint16)
    for c in range(3):
        img = Image.fromarray(arr_16[..., c], mode="I;16")
        img_resized = img.resize((target_w, target_h), Image.BILINEAR)
        out[..., c] = np.array(img_resized)
    return out


# ---------------------------------------------------------------------------
# Per-file conversion
# ---------------------------------------------------------------------------
def decode_kdc_16bit(kdc_path: Path, flash_fired: bool = False, camera: str = "DC120") -> tuple[np.ndarray, dict]:
    """Decode KDC with camera-specific demosaic and noise reduction.

    DC120: Menon2007 demosaic (AMAZE-quality) — better channel alignment
    DC50:  rawpy default demosaic (AHD) — handles 14-bit sensor gamma correctly

    Both: median filter (1 pass), FBDD denoise for non-flash.
    """
    from scipy.ndimage import median_filter, gaussian_filter

    with rawpy.imread(str(kdc_path)) as raw:
        other = raw.other
        sizes = raw.sizes
        white_level = raw.white_level if raw.white_level else 510
        meta = {
            "iso_speed": float(other.iso_speed) if other.iso_speed else None,
            "shutter_speed": float(other.shutter_speed) if other.shutter_speed else None,
            "aperture": float(other.aperture) if other.aperture else None,
            "focal_length": float(other.focal_length) if other.focal_length else None,
            "timestamp": other.timestamp.isoformat() if other.timestamp else None,
            "color_desc": raw.color_desc.decode("ascii", errors="replace"),
            "pixel_aspect": float(sizes.pixel_aspect),
            "raw_width": int(sizes.raw_width),
            "raw_height": int(sizes.raw_height),
            "camera_whitebalance": [float(x) for x in raw.camera_whitebalance],
            "daylight_whitebalance": [float(x) for x in raw.daylight_whitebalance],
            "white_level": int(white_level),
            "source_file": str(kdc_path),
            "source_bytes": kdc_path.stat().st_size,
            "resize_method": "7x_oversample_bicubic_lanczos",
            "noise_reduction": {
                "median_filter_passes": 1,
                "fbdd_noise_reduction": not flash_fired,
            },
        }

        if camera == "DC50":
            # DC50: use rawpy's default demosaic (handles 14-bit gamma correctly)
            arr = raw.postprocess(output_bps=16)
            meta["demosaic_algorithm"] = "AHD (rawpy default)"
            meta["decode_dimensions"] = list(arr.shape[:2])
        else:
            # DC120: use Menon2007 demosaic (AMAZE-quality, better channel alignment)
            from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
            raw_bayer = raw.raw_image.copy().astype(np.float64)
            black_level = max(raw.black_level_per_channel[0], 0)
            pattern = raw.raw_pattern
            meta["demosaic_algorithm"] = "Menon2007"
            meta["decode_dimensions"] = list(raw_bayer.shape)

    if camera != "DC50":
        # Menon2007 demosaic for DC120
        raw_bayer = np.clip(raw_bayer - black_level, 0, white_level) / white_level
        color_map = {0: 'R', 1: 'G', 2: 'B', 3: 'G'}
        bayer_str = (color_map.get(pattern[0][0], 'G') + color_map.get(pattern[0][1], 'R') +
                     color_map.get(pattern[1][0], 'B') + color_map.get(pattern[1][1], 'G'))
        pattern_map = {'GRBG': 'GRBG', 'RGGB': 'RGGB', 'BGGR': 'BGGR', 'GBRG': 'GBRG'}
        bayer_pattern = pattern_map.get(bayer_str, 'GRBG')
        demosaiced = demosaicing_CFA_Bayer_Menon2007(raw_bayer, bayer_pattern)
        arr = np.clip(demosaiced * 65535, 0, 65535).astype(np.uint16)

    # Median filter (1 pass) — smooths zipper artifacts and noise in flat areas
    for c in range(3):
        arr[..., c] = median_filter(arr[..., c], size=3, mode='reflect')

    # FBDD-like noise reduction for non-flash scenes (dark scenes have more noise)
    if not flash_fired:
        arr_float = arr.astype(np.float64)
        for c in range(3):
            ch = arr_float[..., c]
            blurred = gaussian_filter(ch, sigma=0.8)
            local_std = median_filter(np.abs(ch - blurred), size=5)
            blend = np.clip(1.0 - local_std / 500.0, 0, 1) * 0.5
            arr_float[..., c] = ch * (1 - blend) + blurred * blend
        arr = np.clip(arr_float, 0, 65535).astype(np.uint16)

    return arr, meta


def write_tiff(arr: np.ndarray, out_path: Path, metadata: dict, bits: int) -> None:
    if bits == 16:
        if arr.dtype != np.uint16:
            raise ValueError(f"Expected uint16 for 16-bit output, got {arr.dtype}")
        software_tag = "kdc2tiff.py (rawpy 16-bit + per-channel linear color correction)"
    else:
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected uint8 for 8-bit output, got {arr.dtype}")
        software_tag = "kdc2tiff.py (rawpy 16-bit + linear color correction + Floyd-Steinberg dither)"

    description = json.dumps(metadata, ensure_ascii=False, default=str)
    tifffile.imwrite(
        str(out_path), arr,
        photometric=TIFF_PHOTOMETRIC,
        compression=TIFF_COMPRESSION,
        resolution=TIFF_DPI,
        resolutionunit="inch",
        description=description,
        software=software_tag,
        shaped=False,
        metadata=None,
    )


def convert_one(
    kdc_path: Path, out_path: Path, overwrite: bool,
    color_params: Optional[dict] = None,
    bits: int = 16,
    dither: bool = True,
    stretch: bool = True,
) -> str:
    """Convert a single KDC to TIFF."""
    if should_skip(kdc_path, out_path, overwrite):
        return "skipped"
    try:
        # Detect camera and flash
        camera = detect_camera(kdc_path)
        flash_fired = read_flash_tag(kdc_path)
        cam_config = CAMERA_CONFIGS.get(camera, CAMERA_CONFIGS["DC120"])
        output_w = cam_config["output_width"]
        output_h = cam_config["output_height"]

        # Stage 1: Camera-specific demosaic + median filter + FBDD (flash-aware)
        arr_16, meta = decode_kdc_16bit(kdc_path, flash_fired=flash_fired, camera=camera)
        meta["camera"] = camera

        # Stage 2: 7x oversampled resize (camera-specific dimensions)
        arr_16 = resize_16bit_oversampled(arr_16, output_w, output_h)

        # Stage 3: apply color correction (flash-aware, camera-specific)
        if color_params is not None:
            effective = get_effective_params(color_params, flash_fired, camera)
            arr_16 = apply_color_correction(arr_16, effective, stretch=stretch)
            meta["color_correction"] = {
                "applied": True,
                "method": color_params.get("method"),
                "camera": camera,
                "flash_fired": flash_fired,
                "params_used": "flash" if (color_params.get("version") in (18, 19, 20) and flash_fired) else "nonflash" if color_params.get("version") in (18, 19, 20) else "global",
            }
        else:
            meta["color_correction"] = {"applied": False}

        meta["pipeline"] = ["menon2007_demosaic", "resize_7x_oversample"]
        if color_params is not None:
            meta["pipeline"].append("linear_color_correction")
            if stretch and "stretch" in color_params:
                meta["pipeline"].append("percentile_stretch")
        meta["output_bits"] = bits
        if bits == 8:
            meta["pipeline"].append("floyd_steinberg_dither" if dither else "truncate_to_8bit")

        # Stage 4: convert to 8-bit if requested
        if bits == 8:
            if dither:
                arr_out = convert_16bit_to_8bit_dithered(arr_16)
            else:
                arr_out = convert_16bit_to_8bit_simple(arr_16)
        else:
            arr_out = arr_16

        write_tiff(arr_out, out_path, meta, bits)
        return "converted"
    except Exception as e:
        log.error("Failed to convert %s: %s", kdc_path, e, exc_info=True)
        return f"failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kdc2tiff.py",
        description="Convert 1990s Kodak KDC raw files to TIFF (16-bit, no banding, no tint).",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--calibrate", nargs="+", metavar="KDC TIF",
        help="Rebuild reference_lut.json from one or more KDC + reference TIFF pairs, then exit.",
    )
    parser.add_argument("input", type=Path, nargs="?",
                        help="Path to a single .kdc file OR a directory of .kdc files")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output file (file mode) or directory (dir mode).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reconvert files even if the output already exists and is newer.")
    parser.add_argument("--no-color-correction", action="store_true",
                        help="Disable color correction (rawpy output only).")
    parser.add_argument("--params", type=Path, default=None,
                        help=f"Path to color correction params JSON (default: {DEFAULT_LUT_PATH}).")
    parser.add_argument("--bits", type=int, choices=[8, 16], default=16,
                        help="Output bit depth: 16 (default, no banding) or 8 (with dithering).")
    parser.add_argument("--no-dither", action="store_true",
                        help="When --bits 8, disable Floyd-Steinberg dithering (faster but may band).")
    parser.add_argument("--no-stretch", action="store_true",
                        help="Disable the percentile-based stretch (output may look dull; useful for comparison).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    # Special mode: calibrate and exit
    if args.calibrate:
        if len(args.calibrate) < 2 or len(args.calibrate) % 2 != 0:
            log.error("--calibrate requires an even number of args: KDC1 TIF1 [KDC2 TIF2 ...]")
            return 1
        kdc_paths = [Path(p) for p in args.calibrate[::2]]
        ref_paths = [Path(p) for p in args.calibrate[1::2]]
        for kdc, ref in zip(kdc_paths, ref_paths):
            if not kdc.exists() or not ref.exists():
                log.error("Both KDC and reference TIFF must exist (missing %s or %s)", kdc, ref)
                return 1
        out = args.params if args.params else DEFAULT_LUT_PATH
        log.info("Calibrating color correction from %d pair(s) -> %s", len(kdc_paths), out)
        try:
            calibrate_from_pairs(kdc_paths, ref_paths, Path(out))
            return 0
        except Exception as e:
            log.error("Calibration failed: %s", e, exc_info=True)
            return 1

    if args.input is None:
        parser.error("the following arguments are required: input (or use --calibrate)")

    # Load color params
    color_params = None
    if not args.no_color_correction:
        params_path = args.params if args.params else DEFAULT_LUT_PATH
        color_params = load_color_params(Path(params_path))
        if color_params is None:
            log.warning("Color params not found at %s; color correction disabled.", params_path)
        else:
            log.info("Color correction enabled (method %s, cameras: %s)",
                     color_params.get("method"),
                     ", ".join(color_params.get("cameras", {}).keys()) or "DC120")

    try:
        mode, kdc_files, out_files = resolve_input(args.input, args.output)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1

    log.info("Mode: %s, files: %d, output: %s, bits: %d, dither: %s, color_correction: %s, stretch: %s",
             mode, len(kdc_files),
             out_files[0].parent if mode == "dir" else out_files[0],
             args.bits, "off" if args.no_dither else "on",
             "on" if color_params else "off",
             "off" if args.no_stretch else "on")

    failures = []
    skipped = 0
    converted = 0
    start = time.time()

    iterator = kdc_files
    if mode == "dir":
        iterator = tqdm(kdc_files, desc="Converting", unit="file")

    for kdc, out in zip(iterator, out_files):
        try:
            result = convert_one(kdc, out, args.overwrite, color_params,
                                 bits=args.bits, dither=not args.no_dither,
                                 stretch=not args.no_stretch)
        except KeyboardInterrupt:
            log.warning("Interrupted by user.")
            return 130
        if result == "converted":
            converted += 1
            if mode == "file":
                log.info("Converted %s -> %s (%d-bit)", kdc.name, out, args.bits)
        elif result == "skipped":
            skipped += 1
            if mode == "file":
                log.info("Skipped (already converted): %s", out)
        else:
            failures.append((kdc, result))

    elapsed = time.time() - start
    log.info("Done in %.1fs — converted=%d, skipped=%d, failed=%d",
             elapsed, converted, skipped, len(failures))
    if failures:
        log.warning("Failed files:")
        for path, reason in failures:
            log.warning("  %s — %s", path, reason)
        return 2 if converted == 0 else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
