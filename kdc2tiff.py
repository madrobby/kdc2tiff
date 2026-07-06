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

    # Enable noise reduction
    python kdc2tiff.py photo.kdc --noise-reduction

    # Use Lanczos downscaling with oversampling (default is box)
    python kdc2tiff.py photo.kdc --resize lanczos

    # Single-pass resize without oversampling
    python kdc2tiff.py photo.kdc --no-oversample

    # Use 4x oversampling instead of the default 7x
    python kdc2tiff.py photo.kdc --oversample 4

    # Disable sharpening (default is 0.5)
    python kdc2tiff.py photo.kdc --sharpen 0

    # Strong sharpening
    python kdc2tiff.py photo.kdc --sharpen 0.8

    # Recalibrate from reference pairs
    python kdc2tiff.py --calibrate a.kdc a.tif [b.kdc b.tif ...]
"""

from __future__ import annotations

# Suppress specific warnings: deprecation warnings and colour-science's matplotlib notice
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Matplotlib.*not available.*")

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rawpy
import tifffile
from PIL import Image
from tqdm import tqdm
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007

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
                elif type_id == 5:  # RATIONAL (numerator, denominator)
                    if count == 1 and len(vbytes) >= 8:
                        num = _struct.unpack_from('>I', vbytes, 0)[0]
                        denom = _struct.unpack_from('>I', vbytes, 4)[0]
                        return f"{num}/{denom}" if denom else str(num)
                    return vbytes.hex()[:20]
        return None
    except Exception:
        return None


# TIFF/EXIF standard tag IDs
_KDC_TAG_MAKE = 0x010F
_KDC_TAG_MODEL = 0x0110
_KDC_TAG_DATETIME_ORIGINAL = 0x9003
_KDC_TAG_DATETIME_original = 0x132
_KDC_TAG_EXPOSURE_TIME = 0x829A
_KDC_TAG_FNUMBER = 0x829D
_KDC_TAG_ISO_SPEED = 0x8827
_KDC_TAG_FOCAL_LENGTH = 0x920A
_KDC_TAG_FLASH = 0x9209
_KDC_TAG_EXPOSURE_PROGRAM = 0x8822
_KDC_TAG_WHITE_BALANCE = 0x8298
_KDC_TAG_LIGHT_SOURCE = 0x828F

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
# 16-bit bilateral sharpening (only used during oversampled resize)
# ---------------------------------------------------------------------------
def _bilateral_blur(arr: np.ndarray, radius: int, sigma_color: float) -> np.ndarray:
    """Bilateral blur using OpenCV's native C++ implementation.

    Replaces the previous pure-numpy version which allocated (H, W, C, N)
    intermediate arrays and was both memory-heavy and slow.
    """
    import cv2
    # cv2.bilateralFilter only supports 8u and 32f, so convert uint16 → float32
    arr_f32 = arr.astype(np.float32)
    blurred_f32 = cv2.bilateralFilter(arr_f32, d=0, sigmaColor=sigma_color, sigmaSpace=radius)
    return np.clip(blurred_f32, 0, 65535).astype(np.uint16)


def _sharpen_16bit_bilateral(arr: np.ndarray, amount: float, radius: int = 2, sigma_color: float = 3000.0) -> np.ndarray:
    """Bilateral unsharp mask sharpening for 16-bit RGB images.

    Applies `original + amount * (original - bilateral_blur(original))`.
    The bilateral blur preserves edges, so sharpening produces detail
    enhancement without halo artifacts on high-contrast boundaries.

    Only used during oversampled resize (on the upscaled intermediate image),
    where a 2px radius corresponds to ~0.28px on final output — enough to
    recover demosaic softness without amplifying noise.
    """
    if amount <= 0:
        return arr
    arr_f = arr.astype(np.float64)
    blurred = _bilateral_blur(arr, radius, sigma_color)
    result = arr_f + amount * (arr_f - blurred.astype(np.float64))
    return np.clip(result, 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# 16-bit image resize with configurable oversampling
# ---------------------------------------------------------------------------
def resize_16bit_oversampled(arr_16: np.ndarray, target_w: int, target_h: int, down_algo: str = "box", oversample_factor: int = OVERSAMPLE_FACTOR, denoise_fn=None, sharpening: float = 0.0) -> tuple:
    """Resize a 16-bit RGB array with configurable oversampling for higher quality.

    Pipeline:
      1. Upscale each channel to `oversample_factor`x the target size (bicubic interpolation)
      2. Optional denoising on the upscaled image
      3. Optional bilateral unsharp mask sharpening (on the full RGB array)
      4. Downscale back to the target size (specified algorithm)

    Returns (output_array, timing_dict) with per-substep timings.
    """
    t = time.perf_counter
    timing = {}
    oversample_w = target_w * oversample_factor
    oversample_h = target_h * oversample_factor
    algo = _RESIZE_ALGOS[down_algo]

    # Bicubic upscale all channels at once
    t0 = t()
    oversized = np.zeros((oversample_h, oversample_w, 3), dtype=np.uint16)
    for c in range(3):
        img = Image.fromarray(arr_16[..., c], mode="I;16")
        img_oversized = img.resize((oversample_w, oversample_h), Image.BICUBIC)
        oversized[..., c] = np.array(img_oversized)
    timing["upscale"] = t() - t0

    # Denoise and sharpen on the full RGB array (before per-channel downscale)
    if denoise_fn is not None:
        t0 = t()
        oversized = denoise_fn(oversized)
        timing["denoise"] = t() - t0
    if sharpening > 0:
        t0 = t()
        oversized = _sharpen_16bit_bilateral(oversized, sharpening)
        timing["sharpen"] = t() - t0

    # Downscale each channel
    t0 = t()
    out = np.zeros((target_h, target_w, 3), dtype=np.uint16)
    for c in range(3):
        img = Image.fromarray(oversized[..., c], mode="I;16")
        img_final = img.resize((target_w, target_h), algo)
        out[..., c] = np.array(img_final)
    timing["downscale"] = t() - t0

    return out, timing


def resize_16bit(arr_16: np.ndarray, target_w: int, target_h: int, algo: str = "box") -> np.ndarray:
    """Single-pass resize without oversampling."""
    pil_algo = _RESIZE_ALGOS[algo]
    out = np.zeros((target_h, target_w, 3), dtype=np.uint16)
    for c in range(3):
        img = Image.fromarray(arr_16[..., c], mode="I;16")
        img_resized = img.resize((target_w, target_h), pil_algo)
        out[..., c] = np.array(img_resized)
    return out


# ---------------------------------------------------------------------------
# Per-file conversion
# ---------------------------------------------------------------------------
_RAWPY_DEMOSAIC = {
    "ahd": rawpy.DemosaicAlgorithm.AHD,
    "vng": rawpy.DemosaicAlgorithm.VNG,
    "ppg": rawpy.DemosaicAlgorithm.PPG,
    "lmmse": rawpy.DemosaicAlgorithm.LMMSE,
    "amaze": rawpy.DemosaicAlgorithm.AMAZE,
}

_RESIZE_ALGOS = {
    "box": Image.BOX,
    "hamming": Image.HAMMING,
    "lanczos": Image.LANCZOS,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "nearest": Image.NEAREST,
}


def _available_rawpy_demosaics() -> list[str]:
    """Return list of rawpy demosaic algorithm names available in this build."""
    available = []
    for name, algo in _RAWPY_DEMOSAIC.items():
        try:
            algo.checkSupported()
            available.append(name)
        except Exception:
            pass
    return available


def _denoise_16bit(arr: np.ndarray, flash_fired: bool = False) -> np.ndarray:
    """Apply median filter + FBDD-like noise reduction to a 16-bit RGB array."""
    from scipy.ndimage import median_filter, gaussian_filter

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
    return arr


def decode_kdc_16bit(
    kdc_path: Path,
    flash_fired: bool = False,
    camera: str = "DC120",
    demosaic: Optional[str] = None,
) -> tuple[np.ndarray, dict]:
    """Decode KDC with camera-specific demosaic.

    DC120: Menon2007 demosaic by default (AMAZE-quality, better channel alignment);
           rawpy built-in algorithms (ahd/vng/ppg/lmmse/amaze) also available.
    DC50:  rawpy AHD by default (handles 14-bit sensor gamma correctly);
           other rawpy algorithms available via --demosaic.

    Noise reduction is applied separately in the resize stage (on oversampled data)
    or before single-pass resize. Enable with --noise-reduction.
    """

    # Read raw EXIF tags from KDC header before opening with rawpy
    kdc_make = read_tiff_tag(kdc_path, _KDC_TAG_MAKE) or "Eastman Kodak Company"
    kdc_model = read_tiff_tag(kdc_path, _KDC_TAG_MODEL) or ""
    kdc_datetime = read_tiff_tag(kdc_path, _KDC_TAG_DATETIME_ORIGINAL) or read_tiff_tag(kdc_path, _KDC_TAG_DATETIME_original)
    kdc_exposure_time = read_tiff_tag(kdc_path, _KDC_TAG_EXPOSURE_TIME)
    kdc_fnumber = read_tiff_tag(kdc_path, _KDC_TAG_FNUMBER)
    kdc_focal_length = read_tiff_tag(kdc_path, _KDC_TAG_FOCAL_LENGTH)
    kdc_flash = read_tiff_tag(kdc_path, _KDC_TAG_FLASH)
    kdc_exposure_program = read_tiff_tag(kdc_path, _KDC_TAG_EXPOSURE_PROGRAM)
    kdc_white_balance = read_tiff_tag(kdc_path, _KDC_TAG_WHITE_BALANCE)
    kdc_light_source = read_tiff_tag(kdc_path, _KDC_TAG_LIGHT_SOURCE)

    # Determine effective demosaic algorithm per camera
    if camera == "DC120":
        effective_demosaic = demosaic if demosaic else "menon2007"
    else:
        effective_demosaic = demosaic if demosaic else "ahd"

    use_colour_demosaicing = (effective_demosaic == "menon2007" and camera == "DC120")
    use_rawpy_demosaic = effective_demosaic in _RAWPY_DEMOSAIC

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
            # EXIF tags copied from KDC header
            "camera_make": kdc_make,
            "camera_model": kdc_model.rstrip("\x00"),
            "exif_datetime_original": kdc_datetime,
            "exposure_time": kdc_exposure_time,
            "fnumber": kdc_fnumber,
            "exif_focal_length": kdc_focal_length,
            "flash": kdc_flash,
            "exposure_program": kdc_exposure_program,
            "white_balance": str(kdc_white_balance) if kdc_white_balance else None,
            "light_source": kdc_light_source,
        }

        if use_colour_demosaicing:
            # DC120 Menon2007: manual Bayer pipeline
            raw_bayer = raw.raw_image.copy().astype(np.float64)
            black_level = max(raw.black_level_per_channel[0], 0)
            pattern = raw.raw_pattern
            meta["demosaic_algorithm"] = "Menon2007 (colour-demosaicing)"
            meta["decode_dimensions"] = list(raw_bayer.shape)
        elif use_rawpy_demosaic:
            # rawpy built-in demosaic (any camera)
            rawpy_algo = _RAWPY_DEMOSAIC[effective_demosaic]
            try:
                rawpy_algo.checkSupported()
                arr = raw.postprocess(output_bps=16, demosaic_algorithm=rawpy_algo)
                meta["demosaic_algorithm"] = f"{effective_demosaic.upper()} (rawpy)"
            except Exception:
                log.warning(
                    "--demosaic %s is not available in this rawpy build "
                    "(requires GPL pack); falling back to AHD.",
                    effective_demosaic,
                )
                arr = raw.postprocess(output_bps=16)
                meta["demosaic_algorithm"] = "AHD (rawpy default, fallback)"
            meta["decode_dimensions"] = list(arr.shape[:2])
        else:
            # DC50 default: rawpy AHD
            arr = raw.postprocess(output_bps=16)
            meta["demosaic_algorithm"] = "AHD (rawpy default)"
            meta["decode_dimensions"] = list(arr.shape[:2])

    if use_colour_demosaicing:
        # Menon2007 demosaic for DC120
        raw_bayer = np.clip(raw_bayer - black_level, 0, white_level) / white_level
        color_map = {0: 'R', 1: 'G', 2: 'B', 3: 'G'}
        bayer_str = (color_map.get(pattern[0][0], 'G') + color_map.get(pattern[0][1], 'R') +
                     color_map.get(pattern[1][0], 'B') + color_map.get(pattern[1][1], 'G'))
        pattern_map = {'GRBG': 'GRBG', 'RGGB': 'RGGB', 'BGGR': 'BGGR', 'GBRG': 'GBRG'}
        bayer_pattern = pattern_map.get(bayer_str, 'GRBG')
        demosaiced = demosaicing_CFA_Bayer_Menon2007(raw_bayer, bayer_pattern)
        arr = np.clip(demosaiced * 65535, 0, 65535).astype(np.uint16)

    return arr, meta


def _shutter_to_fraction(seconds: float) -> str:
    """Convert shutter speed in seconds to EXIF-friendly fraction string."""
    if seconds <= 0:
        return "0/1"
    reciprocal = round(1.0 / seconds)
    if reciprocal >= 1:
        return f"1/{reciprocal}"
    # Sub-second (long exposure)
    denom = round(seconds * 1000) / 1000
    if denom == int(denom):
        return f"{int(denom)}/1"
    # Fractional: try to find simple fraction
    denom_int = int(denom * 100)
    import math
    gcd = math.gcd(denom_int, 10000)
    return f"{denom_int//gcd}/{10000//gcd}"


def _timestamp_to_exif(dt) -> str:
    """Format a datetime as EXIF DateTimeOriginal string (with colons)."""
    if isinstance(dt, str):
        # Handle ISO format "1994-01-01T00:03:04" -> "1994:01:01 00:03:04"
        d = dt.replace("T", " ", 1)
        parts = d.split(" ")
        if len(parts) == 2:
            return parts[0].replace("-", ":") + " " + parts[1][:8]
        return d.replace("T", " ").replace("-", ":")[:19]
    return dt.strftime("%Y:%m:%d %H:%M:%S")


def _format_fnumber(aperture: float) -> str:
    """Format f-number as EXIF fraction string."""
    import math
    # Multiply to eliminate decimals
    val = round(aperture * 10)
    gcd = math.gcd(val, 10)
    return f"{val//gcd}/{10//gcd}"


def _format_focallength(mm: float) -> str:
    """Format focal length in mm."""
    if mm == int(mm):
        return f"{int(mm)} mm"
    val = round(mm * 10)
    import math
    gcd = math.gcd(val, 10)
    return f"{val//gcd}/{10//gcd} mm"


def exiftool_write_tiff(arr: np.ndarray, out_path: Path, metadata: dict,
                        bits: int) -> None:
    """Write TIFF image and annotate with EXIF data using exiftool.

    tifffile is used for the image; exiftool writes proper EXIF IFD tags
    so tools like identify, exiftool, and standard image viewers can read
    camera metadata (Make, Model, ExposureTime, FNumber, ISO, etc.).
    """
    if bits == 16:
        if arr.dtype != np.uint16:
            raise ValueError(f"Expected uint16 for 16-bit output, got {arr.dtype}")
        software_tag = "kdc2tiff.py (rawpy 16-bit + per-channel linear color correction)"
    else:
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected uint8 for 8-bit output, got {arr.dtype}")
        software_tag = "kdc2tiff.py (rawpy 16-bit + linear color correction + Floyd-Steinberg dither)"

    tifffile.imwrite(
        str(out_path), arr,
        photometric=TIFF_PHOTOMETRIC,
        compression=TIFF_COMPRESSION,
        resolution=TIFF_DPI,
        resolutionunit="inch",
        software=software_tag,
        shaped=False,
        metadata=None,
    )

    # Build exiftool command with proper EXIF tag mapping
    def _sanitize(v):
        """Remove embedded null bytes from string values for exiftool."""
        if isinstance(v, str):
            return v.replace('\x00', '').rstrip()
        return v

    exif_cmd = ["exiftool", "-overwrite_original"]

    # Camera identity
    make = metadata.get("camera_make") or "Eastman Kodak Company"
    exif_cmd.append(f"-Make={_sanitize(make)}")

    model = _sanitize(metadata.get("camera_model", ""))
    if model:
        exif_cmd.append(f"-Model={model}")

    # DateTimeOriginal from KDC header or rawpy
    dt = metadata.get("exif_datetime_original") or metadata.get("timestamp")
    if dt:
        exif_cmd.append(f"-DateTimeOriginal={_timestamp_to_exif(_sanitize(dt))}")

    # ExposureTime
    exposure = metadata.get("exposure_time") or metadata.get("shutter_speed")
    if exposure:
        if isinstance(exposure, str) and "/" in exposure:
            # Already rational format from KDC header (e.g. "2772/100000")
            exif_cmd.append(f"-ExposureTime={_sanitize(exposure)}")
        else:
            try:
                exif_cmd.append(f"-ExposureTime={_shutter_to_fraction(float(exposure))}")
            except (TypeError, ValueError):
                pass

    # FNumber
    fnumber = metadata.get("fnumber") or metadata.get("aperture")
    if fnumber:
        if isinstance(fnumber, str) and "/" in fnumber:
            exif_cmd.append(f"-FNumber={_sanitize(fnumber)}")
        else:
            try:
                exif_cmd.append(f"-FNumber={_format_fnumber(float(fnumber))}")
            except (TypeError, ValueError):
                pass

    # ISO Speed from KDC header tags or rawpy
    iso = metadata.get("iso_speed")
    if iso:
        try:
            exif_cmd.append(f"-ISO={int(float(iso))}")
        except (TypeError, ValueError):
            pass

    # Focal Length
    fl = metadata.get("exif_focal_length") or metadata.get("focal_length")
    if fl:
        if isinstance(fl, str) and "/" in fl:
            # Already rational format (e.g. "37/1")
            exif_cmd.append("-FocalLength=" + _sanitize(fl) + " mm")
        else:
            try:
                exif_cmd.append(f"-FocalLength={_format_focallength(float(fl))}")
            except (TypeError, ValueError):
                pass

    # White Balance
    wb = metadata.get("white_balance")
    if wb:
        exif_cmd.append(f"-WhiteBalance={_sanitize(wb)}")

    # Light Source
    ls = metadata.get("light_source") or metadata.get("lightsource")
    if ls:
        exif_cmd.append(f"-LightSource={_sanitize(ls)}")

    # Exposure Program
    ep = metadata.get("exposure_program")
    if ep:
        exif_cmd.append(f"-ExposureProgram={ep}")

    # Flash - prefer KDC header value if present
    kdc_flash = metadata.get("flash")
    if kdc_flash:
        try:
            flash_val = int(kdc_flash)
        except (ValueError, TypeError):
            flash_val = 31 if metadata.get("flash_fired") else 24
        exif_cmd.append(f"-Flash={flash_val}")

    # Only invoke exiftool if we have tags to write
    if len(exif_cmd) > 2:
        exif_cmd.append(str(out_path))
        result = subprocess.run(
            exif_cmd, capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            log.warning("exiftool failed for %s (ignoring): %s", out_path, result.stderr[:200])


def _print_timing(timing: dict) -> None:
    """Print a per-stage timing breakdown for a converted file."""
    if not timing:
        return
    parts = [f"{k}: {v:.3f}s" for k, v in timing.items()]
    log.info("  timing: %s", ", ".join(parts))


def convert_one(
    kdc_path: Path, out_path: Path, overwrite: bool,
    color_params: Optional[dict] = None,
    bits: int = 16,
    dither: bool = True,
    stretch: bool = True,
    demosaic: Optional[str] = None,
    noise_reduction: bool = False,
    resize_algo: str = "box",
    oversample_factor: int = OVERSAMPLE_FACTOR,
    sharpening: float = 0.0,
) -> tuple:
    """Convert a single KDC to TIFF.

    Returns (status, timing) where status is "converted", "skipped", or
    "failed: ..." and timing is a dict of {stage_name: seconds}.
    """
    if should_skip(kdc_path, out_path, overwrite):
        return "skipped", {}
    try:
        t = time.perf_counter
        timing = {}

        # Detect camera and flash
        t0 = t()
        camera = detect_camera(kdc_path)
        flash_fired = read_flash_tag(kdc_path)
        cam_config = CAMERA_CONFIGS.get(camera, CAMERA_CONFIGS["DC120"])
        output_w = cam_config["output_width"]
        output_h = cam_config["output_height"]
        timing["camera_detect"] = t() - t0

        # Stage 1: Camera-specific demosaic
        t0 = t()
        arr_16, meta = decode_kdc_16bit(
            kdc_path, flash_fired=flash_fired, camera=camera, demosaic=demosaic,
        )
        meta["camera"] = camera
        meta["flash_fired"] = flash_fired
        timing["demosaic"] = t() - t0

        # Stage 2: resize (camera-specific dimensions), with optional denoising and sharpening on oversampled data
        t0 = t()
        denoise_fn = _denoise_16bit if noise_reduction else None
        if oversample_factor > 1:
            arr_16, resize_subtiming = resize_16bit_oversampled(arr_16, output_w, output_h, down_algo=resize_algo, oversample_factor=oversample_factor, denoise_fn=denoise_fn, sharpening=sharpening)
            meta["resize_method"] = f"{oversample_factor}x_bicubic_up_{resize_algo}_down"
        else:
            t0_resize = t()
            if denoise_fn is not None:
                arr_16 = denoise_fn(arr_16, flash_fired=flash_fired)
            arr_16 = resize_16bit(arr_16, output_w, output_h, algo=resize_algo)
            meta["resize_method"] = f"single_pass_{resize_algo}"
            resize_subtiming = {"resize": t() - t0_resize}
        timing.update(resize_subtiming)

        # Stage 3: apply color correction (flash-aware, camera-specific)
        t0 = t()
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
        timing["color_correction"] = t() - t0

        demosaic_name = meta.get("demosaic_algorithm", "unknown")
        meta["pipeline"] = [f"demosaic_{demosaic_name.lower().replace(' ', '_')}"]
        if noise_reduction:
            if oversample_factor > 1:
                meta["pipeline"].append("denoise_oversampled")
            else:
                meta["pipeline"].append("denoise")
        if sharpening > 0 and oversample_factor > 1:
            meta["pipeline"].append("sharpen_bilateral")
        if oversample_factor > 1:
            meta["pipeline"].append(f"resize_{oversample_factor}x_oversample")
        meta["pipeline"].append(f"resize_{resize_algo}")
        if color_params is not None:
            meta["pipeline"].append("linear_color_correction")
            if stretch and "stretch" in color_params:
                meta["pipeline"].append("percentile_stretch")
        meta["output_bits"] = bits
        if bits == 8:
            meta["pipeline"].append("floyd_steinberg_dither" if dither else "truncate_to_8bit")

        # Stage 4: convert to 8-bit if requested
        t0 = t()
        if bits == 8:
            if dither:
                arr_out = convert_16bit_to_8bit_dithered(arr_16)
            else:
                arr_out = convert_16bit_to_8bit_simple(arr_16)
        else:
            arr_out = arr_16
        timing["convert_bits"] = t() - t0

        # Stage 5: write TIFF + EXIF
        t0 = t()
        exiftool_write_tiff(arr_out, out_path, meta, bits)
        timing["write"] = t() - t0

        return "converted", timing
    except Exception as e:
        log.error("Failed to convert %s: %s", kdc_path, e, exc_info=True)
        return f"failed: {type(e).__name__}: {e}", {}


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
                        help="When --bits 8, disable Floyd-Steinberg dithering (may show banding).")
    parser.add_argument("--no-stretch", action="store_true",
                        help="Disable the percentile-based stretch (output may look dull; useful for comparison).")
    parser.add_argument("--noise-reduction", action="store_true",
                        help="Enable median filter and FBDD noise reduction.")
    resize_choices = list(_RESIZE_ALGOS.keys())
    parser.add_argument(
        "--resize",
        choices=resize_choices,
        default="box",
        help=f"Downscaling algorithm (oversampled path) or single-pass algorithm (with --no-oversample). Default: box.",
    )
    parser.add_argument("--no-oversample", action="store_true",
                        help="Alias for --oversample 1 (single-pass resize only).")
    parser.add_argument("--oversample", type=int, default=7,
                        help="Oversampling factor for upscale-then-downscale resize. Default: 7. Use 1 to disable oversampling.")
    parser.add_argument("--sharpen", type=float, default=0.5,
                        help="Bilateral unsharp mask sharpening strength (0 = off, 0.5 = default). Only applied during oversampled resize.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging.")
    rawpy_choices = list(_RAWPY_DEMOSAIC.keys())
    all_demosaic_choices = ["menon2007"] + rawpy_choices
    parser.add_argument(
        "--demosaic",
        choices=all_demosaic_choices,
        default=None,
        help=f"Demosaic algorithm (default: menon2007 for DC120, ahd for DC50). "
             f"Available rawpy algorithms: {', '.join(rawpy_choices)}.",
    )
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

    # Determine default demosaic per camera
    demo_default = "menon2007"  # DC120 default
    demo_effective = args.demosaic if args.demosaic else demo_default

    log.info("Mode: %s, files: %d, output: %s, bits: %d, dither: %s, color_correction: %s, stretch: %s, demosaic: %s, noise_reduction: %s, resize: %s, oversample: %dx, sharpen: %.2f",
             mode, len(kdc_files),
             out_files[0].parent if mode == "dir" else out_files[0],
             args.bits, "off" if args.no_dither else "on",
             "on" if color_params else "off",
             "off" if args.no_stretch else "on",
             demo_effective,
             "on" if args.noise_reduction else "off",
             args.resize,
             1 if args.no_oversample else args.oversample,
             args.sharpen)

    failures = []
    skipped = 0
    converted = 0
    start = time.time()

    # ANSI color codes for progress bar
    ANSI_GREEN = "\033[92m"
    ANSI_YELLOW = "\033[93m"
    ANSI_RED = "\033[91m"
    ANSI_CYAN = "\033[96m"
    ANSI_RESET = "\033[0m"

    iterator = kdc_files
    if mode == "dir":
        iterator = tqdm(
            kdc_files,
            desc=f"{ANSI_CYAN}Converting{ANSI_RESET}",
            unit="file",
            dynamic_ncols=True,
            bar_format=f"{ANSI_GREEN}{{l_bar}}{ANSI_RESET}{{bar}} {ANSI_YELLOW}{{n_fmt}}/{{total_fmt}}{ANSI_RESET} {ANSI_CYAN}{{elapsed}}<{ANSI_CYAN}{{remaining}}{ANSI_RESET}",
        )

    for kdc, out in zip(iterator, out_files):
        try:
            result, timing = convert_one(kdc, out, args.overwrite, color_params,
                                          bits=args.bits, dither=not args.no_dither,
                                          stretch=not args.no_stretch,
                                          demosaic=args.demosaic,
                                          noise_reduction=args.noise_reduction,
                                          resize_algo=args.resize,
                                          oversample_factor=1 if args.no_oversample else args.oversample,
                                          sharpening=args.sharpen)
        except KeyboardInterrupt:
            log.warning("Interrupted by user.")
            if iterator is not None and hasattr(iterator, 'close'):
                iterator.close()
            return 130
        if result == "converted":
            converted += 1
            if mode == "file":
                log.info(f"{ANSI_GREEN}Converted{ANSI_RESET} %s -> %s (%d-bit)", kdc.name, out, args.bits)
                _print_timing(timing)
        elif result == "skipped":
            skipped += 1
            if mode == "file":
                log.info(f"{ANSI_YELLOW}Skipped{ANSI_RESET} (already converted): %s", out)
        else:
            failures.append((kdc, result))

    if iterator is not None and hasattr(iterator, 'close'):
        iterator.close()

    elapsed = time.time() - start
    log.info(f"Done in {ANSI_CYAN}%.1fs{ANSI_RESET} — converted={ANSI_GREEN}%d{ANSI_RESET}, skipped={ANSI_YELLOW}%d{ANSI_RESET}, failed={ANSI_RED}%d{ANSI_RESET}",
             elapsed, converted, skipped, len(failures))
    if failures:
        log.warning(f"{ANSI_RED}Failed files:{ANSI_RESET}")
        for path, reason in failures:
            log.warning(f"  {ANSI_RED}%s{ANSI_RESET} — %s", path, reason)
        return 2 if converted == 0 else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
