
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class PatchSpec:
    size: int = 64
    stride: int = 32
    pad: bool = False


def _maybe_pad2d(x: np.ndarray, size: int, stride: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad bottom/right so sliding window covers the full image."""
    H, W = x.shape[-2], x.shape[-1]
    outH = ((H - size + stride - 1) // stride) * stride + size
    outW = ((W - size + stride - 1) // stride) * stride + size
    padH = max(0, outH - H)
    padW = max(0, outW - W)
    if padH == 0 and padW == 0:
        return x, (0, 0)
    xp = np.pad(x, ((0, 0), (0, padH), (0, padW)) if x.ndim == 3 else ((0, padH), (0, padW)), mode="constant")
    return xp, (padH, padW)


def extract_patches2d(x: np.ndarray, spec: PatchSpec) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Extract sliding window patches from a 2D array.

    Returns:
      patches: (N, size, size)
      coords: list of (top, left) for each patch in the (possibly padded) image.
    """
    if x.ndim != 2:
        raise ValueError("extract_patches2d expects a 2D array")
    img = x
    if spec.pad:
        img, _ = _maybe_pad2d(img, spec.size, spec.stride)

    H, W = img.shape
    patches = []
    coords: List[Tuple[int, int]] = []
    for i in range(0, H - spec.size + 1, spec.stride):
        for j in range(0, W - spec.size + 1, spec.stride):
            patches.append(img[i:i + spec.size, j:j + spec.size])
            coords.append((i, j))
    return np.stack(patches, axis=0), coords


def extract_pair_patches(x: np.ndarray, y: np.ndarray, spec: PatchSpec) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Extract aligned patches from x and y (both 2D)."""
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
    px, coords = extract_patches2d(x, spec)
    py, _ = extract_patches2d(y, spec)
    return px, py, coords


def stitch_patches(patches: np.ndarray, coords: List[Tuple[int, int]], out_shape: Tuple[int, int], spec: PatchSpec) -> np.ndarray:
    """Stitch patches back into an image using simple averaging in overlap regions."""
    H, W = out_shape
    acc = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)
    for p, (i, j) in zip(patches, coords):
        acc[i:i + spec.size, j:j + spec.size] += p.astype(np.float32)
        cnt[i:i + spec.size, j:j + spec.size] += 1.0
    cnt = np.maximum(cnt, 1.0)
    return acc / cnt
