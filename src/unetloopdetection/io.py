
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None


def load_array(path: str | Path) -> np.ndarray:
    """Load 2D/3D array from .npy/.npz or an image (png/tif) if imageio is available."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npy":
        return np.load(path)
    if suf == ".npz":
        z = np.load(path)
        # common convention: first array in archive
        key = list(z.keys())[0]
        return z[key]
    if suf in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
        if iio is None:
            raise ImportError("imageio is required to load images")
        arr = iio.imread(path)
        return np.asarray(arr)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_patches_npz(path: str | Path) -> Dict[str, np.ndarray]:
    """Expect keys: X, y (optional), meta (optional)."""
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    return out


def ensure_float01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype.kind in {"u", "i"}:
        x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / (x.max() + 1e-8)
    return x.astype(np.float32)
