
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import load_patches_npz, ensure_float01


class PatchDataset(Dataset):
    """Dataset for (X, y) stored in a single .npz file.

    Expected shapes:
      X: (N, H, W) or (N, C, H, W)  (float or int)
      y: (N, H, W) or (N, 1, H, W)  (0/1 mask) (optional for inference)
    """

    def __init__(self, npz_path: str | Path, has_labels: bool = True):
        self.npz_path = Path(npz_path)
        d = load_patches_npz(self.npz_path)
        self.X = d["X"]
        self.y = d.get("y", None)
        self.has_labels = has_labels and (self.y is not None)

        if self.X.ndim == 3:
            self.X = self.X[:, None, :, :]  # add channel dim
        if self.has_labels and self.y.ndim == 3:
            self.y = self.y[:, None, :, :]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        x = ensure_float01(self.X[idx])
        x = torch.from_numpy(x).float()

        if self.has_labels:
            y = (self.y[idx] > 0).astype(np.float32)
            y = torch.from_numpy(y).float()
            return x, y
        return x
