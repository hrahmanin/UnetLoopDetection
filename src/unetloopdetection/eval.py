
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PatchDataset
from .model_unet import UNet
from .metrics import binarize, precision_recall_f1, iou


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> Tuple[UNet, Dict]:
    ckpt = torch.load(Path(checkpoint_path), map_location=device)
    cfg = ckpt.get("config", {})
    model = UNet(
        in_channels=int(cfg.get("in_channels", 1)),
        out_channels=int(cfg.get("out_channels", 1)),
        base_channels=int(cfg.get("base_channels", 32)),
        depth=int(cfg.get("depth", 4)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def score_npz(checkpoint_path: str | Path, npz_path: str | Path, out_path: str | Path, batch_size: int = 32, num_workers: int = 2) -> Path:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(checkpoint_path, device=device)

    ds = PatchDataset(npz_path, has_labels=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    probs = []
    for x in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)

    p = np.concatenate(probs, axis=0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, probs=p)
    return out_path


@torch.no_grad()
def evaluate_npz(checkpoint_path: str | Path, npz_path: str | Path, thr: float = 0.5, batch_size: int = 32, num_workers: int = 2) -> Dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(checkpoint_path, device=device)

    ds = PatchDataset(npz_path, has_labels=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(y.numpy())
        ps.append(p)

    y_true = np.concatenate(ys, axis=0)
    probs = np.concatenate(ps, axis=0)
    y_pred = binarize(probs, thr=thr)

    m = precision_recall_f1(y_true, y_pred)
    m["iou"] = float(iou(y_true, y_pred))
    return m
