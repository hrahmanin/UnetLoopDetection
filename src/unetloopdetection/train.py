
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .dataset import PatchDataset
from .model_unet import UNet
from .metrics import binarize, precision_recall_f1, iou


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loss(pos_weight: float, device: torch.device) -> nn.Module:
    w = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=w)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, thr: float) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
        ys.append(y.detach().cpu().numpy())
        ps.append(probs.detach().cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    y_pred = binarize(p, thr=thr)
    m = precision_recall_f1(y_true, y_pred)
    m["iou"] = float(iou(y_true, y_pred))
    return m


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * x.shape[0]
        n += int(x.shape[0])
    return total / max(n, 1)


def main(cfg: TrainConfig) -> Path:
    set_seed(cfg.seed)

    device = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")

    train_ds = PatchDataset(cfg.train_npz, has_labels=True)
    val_ds = PatchDataset(cfg.val_npz, has_labels=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = UNet(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        base_channels=cfg.base_channels,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = make_loss(cfg.pos_weight, device)

    best_f1 = -1.0
    best_path = run_dir / "model_best.pt"

    history = []
    for ep in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_m = evaluate(model, val_loader, device, thr=cfg.threshold)
        row = {"epoch": ep, "train_loss": tr_loss, **val_m}
        history.append(row)

        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

        if cfg.save_best and val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, best_path)

    last_path = run_dir / "model_last.pt"
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, last_path)
    return best_path if cfg.save_best else last_path
