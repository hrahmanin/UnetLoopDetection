
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TrainConfig:
    seed: int = 1337
    device: str = "cuda"
    num_workers: int = 4

    # data
    train_npz: str = "data/train_patches.npz"
    val_npz: str = "data/val_patches.npz"
    batch_size: int = 16

    # model
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    depth: int = 4
    dropout: float = 0.0

    # optimization
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20

    # loss / thresholding
    pos_weight: float = 1.0
    threshold: float = 0.5

    # outputs
    run_dir: str = "outputs/run"
    save_best: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainConfig":
        cfg = TrainConfig()
        for k, v in d.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @staticmethod
    def load_yaml(path: str | Path) -> "TrainConfig":
        path = Path(path)
        with path.open("r") as f:
            d = yaml.safe_load(f) or {}
        return TrainConfig.from_dict(d)
