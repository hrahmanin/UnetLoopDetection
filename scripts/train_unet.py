
from __future__ import annotations

import argparse
from pathlib import Path

from unetloopdetection.config import TrainConfig
from unetloopdetection.train import main as train_main


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="outputs/run_config.yaml")
    p.add_argument("--out", type=str, default="outputs/run")
    return p.parse_args()


def run():
    args = parse_args()
    cfg = TrainConfig.load_yaml(args.config) if Path(args.config).exists() else TrainConfig()
    cfg.run_dir = args.out
    best = train_main(cfg)
    print(str(best))


if __name__ == "__main__":
    run()
