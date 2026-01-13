
from __future__ import annotations

import argparse
from pathlib import Path

from unetloopdetection.eval import score_npz


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--npz", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/scores.npz")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def run():
    args = parse_args()
    out = score_npz(args.ckpt, args.npz, args.out, batch_size=args.batch_size)
    print(str(out))


if __name__ == "__main__":
    run()
