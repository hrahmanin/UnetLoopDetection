
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def binarize(probs: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (probs >= thr).astype(np.uint8)


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = (y_true > 0).astype(np.uint8).ravel()
    yp = (y_pred > 0).astype(np.uint8).ravel()

    tp = float(((yt == 1) & (yp == 1)).sum())
    tn = float(((yt == 0) & (yp == 0)).sum())
    fp = float(((yt == 0) & (yp == 1)).sum())
    fn = float(((yt == 1) & (yp == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    c = confusion(y_true, y_pred)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-8)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}


def iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = (y_true > 0).astype(np.uint8)
    yp = (y_pred > 0).astype(np.uint8)
    inter = float((yt & yp).sum())
    union = float((yt | yp).sum())
    return inter / (union + 1e-8)
