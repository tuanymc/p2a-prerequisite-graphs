# -*- coding: utf-8 -*-
"""Pack subset of editorial helpers for official B7 rebuild. Not the N1–N7 eval."""
from __future__ import annotations

import math

T_B5 = 3.3
TAU = 0.55


def _temp(p: float, t: float) -> float:
    p = min(1 - 1e-6, max(1e-6, float(p)))
    z = math.log(p / (1 - p)) / max(t, 1e-3)
    return 1.0 / (1.0 + math.exp(-z))


def _prf_sets(pred: set, gold: set) -> dict:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "n_pred": len(pred),
        "n_gold": len(gold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }
