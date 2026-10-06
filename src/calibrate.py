# -*- coding: utf-8 -*-
"""S5 — hiệu chỉnh confidence trên val, không dùng test / không dùng evidence test."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _binary_gold(row: dict[str, Any]) -> int:
    return 1 if row.get("gold_label") in {"direct", "transitive"} else 0


def _probs(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([float(r.get("confidence_raw") or r.get("p_A_to_B") or 0.0) for r in rows], dtype=float)


def fit_calibrator(val_rows: list[dict[str, Any]], method: str = "temperature"):
    p = np.clip(_probs(val_rows), 1e-6, 1 - 1e-6)
    y = np.array([_binary_gold(r) for r in val_rows], dtype=float)
    if len(set(y.tolist())) < 2:
        return ("identity", None)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        return ("isotonic", iso)
    if method == "platt":
        lr = LogisticRegression()
        lr.fit(p.reshape(-1, 1), y)
        return ("platt", lr)
    # temperature: logit / T
    logit = np.log(p / (1 - p))

    def nll(t: float) -> float:
        z = logit / max(t, 1e-3)
        q = 1 / (1 + np.exp(-z))
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    best_t, best = 1.0, nll(1.0)
    for t in np.linspace(0.3, 5.0, 48):
        v = nll(float(t))
        if v < best:
            best, best_t = v, float(t)
    return ("temperature", best_t)


def apply_calibrator(rows: list[dict[str, Any]], pack) -> list[dict[str, Any]]:
    kind, model = pack
    out = []
    for r in rows:
        q = dict(r)
        p = float(q.get("confidence_raw") or q.get("p_A_to_B") or 0.0)
        p = min(1 - 1e-6, max(1e-6, p))
        if kind == "identity" or model is None:
            cal = p
        elif kind == "temperature":
            logit = math.log(p / (1 - p))
            cal = 1 / (1 + math.exp(-logit / model))
        elif kind == "isotonic":
            cal = float(model.predict([p])[0])
        else:
            cal = float(model.predict_proba([[p]])[0, 1])
        q["confidence_calibrated"] = round(float(cal), 4)
        out.append(q)
    return out
