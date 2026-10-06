# -*- coding: utf-8 -*-
"""Chỉ số cặp, hướng, calibration, đồ thị."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _y(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([1 if r.get("gold_label") in {"direct", "transitive"} else 0 for r in rows])


def _p(rows: list[dict[str, Any]], key: str = "confidence_calibrated") -> np.ndarray:
    vals = []
    for r in rows:
        v = r.get(key)
        if v is None:
            v = r.get("confidence_raw") or r.get("p_A_to_B") or 0.0
        vals.append(float(v))
    return np.array(vals)


def pair_metrics(rows: list[dict[str, Any]], tau: float = 0.5) -> dict[str, float]:
    if not rows:
        return {}
    y, p = _y(rows), _p(rows, "confidence_raw")
    pred = (p >= tau).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    out = {
        "n": len(rows),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
    }
    if len(set(y.tolist())) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y, p)), 4)
        out["pr_auc"] = round(float(average_precision_score(y, p)), 4)
    return out


def direction_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    pos = [
        r
        for r in rows
        if r.get("gold_label") in {"direct", "transitive"} and r.get("direction_gold") == "A_to_B"
    ]
    if not pos:
        return {"da": None, "frm": None, "n_pos_dir": 0}
    ok = 0
    margins = []
    for r in pos:
        ab, ba = float(r.get("p_A_to_B") or 0), float(r.get("p_B_to_A") or 0)
        if ab > ba:
            ok += 1
        margins.append(ab - ba)
    return {
        "da": round(ok / len(pos), 4),
        "frm": round(float(np.mean(margins)), 4),
        "n_pos_dir": len(pos),
    }


def ece_score(rows: list[dict[str, Any]], n_bins: int = 10, key: str = "confidence_calibrated") -> float:
    y, p = _y(rows), np.clip(_p(rows, key), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return round(float(ece), 4)


def calibration_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    y = _y(rows)
    raw = _p(rows, "confidence_raw")
    cal = _p(rows, "confidence_calibrated")
    out = {
        "ece_raw": ece_score(rows, key="confidence_raw"),
        "ece_cal": ece_score(rows, key="confidence_calibrated"),
        "brier_raw": round(float(brier_score_loss(y, raw)), 4) if len(y) else None,
        "brier_cal": round(float(brier_score_loss(y, cal)), 4) if len(y) else None,
    }
    return out


def graph_metrics(rows: list[dict[str, Any]], graph_stats: dict[str, Any], tau: float = 0.5) -> dict[str, Any]:
    n_e = max(1, sum(1 for r in rows if float(r.get("p_A_to_B") or 0) >= tau))
    rcr = sum(1 for r in rows if "C2_reverse_conflict" in (r.get("flags") or [])) / n_e
    ev_cov = sum(1 for r in rows if r.get("evidence")) / max(1, len(rows))
    return {
        **graph_stats,
        "rcr": round(rcr, 4),
        "evidence_coverage": round(ev_cov, 4),
    }
