# -*- coding: utf-8 -*-
"""Split A (cặp 8:1:1) và Split E (unseen-concept)."""
from __future__ import annotations

import random
from typing import Any


def split_A(pairs: list[dict[str, Any]], ratio=(0.8, 0.1, 0.1), seed: int = 42) -> dict[str, list[str]]:
    ids = [p["pair_id"] for p in pairs]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * ratio[0])
    n_val = int(n * ratio[1])
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def split_E(pairs: list[dict[str, Any]], unseen_frac: float = 0.2, seed: int = 42) -> dict[str, list[str]]:
    skills = sorted({p["source"] for p in pairs} | {p["target"] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(skills)
    n_u = max(1, int(len(skills) * unseen_frac))
    unseen = set(skills[:n_u])
    train, val, test = [], [], []
    for p in pairs:
        a, b = p["source"], p["target"]
        if a in unseen or b in unseen:
            test.append(p["pair_id"])
        else:
            train.append(p["pair_id"])
    rng.shuffle(train)
    cut = max(1, int(0.1 * len(train)))
    val, train = train[:cut], train[cut:]
    return {
        "train": train,
        "val": val,
        "test": test,
        "unseen_skills": sorted(unseen),
    }


def apply_split(pairs: list[dict[str, Any]], spec: dict[str, list[str]], name: str) -> list[dict[str, Any]]:
    loc = {}
    for part in ("train", "val", "test"):
        for pid in spec.get(part, []):
            loc[pid] = part
    out = []
    for p in pairs:
        q = dict(p)
        q["split"] = loc.get(p["pair_id"])
        q["split_scheme"] = name
        out.append(q)
    return out
