# -*- coding: utf-8 -*-
"""C4-lite: destroy gold labels on stored E-test scores.
A frozen scorer cannot fail A vs E; shuffled y should drive AUC to chance.
Not a pretraining-membership test. CPU only. No API."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.io_util import write_json

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_SHUF = 20


def _load(path: Path, test_ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    y, p = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid not in test_ids:
                continue
            raw = r.get("confidence_raw")
            if raw is None:
                raw = max(float(r.get("p_A_to_B") or 0), float(r.get("p_B_to_A") or 0))
            y.append(1 if r.get("gold_label") in {"direct", "transitive"} else 0)
            p.append(float(raw or 0))
    return np.array(y, dtype=int), np.array(p, dtype=float)


def _mention(test_ids: set[str]) -> tuple[np.ndarray, np.ndarray]:
    from src.io_util import read_jsonl

    skills = {s["skill_id"]: s for s in read_jsonl(ROOT / "01_raw/esco_prereq/skills.jsonl")}
    texts = {sid: f"{s.get('label') or ''} {s.get('description') or ''}".lower() for sid, s in skills.items()}
    labels = {sid: (s.get("label") or "").lower() for sid, s in skills.items()}
    y, p = [], []
    for row in read_jsonl(ROOT / "02_interim/candidates.jsonl"):
        if row["pair_id"] not in test_ids:
            continue
        y.append(1 if row.get("gold_label") in {"direct", "transitive"} else 0)
        lab = labels.get(row["source"], "")
        p.append(1.0 if lab and lab in texts.get(row["target"], "") else 0.0)
    return np.array(y, dtype=int), np.array(p, dtype=float)


def _pack(y: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> dict:
    true = {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "base_rate": round(float(y.mean()), 4),
    }
    aucs, prs = [], []
    for _ in range(N_SHUF):
        ys = rng.permutation(y)
        if ys.min() == ys.max():
            continue
        aucs.append(float(roc_auc_score(ys, p)))
        prs.append(float(average_precision_score(ys, p)))
    return {
        "true": true,
        "shuffled": {
            "n": N_SHUF,
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_lo": round(float(np.percentile(aucs, 2.5)), 4),
            "roc_auc_hi": round(float(np.percentile(aucs, 97.5)), 4),
            "pr_auc_mean": round(float(np.mean(prs)), 4),
            "pr_auc_lo": round(float(np.percentile(prs, 2.5)), 4),
            "pr_auc_hi": round(float(np.percentile(prs, 97.5)), 4),
        },
    }


def main() -> None:
    spec = json.loads((ROOT / "05_splits/split_E.json").read_text(encoding="utf-8"))
    test_ids = set(spec["test"])
    rng = np.random.default_rng(SEED)
    out = {"n_test": len(test_ids), "n_shuf": N_SHUF, "seed": SEED, "conditions": {}}
    for name, path in (
        ("B1", ROOT / "06_runs/p2a1_Etv_B1/llm_raw.jsonl"),
        ("B3", ROOT / "06_runs/p2a1_Etv_B3/llm_raw.jsonl"),
        ("B5", ROOT / "06_runs/p2a1_Etv_B5/llm_raw.jsonl"),
    ):
        y, p = _load(path, test_ids)
        out["conditions"][name] = _pack(y, p, rng)
        print(name, out["conditions"][name], flush=True)
    y, p = _mention(test_ids)
    out["conditions"]["mention"] = _pack(y, p, rng)
    print("mention", out["conditions"]["mention"], flush=True)
    write_json(ROOT / "07_eval/c4_label_permute.json", out)
    print("wrote 07_eval/c4_label_permute.json")


if __name__ == "__main__":
    main()
