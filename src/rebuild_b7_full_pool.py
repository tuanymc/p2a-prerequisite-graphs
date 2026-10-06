# -*- coding: utf-8 -*-
"""Rebuild official B7 on E train+val+test after B5 train scoring. CPU, no API."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from src.canonicalize import canonicalize_all
from src.editorial_n123 import TAU, T_B5, _prf_sets, _temp
from src.editorial_n3_finish import finish_dag
from src.graph_validate import validate_graph
from src.io_util import read_jsonl, write_json, write_jsonl
from src.metrics import direction_metrics

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "06_runs/p2a1_Et_B5/llm_raw.jsonl"
ETV = ROOT / "06_runs/p2a1_Etv_B5/llm_raw.jsonl"


def _iter_raw(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_scaled(spec: dict) -> list[dict]:
    loc = {}
    for part in ("train", "val", "test"):
        for pid in spec.get(part, []):
            loc[pid] = part
    seen = set()
    rows = []
    for path in (TRAIN, ETV):
        if not path.exists():
            continue
        for r in _iter_raw(path):
            pid = r.get("pair_id")
            if not pid or pid in seen:
                continue
            if r.get("error") or r.get("rationale") == "request_failed":
                continue
            src = r.get("source")
            tgt = r.get("target")
            if (not src or not tgt) and "__" in pid:
                src, tgt = pid.split("__", 1)
            if not src or not tgt:
                continue
            seen.add(pid)
            pab = float(r.get("p_A_to_B") or 0.0)
            pba = float(r.get("p_B_to_A") or 0.0)
            raw = r.get("confidence_raw")
            if raw is None:
                raw = max(pab, pba)
            rows.append({
                "pair_id": pid,
                "source": src,
                "target": tgt,
                "gold_label": r.get("gold_label"),
                "direction_gold": r.get("direction_gold"),
                "p_A_to_B": _temp(pab, T_B5),
                "p_B_to_A": _temp(pba, T_B5),
                "p_A_to_B_raw": pab,
                "p_B_to_A_raw": pba,
                "confidence_raw": float(raw or 0.0),
                "confidence_calibrated": _temp(float(raw or 0.0), T_B5),
                "split": loc.get(pid, "train"),
                "has_evidence": True,
                "evidence": [{"span": "kept"}],
                "condition": "B5",
            })
    return rows


def main() -> None:
    spec = json.loads((ROOT / "05_splits/split_E.json").read_text(encoding="utf-8"))
    n_train_file = 0
    if TRAIN.exists():
        with TRAIN.open(encoding="utf-8") as f:
            n_train_file = sum(1 for line in f if line.strip())
    print("train file rows", n_train_file, "expect", len(spec["train"]), flush=True)
    skills = canonicalize_all(read_jsonl(ROOT / "01_raw/esco_prereq/skills.jsonl"))
    by_id = {s.skill_id: s for s in skills}
    gold = {
        (g["source"], g["target"])
        for g in read_jsonl(ROOT / "01_raw/esco_prereq/gold_edges.jsonl")
        if g["source"] != g["target"]
    }
    pool = {(p["source"], p["target"]) for p in read_jsonl(ROOT / "02_interim/candidates.jsonl")}
    rows = load_scaled(spec)
    print("scaled rows", len(rows), flush=True)
    graph_rows, gstats = validate_graph(rows, by_id, mu=0.8, tau=TAU, irr_only=False)
    graph_rows, fin, b5, b7 = finish_dag(graph_rows)
    print("graph", gstats, "finished", fin, flush=True)
    write_jsonl(ROOT / "02_interim/graph_validated_B5_s5_full.jsonl", [
        {k: r.get(k) for k in (
            "pair_id", "source", "target", "gold_label", "direction_gold",
            "p_A_to_B", "p_B_to_A", "p_A_to_B_raw", "p_B_to_A_raw",
            "confidence_raw", "confidence_calibrated", "split", "flags",
        )}
        for r in graph_rows
    ])

    test_ids = set(spec["test"])
    etv_ids = set(spec["test"]) | set(spec["val"])
    gold_test = set()
    gold_etv = set()
    gold_train = set()
    for p in read_jsonl(ROOT / "02_interim/candidates.jsonl"):
        key = (p["source"], p["target"])
        if key not in gold:
            continue
        if p["pair_id"] in test_ids:
            gold_test.add(key)
        if p["pair_id"] in etv_ids:
            gold_etv.add(key)
        if p["pair_id"] in spec["train"]:
            gold_train.add(key)
    scopes = {
        "full_gold": gold,
        "in_pool": gold & pool,
        "in_etv_scored": gold_etv,
        "in_test": gold_test,
        "in_train_pool": gold_train,
    }
    table = {
        "B5_at_tau_scaled": {s: _prf_sets(b5, g) for s, g in scopes.items()},
        "B7_accepted_scaled": {s: _prf_sets(b7, g) for s, g in scopes.items()},
    }
    test_rows = [r for r in graph_rows if r.get("split") == "test"]
    y = np.array([1 if r.get("gold_label") in {"direct", "transitive"} else 0 for r in test_rows])
    pred = np.array([1 if (r["source"], r["target"]) in b7 else 0 for r in test_rows])
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    Gg = nx.DiGraph(); Gg.add_edges_from(gold)
    gold_tc = {(n, m) for n in Gg for m in nx.descendants(Gg, n)}
    Gp = nx.DiGraph(); Gp.add_edges_from(b7)
    pred_tc = {(n, m) for n in Gp for m in nx.descendants(Gp, n)}
    out = {
        "n_scored": len(rows),
        "n_train_file": n_train_file,
        "n_train_expected": len(spec["train"]),
        "n_gold_train": len(gold_train),
        "graph_stats": gstats,
        "finished": fin,
        "prf": table,
        "pair_test_B7": {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "n_pred": int(pred.sum()),
            "n_pos": int(y.sum()),
        },
        "direction_test": direction_metrics(test_rows),
        "closure": {
            "n_gold_tc": len(gold_tc),
            "n_pred_tc": len(pred_tc),
            "B7_in_gold_direct": len(b7 & gold),
            "B7_in_gold_tc_not_direct": len((b7 & gold_tc) - gold),
            "B7_outside_gold_tc": len(b7 - gold_tc),
            "gold_direct_in_B7_tc": len(gold & pred_tc),
            "recall_gold_via_B7_tc": round(len(gold & pred_tc) / max(1, len(gold)), 4),
            "precision_B7_vs_gold_tc": round(len(b7 & gold_tc) / max(1, len(b7)), 4),
            "is_dag": fin["is_dag"],
        },
        "n_B5_tau": len(b5),
        "n_B7": len(b7),
    }
    write_json(ROOT / "07_eval/b7_full_pool.json", out)
    print(json.dumps({
        "n_scored": out["n_scored"],
        "n_B5": out["n_B5_tau"],
        "n_B7": out["n_B7"],
        "prf_B7": table["B7_accepted_scaled"],
        "pair": out["pair_test_B7"],
        "closure": out["closure"],
        "finished": fin,
    }, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
