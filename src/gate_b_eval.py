# -*- coding: utf-8 -*-
"""Gate B: graph P/R, transitive closure, bootstrap CI, supervised TF-IDF.
CPU only. No API. Numbers go to 07_eval/gate_b.json — do not invent."""
from __future__ import annotations

import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.svm import LinearSVC

from src.io_util import read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
T_SCALE = {"B1": 3.2, "B3": 3.9, "B5": 3.3}
TAU = {"B1": 0.50, "B3": 0.55, "B5": 0.55}
N_BOOT = 1000
SEED = 42


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


def _load_test_scores(path: Path, test_ids: set[str], name: str) -> dict[str, dict]:
    t = T_SCALE[name]
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("pair_id")
            if pid not in test_ids:
                continue
            src = r.get("source")
            tgt = r.get("target")
            if (not src or not tgt) and pid and "__" in pid:
                src, tgt = pid.split("__", 1)
            if not src or not tgt:
                continue
            pab = float(r.get("p_A_to_B") or 0.0)
            pba = float(r.get("p_B_to_A") or 0.0)
            raw = r.get("confidence_raw")
            if raw is None:
                raw = max(pab, pba)
            raw = float(raw or 0.0)
            y = 1 if r.get("gold_label") in {"direct", "transitive"} else 0
            out[pid] = {
                "y": y,
                "p_raw": raw,
                "p_cal": _temp(raw, t),
                "pab": pab,
                "pba": pba,
                "source": src,
                "target": tgt,
            }
    return out


def _ci(vals: np.ndarray) -> dict:
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    return {
        "mean": round(float(np.mean(vals)), 4),
        "lo": round(float(np.percentile(vals, 2.5)), 4),
        "hi": round(float(np.percentile(vals, 97.5)), 4),
        "n": int(len(vals)),
    }


def _metrics_pack(y: np.ndarray, p_raw: np.ndarray, p_cut: np.ndarray, tau: float) -> dict:
    pred = (p_cut >= tau).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    out = {
        "f1_tau": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }
    if y.min() != y.max():
        out["roc_auc"] = float(roc_auc_score(y, p_raw))
        out["pr_auc"] = float(average_precision_score(y, p_raw))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def _text(skill: dict) -> str:
    return f"{skill.get('label') or ''} {skill.get('description') or ''}".strip()


def graph_and_closure() -> dict:
    print("graph/closure", flush=True)
    gold = {
        (g["source"], g["target"])
        for g in read_jsonl(ROOT / "01_raw/esco_prereq/gold_edges.jsonl")
        if g["source"] != g["target"]
    }
    pool = {(p["source"], p["target"]) for p in read_jsonl(ROOT / "02_interim/candidates.jsonl")}
    spec = json.loads((ROOT / "05_splits/split_E.json").read_text(encoding="utf-8"))
    test_ids = set(spec["test"])
    val_ids = set(spec["val"])
    train_ids = set(spec["train"])
    etv_ids = test_ids | val_ids

    cand_by_id = {}
    gold_test = set()
    gold_etv = set()
    gold_train_pool = set()
    for p in read_jsonl(ROOT / "02_interim/candidates.jsonl"):
        cand_by_id[p["pair_id"]] = (p["source"], p["target"])
        key = (p["source"], p["target"])
        if key not in gold:
            continue
        if p["pair_id"] in test_ids:
            gold_test.add(key)
        if p["pair_id"] in etv_ids:
            gold_etv.add(key)
        if p["pair_id"] in train_ids:
            gold_train_pool.add(key)

    gold_pool = gold & pool
    b5_tau = set()
    b7 = set()
    n_c3 = 0
    for r in read_jsonl(ROOT / "02_interim/graph_validated_B5.jsonl"):
        flags = set(r.get("flags") or [])
        pab = float(r.get("p_A_to_B") or 0.0)
        if r["source"] == r["target"] or "C1_self_loop" in flags:
            continue
        if pab >= 0.55:
            key = (r["source"], r["target"])
            b5_tau.add(key)
            if "C3_cycle_removed" in flags:
                n_c3 += 1
            else:
                b7.add(key)

    scopes = {
        "full_gold": gold,
        "in_pool": gold_pool,
        "in_etv_scored": gold_etv,
        "in_test": gold_test,
    }
    preds = {"B5_at_tau_directed": b5_tau, "B7_accepted": b7}
    table = {}
    for pname, pred in preds.items():
        table[pname] = {sname: _prf_sets(pred, gset) for sname, gset in scopes.items()}

    Gg = nx.DiGraph()
    Gg.add_edges_from(gold)
    gold_cycles = not nx.is_directed_acyclic_graph(Gg) if Gg.number_of_nodes() else False
    gold_tc = set()
    for n in Gg:
        for m in nx.descendants(Gg, n):
            gold_tc.add((n, m))

    Gp = nx.DiGraph()
    Gp.add_edges_from(b7)
    pred_tc = set()
    for n in Gp:
        for m in nx.descendants(Gp, n):
            pred_tc.add((n, m))

    pred_in_direct = b7 & gold
    pred_in_tc_only = (b7 & gold_tc) - gold
    pred_outside_tc = b7 - gold_tc
    gold_hit_by_pred_tc = gold & pred_tc

    c6 = {"skipped": True}
    if not gold_cycles and Gg.number_of_edges():
        red = nx.transitive_reduction(Gg)
        red_e = set(red.edges())
        shortcuts = gold - red_e
        c6 = {
            "skipped": False,
            "n_gold": len(gold),
            "n_reduction": len(red_e),
            "n_shortcut_transitive": len(shortcuts),
            "frac_shortcut": round(len(shortcuts) / max(1, len(gold)), 4),
        }

    return {
        "n_gold": len(gold),
        "n_gold_pool": len(gold_pool),
        "n_gold_etv": len(gold_etv),
        "n_gold_test": len(gold_test),
        "n_gold_train_pool_unscored": len(gold_train_pool),
        "coverage_s2": round(len(gold_pool) / max(1, len(gold)), 4),
        "n_B5_tau_directed": len(b5_tau),
        "n_B7": len(b7),
        "n_C3_removed_from_tau": n_c3,
        "note": (
            "B7/B5 graphs are induced only from E val+test scored pairs "
            "(58,219). Train-pool gold cannot appear in the accepted DAG."
        ),
        "prf": table,
        "closure": {
            "gold_is_cyclic": gold_cycles,
            "n_gold_tc": len(gold_tc),
            "n_pred_tc": len(pred_tc),
            "B7_in_gold_direct": len(pred_in_direct),
            "B7_in_gold_tc_not_direct": len(pred_in_tc_only),
            "B7_outside_gold_tc": len(pred_outside_tc),
            "gold_direct_in_B7_tc": len(gold_hit_by_pred_tc),
            "recall_gold_via_B7_tc": round(len(gold_hit_by_pred_tc) / max(1, len(gold)), 4),
            "precision_B7_vs_gold_tc": round(len(b7 & gold_tc) / max(1, len(b7)), 4),
        },
        "C6_gold_heuristic": c6,
    }


def bootstrap() -> dict:
    print("bootstrap", flush=True)
    spec = json.loads((ROOT / "05_splits/split_E.json").read_text(encoding="utf-8"))
    test_ids = set(spec["test"])
    b1 = _load_test_scores(ROOT / "06_runs/p2a1_Etv_B1/llm_raw.jsonl", test_ids, "B1")
    b3 = _load_test_scores(ROOT / "06_runs/p2a1_Etv_B3/llm_raw.jsonl", test_ids, "B3")
    b5 = _load_test_scores(ROOT / "06_runs/p2a1_Etv_B5/llm_raw.jsonl", test_ids, "B5")
    common = sorted(set(b1) & set(b3) & set(b5) & test_ids)
    print("aligned test", len(common), flush=True)

    skills = {s["skill_id"]: s for s in read_jsonl(ROOT / "01_raw/esco_prereq/skills.jsonl")}
    texts = {sid: _text(s).lower() for sid, s in skills.items()}
    labels = {sid: (s.get("label") or "").lower() for sid, s in skills.items()}

    y = np.array([b1[pid]["y"] for pid in common], dtype=int)
    packs = {}
    for name, store, tau in (("B1", b1, 0.50), ("B3", b3, 0.55), ("B5", b5, 0.55)):
        packs[name] = {
            "p_raw": np.array([store[pid]["p_raw"] for pid in common], dtype=float),
            "p_cal": np.array([store[pid]["p_cal"] for pid in common], dtype=float),
            "tau": tau,
        }
    mention = np.array(
        [
            1.0
            if labels.get(b1[pid]["source"], "")
            and labels.get(b1[pid]["source"], "") in texts.get(b1[pid]["target"], "")
            else 0.0
            for pid in common
        ],
        dtype=float,
    )
    packs["mention"] = {"p_raw": mention, "p_cal": mention, "tau": 0.50}

    point = {}
    for name, pack in packs.items():
        m = _metrics_pack(y, pack["p_raw"], pack["p_cal"], pack["tau"])
        point[name] = {k: (None if isinstance(v, float) and math.isnan(v) else round(v, 4)) for k, v in m.items()}
        point[name]["tau"] = pack["tau"]
        print("point", name, point[name], flush=True)

    rng = np.random.default_rng(SEED)
    n = len(y)
    bags = {name: {"roc_auc": [], "pr_auc": [], "f1_tau": []} for name in packs}
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for name, pack in packs.items():
            m = _metrics_pack(yy, pack["p_raw"][idx], pack["p_cal"][idx], pack["tau"])
            bags[name]["roc_auc"].append(m["roc_auc"])
            bags[name]["pr_auc"].append(m["pr_auc"])
            bags[name]["f1_tau"].append(m["f1_tau"])
        if (i + 1) % 200 == 0:
            print("boot", i + 1, flush=True)

    ci = {}
    for name, bag in bags.items():
        ci[name] = {k: _ci(np.array(v, dtype=float)) for k, v in bag.items()}
        ci[name]["tau"] = packs[name]["tau"]
        ci[name]["point"] = point[name]
    return {"n": n, "n_pos": int(y.sum()), "n_boot": N_BOOT, "seed": SEED, "conditions": ci}


def supervised() -> dict:
    print("supervised", flush=True)
    skills = {s["skill_id"]: s for s in read_jsonl(ROOT / "01_raw/esco_prereq/skills.jsonl")}
    spec = json.loads((ROOT / "05_splits/split_E.json").read_text(encoding="utf-8"))
    loc = {}
    for part in ("train", "val", "test"):
        for pid in spec[part]:
            loc[pid] = part

    def pair_text(p: dict) -> str:
        a = skills.get(p["source"], {})
        b = skills.get(p["target"], {})
        return f"{_text(a)} [SEP] {_text(b)}"

    buckets = {"train": [], "val": [], "test": []}
    for p in read_jsonl(ROOT / "02_interim/candidates.jsonl"):
        part = loc.get(p["pair_id"])
        if part not in buckets:
            continue
        y = 1 if p.get("gold_label") in {"direct", "transitive"} else 0
        buckets[part].append((pair_text(p), y))

    print({k: (len(v), sum(y for _, y in v)) for k, v in buckets.items()}, flush=True)
    vec = TfidfVectorizer(min_df=3, max_df=0.95, ngram_range=(1, 2), max_features=80000)
    Xtr = vec.fit_transform([t for t, _ in buckets["train"]])
    ytr = np.array([y for _, y in buckets["train"]], dtype=int)
    Xva = vec.transform([t for t, _ in buckets["val"]])
    yva = np.array([y for _, y in buckets["val"]], dtype=int)
    Xte = vec.transform([t for t, _ in buckets["test"]])
    yte = np.array([y for _, y in buckets["test"]], dtype=int)

    out = {"n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)), "models": {}}
    for tag, clf in (
        ("logreg", LogisticRegression(max_iter=400, solver="liblinear")),
        ("logreg_balanced", LogisticRegression(max_iter=400, solver="liblinear", class_weight="balanced")),
        ("linear_svm", LinearSVC(max_iter=2000)),
    ):
        print("fit", tag, flush=True)
        clf.fit(Xtr, ytr)
        if tag.startswith("logreg"):
            pva = clf.predict_proba(Xva)[:, 1]
            pte = clf.predict_proba(Xte)[:, 1]
        else:
            dva = clf.decision_function(Xva)
            dte = clf.decision_function(Xte)
            pva = 1.0 / (1.0 + np.exp(-dva))
            pte = 1.0 / (1.0 + np.exp(-dte))
        best = {"tau": 0.5, "f1": -1.0}
        for i in range(6, 19):
            tau = i / 20.0
            f1 = float(f1_score(yva, (pva >= tau).astype(int), zero_division=0))
            if f1 > best["f1"]:
                best = {"tau": tau, "f1": round(f1, 4)}
        pred = (pte >= best["tau"]).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        rec_m = {
            "val_best_tau": best["tau"],
            "val_f1": best["f1"],
            "f1_at_0.5": round(float(f1_score(yte, (pte >= 0.5).astype(int), zero_division=0)), 4),
            "f1_tau": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "roc_auc": round(float(roc_auc_score(yte, pte)), 4),
            "pr_auc": round(float(average_precision_score(yte, pte)), 4),
        }
        out["models"][tag] = rec_m
        print(tag, rec_m, flush=True)
    return out


def main() -> None:
    dest: dict = {}
    dest["graph"] = graph_and_closure()
    write_json(ROOT / "07_eval/gate_b.json", dest)
    dest["bootstrap"] = bootstrap()
    write_json(ROOT / "07_eval/gate_b.json", dest)
    dest["supervised"] = supervised()
    miss = json.loads((ROOT / "07_eval/s2_miss_analysis.json").read_text(encoding="utf-8"))
    dest["s2_miss"] = {
        k: miss[k]
        for k in (
            "n_gold",
            "n_hit",
            "n_missed",
            "miss_containment",
            "miss_share_bigram",
            "miss_overlap_ge_1",
            "miss_overlap_ge_2",
            "miss_jaccard_ge_0.2",
            "miss_same_tax2",
            "recoverable_cpu",
            "frac_recoverable_cpu",
        )
        if k in miss
    }
    write_json(ROOT / "07_eval/gate_b.json", dest)
    print("wrote 07_eval/gate_b.json", flush=True)


if __name__ == "__main__":
    main()
