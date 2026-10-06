# -*- coding: utf-8 -*-
"""CPU-only extras while Qwen E-test runs. Do not touch 06_runs/p2a1_Et_*."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.io_util import read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]


def _text(s: dict) -> str:
    return f"{s.get('label') or ''} {s.get('description') or ''}".lower()


def _prf_at_val(yva, pva, yte, pte) -> dict:
    best = {"tau": 0.5, "f1": -1.0}
    for i in range(1, 20):
        tau = i / 20.0
        f1 = float(f1_score(yva, (pva >= tau).astype(int), zero_division=0))
        if f1 > best["f1"]:
            best = {"tau": round(tau, 2), "f1": round(f1, 4)}
    pred = (pte >= best["tau"]).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    return {
        "val_best_tau": best["tau"],
        "val_f1": best["f1"],
        "f1_at_0.5": round(float(f1_score(yte, (pte >= 0.5).astype(int), zero_division=0)), 4),
        "f1_tau": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "roc_auc": round(float(roc_auc_score(yte, pte)), 4) if len(set(yte.tolist())) > 1 else None,
        "pr_auc": round(float(average_precision_score(yte, pte)), 4) if len(set(yte.tolist())) > 1 else None,
    }


def _shared_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def taxonomy_and_refd() -> dict:
    print("taxonomy/refd", flush=True)
    skills = {s["skill_id"]: s for s in read_jsonl(ROOT / "01_raw/esco_prereq/skills.jsonl")}
    spec = json.loads((ROOT / "05_splits/split_E.json").read_text(encoding="utf-8"))
    loc = {}
    for part in ("train", "val", "test"):
        for pid in spec[part]:
            loc[pid] = part

    path_lens = [len(s.get("taxonomy_path") or []) for s in skills.values()]
    empty = sum(1 for n in path_lens if n == 0)

    # Mention graph: skill i is related to j if label(j) (len>=6) occurs in text(i).
    labels = {sid: (s.get("label") or "").strip().lower() for sid, s in skills.items()}
    texts = {sid: _text(s) for sid, s in skills.items()}
    usable = {sid: lab for sid, lab in labels.items() if len(lab) >= 6}
    related = defaultdict(set)  # sid -> ids whose label appears in sid's text
    inv = defaultdict(set)  # sid -> ids that mention this label
    items = list(usable.items())
    for sid, text in texts.items():
        for oid, lab in items:
            if oid == sid:
                continue
            if lab in text:
                related[sid].add(oid)
                inv[oid].add(sid)
    print("mention-graph edges", sum(len(v) for v in related.values()), flush=True)

    def tax_score(src: str, tgt: str) -> float:
        a = skills.get(src, {}).get("taxonomy_path") or []
        b = skills.get(tgt, {}).get("taxonomy_path") or []
        if not a or not b:
            return 0.0
        return _shared_prefix(a, b) / max(len(a), len(b))

    def refd(src: str, tgt: str) -> float:
        ra, rb = inv.get(src) or set(), inv.get(tgt) or set()
        if not ra and not rb:
            return 0.0

        def mean_rel(neigh: set, other: str) -> float:
            if not neigh:
                return 0.0
            hits = 0
            for n in neigh:
                if other in related.get(n, ()) or n == other:
                    hits += 1
            return hits / len(neigh)

        return float(mean_rel(ra, tgt) - mean_rel(rb, src))

    buckets = {"val": {"tax": [], "refd": [], "y": []}, "test": {"tax": [], "refd": [], "y": []}}
    n_pairs = 0
    for p in read_jsonl(ROOT / "02_interim/candidates.jsonl"):
        part = loc.get(p["pair_id"])
        if part not in buckets:
            continue
        y = 1 if p.get("gold_label") in {"direct", "transitive"} else 0
        buckets[part]["tax"].append(tax_score(p["source"], p["target"]))
        buckets[part]["refd"].append(refd(p["source"], p["target"]))
        buckets[part]["y"].append(y)
        n_pairs += 1
    print("scored pairs", n_pairs, flush=True)

    out = {
        "taxonomy_paths": {
            "n_skills": len(skills),
            "n_empty": empty,
            "mean_len": round(float(np.mean(path_lens)), 3),
            "p50_len": int(np.median(path_lens)),
            "max_len": int(max(path_lens) if path_lens else 0),
        },
        "mention_graph": {
            "min_label_chars": 6,
            "n_nodes_with_related": len(related),
            "n_directed_mention_links": int(sum(len(v) for v in related.values())),
            "note": (
                "RefD-lite uses a mention graph, not Wikipedia in/out links. "
                "Not comparable to Liang 2015 Table numbers."
            ),
        },
        "baselines": {},
    }
    yva = np.array(buckets["val"]["y"])
    yte = np.array(buckets["test"]["y"])
    for name in ("tax", "refd"):
        pva = np.array(buckets["val"][name], dtype=float)
        pte = np.array(buckets["test"][name], dtype=float)
        rec = _prf_at_val(yva, pva, yte, pte)
        out["baselines"][name] = rec
        print(name, rec, flush=True)
    return out


def c6_on_b7() -> dict:
    print("C6 on B7", flush=True)
    gold = {
        (g["source"], g["target"])
        for g in read_jsonl(ROOT / "01_raw/esco_prereq/gold_edges.jsonl")
        if g["source"] != g["target"]
    }
    b7 = []
    for r in read_jsonl(ROOT / "02_interim/graph_validated_B5.jsonl"):
        flags = set(r.get("flags") or [])
        if r["source"] == r["target"] or "C1_self_loop" in flags or "C3_cycle_removed" in flags:
            continue
        if float(r.get("p_A_to_B") or 0) < 0.55:
            continue
        b7.append((r["source"], r["target"]))
    G = nx.DiGraph()
    G.add_edges_from(b7)
    transitive = []
    directish = []
    for u, v in b7:
        G.remove_edge(u, v)
        alt = nx.has_path(G, u, v) if G.has_node(u) and G.has_node(v) else False
        G.add_edge(u, v)
        if alt:
            transitive.append((u, v))
        else:
            directish.append((u, v))
    gold_tc = set()
    Gg = nx.DiGraph()
    Gg.add_edges_from(gold)
    for n in Gg:
        for m in nx.descendants(Gg, n):
            gold_tc.add((n, m))
    pred_t = set(transitive)
    return {
        "n_B7": len(b7),
        "n_heuristic_transitive": len(transitive),
        "n_heuristic_direct": len(directish),
        "frac_transitive": round(len(transitive) / max(1, len(b7)), 4),
        "heuristic_trans_in_gold_direct": len(pred_t & gold),
        "heuristic_trans_in_gold_tc": len(pred_t & gold_tc),
        "note": "C6 heuristic: alternate path in the accepted DAG after removing the edge. Not human t_ij.",
    }


def main() -> None:
    dest = {
        "structural": taxonomy_and_refd(),
        "C6_B7": c6_on_b7(),
    }
    write_json(ROOT / "07_eval/parallel_cpu_baselines.json", dest)
    print(json.dumps({"C6_B7": dest["C6_B7"], "baselines": dest["structural"]["baselines"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
