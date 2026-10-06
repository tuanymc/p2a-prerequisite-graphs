# -*- coding: utf-8 -*-
"""Ghi mọi cạnh gold thành cặp đánh giá (kể cả ngoài pool) — không đưa vào S2."""
from __future__ import annotations

from src.io_util import read_jsonl, write_json, write_jsonl


def main() -> None:
    gold = read_jsonl("01_raw/esco_prereq/gold_edges.jsonl")
    cands = read_jsonl("02_interim/candidates.jsonl")
    pool = {(p["source"], p["target"]) for p in cands}
    rows = []
    for g in gold:
        if g["source"] == g["target"]:
            continue
        rows.append(
            {
                "pair_id": f"{g['source']}__{g['target']}",
                "source": g["source"],
                "target": g["target"],
                "gold_label": g.get("gold_label", "direct"),
                "direction_gold": g.get("direction_gold", "A_to_B"),
                "in_candidate_pool": (g["source"], g["target"]) in pool,
                "eval_only": (g["source"], g["target"]) not in pool,
            }
        )
    write_jsonl("02_interim/gold_eval_pairs.jsonl", rows)
    n_in = sum(1 for r in rows if r["in_candidate_pool"])
    write_json(
        "07_eval/gold_eval_overlay.json",
        {"n_gold": len(rows), "n_in_pool": n_in, "n_eval_only": len(rows) - n_in},
    )
    print("gold_eval", len(rows), "in_pool", n_in)


if __name__ == "__main__":
    main()
