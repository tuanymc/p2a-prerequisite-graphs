# -*- coding: utf-8 -*-
"""C4 cutoff: URI set-difference vs ESCO v1.1.1 (Sept 2022).

Not last-modifiedDate (v1.2 republish stamps ~10k skills in 2024).
v1.1.2 (Feb 2024) added no new content. URIs absent from v1.1.1 are
therefore post-gpt-4o-mini-cutoff only if they entered in v1.2 (May 2024).
CPU only. No API.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

from src.io_util import read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
V111 = ROOT / "01_raw/esco_v111/skills.csv"
V111_URL = (
    "https://raw.githubusercontent.com/tabiya-tech/tabiya-open-dataset/"
    "main/tabiya-esco-v1.1.1/csv/skills.csv"
)


def _ensure_v111() -> Path:
    V111.parent.mkdir(parents=True, exist_ok=True)
    if V111.exists() and V111.stat().st_size > 1000:
        return V111
    urllib.request.urlretrieve(V111_URL, V111)
    return V111


def _v111_uris() -> set[str]:
    out: set[str] = set()
    with _ensure_v111().open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            u = (row.get("ORIGINURI") or "").strip()
            if u:
                out.add(u)
    return out


def main() -> None:
    old = _v111_uris()
    skills = read_jsonl(ROOT / "01_raw/esco_prereq/skills.jsonl")
    ours = [s["skill_id"] for s in skills]
    http = [u for u in ours if str(u).startswith("http://data.europa.eu/esco/skill/")]
    new = sorted(u for u in http if u not in old)
    by_id = {s["skill_id"]: s for s in skills}
    spec = json.loads((ROOT / "05_splits/split_E.json").read_text(encoding="utf-8"))
    test_ids = set(spec.get("test") or [])
    new_set = set(new)
    n_touch = n_pos = 0
    for row in read_jsonl(ROOT / "02_interim/candidates.jsonl"):
        if row.get("pair_id") not in test_ids:
            continue
        if row.get("source") in new_set or row.get("target") in new_set:
            n_touch += 1
            if row.get("gold_label") in {"direct", "transitive"}:
                n_pos += 1
    out = {
        "protocol": "URI set-difference vs ESCO v1.1.1 (Tabiya transform)",
        "v111_source": V111_URL,
        "v111_n_uris": len(old),
        "dump_n_skills": len(ours),
        "dump_n_http_uris": len(http),
        "n_new_uris": len(new),
        "new_uris": [
            {"skill_id": u, "label": (by_id.get(u) or {}).get("label")}
            for u in new
        ],
        "e_test_pairs_touching_new": n_touch,
        "e_test_gold_pos_touching_new": n_pos,
        "ranking_possible": n_pos >= 2,
        "note": (
            "ESCO last-modifiedDate is not an introduction date "
            "(10,770 of 13,960 skills stamped 2024). "
            "A ranking test needs gold positives; this dump has none "
            "on the post-v1.1.1 URI slice."
        ),
    }
    write_json(ROOT / "07_eval/c4_cutoff_slice.json", out)
    print(json.dumps({k: out[k] for k in out if k != "new_uris"}, ensure_ascii=False, indent=2))
    print("new", out["new_uris"])


if __name__ == "__main__":
    main()
