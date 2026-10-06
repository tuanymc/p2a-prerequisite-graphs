# -*- coding: utf-8 -*-
"""Wikipedia RefD coverage via Wikidata P4644 + enwiki sitelink.

Not a ranking run: needs both endpoints mapped and gold positives.
CPU + one SPARQL GET. No Wikipedia dump, no LLM.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from urllib.parse import quote

from src.io_util import read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "01_raw/wikidata_p4644_enwiki.json"
SPARQL = """SELECT ?esco ?title WHERE {
  ?item wdt:P4644 ?esco .
  ?sitelink schema:about ?item ;
            schema:isPartOf <https://en.wikipedia.org/> ;
            schema:name ?title .
}"""


def _load_map() -> dict[str, str]:
    if not CACHE.exists() or CACHE.stat().st_size < 100:
        url = (
            "https://query.wikidata.org/sparql?query="
            + quote(SPARQL)
            + "&format=json"
        )
        subprocess.check_call(
            [
                "curl.exe", "-sS", "-A",
                "P2A-IJAIT-research/1.0",
                "--max-time", "120", url, "-o", str(CACHE),
            ]
        )
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    mapped: dict[str, str] = {}
    for b in data["results"]["bindings"]:
        mapped[b["esco"]["value"].strip()] = b["title"]["value"]
    return mapped


def _uuid(sid: str) -> str:
    return sid.rsplit("/", 1)[-1]


def main() -> None:
    mapped = _load_map()
    ours = []
    for s in read_jsonl(ROOT / "01_raw/esco_prereq/skills.jsonl"):
        sid = s["skill_id"]
        if str(sid).startswith("http://data.europa.eu/esco/skill/"):
            ours.append(_uuid(sid))
    hit = {u: mapped[u] for u in ours if u in mapped}
    spec = json.loads((ROOT / "05_splits/split_E.json").read_text(encoding="utf-8"))
    test = set(spec.get("test") or [])
    n = n_both = n_pos = n_pos_both = 0
    for p in read_jsonl(ROOT / "02_interim/candidates.jsonl"):
        if p.get("pair_id") not in test:
            continue
        n += 1
        both = _uuid(p["source"]) in hit and _uuid(p["target"]) in hit
        pos = p.get("gold_label") in {"direct", "transitive"}
        n_both += int(both)
        n_pos += int(pos)
        n_pos_both += int(both and pos)
    out = {
        "protocol": "Wikidata P4644 + English Wikipedia sitelink",
        "wd_enwiki_items": len(mapped),
        "dump_http_skills": len(ours),
        "dump_with_enwiki": len(hit),
        "e_test_pairs": n,
        "e_test_both_mapped": n_both,
        "e_test_gold_pos": n_pos,
        "e_test_gold_pos_both_mapped": n_pos_both,
        "ranking_possible": n_pos_both >= 20,
        "note": (
            "Classic RefD needs Wikipedia in/out links on a concept map. "
            "Official ESCO-to-enwiki coverage on this dump leaves 5 gold "
            "positives with both ends mapped; not a ranking test."
        ),
    }
    write_json(ROOT / "07_eval/c4_wiki_refd_coverage.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
