# -*- coding: utf-8 -*-
"""Tải/chuyển ESCO-PrereqSkill → skills / gold_edges / docs JSONL (schema P2A)."""
from __future__ import annotations

import ast
import csv
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_CSV = ROOT / "01_raw" / "esco_prereq" / "_upstream" / "dataset" / "prerequisite_skill_dataset.csv"
ESCO_SKILLS = Path(r"D:\0.NCS ThuDT\P0_project\03_taxonomies\esco\skills_en.csv")
OUT = ROOT / "01_raw" / "esco_prereq"


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "").strip())


def parse_prereqs(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        val = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return [nfc(raw)]
    if isinstance(val, str):
        return [nfc(val)] if val else []
    return [nfc(x) for x in val if nfc(str(x))]


def load_esco(path: Path) -> tuple[dict[str, dict], dict[str, list[str]]]:
    by_uri: dict[str, dict] = {}
    by_label: dict[str, list[str]] = {}
    if not path.exists():
        return by_uri, by_label
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uri = nfc(row.get("conceptUri") or "")
            if not uri:
                continue
            label = nfc(row.get("preferredLabel") or "")
            rec = {
                "uri": uri,
                "label": label,
                "description": nfc(row.get("description") or row.get("definition") or ""),
                "skill_type": nfc(row.get("skillType") or ""),
                "reuse": nfc(row.get("reuseLevel") or ""),
            }
            by_uri[uri] = rec
            keys = {label.lower()}
            for alt in (row.get("altLabels") or "").split("\n"):
                a = nfc(alt)
                if a:
                    keys.add(a.lower())
            for k in keys:
                by_label.setdefault(k, []).append(uri)
    return by_uri, by_label


def resolve_name(name: str, by_uri: dict, by_label: dict) -> str:
    hits = by_label.get(name.lower()) or []
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # ưu tiên knowledge rồi skill/competence
        def rank(u: str) -> int:
            t = by_uri[u].get("skill_type", "")
            return 0 if "knowledge" in t else 1

        hits = sorted(set(hits), key=rank)
        return hits[0]
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")
    return f"ESCO-NAME:{slug}"


def main() -> None:
    esco, by_label = load_esco(ESCO_SKILLS)
    skills: dict[str, dict] = {}
    edges: list[dict] = []
    unresolved = 0
    multi = 0

    with SRC_CSV.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        uri = nfc(row["skill_url"])
        label = nfc(row["skill_name"])
        desc = nfc(row["skill_description"])
        meta = esco.get(uri, {})
        path = [p for p in (meta.get("skill_type"), meta.get("reuse")) if p]
        skills[uri] = {
            "skill_id": uri,
            "label": label or meta.get("label") or uri,
            "description": desc or meta.get("description") or "",
            "taxonomy_path": path,
            "domain": meta.get("reuse") or "",
            "source": "ESCO-PrereqSkill",
        }
        for pname in parse_prereqs(row.get("prerequisite_name") or ""):
            hits = by_label.get(pname.lower()) or []
            if len(hits) > 1:
                multi += 1
            pid = resolve_name(pname, esco, by_label)
            if pid.startswith("ESCO-NAME:"):
                unresolved += 1
                if pid not in skills:
                    skills[pid] = {
                        "skill_id": pid,
                        "label": pname,
                        "description": "",
                        "taxonomy_path": [],
                        "domain": "",
                        "source": "ESCO-PrereqSkill-name",
                    }
            elif pid not in skills:
                m = esco.get(pid, {})
                skills[pid] = {
                    "skill_id": pid,
                    "label": m.get("label") or pname,
                    "description": m.get("description") or "",
                    "taxonomy_path": [p for p in (m.get("skill_type"), m.get("reuse")) if p],
                    "domain": m.get("reuse") or "",
                    "source": "ESCO",
                }
            if pid == uri:
                continue
            edges.append(
                {
                    "source": pid,
                    "target": uri,
                    "gold_label": "direct",
                    "direction_gold": "A_to_B",
                    "relation": "prerequisite",
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    sk_path = OUT / "skills.jsonl"
    ge_path = OUT / "gold_edges.jsonl"
    doc_path = OUT / "docs.jsonl"
    with sk_path.open("w", encoding="utf-8") as f:
        for rec in sorted(skills.values(), key=lambda x: x["skill_id"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with ge_path.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with doc_path.open("w", encoding="utf-8") as f:
        for rec in skills.values():
            text = rec["label"]
            if rec["description"]:
                text = f"{rec['label']}. {rec['description']}"
            if not rec["description"] and rec["skill_id"].startswith("ESCO-NAME:"):
                continue
            doc = {
                "document_id": f"DOC:{rec['skill_id']}",
                "source_type": "ontology",
                "text": text,
                "skill_ids": [rec["skill_id"]],
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    stats = {
        "source_csv": str(SRC_CSV),
        "github": "https://github.com/lengocluyen/ESCO-PrereqSkill",
        "paper": "arXiv:2507.18479 (Le & Abel); paper ghi 3.196 skill — dump GitHub có nhiều hơn",
        "n_target_rows": len(rows),
        "n_skills_jsonl": len(skills),
        "n_gold_edges": len(edges),
        "n_docs": sum(1 for r in skills.values() if r["description"] or not r["skill_id"].startswith("ESCO-NAME:")),
        "n_prereq_unresolved_name": unresolved,
        "n_prereq_ambiguous_label": multi,
        "esco_skills_join": str(ESCO_SKILLS) if ESCO_SKILLS.exists() else None,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
