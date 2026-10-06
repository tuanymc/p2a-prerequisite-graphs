# -*- coding: utf-8 -*-
"""S1 — chuẩn hoá skill: Unicode NFC, id ổn định, không viết lại description."""
from __future__ import annotations

import unicodedata
from typing import Any

from src.schema import Skill


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", (text or "").strip())


def canonicalize_skill(row: dict[str, Any]) -> Skill:
    path = row.get("taxonomy_path") or []
    if isinstance(path, str):
        path = [p.strip() for p in path.split(">") if p.strip()]
    return Skill(
        skill_id=nfc(str(row["skill_id"])),
        label=nfc(row.get("label") or row.get("name") or ""),
        description=nfc(row.get("description") or ""),
        taxonomy_path=[nfc(x) for x in path],
        domain=nfc(row.get("domain") or ""),
        source=nfc(row.get("source") or "ESCO"),
    )


def canonicalize_all(rows: list[dict[str, Any]]) -> list[Skill]:
    seen: dict[str, Skill] = {}
    for row in rows:
        sk = canonicalize_skill(row)
        seen[sk.skill_id] = sk
    return list(seen.values())
