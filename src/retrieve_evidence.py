# -*- coding: utf-8 -*-
"""S3 — truy hồi 2 tầng, giữ span gốc. Không dùng câu LLM viết làm evidence."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.schema import Skill

ORDER_HINTS = (
    "before",
    "after",
    "prerequisite",
    "first",
    "then",
    "requires",
    "trước",
    "sau",
    "tiên quyết",
    "phải học",
)


class EvidenceIndex:
    """Lọc doc theo skill_id / nhãn — tránh quét 5k doc × 62k cặp."""

    def __init__(self, docs: list[dict[str, Any]], skills: list[Skill] | None = None):
        self.docs = docs
        self.by_skill: dict[str, list[int]] = defaultdict(list)
        self.by_label: dict[str, list[int]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for sid in doc.get("skill_ids") or []:
                self.by_skill[sid].append(i)
            text = (doc.get("text") or "").lower()
            if text:
                head = text[:240].strip()
                if head:
                    self.by_label[head].append(i)
        if skills:
            for s in skills:
                lab = (s.label or "").lower().strip()
                if not lab or s.skill_id in self.by_skill:
                    continue
                for i, doc in enumerate(docs):
                    if lab in (doc.get("text") or "").lower():
                        self.by_skill[s.skill_id].append(i)

    def docs_for(self, *skills: Skill) -> list[dict[str, Any]]:
        idxs: set[int] = set()
        for s in skills:
            idxs.update(self.by_skill.get(s.skill_id, ()))
            lab = (s.label or "").lower().strip()
            if lab:
                idxs.update(self.by_label.get(lab, ()))
        if not idxs:
            return self.docs
        return [self.docs[i] for i in sorted(idxs)]


def _windows(text: str, needle: str, max_chars: int) -> list[tuple[int, int, str]]:
    low = text.lower()
    key = needle.lower()
    if not key:
        return []
    out = []
    start = 0
    while True:
        i = low.find(key, start)
        if i < 0:
            break
        a = max(0, i - max_chars // 3)
        b = min(len(text), i + len(needle) + max_chars // 2)
        out.append((a, b, text[a:b].strip()))
        start = i + len(needle)
        if len(out) >= 3:
            break
    return out


def retrieve_for_pair(
    source: Skill,
    target: Skill,
    docs: list[dict[str, Any]],
    *,
    top_k: int = 4,
    max_span_chars: int = 280,
    index: EvidenceIndex | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """E_AB và E_BA riêng — bằng chứng có thể bất đối xứng."""
    pool = index.docs_for(source, target) if index is not None else docs
    fwd, rev = [], []
    for doc in pool:
        text = doc.get("text") or ""
        did = doc.get("document_id", "")
        stype = doc.get("source_type", "documentation")
        for start, end, span in _windows(text, source.label, max_span_chars):
            score = 0.4 + 0.2 * any(h in span.lower() for h in ORDER_HINTS)
            if target.label.lower() in span.lower():
                score += 0.3
            item = {
                "source_id": f"{did}:{start}",
                "source_type": stype,
                "document_id": did,
                "span": span,
                "char_start": start,
                "char_end": end,
                "evidence_type": "explicit_dependency"
                if any(h in span.lower() for h in ORDER_HINTS)
                else "definition",
                "retrieval_score": round(min(score, 1.0), 3),
                "llm_used_as": "none",
            }
            fwd.append(item)
        for start, end, span in _windows(text, target.label, max_span_chars):
            score = 0.4 + 0.2 * any(h in span.lower() for h in ORDER_HINTS)
            if source.label.lower() in span.lower():
                score += 0.3
            item = {
                "source_id": f"{did}:{start}:rev",
                "source_type": stype,
                "document_id": did,
                "span": span,
                "char_start": start,
                "char_end": end,
                "evidence_type": "explicit_dependency"
                if any(h in span.lower() for h in ORDER_HINTS)
                else "definition",
                "retrieval_score": round(min(score, 1.0), 3),
                "llm_used_as": "none",
            }
            rev.append(item)

    def _top(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items.sort(key=lambda x: x["retrieval_score"], reverse=True)
        seen = set()
        uniq = []
        for it in items:
            key = (it["document_id"], it["char_start"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(it)
            if len(uniq) >= top_k:
                break
        return uniq

    return _top(fwd), _top(rev)
