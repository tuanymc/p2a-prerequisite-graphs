# -*- coding: utf-8 -*-
"""S2 — sinh ứng viên. Cấm dùng gold. Không duyệt N² trên ESCO đầy đủ."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from src.schema import Skill

STOP = {
    "the", "and", "or", "of", "to", "in", "for", "a", "an", "on", "with", "by",
    "as", "is", "are", "be", "that", "this", "from", "at", "it", "into", "such",
}


def _tokens(text: str) -> set[str]:
    raw = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    return {t for t in raw if len(t) > 2 and t not in STOP}


def _words(text: str) -> list[str]:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split()


def _bigrams(text: str) -> set[str]:
    w = [t for t in _words(text) if len(t) > 1 and t not in STOP]
    return {" ".join(w[i : i + 2]) for i in range(len(w) - 1)}


def _link(neighbor: dict[str, set[str]], a: str, b: str) -> None:
    if a and b and a != b:
        neighbor[a].add(b)
        neighbor[b].add(a)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def generate_complete_pairs(skills: list[Skill], universe_ids: set[str]) -> list[dict[str, Any]]:
    """Moi cap huong trong tap concept cong khai (vd. 208 topic LectureBank). Khong dung nhan gold."""
    ids = [s.skill_id for s in skills if s.skill_id in universe_ids]
    tok = {s.skill_id: _tokens(s.label + " " + s.description) for s in skills}
    pairs = []
    seen: set[tuple[str, str]] = set()
    for src in ids:
        for tgt in ids:
            if src == tgt or (src, tgt) in seen:
                continue
            seen.add((src, tgt))
            pairs.append(
                {
                    "pair_id": f"{src}__{tgt}",
                    "source": src,
                    "target": tgt,
                    "score_sim": round(jaccard(tok.get(src, set()), tok.get(tgt, set())), 4),
                    "in_candidate_pool": True,
                    "gen": "complete_universe",
                }
            )
    return pairs


def generate_candidates(
    skills: list[Skill],
    *,
    top_k: int = 8,
    use_taxonomy: bool = True,
    use_cooccurrence: bool = True,
    docs: list[dict[str, Any]] | None = None,
    mode: str = "index",
    universe_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Hàng xóm inverted-index, hoặc complete_universe (LectureBank 208 topic)."""
    if mode == "complete_universe":
        if not universe_ids:
            raise ValueError("complete_universe needs universe_ids")
        return generate_complete_pairs(skills, universe_ids)
    tok = {s.skill_id: _tokens(s.label + " " + s.description) for s in skills}
    df: Counter[str] = Counter()
    for tset in tok.values():
        df.update(tset)
    n = max(1, len(skills))
    max_df = max(12, int(0.015 * n))

    inv: dict[str, list[str]] = defaultdict(list)
    for sid, tset in tok.items():
        for t in tset:
            if 2 <= df[t] <= max_df:
                inv[t].append(sid)

    soft: dict[str, set[str]] = defaultdict(set)
    hard: dict[str, set[str]] = defaultdict(set)
    for ids in inv.values():
        if len(ids) > 60:
            continue
        for a, b in combinations(ids, 2):
            _link(soft, a, b)

    if use_taxonomy:
        by_tax: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for s in skills:
            if len(s.taxonomy_path) >= 3:
                by_tax[tuple(s.taxonomy_path[:3])].append(s.skill_id)
        for ids in by_tax.values():
            if 2 <= len(ids) <= 40:
                for a, b in combinations(ids, 2):
                    _link(soft, a, b)

    # Label-only: bigram / containment / overlap>=2 — luon giu, khong cat top_k.
    lab_tok = {s.skill_id: _tokens(s.label) for s in skills}
    inv_bg: dict[str, list[str]] = defaultdict(list)
    inv_lab: dict[str, list[str]] = defaultdict(list)
    df_lab: Counter[str] = Counter()
    for sid, tset in lab_tok.items():
        df_lab.update(tset)
        for t in tset:
            inv_lab[t].append(sid)
    for s in skills:
        for bg in _bigrams(s.label):
            inv_bg[bg].append(s.skill_id)
    for ids in inv_bg.values():
        if 2 <= len(ids) <= 80:
            for a, b in combinations(ids, 2):
                _link(hard, a, b)

    max_lab_df = max(40, int(0.08 * n))
    for sid, tset in lab_tok.items():
        cnt: Counter[str] = Counter()
        for t in tset:
            if df_lab[t] > max_lab_df:
                continue
            for oid in inv_lab[t]:
                if oid != sid:
                    cnt[oid] += 1
        for oid, c in cnt.items():
            ot = lab_tok.get(oid, set())
            if c >= 2 or (tset and ot and (tset <= ot or ot <= tset)):
                _link(hard, sid, oid)

    if use_cooccurrence and docs:
        idset = {s.skill_id for s in skills}
        for doc in docs:
            sids = [x for x in (doc.get("skill_ids") or []) if x in idset]
            if len(sids) < 2:
                continue
            for a, b in combinations(sorted(set(sids)), 2):
                _link(soft, a, b)

    pairs = []
    seen: set[tuple[str, str]] = set()

    def emit(src: str, tgt: str, gen: str) -> None:
        key = (src, tgt)
        if src == tgt or key in seen:
            return
        seen.add(key)
        pairs.append(
            {
                "pair_id": f"{src}__{tgt}",
                "source": src,
                "target": tgt,
                "score_sim": round(jaccard(tok.get(src, set()), tok.get(tgt, set())), 4),
                "in_candidate_pool": True,
                "gen": gen,
            }
        )

    for s in skills:
        ranked = sorted(
            soft[s.skill_id],
            key=lambda oid: jaccard(tok[s.skill_id], tok.get(oid, set())),
            reverse=True,
        )[:top_k]
        for oid in ranked:
            emit(s.skill_id, oid, "index")
            emit(oid, s.skill_id, "index")
        for oid in hard[s.skill_id]:
            emit(s.skill_id, oid, "label_signal")
            emit(oid, s.skill_id, "label_signal")
    return pairs


def attach_gold(
    pairs: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Gắn nhãn sau khi đã sinh ứng viên — không ngược lại."""
    gmap = {(g["source"], g["target"]): g for g in gold_rows}
    out = []
    for p in pairs:
        g = gmap.get((p["source"], p["target"]))
        q = dict(p)
        if g:
            q["gold_label"] = g.get("gold_label", "direct")
            q["direction_gold"] = g.get("direction_gold", "A_to_B")
        else:
            q["gold_label"] = "none"
            q["direction_gold"] = "none"
        out.append(q)
    return out


def gold_coverage(
    pairs: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pool = {(p["source"], p["target"]) for p in pairs}
    n = len(gold_rows)
    hit = sum(1 for g in gold_rows if (g["source"], g["target"]) in pool)
    return {"n_gold": n, "n_gold_in_pool": hit, "coverage": round(hit / n, 4) if n else 0.0}
