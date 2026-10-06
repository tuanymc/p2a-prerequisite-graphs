# -*- coding: utf-8 -*-
"""Lexical baselines on the same ESCO Split E test pool. No LLM. CPU only."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from src.io_util import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _text(skill: dict) -> str:
    return f"{skill.get('label') or ''} {skill.get('description') or ''}".strip()


def _best_f1(y, p) -> dict:
    best = {"threshold": 0.0, "f1": 0.0}
    for i in range(1, 20):
        tau = i / 20.0
        pred = (p >= tau).astype(int)
        f1 = float(f1_score(y, pred, zero_division=0))
        if f1 > best["f1"]:
            best = {"threshold": tau, "f1": round(f1, 4)}
    return best


def main() -> None:
    skills = {s["skill_id"]: s for s in read_jsonl(ROOT / "01_raw/esco_prereq/skills.jsonl")}
    spec = json.loads((ROOT / "05_splits/split_E.json").read_text(encoding="utf-8"))
    test_ids = set(spec["test"])
    pairs = [p for p in read_jsonl(ROOT / "02_interim/candidates.jsonl") if p["pair_id"] in test_ids]
    y = [1 if p.get("gold_label") in {"direct", "transitive"} else 0 for p in pairs]
    print("n_test", len(pairs), "pos", sum(y))

    jacc = [float(p.get("score_sim") or 0.0) for p in pairs]
    mention = []
    texts = {sid: _text(s).lower() for sid, s in skills.items()}
    labels = {sid: (s.get("label") or "").lower() for sid, s in skills.items()}
    for p in pairs:
        src, tgt = p["source"], p["target"]
        lab = labels.get(src, "")
        mention.append(1.0 if lab and lab in texts.get(tgt, "") else 0.0)

    corpus = [_text(skills[sid]) for sid in skills]
    vec = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2))
    mat = vec.fit_transform(corpus)
    index = {sid: i for i, sid in enumerate(skills)}
    tfidf = []
    for p in pairs:
        i, j = index[p["source"]], index[p["target"]]
        tfidf.append(float(cosine_similarity(mat[i], mat[j])[0, 0]))

    out = {"n": len(pairs), "n_pos": int(sum(y))}
    import numpy as np

    y_arr = np.array(y)
    for name, scores in (("jaccard_label_desc", jacc), ("mention_src_in_tgt", mention), ("tfidf_cosine", tfidf)):
        p = np.array(scores)
        rec = {
            "roc_auc": round(float(roc_auc_score(y_arr, p)), 4) if len(set(y)) > 1 else None,
            "pr_auc": round(float(average_precision_score(y_arr, p)), 4) if len(set(y)) > 1 else None,
            "f1_at_0.5": round(float(f1_score(y_arr, (p >= 0.5).astype(int), zero_division=0)), 4),
            "best_val_style_sweep": _best_f1(y_arr, p),
        }
        out[name] = rec
        print(name, rec)

    dest = ROOT / "07_eval/lexical_baseline_E_test.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
