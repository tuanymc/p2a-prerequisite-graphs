# -*- coding: utf-8 -*-
"""Cham LLM co resume, loc split, va worker song song."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.io_util import append_jsonl, read_jsonl, resolve, write_json
from src.llm_reason import reason_pair
from src.retrieve_evidence import EvidenceIndex, retrieve_for_pair


def filter_by_split(pairs: list[dict], spec: dict, parts: list[str]) -> list[dict]:
    keep: set[str] = set()
    for part in parts:
        keep.update(spec.get(part) or [])
    return [p for p in pairs if p.get("pair_id") in keep]


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row.get("pair_id")
            if pid:
                ids.add(pid)
    return ids


def score_pairs(
    pairs: list[dict],
    *,
    by_id: dict,
    docs: list[dict],
    cfg: dict,
    flags: dict,
    cond_name: str,
    out_jsonl: str,
    resume: bool = True,
    workers: int = 8,
    ckpt_every: int = 50,
) -> list[dict]:
    out = resolve(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_ids(out) if resume else set()
    if not resume and out.exists():
        out.unlink()

    todo = [p for p in pairs if p["pair_id"] not in done]
    print("score", cond_name, "todo", len(todo), "resume", len(done), "workers", workers, flush=True)

    ev_path = resolve((cfg.get("paths") or {}).get("evidence", "02_interim/evidence.jsonl"))
    cached: dict[str, dict] = {}
    if flags.get("use_evidence") and ev_path.exists():
        cached = {r["pair_id"]: r for r in read_jsonl(ev_path)}
    index = EvidenceIndex(docs, list(by_id.values())) if flags.get("use_evidence") else None

    lock = threading.Lock()
    n_ok = 0
    n_err = 0

    def one(p: dict) -> dict:
        a, b = by_id[p["source"]], by_id[p["target"]]
        if flags.get("use_evidence"):
            hit = cached.get(p["pair_id"])
            if hit:
                ev_ab, ev_ba = hit.get("evidence") or [], hit.get("evidence_reverse") or []
            else:
                ev_ab, ev_ba = retrieve_for_pair(
                    a, b, docs,
                    top_k=cfg.get("retrieval", {}).get("top_k_docs", 4),
                    max_span_chars=cfg.get("retrieval", {}).get("max_span_chars", 280),
                    index=index,
                )
        else:
            ev_ab, ev_ba = [], []
        pred = reason_pair(a, b, ev_ab, ev_ba, cfg["llm"], flags)
        return {**p, **pred, "evidence": ev_ab, "evidence_reverse": ev_ba, "condition": cond_name}

    if not todo:
        return read_jsonl(out) if out.exists() else []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(one, p): p["pair_id"] for p in todo}
        for fut in as_completed(futs):
            pid = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {
                    "pair_id": pid,
                    "error": str(exc)[:400],
                    "p_A_to_B": 0.0,
                    "p_B_to_A": 0.0,
                    "relation_type": "none",
                    "direction": "none",
                    "confidence_raw": 0.0,
                    "rationale": "request_failed",
                    "evidence_ids": [],
                    "condition": cond_name,
                }
                n_err += 1
            with lock:
                append_jsonl(out, [row])
                n_ok += 1
                if n_ok % ckpt_every == 0 or n_ok == len(todo):
                    write_json(
                        out.parent / "progress.json",
                        {"done": len(done) + n_ok, "todo_this_run": len(todo), "errors": n_err},
                    )
                    print("ckpt", len(done) + n_ok, "/", len(done) + len(todo), "err", n_err, flush=True)

    return read_jsonl(out)
