# -*- coding: utf-8 -*-
"""Dieu phoi P2A-0 ... P2A-4 va bay buoc S1-S7."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.calibrate import apply_calibrator, fit_calibrator
from src.candidates import attach_gold, generate_candidates, gold_coverage
from src.canonicalize import canonicalize_all
from src.graph_validate import validate_graph
from src.io_util import load_dotenv, read_jsonl, read_yaml, resolve, write_json, write_jsonl
from src.metrics import calibration_metrics, direction_metrics, graph_metrics, pair_metrics
from src.retrieve_evidence import EvidenceIndex, retrieve_for_pair
from src.score_loop import filter_by_split, score_pairs
from src.selective_verify import queues, reliability_at_budget
from src.splits import apply_split, split_A, split_E


def load_cfg(path: str) -> dict:
    return read_yaml(path)


def P(cfg: dict, key: str, default: str) -> str:
    return (cfg.get("paths") or {}).get(key, default)


def load_skills(cfg: dict):
    rows = read_jsonl(cfg["data"]["skills"])
    skills = canonicalize_all(rows)
    by_id = {s.skill_id: s for s in skills}
    return skills, by_id


def _is_real_dump(cfg: dict) -> bool:
    skills = str(cfg.get("data", {}).get("skills", ""))
    return "esco_prereq" in skills or "lecturebank" in skills


def _guard_llm(cfg: dict, *, allow_mock: bool) -> None:
    mode = (cfg.get("llm") or {}).get("mode", "mock")
    if mode == "mock" and _is_real_dump(cfg) and not allow_mock:
        raise SystemExit(
            "Refuse mock LLM on real dump. Set llm.mode=openai_compat and P2A_LLM_KEY, "
            "or pass --allow-mock (fixture / smoke only; not paper numbers)."
        )
    if mode == "openai_compat":
        if not (os.environ.get("P2A_LLM_KEY") or os.environ.get("OPENAI_API_KEY")):
            raise SystemExit("Missing P2A_LLM_KEY / OPENAI_API_KEY for openai_compat")


def phase0(cfg: dict) -> None:
    skills, _ = load_skills(cfg)
    gold = read_jsonl(cfg["data"]["gold"])
    docs = read_jsonl(cfg["data"]["docs"])
    write_jsonl(P(cfg, "skills_canonical", "03_taxonomy/skills_canonical.jsonl"), [s.to_dict() for s in skills])

    cand_cfg = cfg.get("candidates") or {}
    universe_ids = None
    uni_path = cand_cfg.get("universe")
    if uni_path:
        universe_ids = {r["skill_id"] for r in read_jsonl(uni_path) if r.get("skill_id")}
    pairs = generate_candidates(
        skills,
        top_k=cand_cfg.get("top_k_similar", 8),
        use_taxonomy=cand_cfg.get("use_taxonomy_neighbors", True),
        use_cooccurrence=cand_cfg.get("use_cooccurrence", False),
        docs=docs,
        mode=cand_cfg.get("mode") or "index",
        universe_ids=universe_ids,
    )
    pairs = attach_gold(pairs, gold)
    cov = gold_coverage(pairs, gold)
    spec_a = split_A(pairs, tuple(cfg["splits"]["pair_split_ratio"]), cfg["seed"])
    spec_e = split_E(pairs, cfg["splits"]["unseen_concept_frac"], cfg["seed"])
    write_json(P(cfg, "split_a", "05_splits/split_A.json"), spec_a)
    write_json(P(cfg, "split_e", "05_splits/split_E.json"), spec_e)
    write_json(P(cfg, "coverage", "07_eval/candidate_coverage.json"), cov)
    write_jsonl(P(cfg, "candidates", "02_interim/candidates.jsonl"), pairs)
    print("P2A-0 candidates", len(pairs), "gold_coverage", cov)
    from src.inspect_p2a0 import sample_inspect

    out = sample_inspect(
        skills_path=P(cfg, "skills_canonical", "03_taxonomy/skills_canonical.jsonl"),
        gold_path=cfg["data"]["gold"],
        docs_path=cfg["data"]["docs"],
        cands_path=P(cfg, "candidates", "02_interim/candidates.jsonl"),
        out_md=P(cfg, "inspect_md", "07_eval/inspect_p2a0.md"),
        out_sample=P(cfg, "inspect_sample", "07_eval/inspect_p2a0_sample.jsonl"),
        title=cfg.get("name") or "dump",
    )
    print("P2A-0 done", out)


def phase_s3(cfg: dict, *, limit: int | None = None) -> Path:
    skills, by_id = load_skills(cfg)
    docs = read_jsonl(cfg["data"]["docs"])
    pairs = read_jsonl(P(cfg, "candidates", "02_interim/candidates.jsonl"))
    if limit:
        pairs = pairs[:limit]
    index = EvidenceIndex(docs, skills)
    rows = []
    n_ab = n_ba = 0
    for p in pairs:
        a, b = by_id[p["source"]], by_id[p["target"]]
        ev_ab, ev_ba = retrieve_for_pair(
            a, b, docs,
            top_k=cfg["retrieval"]["top_k_docs"],
            max_span_chars=cfg["retrieval"]["max_span_chars"],
            index=index,
        )
        n_ab += int(bool(ev_ab))
        n_ba += int(bool(ev_ba))
        rows.append({"pair_id": p["pair_id"], "evidence": ev_ab, "evidence_reverse": ev_ba})
    out = write_jsonl(P(cfg, "evidence", "02_interim/evidence.jsonl"), rows)
    stats = {
        "n_pairs": len(rows),
        "n_with_span_ab": n_ab,
        "n_with_span_ba": n_ba,
        "frac_span_ab": round(n_ab / max(1, len(rows)), 4),
        "frac_span_ba": round(n_ba / max(1, len(rows)), 4),
    }
    write_json(P(cfg, "evidence_stats", "07_eval/s3_evidence_stats.json"), stats)
    print("S3 evidence", stats)
    return out


def _normalize_pairs(rows: list[dict], by_id: dict) -> list[dict]:
    out = []
    for r in rows:
        src, tgt = r.get("source"), r.get("target")
        if not src or not tgt or src not in by_id or tgt not in by_id or src == tgt:
            continue
        q = dict(r)
        q["pair_id"] = q.get("pair_id") or f"{src}__{tgt}"
        q.setdefault("gold_label", "none")
        if q.get("gold_label") in {"direct", "transitive"} and q.get("direction_gold") in (None, "", "none"):
            q["direction_gold"] = "A_to_B"
        q.setdefault("direction_gold", "none")
        out.append(q)
    return out


def _run_condition(
    cfg: dict,
    cond_name: str,
    phase: str,
    *,
    limit: int | None = None,
    pairs_path: str | None = None,
    scheme: str | None = None,
    parts: list[str] | None = None,
    resume: bool = True,
    workers: int = 8,
) -> dict:
    flags = dict(cfg["conditions"][cond_name])
    skills, by_id = load_skills(cfg)
    docs = read_jsonl(cfg["data"]["docs"])
    raw = read_jsonl(pairs_path or P(cfg, "candidates", "02_interim/candidates.jsonl"))
    pairs = _normalize_pairs(raw, by_id)
    spec_e = __import__("json").loads(resolve(P(cfg, "split_e", "05_splits/split_E.json")).read_text(encoding="utf-8"))
    spec_a = __import__("json").loads(resolve(P(cfg, "split_a", "05_splits/split_A.json")).read_text(encoding="utf-8"))
    if scheme and parts:
        spec = spec_e if scheme == "E" else spec_a
        pairs = filter_by_split(pairs, spec, parts)
        print("filter", scheme, parts, "n", len(pairs))
    if limit:
        pairs = pairs[:limit]

    run_tag = phase
    if pairs_path:
        run_tag = f"{phase}_inspect" if "inspect" in pairs_path.replace("\\", "/") else f"{phase}_custom"
    elif scheme and parts:
        run_tag = f"{phase}_{scheme}{''.join(p[0] for p in parts)}"
    cfg_name = str(cfg.get("name") or "").strip()
    run_id = f"{run_tag}_{cond_name}"
    if cfg_name and cfg_name != "mvd":
        run_id = f"{run_id}_{cfg_name}"
    scored = score_pairs(
        pairs,
        by_id=by_id,
        docs=docs,
        cfg=cfg,
        flags=flags,
        cond_name=cond_name,
        out_jsonl=f"06_runs/{run_id}/llm_raw.jsonl",
        resume=resume,
        workers=workers,
    )

    use_cal = flags.get("use_calibration")
    use_graph = flags.get("use_graph")
    metrics = {
        "condition": cond_name,
        "phase": run_tag,
        "n_scored": len(scored),
        "llm_mode": cfg["llm"].get("mode"),
        "pairs_path": pairs_path,
        "all_scored": {
            "pair": pair_metrics(scored),
            "direction": direction_metrics(scored),
        },
    }

    for scheme, spec in (("E", spec_e), ("A", spec_a)):
        tagged = apply_split(scored, spec, scheme)
        val = [r for r in tagged if r.get("split") == "val"]
        test = [r for r in tagged if r.get("split") == "test"]
        work = tagged
        if use_cal:
            pack = fit_calibrator(val or tagged, cfg["calibration"]["method"])
            work = apply_calibrator(work, pack)
            test = apply_calibrator(test, pack)
            val = apply_calibrator(val, pack)
        graph_stats = {}
        if use_graph:
            irr_only = use_graph == "irr_only"
            work, graph_stats = validate_graph(
                work, by_id,
                mu=cfg["graph"]["mu"],
                tau=cfg["graph"]["tau_conflict"],
                irr_only=irr_only,
            )
            test = [r for r in work if r.get("pair_id") in set(spec.get("test", []))]
        block = {
            "pair": pair_metrics(test),
            "direction": direction_metrics(test),
        }
        if use_cal:
            block["calibration"] = calibration_metrics(test or work)
        if use_graph:
            block["graph"] = graph_metrics(work, graph_stats, cfg["graph"]["tau_conflict"])
        metrics[f"split_{scheme}"] = block
        ds = str(cfg.get("dataset") or "")
        interim = f"02_interim/{ds}" if ds in {"lecturebank", "ucd", "mooc"} else "02_interim"
        write_jsonl(f"{interim}/llm_calibrated_{scheme}.jsonl", work if use_cal or use_graph else tagged)

    if use_graph is True:
        ds = str(cfg.get("dataset") or "")
        interim = f"02_interim/{ds}" if ds in {"lecturebank", "ucd", "mooc"} else "02_interim"
        write_jsonl(f"{interim}/graph_validated.jsonl", work)
        write_json("05_release/graph.json", {
            "nodes": [s.to_dict() for s in skills],
            "edges": [
                r for r in work
                if float(r.get("p_A_to_B") or 0) >= cfg["graph"]["tau_conflict"]
                and "C3_cycle_removed" not in (r.get("flags") or [])
            ],
        })

    metrics_name = f"{run_id}_metrics.json" if run_id != f"{run_tag}_{cond_name}" else f"{run_tag}_{cond_name}_metrics.json"
    eval_dir = "07_eval/lecturebank" if str(cfg.get("dataset") or "") == "lecturebank" else "07_eval"
    write_json(f"{eval_dir}/{metrics_name}", metrics)
    print(cond_name, "all", metrics.get("all_scored"), "E", metrics.get("split_E", {}).get("pair"))
    return metrics


def phase4(cfg: dict) -> None:
    path = resolve("02_interim/graph_validated.jsonl")
    if not path.exists():
        path = resolve("02_interim/llm_calibrated_E.jsonl")
    rows = read_jsonl(path)
    qs = queues(rows, cfg["seed"])
    curves = []
    for name, q in qs.items():
        write_jsonl(f"04_annotation/review_queue_{name}.jsonl", q)
        curves.extend(reliability_at_budget(q, cfg["review"]["budgets"]))
    write_json("07_eval/cost_curve.json", curves)
    lines = ["policy,budget,n_reviewed,reliability"]
    for row in curves:
        lines.append(f"{row['policy']},{row['budget']},{row['n_reviewed']},{row['reliability']}")
    out = resolve("07_eval/cost_curve.csv")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("P2A-4 done", out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline P2A")
    ap.add_argument("--config", default="configs/mvd.yaml")
    ap.add_argument("--phase", choices=["0", "1", "2", "3", "4", "s3"])
    ap.add_argument("--condition", default=None)
    ap.add_argument("--all-mvd", action="store_true")
    ap.add_argument("--evidence-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pairs", default=None, help="JSONL cap (vd. 07_eval/inspect_p2a0_sample.jsonl)")
    ap.add_argument("--scheme", choices=["A", "E"], default=None)
    ap.add_argument("--parts", default=None, help="vd. test,val")
    ap.add_argument("--only", action="store_true", help="Chi chay --condition, khong tu chay B1+B3")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--allow-mock", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    if "openrouter" in args.config.replace("\\", "/").lower():
        load_dotenv(".env.openrouter", override=True)
    cfg = load_cfg(args.config)
    cand = resolve(P(cfg, "candidates", "02_interim/candidates.jsonl"))

    if args.evidence_only or args.phase == "s3":
        if not cand.exists():
            phase0(cfg)
        phase_s3(cfg, limit=args.limit)
        return

    if args.all_mvd or args.phase == "0":
        phase0(cfg)
    if args.all_mvd or args.phase == "1":
        _guard_llm(cfg, allow_mock=args.allow_mock or bool(args.all_mvd))
        if not cand.exists():
            phase0(cfg)
        cond = args.condition or "B3"
        parts = [x.strip() for x in (args.parts or "").split(",") if x.strip()] or None
        kw = dict(
            limit=args.limit,
            pairs_path=args.pairs,
            scheme=args.scheme,
            parts=parts,
            resume=not args.no_resume,
            workers=args.workers,
        )
        if args.only:
            _run_condition(cfg, cond, "p2a1", **kw)
        else:
            _run_condition(cfg, "B1", "p2a1", **kw)
            _run_condition(cfg, cond if cond in {"B1", "B3"} else "B3", "p2a1", **kw)
    if args.all_mvd or args.phase == "2":
        _guard_llm(cfg, allow_mock=args.allow_mock or bool(args.all_mvd))
        if not cand.exists():
            phase0(cfg)
        _run_condition(cfg, args.condition or "B5", "p2a2", limit=args.limit)
    if args.all_mvd or args.phase == "3":
        _guard_llm(cfg, allow_mock=args.allow_mock or bool(args.all_mvd))
        if not cand.exists():
            phase0(cfg)
        _run_condition(cfg, "B6", "p2a3", limit=args.limit)
        _run_condition(cfg, "S4lite", "p2a3", limit=args.limit)
        _run_condition(cfg, args.condition or "B7", "p2a3", limit=args.limit)
    if args.all_mvd or args.phase == "4":
        if not resolve("02_interim/graph_validated.jsonl").exists():
            raise SystemExit("Need graph_validated.jsonl from P2A-3 before phase 4")
        phase4(cfg)


if __name__ == "__main__":
    main()
