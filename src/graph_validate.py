# -*- coding: utf-8 -*-
"""S6 — C1–C6 + L_irr (ProPRL)."""
from __future__ import annotations

from typing import Any

import networkx as nx


def l_irr(rows: list[dict[str, Any]], mu: float = 0.8) -> float:
    pos = [r for r in rows if r.get("gold_label") in {"direct", "transitive"}]
    if not pos:
        pos = rows
    if not pos:
        return 0.0
    s = 0.0
    for r in pos:
        s += max(0.0, float(r.get("p_A_to_B") or 0) + float(r.get("p_B_to_A") or 0) - mu)
    return s / len(pos)


def validate_graph(
    rows: list[dict[str, Any]],
    skills_by_id: dict[str, Any],
    *,
    mu: float = 0.8,
    tau: float = 0.5,
    irr_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    stats = {
        "n": len(rows),
        "self_loop": 0,
        "reverse_conflict": 0,
        "no_evidence": 0,
        "taxonomy_flag": 0,
        "l_irr": round(l_irr(rows, mu), 4),
    }
    accept = []
    for r in rows:
        q = dict(r)
        flags = list(q.get("flags") or [])
        src, tgt = q["source"], q["target"]
        p_ab = float(q.get("p_A_to_B") or 0)
        p_ba = float(q.get("p_B_to_A") or 0)
        conf = float(q.get("confidence_calibrated") or q.get("confidence_raw") or p_ab)

        if src == tgt:
            flags.append("C1_self_loop")
            stats["self_loop"] += 1
        if p_ab > tau and p_ba > tau:
            flags.append("C2_reverse_conflict")
            stats["reverse_conflict"] += 1
            if abs(p_ab - p_ba) < 0.05:
                q["verification_status"] = "direction_conflict"
            elif p_ab >= p_ba:
                q["p_B_to_A"] = min(p_ba, max(0.0, mu - p_ab))
            else:
                q["p_A_to_B"] = min(p_ab, max(0.0, mu - p_ba))
        ev = q.get("evidence") or []
        ev_rev = q.get("evidence_reverse") or []
        if not ev and not ev_rev:
            flags.append("C5_no_evidence")
            stats["no_evidence"] += 1
        sa = skills_by_id.get(src)
        sb = skills_by_id.get(tgt)
        if sa and sb and sa.taxonomy_path and sb.taxonomy_path:
            if sa.taxonomy_path[:1] != sb.taxonomy_path[:1] and conf > 0.8:
                flags.append("C4_taxonomy")
                stats["taxonomy_flag"] += 1
        q["flags"] = flags
        keep = src != tgt and (p_ab >= tau or p_ba >= tau)
        if keep and "C1_self_loop" not in flags:
            accept.append(q)
        out.append(q)

    if not irr_only:
        g = nx.DiGraph()
        for q in accept:
            p_ab = float(q.get("p_A_to_B") or 0)
            p_ba = float(q.get("p_B_to_A") or 0)
            if p_ab >= p_ba and p_ab >= tau:
                g.add_edge(q["source"], q["target"], u=1 - float(q.get("confidence_calibrated") or p_ab))
        broken = set()
        n_cycles = 0
        print("C3 start nodes", g.number_of_nodes(), "edges", g.number_of_edges(), flush=True)
        for _round in range(80):
            sccs = [c for c in nx.strongly_connected_components(g) if len(c) > 1]
            if not sccs:
                break
            if _round == 0 or _round % 10 == 0:
                print("C3 round", _round, "sccs", len(sccs), "max", max(len(c) for c in sccs), flush=True)
            for comp in sccs:
                inside = [(u, v) for u, v in g.edges(comp) if v in comp]
                if not inside:
                    continue
                inside.sort(key=lambda e: g[e[0]][e[1]]["u"], reverse=True)
                k = max(1, min(250, (len(inside) + 19) // 20))
                for e in inside[:k]:
                    if g.has_edge(*e):
                        g.remove_edge(*e)
                        broken.add(e)
                        n_cycles += 1
        stats["n_cycles"] = n_cycles
        stats["edges_broken_for_dag"] = len(broken)
        for q in out:
            key = (q["source"], q["target"])
            if key in broken:
                q.setdefault("flags", []).append("C3_cycle_removed")
        try:
            dag = nx.DiGraph()
            for q in out:
                if "C3_cycle_removed" in (q.get("flags") or []):
                    continue
                p_ab = float(q.get("p_A_to_B") or 0)
                if q["source"] != q["target"] and p_ab >= tau and "C1_self_loop" not in (q.get("flags") or []):
                    dag.add_edge(q["source"], q["target"])
            stats["is_dag"] = nx.is_directed_acyclic_graph(dag) if dag.number_of_nodes() else True
        except nx.NetworkXError:
            stats["is_dag"] = False
    else:
        stats["n_cycles"] = None
        stats["is_dag"] = None

    stats["accepted"] = sum(
        1
        for q in out
        if float(q.get("p_A_to_B") or 0) >= tau and "C1_self_loop" not in (q.get("flags") or [])
        and "C3_cycle_removed" not in (q.get("flags") or [])
    )
    return out, stats
