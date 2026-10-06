# -*- coding: utf-8 -*-
"""Finish official B7 as a DAG after validate_graph C3 rounds."""
from __future__ import annotations

import networkx as nx

TAU = 0.55


def finish_dag(rows: list[dict]) -> tuple[list[dict], dict, set, set]:
    g = nx.DiGraph()
    conf = {}
    for r in rows:
        flags = set(r.get("flags") or [])
        if "C1_self_loop" in flags or r["source"] == r["target"]:
            continue
        if "C3_cycle_removed" in flags:
            continue
        pab = float(r.get("p_A_to_B") or 0.0)
        if pab >= TAU:
            g.add_edge(r["source"], r["target"])
            conf[(r["source"], r["target"])] = float(
                r.get("confidence_calibrated") or pab
            )
    extra = []
    n_round = 0
    while True:
        sccs = [c for c in nx.strongly_connected_components(g) if len(c) > 1]
        if not sccs:
            break
        n_round += 1
        for comp in sccs:
            inside = [(u, v) for u, v in g.edges(comp) if v in comp]
            if not inside:
                continue
            inside.sort(key=lambda e: 1.0 - conf.get(e, 0.0), reverse=True)
            e = inside[0]
            if g.has_edge(*e):
                g.remove_edge(*e)
                extra.append(e)
        if n_round > 20000:
            break
    extra_set = set(extra)
    for r in rows:
        key = (r["source"], r["target"])
        if key in extra_set and "C3_cycle_removed" not in (r.get("flags") or []):
            r.setdefault("flags", []).append("C3_cycle_removed")
    accepted = 0
    b5 = set()
    b7 = set()
    n_c3 = 0
    for r in rows:
        flags = set(r.get("flags") or [])
        if r["source"] == r["target"] or "C1_self_loop" in flags:
            continue
        if float(r.get("p_A_to_B") or 0) >= TAU:
            key = (r["source"], r["target"])
            b5.add(key)
            if "C3_cycle_removed" in flags:
                n_c3 += 1
            else:
                b7.add(key)
                accepted += 1
    dag = nx.DiGraph()
    dag.add_edges_from(b7)
    stats = {
        "extra_deleted": len(extra),
        "n_rounds": n_round,
        "n_B5_tau": len(b5),
        "n_B7": len(b7),
        "n_C3": n_c3,
        "is_dag": nx.is_directed_acyclic_graph(dag) if dag.number_of_nodes() else True,
        "accepted": accepted,
    }
    return rows, stats, b5, b7
