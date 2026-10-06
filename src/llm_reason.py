# -*- coding: utf-8 -*-
"""S4 — lý luận cặp: bắt buộc p_AB và p_BA. mock | openai_compat."""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from typing import Any

from src.schema import Skill


SYSTEM = """You judge directed prerequisite relations.
Return JSON only. Do not invent quotations. Cite only provided evidence_ids.
Fields:
- p_A_to_B, p_B_to_A: numeric floats in [0,1], never strings or labels
- relation_type: direct | transitive | none | ambiguous
- direction: A_to_B | B_to_A | none
- confidence_raw: numeric float in [0,1]
- rationale: short string
- evidence_ids: array of provided ids
"""


def _as_prob(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return _clip(float(value))
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan", "n/a", "na"}:
        return default
    try:
        return _clip(float(text))
    except ValueError:
        return default


def _parse_llm_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a < 0 or b <= a:
            raise
        obj = json.loads(text[a : b + 1])
    if not isinstance(obj, dict):
        raise ValueError("LLM JSON is not an object")
    p_ab = _as_prob(obj.get("p_A_to_B"))
    p_ba = _as_prob(obj.get("p_B_to_A"))
    conf = _as_prob(obj.get("confidence_raw"), max(p_ab, p_ba))
    rtype = str(obj.get("relation_type") or "none")
    if rtype not in {"direct", "transitive", "none", "ambiguous"}:
        rtype = "none"
    direction = str(obj.get("direction") or "none")
    if direction not in {"A_to_B", "B_to_A", "none"}:
        direction = "A_to_B" if p_ab > p_ba + 0.05 else ("B_to_A" if p_ba > p_ab + 0.05 else "none")
    eids = obj.get("evidence_ids") or []
    if isinstance(eids, str):
        eids = [eids]
    return {
        "p_A_to_B": p_ab,
        "p_B_to_A": p_ba,
        "relation_type": rtype,
        "direction": direction,
        "confidence_raw": conf,
        "rationale": str(obj.get("rationale") or "")[:800],
        "evidence_ids": [str(x) for x in eids if x],
    }


def build_prompt(
    a: Skill,
    b: Skill,
    ev_ab: list[dict[str, Any]],
    ev_ba: list[dict[str, Any]],
    *,
    use_evidence: bool,
    use_taxonomy: bool,
) -> str:
    parts = [f"Skill A: {a.skill_id} | {a.label}\n{a.description}"]
    parts.append(f"Skill B: {b.skill_id} | {b.label}\n{b.description}")
    if use_taxonomy:
        parts.append(f"Taxonomy A: {' > '.join(a.taxonomy_path)}")
        parts.append(f"Taxonomy B: {' > '.join(b.taxonomy_path)}")
    if use_evidence:
        parts.append("Evidence A→B:")
        for e in ev_ab:
            parts.append(f"- [{e.get('source_id')}] {e.get('span')}")
        parts.append("Evidence B→A:")
        for e in ev_ba:
            parts.append(f"- [{e.get('source_id')}] {e.get('span')}")
    else:
        parts.append("No retrieved evidence. Do not fabricate spans.")
    parts.append("Decide both directions. JSON only.")
    return "\n".join(parts)


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def mock_reason(
    a: Skill,
    b: Skill,
    ev_ab: list[dict[str, Any]],
    ev_ba: list[dict[str, Any]],
    *,
    use_evidence: bool,
    use_taxonomy: bool,
) -> dict[str, Any]:
    """Heuristic để thông pipeline — không dùng làm số paper."""
    score_ab = 0.35
    score_ba = 0.25
    eids = []
    if use_evidence:
        if ev_ab:
            score_ab += 0.25 * max(e.get("retrieval_score", 0) for e in ev_ab)
            eids.extend(e.get("source_id") for e in ev_ab[:2])
        if ev_ba:
            score_ba += 0.15 * max(e.get("retrieval_score", 0) for e in ev_ba)
    if use_taxonomy and a.taxonomy_path and b.taxonomy_path:
        if a.taxonomy_path == b.taxonomy_path[: len(a.taxonomy_path)]:
            score_ab += 0.1
        if len(a.taxonomy_path) < len(b.taxonomy_path):
            score_ab += 0.05
    p_ab, p_ba = _clip(score_ab), _clip(score_ba)
    direction = "A_to_B" if p_ab > p_ba else ("B_to_A" if p_ba > p_ab else "none")
    rtype = "direct" if abs(p_ab - p_ba) > 0.12 and max(p_ab, p_ba) > 0.5 else "none"
    return {
        "p_A_to_B": round(p_ab, 4),
        "p_B_to_A": round(p_ba, 4),
        "relation_type": rtype,
        "direction": direction,
        "confidence_raw": round(max(p_ab, p_ba), 4),
        "rationale": "mock heuristic; replace with real LLM for paper numbers",
        "evidence_ids": [x for x in eids if x],
    }


def openai_compat_reason(prompt: str, model: str, temperature: float = 0.0) -> dict[str, Any]:
    url = os.environ.get("P2A_LLM_URL", "https://api.openai.com/v1/chat/completions")
    key = os.environ.get("P2A_LLM_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Thiếu P2A_LLM_KEY / OPENAI_API_KEY")
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    if os.environ.get("P2A_LLM_JSON_OBJECT", "1") != "0":
        body["response_format"] = {"type": "json_object"}
    timeout = 300 if "11434" in url or "localhost" in url or "127.0.0.1" in url else 120
    data = json.dumps(body).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(6):
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if "openrouter.ai" in url:
            headers["HTTP-Referer"] = os.environ.get("P2A_LLM_HTTP_REFERER", "https://localhost")
            headers["X-Title"] = os.environ.get("P2A_LLM_TITLE", "P2A")
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"]
            return _parse_llm_json(text)
        except Exception as exc:
            last_err = exc
            code = getattr(exc, "code", None)
            if code and int(code) not in {408, 409, 429, 500, 502, 503, 529}:
                if attempt >= 2:
                    break
            time.sleep(min(32.0, 1.5 * (2 ** attempt)))
    raise RuntimeError(f"LLM request failed after retries: {type(last_err).__name__}")


def reason_pair(
    a: Skill,
    b: Skill,
    ev_ab: list[dict[str, Any]],
    ev_ba: list[dict[str, Any]],
    cfg: dict[str, Any],
    flags: dict[str, bool],
) -> dict[str, Any]:
    prompt = build_prompt(
        a, b, ev_ab, ev_ba,
        use_evidence=flags.get("use_evidence", True),
        use_taxonomy=flags.get("use_taxonomy", True),
    )
    mode = cfg.get("mode", "mock")
    if mode == "mock":
        return mock_reason(
            a, b, ev_ab, ev_ba,
            use_evidence=flags.get("use_evidence", True),
            use_taxonomy=flags.get("use_taxonomy", True),
        )
    return openai_compat_reason(
        prompt,
        cfg.get("model_open") or "gpt-4o-mini",
        float(cfg.get("temperature") or 0.0),
    )
