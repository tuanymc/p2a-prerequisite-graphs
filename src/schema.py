# -*- coding: utf-8 -*-
"""Schema P2A — khớp design v2.0 mục 6. Evidence không được là span do LLM viết."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

GoldLabel = Literal["none", "direct", "transitive", "ambiguous"]
Direction = Literal["A_to_B", "B_to_A", "none"]
EvidenceType = Literal[
    "definition", "explicit_dependency", "instructional_sequence", "conceptual"
]
LlmUsedAs = Literal["query", "none"]


@dataclass
class Skill:
    skill_id: str
    label: str
    description: str
    taxonomy_path: list[str] = field(default_factory=list)
    domain: str = ""
    source: str = "ESCO"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    source_id: str
    source_type: str
    document_id: str
    span: str
    char_start: int = 0
    char_end: int = 0
    evidence_type: EvidenceType | str = "conceptual"
    retrieval_score: float = 0.0
    llm_used_as: LlmUsedAs = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PairRecord:
    pair_id: str
    source: str
    target: str
    gold_label: GoldLabel | str | None = None
    direction_gold: Direction | str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    evidence_reverse: list[dict[str, Any]] = field(default_factory=list)
    p_A_to_B: float | None = None
    p_B_to_A: float | None = None
    confidence_raw: float | None = None
    confidence_calibrated: float | None = None
    relation_type: str | None = None
    rationale: str | None = None
    flags: list[str] = field(default_factory=list)
    verification_status: str | None = None
    split: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
