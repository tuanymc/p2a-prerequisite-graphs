# P2A reproduction pack (IJAIT)

This folder is the manuscript companion for
*From Pair Scores to Prerequisite Graphs: When Retrieved Evidence Helps*.

Scoring logs (`06_runs/`) stay in the project archive and will be
published on acceptance. Do not commit `.env` or OpenRouter keys.

## What is included

- `src/` — conversion, candidate generation, retrieval, S4 scoring,
  temperature scaling, C1–C5, metrics, lexical baselines
- `prompts/` — S4 system message and user-template notes
- `configs/mvd.yaml` — seed 42, unseen-concept fraction 0.2, k=4,
  window 280 characters, temperature 0
- `splits/split_E_spec.json` — counts and the 1,125 held-out skill ids
  (not the 135k pair-id lists)
- `examples/four_pairs.json` — the four inspect pairs in Table of
  worked examples
- `requirements.txt`

## What is not included

- API keys, `.env`, OpenRouter keys
- `01_raw/esco_prereq/` dump (download ESCO-PrereqSkill from Le and Abel)
- `06_runs/*.jsonl` scoring logs
- the 240-pair human sheet

## Rebuild split E

```
from src.splits import split_E
spec = split_E(pairs, unseen_frac=0.2, seed=42)
```

Held-out skills are 20% of identifiers that appear in the candidate
pool after a `random.Random(42)` shuffle of the sorted skill id list.
Any pair that touches a held-out skill is test.
Remaining pairs are split 90:10 into train/validation.

## Do not

- train on the 135,726-pair pool in this paper's protocol
- treat mock / smoke / inspect-200 numbers as paper metrics
- mix Qwen test numbers into the gpt-4o-mini test table
- cite arXiv:2502.19915
