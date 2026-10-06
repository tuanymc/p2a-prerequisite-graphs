# P2A reproduction pack (IJAIT)

This folder is the manuscript companion for
*From Pair Scores to Prerequisite Graphs on Skill-Taxonomy Descriptions*.

Public repository:
https://github.com/tuanymc/p2a-prerequisite-graphs

Scoring logs (`06_runs/`) stay in the project archive and will be
published on acceptance. Do not commit `.env` or OpenRouter keys.

## What is included

- `src/` — conversion, candidate generation, retrieval, S4 scoring,
  temperature scaling, C1–C5, official B7 rebuild, metrics, lexical baselines
- `prompts/` — S4 system message and user-template notes
- `configs/mvd.yaml` — seed 42, unseen-concept fraction 0.2, k=4,
  window 280 characters, temperature 0 (`gpt-4o-mini`)
- `configs/qwen_openrouter.yaml` — Qwen 2.5 7B Instruct, separate table,
  never mixed with mini numbers
- `splits/split_E_spec.json` — counts and the 1,125 held-out skill ids
  (not the 135k pair-id lists)
- `examples/four_pairs.json` — the four inspect pairs in Table of
  worked examples
- `eval/b7_full_pool.json` — official scaled B7 on the 135,726 scored
  E train+val+test pairs
- `eval/gate_b.json` — bootstrap / SVM / earlier val+test graph snapshot
- `requirements.txt`

Official graph (temperature-scale B5 `p_A_to_B`/`p_B_to_A` at T=3.3,
then C2+C3 at τ=0.55): B7 has 8,906 edges, precision 0.100, full-gold
recall 0.154, gold-in-pool recall 0.318. Pair F1@τ on E test DAG
membership is 0.163.

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

- train a GNN on the 135,726-pair pool in this paper's protocol
- treat mock / smoke / inspect-200 numbers as paper metrics
- mix Qwen numbers into the gpt-4o-mini test table
- cite arXiv:2502.19915
- replace the Qwen 2.5 7B open-family table with a larger Qwen SKU
