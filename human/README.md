# Human sheets and slim scores

`round1/` is the 240-pair sample in Table 9 of the IJAIT manuscript
(authors as annotators; Hien adjudicated disagreements).
`round2/` is the second 240-pair sheet (C6 / shortcut oversample).

Filled CSV forms are the sheets the annotators submitted. JSONL gold
files are the adjudicated labels used in the paper. `iaa.json` stores
the 80-pair overlap counts.

`scores/` holds one JSONL per E-test condition with
`pair_id`, gold, and JSON probabilities only (no prompts, spans, or
rationales). Full scoring logs stay in the project archive.
