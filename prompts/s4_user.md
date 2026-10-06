# S4 user prompt variants

Always start with:

```
Skill A: {skill_id} | {label}
{description}
Skill B: {skill_id} | {label}
{description}
```

Then:

- **B5 only:** `Taxonomy A: {path joined by >}` and `Taxonomy B: {path}`
- **B3 and B5:** `Evidence A→B:` / `Evidence B→A:` lines `- [{source_id}] {span}`
- **B1:** `No retrieved evidence. Do not fabricate spans.`

Every variant ends with `Decide both directions. JSON only.`

Decoding temperature is 0. One call returns both directions.
See `src/llm_reason.py` (`SYSTEM`, `build_prompt`).
