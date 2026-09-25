# design.md

This project has no end-user UI — but it does have plots, a methodology doc, and
possibly a small local review tool for eyeballing false positives/negatives. Keeping
these consistent makes error analysis faster to scan and the final documentation look
deliberate rather than default-matplotlib.

## 1. Color Palette (semantic — use everywhere: plots, review tool, doc tables)

| Meaning | Color | Hex |
|---|---|---|
| True positive / correct match | Teal green | `#2E9E7A` |
| False positive (wrong merge) | Coral red | `#E4572E` |
| False negative (missed match) | Amber | `#F2A93B` |
| Neutral / general data | Slate blue | `#3B5BA5` |
| Background / gridlines | Warm gray | `#F5F3EF` / `#DAD6CE` |

Use this mapping consistently across every chart in the EDA notebook and the
Documentation_template.md appendix charts — a reader should learn "red = false
positive" once and reuse it everywhere.

## 2. Chart Style (matplotlib/seaborn)
- `sns.set_theme(style="whitegrid")`, override gridline color to the warm gray above
- Consistent figure size: `(8, 5)` for single charts, `(12, 5)` for side-by-side
  precision/recall or threshold-sweep plots
- Legend: bottom-right or outside-right, never overlapping data
- Always label axes with units (e.g. "Similarity threshold", "Macro F0.5") — no bare
  numbers
- Threshold-sweep plot: x = threshold, y = F0.5, vertical dashed line at the chosen
  max — this is the one chart worth polishing since it directly justifies a pipeline
  decision in the writeup

## 3. Typography (docs)
- Headings in the `.md` docs: default GitHub/VSCode markdown rendering is fine — don't
  over-engineer this, it's read as plain markdown or exported to PDF
- If exporting `Documentation_template.md` to PDF: body text **Inter** (or system
  sans-serif fallback), code/tables in **JetBrains Mono** (or any monospace fallback)
- Keep code blocks monospace with syntax highlighting when pasting pipeline snippets
  into the methodology doc

## 4. If You Build a Local Review Tool (optional, e.g. Streamlit/Gradio)
Only build this if manually scrolling through false positives/negatives in a notebook
gets painful — not a required deliverable.
- Dark background (`#1E1E1E`) — easier on the eyes for long error-analysis sessions
- Each candidate pair row color-coded by verdict using the palette in §1
  (green border = true positive, red = false positive, amber = false negative)
- Show: S1 record, matched S2/S3 record side by side, model score, and the specific
  feature values that drove the decision (so you can spot patterns like "always fails
  on address_jaccard when PIN code is missing")
- No polish beyond this — it's a debugging tool for you, not a submission deliverable
