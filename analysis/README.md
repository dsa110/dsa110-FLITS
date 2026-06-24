# Manuscript figures & sections

Each manuscript section lives in its own `analysis/<topic>/` directory and is **self-describing** via
its `figures.manifest.json` (the same file the figure-review Stop gate uses). `build_manuscript.py`
discovers those manifests and assembles two outputs:

- **`analysis/manuscript_figures.html`** — an SVG-first gallery for easy viewing in a browser
  (vector where available, so it scales cleanly); missing figures are flagged, not shown broken.
- **`analysis/manuscript.tex`** — the per-section `*.tex` fragments `\input` in order, with a
  `\graphicspath` so each `\includegraphics{<stem>}` resolves its sibling figure. Compile with
  `cd analysis && pdflatex manuscript.tex`.

Both are auto-generated — **do not hand-edit them**; edit the per-section files and rebuild.

## View / update

```bash
python analysis/build_manuscript.py            # rebuild gallery + manuscript.tex
python analysis/build_manuscript.py --regen    # re-run each section's `regen` first, then rebuild
python analysis/build_manuscript.py --open      # rebuild, then open the gallery in a browser
```

## Add a section

1. `mkdir analysis/<topic>/`.
2. A plot script that writes a figure — **emit `.svg`** for crisp gallery display (and `.pdf` if you
   want the assembled LaTeX to embed vector); see `chime_dm/plot_dm_grid.py` for the pattern.
3. `figures.manifest.json` listing each `figures[].path` + its `expectation` (the gate contract).
4. Optionally a `<topic>.tex` (prose + figure float) to include the section in `manuscript.tex`.
5. Rebuild.

### Optional manifest keys (steer the build)

| key | effect |
|---|---|
| `"manuscript_order": <int>` | sort position in the gallery + assembled manuscript (default: last) |
| `"regen": "<shell cmd>"` | command (run from repo root) that rebuilds the section's figures; run by `--regen`. Omit if the generator needs data not in-tree — the section is then shown as-is. |

The gallery prefers a sibling `<stem>.svg` over the manifest's listed raster, so listing a `.png` in
`figures[].path` still displays the vector version when one exists next to it.
