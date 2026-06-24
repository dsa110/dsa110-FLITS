# Per-Burst Association Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one 1 by 2 association figure per CHIME/DSA co-detected burst and place the figures in the Faber2026 manuscript.

**Architecture:** Reuse `crossmatching/toa_crossmatch_results.json`, `crossmatching/association_report.json`, `crossmatching/chime_side_inputs.json`, `crossmatching/notebook_reproduction_fixture.json`, and existing beam helpers in `analysis/`. A new plotting script writes PDF/PNG outputs under `crossmatching/association_cards/`; the manuscript copies PDFs into `Faber2026/figures/association_cards/` and includes them from `sections/toa.tex`.

**Tech Stack:** Python, matplotlib, astropy, existing FLITS JSON fixtures, existing CHIME/DSA beam helpers, LaTeX.

---

### Task 1: Figure Generator

**Files:**
- Create: `crossmatching/plot_association_cards.py`
- Output: `crossmatching/association_cards/*.pdf`
- Output: `crossmatching/association_cards/*.png`

- [ ] Load TOA, association, CHIME-side, and fixture JSON files.
- [ ] For each burst, compute residual and uncertainty from the existing TOA result.
- [ ] Draw left panel: TOA residual axis with CHIME and DSA markers, uncertainty band, residual annotation, and compact DM inset only when CHIME DM is constrained.
- [ ] Draw right panel: DSA coordinate, CHIME tied-beam coordinate, CHIME localization radius, CHIME approximate beam contour, DSA measured beam contour when the local beam cube is present, and separation annotation.
- [ ] Write one PDF and one PNG per burst.

### Task 2: Manuscript Wiring

**Files:**
- Create/copy: `/Users/jakobfaber/Developer/overleaf/Faber2026/figures/association_cards/*.pdf`
- Modify: `/Users/jakobfaber/Developer/overleaf/Faber2026/sections/toa.tex`

- [ ] Copy all generated PDFs into the manuscript figures tree.
- [ ] Add a compact multi-page figure block after the TOA residual table.
- [ ] Keep chance-coincidence probabilities in the table/text path, not as figure panels.

### Task 3: Verification

**Commands:**
- `python crossmatching/plot_association_cards.py`
- `pytest tests/test_association.py tests/test_crossmatching_notebook_reproduction.py -q`
- Build or dry-run the manuscript references from `/Users/jakobfaber/Developer/overleaf/Faber2026`.

- [ ] Confirm 12 PDF figures and 12 PNG figures are generated.
- [ ] Confirm all manuscript `\includegraphics` targets exist.
- [ ] Commit only scoped FLITS and Faber2026 changes, leaving unrelated dirty files untouched.
