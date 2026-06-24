# Handoff prompt — drive h17 over SSH from a laptop agent

You're a coding agent running on a laptop. **All compute, data, docker, and the git repo live on a remote
host `lxd110h17` (alias `h17`); you operate it over SSH.** Nothing is local except your own scratch space
for viewing files you pull back. Pick up the CHIME–DSA co-detection manuscript-figure work below.

## Remote facts (verified 2026-06-24)

- **SSH target:** `lxd110h17` (the alias resolves; `scp lxd110h17:…` works).
- **Repo on h17:** `/home/ubuntu/Developer/repos/github.com/jakobtfaber/dsa110-FLITS`, branch
  `feat/custom-dm-tool` (PR #41). Remote `origin` = `git@github.com:jakobtfaber/dsa110-FLITS.git`.
- **Python env:** conda **`casa6`** at `/opt/miniforge` (NOT "flits" — that env doesn't exist here).
  It has numpy 1.26 / scipy 1.15 / matplotlib 3.10. A bare SSH shell does **not** auto-activate it.
- **Docker:** `chimefrb/baseband-analysis:latest` present and runnable by user `ubuntu` (only needed to
  re-do the dedispersion — Path B; you almost never need it).
- **Headless browser on h17:** `/usr/bin/google-chrome` (use it to rasterize the HTML gallery for viewing).

## The one rule for running anything on h17

A non-interactive SSH shell won't have conda or the env. Always prefix with the env + repo `cd`:

```bash
PFX='source /opt/miniforge/etc/profile.d/conda.sh && conda activate casa6 && cd /home/ubuntu/Developer/repos/github.com/jakobtfaber/dsa110-FLITS'
ssh lxd110h17 "$PFX && <your command>"
```

(The `matplotlibrc` line-16 "axes.prop_cycle … not a valid cycler" warning is a **pre-existing, harmless**
parse quirk — figures still render fully styled. Filter it with `2>&1 | grep -v matplotlibrc`.)

## State — what's already done (don't redo)

On `feat/custom-dm-tool`, the independent CHIME-side DM (association Pillar 2) is landed: **8/12 bursts
constrain the DM** (all consistent with DSA within the 1 pc/cm³ floor), 4/12 honest non-detections.
This session added the manuscript figure + a systematic figure pipeline, **currently uncommitted**:

```
analysis/chime_dm/plot_dm_grid.py            renders the 12-panel SVG/PDF/PNG (data default path is on h17;
                                             knobs: CHIME_DM_TSMOOTH=12 default, CHIME_DM_DATA/_OUT/_SUFFIX/_EXTS)
analysis/chime_dm/chime_dm_grid.{svg,pdf,png}  the figure (ready to use)
analysis/chime_dm/chime_dm.tex               section prose + figure float
analysis/chime_dm/figures.manifest.json      figure contract (+ manuscript_order/regen keys)
analysis/chime_dm/figures.review.json        figure-review verdict (s12 smoothing)
analysis/build_manuscript.py                 discovers analysis/*/figures.manifest.json -> SVG gallery + manuscript.tex
analysis/README.md                           the add/update/view convention
analysis/manuscript_figures.html             generated SVG gallery (open in a browser)
analysis/manuscript.tex                      generated assembled sections (cd analysis && pdflatex)
analysis/burst_energies/figures.manifest.json  +manuscript_order
outputs/handoff_manuscript-figures_2026-06-24/HANDOFF.md  (this file)
```

The bridge data the figure renders from already sits on h17 — no need to move or set anything:
`…/chime-dsa-codetections/results/chime_dm_grid_fits.json` (8 KB) + `chime_dm_grid_waterfalls.npz` (1.4 MB).
`plot_dm_grid.py`'s default `CHIME_DM_DATA` points there, so on h17 it Just Works.

## Run the pipeline (all remote)

```bash
# regenerate the 12-panel figure (SVG+PDF+PNG) from the bridge artifacts:
ssh lxd110h17 "$PFX && python analysis/chime_dm/plot_dm_grid.py"

# build the SVG gallery + assembled manuscript.tex:
ssh lxd110h17 "$PFX && python analysis/build_manuscript.py"

# add/update sections systematically (re-runs each section's `regen`, then rebuilds):
ssh lxd110h17 "$PFX && python analysis/build_manuscript.py --regen"

# optional: compile the assembled manuscript section to PDF:
ssh lxd110h17 "$PFX && cd analysis && pdflatex -interaction=nonstopmode manuscript.tex"
```

## How to VIEW a figure or the gallery (you're on the laptop; h17 has no display you can see)

Render/keep the artifact on h17, then `scp` it local and open it with your image tool:

```bash
# the figure PNG (or .svg) directly:
scp lxd110h17:/home/ubuntu/Developer/repos/github.com/jakobtfaber/dsa110-FLITS/analysis/chime_dm/chime_dm_grid.png /tmp/

# the HTML gallery -> rasterize on h17 with chrome, then pull the PNG:
ssh lxd110h17 "$PFX && google-chrome --headless --no-sandbox --disable-gpu --window-size=1100,2400 \
  --screenshot=/tmp/gallery.png file://\$PWD/analysis/manuscript_figures.html"
scp lxd110h17:/tmp/gallery.png /tmp/
```

If you have the repo's `.claude` figure-review Stop gate active in *your* session, you must actually look
at each pulled PNG and write the dir's `figures.review.json` before finishing — see `.claude/hooks/figure-review-gate.sh`.

## How to EDIT code

Your Edit/Write tools act on the **laptop** filesystem, but the repo lives on **h17** — so edit through git,
don't edit blind over SSH:

- **Preferred:** keep a local clone, edit with normal tools, `git push`, then sync h17:
  `ssh lxd110h17 "$PFX && git pull --ff-only"`. Run/verify remotely, pull artifacts back to view.
- **Tiny one-off remote edits** are possible with `ssh … "sed -i …"` or a heredoc, but that's fragile —
  prefer the git path for anything non-trivial.

Watch the post-edit autoformatter on h17: it strips an import added in an edit that has no consumer yet
(it removed `scipy.ndimage` here once). Add any import in the same edit as its first use.

## First thing to do

Confirm the link and state, then show me the current figure:
```bash
ssh lxd110h17 "$PFX && git status --short && python analysis/build_manuscript.py"
scp lxd110h17:/home/ubuntu/Developer/repos/github.com/jakobtfaber/dsa110-FLITS/analysis/chime_dm/chime_dm_grid.png /tmp/
```
…then open `/tmp/chime_dm_grid.png`. Ask me before committing/pushing the uncommitted work to the shared PR branch.

## Path B — re-doing the dedispersion (rare; needs docker + baseband data, both on h17)

```bash
ssh lxd110h17 "cd /data/research/astrophysics/frbs/chime-dsa-codetections && bin/baseband_analysis_python.sh scripts/dump_grid_data.py"
```
≈1–2 min (`Pool(6)`), uniform `TDS=32 / N_SB=6`, `MIN_SNR=4`, `MIN_GOOD=3`, 1 pc/cm³ floor. Rewrites the two
`results/chime_dm_grid_*` bridge files in place; then re-run the figure step. See `.agents/audit-chime-side-dm.md` (P5).
```
