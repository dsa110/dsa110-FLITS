# Figure-review protocol (visual-assessment gate)

A figure is not "validated" until a human-equivalent has actually **looked** at it and
compared it to what it was supposed to show. This repo enforces that mechanically so
"I made a plot" can never silently become "the result checks out" — the exact failure
mode where a band-wide spectral artifact in a simulated burst went unnoticed until
someone opened the PNG.

## The contract

1. **Producer writes a manifest.** Any script that saves figures calls
   `tools.figure_manifest.write_manifest(out_dir, [(png, expectation), ...])`, emitting
   `<out_dir>/figures.manifest.json`. Each entry pairs a PNG with a one-line statement of
   what it should show (peaks, tails, scales, smoothness, the feature that supports the
   claim). Write the *expectation*, not a description of the code.

2. **Reviewer looks and records verdicts.** Before the figures can be cited, each PNG is
   opened (read as an image, so it renders) and assessed panel-by-panel against its
   expectation, the way a skeptical referee would. The result is written to
   `<out_dir>/figures.review.json`:

   ```json
   {"reviewed_by": "...",
    "verdicts": [{"path": "<png>", "verdict": "match" | "anomaly" | "skipped:<why>",
                  "panels": {"<panel>": "<what was seen>"}, "notes": "..."}]}
   ```

   Use the `figure-reviewer` subagent (`Agent` tool, `subagent_type: "figure-reviewer"`)
   whose only job is to view images and judge them — or do it inline. "Looks fine" is not
   a verdict; name what you saw.

3. **The Stop hook enforces it.** `.claude/hooks/figure-review-gate.sh` (registered as a
   project `Stop` hook in `.claude/settings.json`) scans for every `figures.manifest.json`
   under the repo and **blocks end-of-turn** if any manifest is newer than its
   `figures.review.json` (or has none). Pure mtime check — no-op when no figures exist.

## Escape hatch

If a figure genuinely cannot be assessed (corrupt, irrelevant, intentionally skipped),
record `"verdict": "skipped:<reason>"` for it. The gate clears once a review exists that
is at least as new as the manifest.

## Notes

- Output dirs (e.g. `simulation/validation_out/`) are gitignored, so the manifest/review
  JSON live next to the (also gitignored) PNGs and are regenerated per run.
- The hook and the `figure-reviewer` agent are loaded at session start; after adding or
  editing them, reload the Claude Code session in this repo for them to take effect.
- Scope is this repo only (project `.claude/`). Promote to `~/.claude/` to make it global.
