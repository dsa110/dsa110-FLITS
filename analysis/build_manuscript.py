#!/usr/bin/env python
r"""Assemble the co-detection manuscript figures into an easy-view SVG gallery (+ a master
manuscript.tex), driven entirely by the per-section ``analysis/<topic>/figures.manifest.json`` files.

A *manuscript section* is any dir under ``analysis/`` with a ``figures.manifest.json``. Two optional
manifest keys steer the build (everything else is the existing figure-review-gate contract):

  "manuscript_order": <int>   sort position in the gallery + assembled manuscript (default: last)
  "regen": "<shell cmd>"      command (run from the repo root) that rebuilds that section's figures;
                              sections without it are shown as-is and skipped by --regen

Figures come from ``figures[].path``; the gallery prefers a sibling ``<stem>.svg`` (vector, crisp) and
falls back to the listed raster. The assembled ``manuscript.tex`` \input's each section's ``*.tex``
(ordered) with a ``\graphicspath`` so its ``\includegraphics{<stem>}`` resolves the sibling PDF/PNG;
the gallery caption uses that tex's ``\caption{}`` when present, else the manifest ``expectation``.

Run from the repo root:
  python analysis/build_manuscript.py            # build gallery + manuscript.tex
  python analysis/build_manuscript.py --regen    # first re-run each section's `regen` command
  python analysis/build_manuscript.py --open     # build, then open the gallery in a browser
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import webbrowser
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
REPO = ANALYSIS.parent
GALLERY = ANALYSIS / "manuscript_figures.html"
MASTER = ANALYSIS / "manuscript.tex"

_HTML = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Co-detection manuscript figures</title>
<style>
  body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
  h1{font-size:1.6rem} h2{font-size:1.1rem;margin:.2rem 0;color:#0C5DA5}
  small{color:#9e9e9e;font-weight:400;font-size:.8rem}
  section{border:1px solid #e3e3e3;border-radius:10px;padding:1rem 1.2rem;margin:1.2rem 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}
  img{display:block;width:100%;height:auto;margin:.8rem 0;background:#fff}
  p{color:#444;font-size:.92rem} code{background:#f4f4f4;padding:.1rem .4rem;border-radius:4px;font-size:.82rem;color:#666}
  .head{color:#555;font-size:.9rem}
  .missing{margin:.8rem 0;padding:2rem;text-align:center;color:#FF2C00;background:#fff5f5;border:1px dashed #ffc2bd;border-radius:8px}
</style>
<h1>Co-detection manuscript figures</h1>
<p class="head">__COUNT__ figures across __NSEC__ sections. Built by <code>analysis/build_manuscript.py</code> &mdash; SVG where available (scale freely). Re-run with <code>--regen</code> to refresh from source.</p>
__BODY__
</html>
"""

_TEX = r"""% Auto-assembled by analysis/build_manuscript.py -- do not hand-edit; edit the per-section *.tex.
% Compile from the analysis/ directory (graphicspath + \input are relative to it):
%   cd analysis && pdflatex manuscript.tex
\documentclass{article}
\usepackage{graphicx,amsmath}
\usepackage[margin=1in]{geometry}
\graphicspath{__GRAPHICSPATH__}
\begin{document}
\section*{CHIME--DSA co-detection: assembled sections}
__BODY__
\end{document}
"""


def sections() -> list[dict]:
    out = []
    for man in sorted(ANALYSIS.rglob("figures.manifest.json")):
        d = man.parent
        j = json.loads(man.read_text())
        tex = next(iter(sorted(d.glob("*.tex"))), None)
        out.append({"dir": d, "man": j, "tex": tex, "order": j.get("manuscript_order", 1_000)})
    out.sort(key=lambda s: (s["order"], str(s["dir"])))
    return out


def _svg_or(path: Path) -> Path:
    svg = path.with_suffix(".svg")
    return svg if svg.exists() else path


def _caption(tex: Path | None) -> str | None:
    if not tex or not tex.exists():
        return None
    t = tex.read_text()
    i = t.find(r"\caption{")
    if i < 0:
        return None
    j, depth = i + 9, 1
    while j < len(t) and depth:  # brace-balance to the matching close
        depth += {"{": 1, "}": -1}.get(t[j], 0)
        j += 1
    return re.sub(r"\s+", " ", t[i + 9 : j - 1]).strip()


def regen(secs: list[dict]) -> None:
    for s in secs:
        cmd = s["man"].get("regen")
        if not cmd:
            continue
        print(f"[regen] {s['dir'].name}: {cmd}")
        subprocess.run(cmd, shell=True, cwd=REPO, check=True)


def build_gallery(secs: list[dict]) -> int:
    cards, n = [], 0
    for s in secs:
        d, cap = s["dir"], _caption(s["tex"])
        for f in s["man"].get("figures", []):
            fig = _svg_or(d / f["path"])
            note = cap or f.get("expectation", "")
            if fig.exists():
                tag = fig.suffix.lstrip(".").upper()
                media = f'<img src="{html.escape(str(fig.relative_to(ANALYSIS)))}" alt="{html.escape(f["path"])}">'
            else:  # manifest points at a figure not present in the tree (gitignored / not yet generated)
                tag, hint = (
                    "MISSING",
                    "regen" if s["man"].get("regen") else "no `regen` cmd in manifest",
                )
                media = f'<div class="missing">figure not on disk &mdash; run <code>--regen</code> ({hint})</div>'
            cards.append(
                f"<section><h2>{html.escape(d.name)} <small>{tag}</small></h2>{media}"
                f"<p>{html.escape(note)}</p>"
                f"<code>{html.escape(str(fig.relative_to(REPO)))}</code></section>"
            )
            n += 1
    GALLERY.write_text(
        _HTML.replace("__BODY__", "\n".join(cards))
        .replace("__COUNT__", str(n))
        .replace("__NSEC__", str(len(secs)))
    )
    print(f"wrote {GALLERY.relative_to(REPO)} ({n} figures, {len(secs)} sections)")
    return n


def build_tex(secs: list[dict]) -> None:
    inc = [s for s in secs if s["tex"]]
    gp = "".join(f"{{{s['dir'].relative_to(ANALYSIS)}/}}" for s in inc)
    body = "\n".join(rf"\input{{{s['tex'].relative_to(ANALYSIS)}}}" for s in inc)
    MASTER.write_text(_TEX.replace("__GRAPHICSPATH__", gp).replace("__BODY__", body))
    print(f"wrote {MASTER.relative_to(REPO)} ({len(inc)} sections with .tex)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--regen", action="store_true", help="re-run each section's `regen` command first"
    )
    ap.add_argument(
        "--open", action="store_true", help="open the gallery in a browser after building"
    )
    a = ap.parse_args()
    secs = sections()
    if a.regen:
        regen(secs)
    build_gallery(secs)
    build_tex(secs)
    if a.open:
        try:
            webbrowser.open(GALLERY.as_uri())
        except Exception as e:  # headless: just print the path
            print(f"(open failed: {e}) view: {GALLERY}")


if __name__ == "__main__":
    main()
