"""Emit foreground_table.tex from a structured single-source data file.

Rationale
---------
Like the budget table, the manuscript's ``foreground_table.tex`` was
hand-transcribed. This module makes it *generated*: the 28 tabulated census
systems live in ``foreground_table_data.json`` (one place to review and edit)
and the AASTeX ``deluxetable`` markup is assembled here. Each row's object ID and
verdict are cross-checked against the committed census registry
(``data/intervening_census_registry.csv``) by
``tests/test_foreground_table_emitter.py``, tying the manuscript table to a
recomputable validation product rather than a hand transcription.

The table is a *curated subset* of the registry's confirmed+inconclusive
systems (refuted candidates and stellar classifications are omitted per
Section 2), so the emitter renders the vetted ``rows`` verbatim rather than
re-deriving the selection; the parity test guards value drift on the rows that
ARE tabulated.

Usage
-----
    from galaxies.foreground.foreground_table_emitter import format_foreground_table_tex
    tex = format_foreground_table_tex()

CLI:
    python -m galaxies.foreground.foreground_table_emitter            # writes exports/foreground_table.tex
    python -m galaxies.foreground.foreground_table_emitter --check    # parity check, non-zero exit on drift
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DATA_PATH = HERE / "foreground_table_data.json"
REGISTRY_PATH = HERE / "data" / "intervening_census_registry.csv"
EXPORT_PATH = HERE.parents[1] / "exports" / "foreground_table.tex"

# --- Static AASTeX markup, copied verbatim from the manuscript's committed
# --- foreground_table.tex so the parity diff isolates VALUE drift from MARKUP
# --- drift. Edit these blocks (and the manuscript copy) in one commit if the
# --- table's columns/caption/footnotes change.
_HEAD = r"""% !! GENERATED FILE -- do not edit by hand. Values live in
%    galaxies/foreground/foreground_table_data.json; markup in foreground_table_emitter.py.
%    Regenerate: python -m galaxies.foreground.foreground_table_emitter --out <this file>
%    Object IDs + verdicts are cross-checked against data/intervening_census_registry.csv.
% Foreground census table. Values from the dsa110-FLITS foreground validation
% (LS DR9 / DESI DR1 / NED /
% PS1-STRM); see sec:obs-fg. Clusters are the WenHan2024 DESI Legacy/WISE
% catalog, restricted to sightlines passing within R500 (only FRB 20230307A
% J115120.4+714435 qualifies); 14 further foreground clusters at b>R500 are omitted.
% Only confirmed and inconclusive systems are tabulated; the 7 candidates
% refuted as background (and stellar classifications) are described in the text
% (sec:obs-fg), matching the treatment of the other categorical cuts.
\startlongtable
\begin{deluxetable*}{lllrrlllll}
\tabletypesize{\tiny}
\tablecaption{Intervening foreground halos and clusters along the sightlines to the 12 CHIME/DSA co-detected FRBs, validated against DESI Legacy DR9 (Zhou+2021 photo-z), DESI DR1 spec-z, NED, and PS1-STRM. Verdicts: \emph{confirmed} (catalog $z<z_{\rm host}$) and \emph{inconclusive} ($z$ within $1\sigma$ of host, host $z$ unknown, or no trustworthy $z$). Candidates refuted as background are omitted (Section~\ref{sec:obs-fg}). \label{tab:foreground}}
\tablehead{\colhead{Burst} & \colhead{Type} & \colhead{Obj ID} & \colhead{$b$} & \colhead{$b/R_{500}$} & \colhead{$z$} & \colhead{$z$ src} & \colhead{Class} & \colhead{Verdict} & \colhead{Note}}
\startdata
"""

_TAIL = r"""\enddata
\tablecomments{All 28 tabulated systems exist in $\geq$1 public catalog; every redshift is DESI/Legacy spectroscopic or LS~DR9/PS1-STRM photometric. $b$ is the proper impact parameter (kpc) at the object redshift. The 7 candidates refuted as background by the validation stage are omitted, as are candidates spectroscopically classified as stars (Section~\ref{sec:obs-fg}). Clusters are drawn from the \citet{WenHan2024} DESI Legacy/WISE catalog and restricted to the one sightline (FRB~20230307A) passing within $R_{500}$; 14 further spectroscopically confirmed foreground clusters lie at $b>R_{500}$ and are omitted as they contribute negligibly to $\mathrm{DM_{int}}$.}
\end{deluxetable*}
"""


def _load_rows(data_path: pathlib.Path | str | None = None) -> list[list[str]]:
    path = pathlib.Path(data_path) if data_path is not None else DATA_PATH
    return json.loads(path.read_text())["rows"]


def render_row(cells: list[str]) -> str:
    """Join the ten verbatim deluxetable cells into a table row."""
    if len(cells) != 10:
        raise ValueError(f"expected 10 cells, got {len(cells)}: {cells!r}")
    return " & ".join(cells) + r" \\"


def format_foreground_table_tex(data_path: pathlib.Path | str | None = None) -> str:
    """Return the complete foreground_table.tex as a string (values from JSON)."""
    rows = _load_rows(data_path)
    body = "\n".join(render_row(r) for r in rows)
    return _HEAD + body + "\n" + _TAIL


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--out", type=pathlib.Path, default=EXPORT_PATH,
                   help=f"output path (default: {EXPORT_PATH})")
    p.add_argument("--check", action="store_true",
                   help="compare against --out instead of writing; exit 1 on drift")
    args = p.parse_args(argv)
    tex = format_foreground_table_tex()
    if args.check:
        if not args.out.exists():
            print(f"MISSING: {args.out}", file=sys.stderr)
            return 1
        if args.out.read_text() == tex:
            print(f"OK: emitter matches {args.out}")
            return 0
        print(f"DRIFT: emitter output differs from {args.out}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex)
    print(f"wrote {args.out} ({len(_load_rows())} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
