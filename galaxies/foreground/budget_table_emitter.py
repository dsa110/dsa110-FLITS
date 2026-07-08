"""Emit budget_table.tex from a structured single-source data file.

Rationale
---------
The manuscript's ``budget_table.tex`` was historically hand-transcribed, which
let values drift from the pipeline (e.g. the DR8/DR9 survey-release mismatch
caught in ``language_audit.md``). This module makes the table *generated*: the
values live in ``budget_table_data.json`` (one place to review and edit) and the
AASTeX ``deluxetable`` markup is assembled here. The DM_host posterior column is
cross-checked against the forward-model output of
``scripts/dm_budget_uncertainty.py`` by ``tests/test_budget_table_emitter.py``.

Usage
-----
    from galaxies.foreground.budget_table_emitter import format_budget_table_tex
    tex = format_budget_table_tex()          # -> full .tex as a string

CLI (also wired as ``--emit-tex`` on the sightline_budget entry point):
    python -m galaxies.foreground.budget_table_emitter            # writes exports/budget_table.tex
    python -m galaxies.foreground.budget_table_emitter --check    # parity check, non-zero exit on drift
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DATA_PATH = HERE / "budget_table_data.json"
# FLITS-tree export target (parity anchor for the regression test).
EXPORT_PATH = HERE.parents[1] / "exports" / "budget_table.tex"

# --- Static AASTeX markup, copied verbatim from the manuscript's committed
# --- budget_table.tex so the first parity diff isolates VALUE drift from
# --- MARKUP drift. If the table's columns/caption/footnotes change, edit these
# --- two blocks (and the manuscript copy) in one commit.
_HEAD = r"""% !! GENERATED FILE -- do not edit by hand. Values live in
%    galaxies/foreground/budget_table_data.json; markup in budget_table_emitter.py.
%    Regenerate: python -m galaxies.foreground.budget_table_emitter --out <this file>
% Derived from the analysis results used in this work. DM is in pc\,cm$^{-3}$;
% $\tau_{\mathrm{int}}$ is the predicted intervening pulse-broadening at 1\,GHz.
% Update by regenerating the table from the analysis outputs, not by hand.
% NOTE(V1 gate): the measured tau_obs column (joint-fit tau_1GHz) and its
% tab:beta cross-reference are withheld until the fit re-validation ladder (V1)
% clears; restore them by regenerating the full table, not by hand-editing.
\begin{deluxetable*}{lrrrrrllrr}
\tabletypesize{\scriptsize}
\tablecaption{Per-sightline dispersion measure budget for the twelve co-detections,
from the decomposition of Eq.~\ref{eq:dmbudget}. $\mathrm{DM_{obs}}$ is the
DSA-110 catalog dispersion measure under the shared DSA-DM reference
convention of Section~\ref{sec:toa}. $\mathrm{DM_{MW}}$ is the NE2025
\citep{Ocker2026} disk (integrated to $30\,\mathrm{kpc}$) plus a
$40\,\mathrm{pc\,cm^{-3}}$ halo; $\langle\mathrm{DM_{cos}}\rangle$ is
the Macquart mean at the host redshift; $\mathrm{DM_{int}}$ is the intervening
circumgalactic column summed over the confirmed foreground systems (two-phase
hot mNFW + cool, capped at $b=0.1\,R_{\mathrm{vir}}$ in the galaxy-interior
regime); $\mathrm{DM_{host}}$ is the forward-modeled host posterior (median with
$16$th--$84$th-percentile interval; Appendix~\ref{app:host-forward-model}),
which supersedes the arithmetic residual by sampling the full
$P(\mathrm{DM_{cosmic}}\,|\,z)$ and the Galactic-disk, Galactic-halo, and
intervening priors rather than subtracting the (skewed) cosmological mean.
``regime'' and ``mass'' give
the intervening column's geometry (CGM, galaxy-interior, or none) and whether
the dominant halo's mass is measured or assumed. $\tau_{\mathrm{int}}$ is the
predicted intervening pulse-broadening at $1\,\mathrm{GHz}$ from the
thin-screen two-phase model (Section~\ref{sec:scattering}); the measured
scattering comparison is deferred to the scattering analysis
(Section~\ref{sec:results-alpha}). DM in
$\mathrm{pc\,cm^{-3}}$, $\tau$ in ms.
\label{tab:budget}}
\tablehead{\colhead{Burst} & \colhead{$z$} & \colhead{$\mathrm{DM_{obs}}$} &
\colhead{$\mathrm{DM_{MW}}$} & \colhead{$\langle\mathrm{DM_{cos}}\rangle$} &
\colhead{$\mathrm{DM_{int}}$} & \colhead{regime} & \colhead{mass} &
\colhead{$\mathrm{DM_{host}}$} & \colhead{$\tau_{\mathrm{int}}$}}
\startdata
"""

_TAIL = r"""\enddata
\tablenotetext{p}{Host redshift unknown (placeholder); the cosmological and host
terms cannot be computed, so this sightline is excluded from any distance-dependent
quantity.}
\tablenotetext{c}{Sightline in the galaxy-interior regime ($b<0.1\,R_{\mathrm{vir}}$,
$b\approx4.6$\,kpc); $\mathrm{DM_{int}}$ is capped at the $b=0.1\,R_{\mathrm{vir}}$
value (raw core-extrapolated column $71\,\mathrm{pc\,cm^{-3}}$).}
\tablenotetext{u}{Position lies outside the deep-imaging survey footprints
(DESI Legacy DR8-North; SDSS DR12 returns no redshifts here), so only the
shallow all-sky catalogs (NED, GLADE+) constrain the sightline: the
intervening term is \emph{unconstrained by the searched surveys}, not
excluded---absence of coverage is not absence of foreground
(Section~\ref{sec:obs-fg}).}
\tablenotetext{m}{Confirmed foreground halos are present but lie beyond the
virial radii implied by their assumed halo masses
($b/R_{\mathrm{vir}}\approx1.3$--$1.8$ at the fallback mass), so the modeled
two-phase column vanishes. The fallback mass is a fiducial stellar mass
($\log M_\star/M_\odot\approx10$, the midpoint of the $9$--$11$ range spanned by
the sensitivity prior families) converted to halo mass through the
\citet{Moster2013} stellar-to-halo-mass relation. This is a median guess, not a
conservative lower bound, so its sign is not one-directional:
$\mathrm{DM_{int}}=0$ here is conditional on the assumed mass, and a more
massive (or a measured) halo would bring the sightline inside $R_{\mathrm{vir}}$
and make the column nonzero.}
\tablecomments{Because the cosmological term is drawn from a highly skewed
distribution, the host posteriors are broad and asymmetric and their medians
exceed the naive mean-subtracted residuals. The two highest-redshift sightlines
(FRB 20220310F, FRB 20221203A; $z=0.48$, $0.51$), where the Macquart term
dominates the budget, have host posteriors consistent with zero
($P(\mathrm{DM_{host}}<0)\approx0.45$): their formerly negative point residuals
reflect sightline-to-sightline scatter about the cosmological mean rather than an
unphysical host. Per-sightline $P(\mathrm{DM_{host}}<0)$ and the forward-model
priors are tabulated in Appendix~\ref{app:host-forward-model}.}
\end{deluxetable*}
"""


def _load_rows(data_path: pathlib.Path | str | None = None) -> list[dict]:
    path = pathlib.Path(data_path) if data_path is not None else DATA_PATH
    return json.loads(path.read_text())["rows"]


def render_cells(r: dict) -> list[str]:
    """Render one structured row into its ten deluxetable cells."""
    burst = r["burst"] + (rf"\tablenotemark{{{r['burst_note']}}}" if r.get("burst_note") else "")
    z = r"\nodata\tablenotemark{p}" if r["z"] is None else f"${r['z']:.3f}$"
    dm_cos = r"\nodata" if r["dm_cos"] is None else f"{r['dm_cos']}"
    dm_int = f"{r['dm_int']}" + (rf"\tablenotemark{{{r['dm_int_note']}}}" if r.get("dm_int_note") else "")
    mass = r"\nodata" if not r.get("mass") else r["mass"]
    if r["dm_host"] is None:
        host = r"\nodata"
    else:
        med, plus, minus = r["dm_host"]
        host = f"${med}^{{+{plus}}}_{{-{minus}}}$"
    return [
        burst, z, f"{r['dm_obs']}", f"{r['dm_mw']}", dm_cos, dm_int,
        r["regime"], mass, host, f"${r['tau_int']}$",
    ]


def render_row(r: dict) -> str:
    return " & ".join(render_cells(r)) + r" \\"


def format_budget_table_tex(data_path: pathlib.Path | str | None = None) -> str:
    """Return the complete budget_table.tex as a string (values from JSON)."""
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
    tex = format_budget_table_tex()
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
