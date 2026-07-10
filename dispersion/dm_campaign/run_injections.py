"""Phase-0 injection matrix: every in-tree DM estimator vs known truth.

Runs the three in-tree estimators (arrival regression, DM-phase variant,
DM-power variant) over an injection matrix on both instrument geometries and
emits per-estimator recovery JSON + recovery-vs-truth grids + a contact sheet
for owner visual review (plan-dm-measurement-methods Phase 0; visual vetting
is a first-class acceptance criterion, not just the numeric gate).

Usage (from pipeline/):
    conda run -n flits python -m dispersion.dm_campaign.run_injections --quick

Geometry notes:
- Both geometries use analysis-product resolution (256 ch, 163.84 us), the
  resolution the Phase-1 battery will run on — not raw resolution.
- CHIME truth is drawn from +-1.5 (not the plan's +-3): the 82 ms singlebeam
  cutout bounds the usable sweep (+-1.5 -> +-29 ms at 400-800 MHz already
  approaches the window edge). DSA truth +-2.5 (delay span ~1.4 ms, unbound).
- Search windows: +-5 (DSA) / +-4 (CHIME), identical for all estimators
  (uniform-battery constraint) and sized so the CHIME coarse scan cannot
  shift a real burst fully out of the cutout.
- Noise is synthetic Gaussian here; real off-pulse bootstrap
  (make_noise_from_offpulse) is exercised against real products in Phase 1.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# One BLAS thread per worker, set before the spawned workers import numpy:
# N workers x default all-core BLAS oversubscribes (froze a 12-core laptop,
# 2026-07-09) and is slower than pinned single-threaded workers.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

from dispersion.chime_dm import measure_dm
from dispersion.dm_campaign.injection import InjectionSpec, inject_pulse
from dispersion.dm_power_analysis import measure_dm_power
from dispersion.dmphasev2 import DMPhaseEstimator, dmphase_trial_to_physical_residual_dm

DM_REF = 500.0  # arbitrary fiducial; estimators measure dm_ref + injected offset
DM_STEP = 0.25
SNRS = (8.0, 25.0, 80.0)
WIDTHS_MS = (0.3, 2.0)
TAU600_MS = (0.0, 5.0, 20.0)  # scattering at 600 MHz; tau_1GHz = tau_600 * 0.6^4
COMPONENTS = (1, 2)
INSTRUMENTS = {
    "dsa": dict(f_lo_ghz=1.31125, f_hi_ghz=1.49875, nchan=256, dt_ms=0.16384, ntime=1024,
                window=5.0, truth_dm=2.5),
    "chime": dict(f_lo_ghz=0.40019, f_hi_ghz=0.80019, nchan=256, dt_ms=0.16384, ntime=512,
                  window=4.0, truth_dm=1.5),
}
ESTIMATORS = ("arrival_regression", "dmphase_variant_intree", "dmpower_variant_intree")


def run_cell(cell):
    """Inject one known-truth burst and run all three estimators on the SAME waterfall."""
    geom = INSTRUMENTS[cell["instrument"]]
    rng = np.random.default_rng(cell["cell_seed"])
    freq_ghz = np.linspace(geom["f_lo_ghz"], geom["f_hi_ghz"], geom["nchan"])
    freqs_mhz = freq_ghz * 1e3
    dt_s = geom["dt_ms"] * 1e-3

    noise = rng.normal(size=(geom["nchan"], geom["ntime"])).astype(np.float32)
    dm_off = float(rng.uniform(-geom["truth_dm"], geom["truth_dm"]))
    spec = InjectionSpec(
        dm_offset=dm_off,
        snr=cell["snr"],
        width_ms=cell["width_ms"],
        tau_1ghz_ms=cell["tau600_ms"] * 0.6**4,
        components=cell["components"],
    )
    wf, truth = inject_pulse(noise, freq_ghz, geom["dt_ms"], spec, rng)
    dm_true = DM_REF + truth["dm_offset"]
    grid = np.arange(-geom["window"], geom["window"] + DM_STEP, DM_STEP)

    rows = []
    for name in ESTIMATORS:
        t_start = time.perf_counter()
        rec, sig, reason = None, None, "ok"
        try:
            if name == "arrival_regression":
                res = measure_dm(wf, freqs_mhz, dt_s, DM_REF,
                                 dm_window=geom["window"], dm_step=DM_STEP)
                rec, sig, reason = res["dm"], res["dm_err"], res["reason"]
            elif name == "dmphase_variant_intree":
                est = DMPhaseEstimator(wf.T, freqs_mhz, dt_s, grid,
                                       n_boot=cell["n_boot"], random_state=cell["cell_seed"])
                rec = DM_REF + dmphase_trial_to_physical_residual_dm(est.dm_best)
                sig = float(est.dm_sigma)
            else:
                res = measure_dm_power(wf, freqs_mhz, dt_s, DM_REF, grid,
                                       n_boot=cell["n_boot"], random_state=cell["cell_seed"])
                rec, sig, reason = res["dm"], res["dm_err"], res["reason"]
        except Exception as e:  # a crash is itself a finding, not a run failure
            reason = f"{type(e).__name__}: {e}"
        rows.append({
            "estimator": name,
            **{k: cell[k] for k in
               ("instrument", "snr", "width_ms", "tau600_ms", "components", "seed")},
            "dm_offset_true": truth["dm_offset"],
            "dm_true": dm_true,
            "dm_rec": None if rec is None else float(rec),
            "dm_sigma": None if sig is None else float(sig),
            "err": None if rec is None else float(rec - dm_true),
            "constrained": rec is not None,
            "reason": reason,
            "runtime_s": round(time.perf_counter() - t_start, 3),
        })
    return rows


def summarize(rows):
    """Numeric gate per (estimator, instrument) on constrained S/N>=25 cells.

    '68% coverage within [0.5, 1.5]' is read as: the 68th percentile of
    |err|/sigma must land in [0.5, 1.5] (well-calibrated errors give ~1).
    """
    out = {}
    for name in ESTIMATORS:
        for instr in INSTRUMENTS:
            sel = [r for r in rows if r["estimator"] == name and r["instrument"] == instr]
            ok = [r for r in sel if r["constrained"] and r["snr"] >= 25
                  and r["dm_sigma"] and r["dm_sigma"] > 0]
            entry = {"n_cells": len(sel), "n_constrained_snr25": len(ok),
                     "n_unconstrained": sum(not r["constrained"] for r in sel)}
            if ok:
                errs = np.array([r["err"] for r in ok])
                sigs = np.array([r["dm_sigma"] for r in ok])
                entry.update(
                    median_bias=float(np.median(errs)),
                    median_sigma=float(np.median(sigs)),
                    p68_abs_z=float(np.percentile(np.abs(errs / sigs), 68)),
                )
                entry["pass_numeric"] = bool(
                    abs(entry["median_bias"]) < entry["median_sigma"]
                    and 0.5 <= entry["p68_abs_z"] <= 1.5
                )
            else:
                entry["pass_numeric"] = False
            out[f"{name}/{instr}"] = entry
    return out


def plot_recovery_grid(rows, name, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_color = {8.0: "tab:grey", 25.0: "tab:blue", 80.0: "tab:red"}
    fig, axes = plt.subplots(2, len(TAU600_MS), figsize=(11, 6),
                             sharex="row", sharey=True, constrained_layout=True)
    for i, instr in enumerate(INSTRUMENTS):
        for j, tau in enumerate(TAU600_MS):
            ax = axes[i, j]
            sel = [r for r in rows if r["estimator"] == name
                   and r["instrument"] == instr and r["tau600_ms"] == tau]
            for r in sel:
                if not r["constrained"]:
                    ax.plot(r["dm_offset_true"], 0, "x", color="k", ms=5, alpha=0.4)
                    continue
                mk = "o" if r["components"] == 1 else "s"
                ax.errorbar(r["dm_offset_true"], r["err"], yerr=r["dm_sigma"],
                            fmt=mk, ms=4, color=snr_color[r["snr"]],
                            mfc="none" if r["width_ms"] > 1 else None,
                            elinewidth=0.7, capsize=0, alpha=0.85)
            ax.axhline(0, color="k", lw=0.6)
            if i == 0:
                ax.set_title(f"tau600 = {tau:g} ms", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{instr.upper()}\nrecovered $-$ truth [pc cm$^{{-3}}$]")
            ax.set_xlabel("true $\\Delta$DM [pc cm$^{-3}$]")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"S/N {int(s)}")
               for s, c in snr_color.items()]
    handles += [plt.Line2D([], [], marker="s", ls="", color="k", label="2 components"),
                plt.Line2D([], [], marker="x", ls="", color="k", label="unconstrained")]
    fig.legend(handles=handles, loc="lower center", ncols=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"{name}: DM recovery vs truth (open markers = 2 ms width)", y=1.04)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_contact_sheet(rows, summary, path):
    """Owner-review sheet: one panel per (estimator, instrument), one question each —
    does the estimator recover truth, and are its errors honest? Morphology is the
    only color dimension (it drove the dmphase finding); S/N 8 is de-emphasized."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    morph_color = {(0.3, 1): "tab:blue", (2.0, 1): "tab:orange",
                   (0.3, 2): "tab:purple", (2.0, 2): "tab:red"}
    morph_label = {(0.3, 1): "narrow, 1 comp", (2.0, 1): "wide (2 ms), 1 comp",
                   (0.3, 2): "narrow, 2 comp", (2.0, 2): "wide, 2 comp"}
    ylim = {"arrival_regression": 3.0, "dmphase_variant_intree": 6.5,
            "dmpower_variant_intree": 6.5}

    fig, axes = plt.subplots(len(ESTIMATORS), 2, figsize=(12, 13), sharex="col")
    for i, name in enumerate(ESTIMATORS):
        for j, instr in enumerate(INSTRUMENTS):
            ax = axes[i, j]
            lim = ylim[name]
            sel = [r for r in rows if r["estimator"] == name and r["instrument"] == instr]
            n_unc = sum(not r["constrained"] for r in sel)
            for r in sel:
                if not r["constrained"]:
                    continue
                lo_snr = r["snr"] < 25
                c = "0.75" if lo_snr else morph_color[(r["width_ms"], r["components"])]
                y = float(np.clip(r["err"], -lim, lim))
                clipped = abs(r["err"]) > lim
                ax.errorbar(r["dm_offset_true"], y,
                            yerr=None if clipped else r["dm_sigma"],
                            fmt="^" if clipped else "o", ms=5 if lo_snr else 7,
                            color=c, elinewidth=1.0, capsize=0,
                            alpha=0.45 if lo_snr else 0.9, zorder=2 if lo_snr else 3)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_ylim(-1.05 * lim, 1.05 * lim)
            g = summary[f"{name}/{instr}"]
            if "median_bias" in g:
                verdict = "PASS" if g["pass_numeric"] else "FAIL"
                txt = (f"{verdict}   bias {g['median_bias']:+.2f}   "
                       f"$\\sigma_{{med}}$ {g['median_sigma']:.2f}   "
                       f"p68$|z|$ {g['p68_abs_z']:.1f}")
            else:
                verdict, txt = "FAIL", "FAIL   no constrained S/N$\\geq$25 cells"
            ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", lw=0,
                              fc="#c8e6c9" if verdict == "PASS" else "#ffcdd2"))
            ax.text(0.98, 0.03, f"{n_unc}/{len(sel)} unconstrained",
                    transform=ax.transAxes, ha="right", fontsize=9, color="0.4")
            if i == 0:
                ax.set_title(instr.upper(), fontsize=14)
            if j == 0:
                ax.set_ylabel(f"{name}\nrecovered $-$ truth [pc cm$^{{-3}}$]", fontsize=11)
            if i == len(ESTIMATORS) - 1:
                ax.set_xlabel("true $\\Delta$DM [pc cm$^{-3}$]", fontsize=12)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=c, label=morph_label[k])
               for k, c in morph_color.items()]
    handles += [plt.Line2D([], [], marker="o", ls="", ms=5, color="0.75", alpha=0.6,
                           label="S/N 8 (any morphology)"),
                plt.Line2D([], [], marker="^", ls="", ms=7, color="k",
                           label="off-scale (clipped)")]
    fig.legend(handles=handles, loc="upper center", ncols=3, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("DM injection recovery -- error bars should cover the scatter;"
                 " points off the zero line at S/N$\\geq$25 are misses", y=1.045,
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds/cell, reduced bootstraps (local smoke of the full matrix)")
    ap.add_argument("--out", type=Path, default=Path("results/dm_campaign/injections"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--replot", action="store_true",
                    help="re-render figures from the JSONs already in --out (no compute)")
    args = ap.parse_args(argv)

    if args.replot:
        rows = [r for name in ESTIMATORS
                for r in json.loads((args.out / f"{name}.json").read_text())]
        for name in ESTIMATORS:
            plot_recovery_grid(rows, name, args.out / f"{name}_recovery.png")
        plot_contact_sheet(rows, summarize(rows), args.out / "contact_sheet.png")
        print(f"replotted {args.out}")
        return

    n_seeds = 2 if args.quick else 5
    n_boot = 20 if args.quick else 100
    cells = []
    for instr in INSTRUMENTS:
        for snr in SNRS:
            for width in WIDTHS_MS:
                for tau in TAU600_MS:
                    for comp in COMPONENTS:
                        for seed in range(n_seeds):
                            cells.append(dict(
                                instrument=instr, snr=snr, width_ms=width, tau600_ms=tau,
                                components=comp, seed=seed, n_boot=n_boot,
                                # crc32, not hash(): hash() is salted per process
                                cell_seed=zlib.crc32(
                                    f"{instr}:{snr}:{width}:{tau}:{comp}:{seed}".encode()),
                            ))
    print(f"{len(cells)} cells x {len(ESTIMATORS)} estimators "
          f"({'quick' if args.quick else 'full'} mode)")

    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, cell_rows in enumerate(pool.map(run_cell, cells)):
            rows.extend(cell_rows)
            if (i + 1) % 24 == 0:
                print(f"  {i + 1}/{len(cells)} cells ({time.perf_counter() - t0:.0f}s)")

    args.out.mkdir(parents=True, exist_ok=True)
    for name in ESTIMATORS:
        (args.out / f"{name}.json").write_text(json.dumps(
            [r for r in rows if r["estimator"] == name], indent=1))
        plot_recovery_grid(rows, name, args.out / f"{name}_recovery.png")

    summary = summarize(rows)
    (args.out / "injection_summary.json").write_text(json.dumps(
        {"mode": "quick" if args.quick else "full", "n_cells": len(cells),
         "dm_ref": DM_REF, "gate": summary}, indent=1))

    plot_contact_sheet(rows, summary, args.out / "contact_sheet.png")

    for key, entry in summary.items():
        print(f"{key}: {entry}")
    print(f"wrote {args.out} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
