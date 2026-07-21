"""Phase-1 uniform battery: every adapter on every co-detection product.

All 24 products (12 bursts x 2 telescopes) x all five adapters, one config
(configs/battery.yaml), identical windows and downsampling per instrument.
Emits battery_results.json, a 4-panel diagnostic per run, and a per-method
contact sheet for owner visual review. Residuals are relative to each
product's file-stem reference DM (per-instrument-optimized; see
docs/rse/specs/research-dm-measurement-methods.md on that convention).

Usage (from pipeline/):
    conda run -n flits python -m dispersion.dm_campaign.run_battery
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import yaml

from dispersion.dm_campaign.adapters import ADAPTERS
from dispersion.dm_campaign.adaptive_arrival import product_path
from dispersion.dm_power_analysis import (
    CHIME_DT_S,
    DSA_DT_S,
    _dm_ref,
    _dm_ref_source,
    _downsample_for_fit,
    _freq_grid_mhz,
    _orient_waterfall_to_ascending_frequency,
    load_manifest_rows,
    shift_waterfall_residual_dm,
)

_CFG_PATH = Path(__file__).parent / "configs" / "battery.yaml"


def _load_product(row, cfg):
    path = product_path(row, cfg)
    if not path.exists():
        return None
    wf = _orient_waterfall_to_ascending_frequency(
        np.asarray(np.load(path, mmap_mode="r"), dtype=float), row["telescope"])
    freq = _freq_grid_mhz(row["telescope"], wf.shape[0])
    dt_s = CHIME_DT_S if row["telescope"] == "chime" else DSA_DT_S
    return _downsample_for_fit(wf, freq, dt_s, cfg["max_channels"], cfg["max_time"])


def run_product(job):
    row, cfg = job
    burst, tel = row["burst"], row["telescope"]
    loaded = _load_product(row, cfg)
    if loaded is None:
        return [dict(burst=burst, telescope=tel, adapter=a, dm=None, sigma=None,
                     reason="input file missing") for a in cfg["adapters"]]
    wf, freq, dt_s, factors = loaded
    dm_ref = _dm_ref(row)
    window = cfg["windows"][tel]
    # regression-stage fit-quality payload (arrival_regression only): enough to
    # reconstruct the subband-arrival fit line and show gated subbands.
    fit_keys = ("subbands", "subbands_dropped", "coarse_dm", "nu_ref_mhz",
                "peak_snr", "chi2_red", "intercept_s", "railed", "n_good_subbands")
    out = []
    for name in cfg["adapters"]:
        t0 = time.perf_counter()
        try:
            res = ADAPTERS[name].measure(wf, freq_ghz=freq / 1e3, dt_ms=dt_s * 1e3,
                                         dm_ref=dm_ref, window=window)
            rec = dict(dm=res.dm, sigma=res.sigma,
                       residual=None if res.dm is None else res.dm - dm_ref,
                       reason=res.meta.get("reason", "ok") if res.dm is None else "ok",
                       curve={k: np.asarray(v).tolist() for k, v in res.curve.items()},
                       # "subbands" marks the arrival-regression meta; other
                       # adapters share some fit_keys and must not get a fit blob
                       fit={k: res.meta[k] for k in fit_keys if k in res.meta}
                       if "subbands" in res.meta else None)
        except Exception as e:  # a crash on real data is a finding
            rec = dict(dm=None, sigma=None, residual=None,
                       reason=f"{type(e).__name__}: {e}", curve={}, fit=None)
        out.append(dict(burst=burst, telescope=tel, adapter=name, dm_ref=dm_ref,
                        dm_ref_provenance=_dm_ref_source(row),
                        downsample=factors, runtime_s=round(time.perf_counter() - t0, 2),
                        **rec))
    return out


def _zscore(wf):
    med = np.nanmedian(wf, axis=1)[:, None]
    mad = np.nanmedian(np.abs(wf - med), axis=1)[:, None]
    return (wf - med) / np.where(mad > 0, 1.4826 * mad, 1.0)


def plot_run_panels(row, cfg, results, out_dir):
    """4-panel diagnostic per adapter for one product: search curve, waterfall
    at candidate DM, waterfall at reference DM, band profile overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loaded = _load_product(row, cfg)
    if loaded is None:
        return
    wf, freq, dt_s, _ = loaded
    z = _zscore(wf)
    t_ms = np.arange(wf.shape[1]) * dt_s * 1e3
    ext = (0, t_ms[-1], freq[0], freq[-1])
    for r in results:
        if r["burst"] != row["burst"] or r["telescope"] != row["telescope"]:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
        x = r["curve"].get("residual_dm")
        ykey = next((k for k in r["curve"] if k != "residual_dm"), None)
        if x is not None and ykey:
            axes[0].plot(x, r["curve"][ykey], lw=1)
        elif x is not None:
            # released DM_phase exposes no curve; show the searched window
            axes[0].set_xlim(min(x), max(x))
            axes[0].set_yticks([])
            axes[0].text(0.5, 0.5, "package exposes\nno search curve",
                         transform=axes[0].transAxes, ha="center", va="center",
                         fontsize=8, color="0.5")
        if r["residual"] is not None:
            axes[0].axvline(r["residual"], color="r", lw=0.8)
        axes[0].set_xlabel("residual DM [pc cm$^{-3}$]", fontsize=8)
        axes[0].set_title(f"search curve ({r['adapter']})", fontsize=9)
        for ax, ddm, label in ((axes[1], r["residual"], "candidate DM"),
                               (axes[2], 0.0, "reference DM")):
            shown = z if not ddm else _zscore(
                shift_waterfall_residual_dm(wf, freq, dt_s, float(ddm)))
            ax.imshow(shown, aspect="auto", origin="lower", extent=ext,
                      vmin=-1, vmax=6, cmap="magma")
            ax.set_title(label, fontsize=9)
        prof_ref = z.sum(0)
        axes[3].plot(t_ms, prof_ref, color="0.6", lw=0.8, label="at reference")
        if r["residual"]:
            axes[3].plot(t_ms, _zscore(shift_waterfall_residual_dm(
                wf, freq, dt_s, float(r["residual"]))).sum(0), color="r", lw=0.8,
                label="at candidate")
        axes[3].legend(fontsize=7, frameon=False)
        res_txt = ("unconstrained" if r["residual"] is None
                   else f"$\\Delta$DM = {r['residual']:+.3f} $\\pm$ {r['sigma']:.3f}")
        fig.suptitle(f"{row['burst']} / {row['telescope'].upper()} / {r['adapter']}: "
                     f"{res_txt} (ref {r['dm_ref']:.3f})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{row['burst']}_{row['telescope']}_{r['adapter']}.png",
                    dpi=110)
        plt.close(fig)


def plot_regression_panel(r, path):
    """Fit-quality view of the arrival-regression stage for one product:
    subband arrivals vs dispersion delay-per-DM with the fitted line (whose
    slope IS the fine residual DM), normalized residuals, and gated subbands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dispersion.chime_dm import K_DM

    fit = r["fit"]
    nu_ref = fit["nu_ref_mhz"]
    x_of = lambda f_mhz: K_DM * (1.0 / np.asarray(f_mhz) ** 2 - 1.0 / nu_ref**2) * 1e3
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True,
                                  height_ratios=[2.4, 1.0])
    sub = fit["subbands"]
    if sub:
        xs = x_of([s["freq_mhz"] for s in sub])
        ys = np.array([s["t0_s"] for s in sub]) * 1e3
        es = np.array([s["t0_err_s"] for s in sub]) * 1e3
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=4, lw=1, capsize=2, color="k")
        for s, xi, yi in zip(sub, xs, ys, strict=True):
            ax.annotate(f"{s['freq_mhz']:.0f} MHz (S/N {s['snr']:.0f})",
                        (xi, yi), textcoords="offset points", xytext=(4, 4),
                        fontsize=6, color="0.4")
    for d in fit["subbands_dropped"]:
        for a in (ax, axr):
            a.axvline(x_of(d["freq_mhz"]), color="0.8", ls="--", lw=0.8, zorder=0)
    constrained = fit.get("chi2_red") is not None
    if constrained and sub:
        # slope has DM units, x is ms per unit DM -> the product is ms
        slope = r["dm"] - fit["coarse_dm"]
        grid = np.linspace(min(0, xs.min()) * 1.05, max(xs.max(), 0) * 1.05, 50)
        ax.plot(grid, slope * grid + fit["intercept_s"] * 1e3, color="r", lw=1,
                label=f"fit: fine $\\Delta$DM {slope:+.4f}")
        yfit = slope * xs + fit["intercept_s"] * 1e3
        axr.errorbar(xs, (ys - yfit) / es, yerr=1.0, fmt="o", ms=4, lw=1,
                     capsize=2, color="k")
        axr.axhspan(-1, 1, color="0.92", zorder=0)
        ax.legend(fontsize=7, frameon=False)
    axr.axhline(0, color="r", lw=0.8)
    axr.set_xlabel("dispersion delay per unit DM, $K(\\nu^{-2}-\\nu_{\\rm ref}^{-2})$ "
                   "[ms/(pc cm$^{-3}$)]", fontsize=8)
    ax.set_ylabel("subband arrival $t_0$ [ms]", fontsize=8)
    axr.set_ylabel("(data $-$ fit)/$\\sigma$", fontsize=8)
    stat = (f"$\\chi^2_\\nu$ = {fit['chi2_red']:.2f}, "
            f"$\\Delta$DM = {r['residual']:+.4f} $\\pm$ {r['sigma']:.4f}"
            if constrained else f"unconstrained: {r['reason']}")
    fig.suptitle(
        f"{r['burst']} / {r['telescope'].upper()} arrival regression -- "
        f"{fit['n_good_subbands']}/{fit['n_good_subbands'] + len(fit['subbands_dropped'])} "
        f"subbands, coarse $\\Delta$DM {fit['coarse_dm'] - r['dm_ref']:+.2f} "
        f"(peak S/N {fit['peak_snr']:.0f})\n{stat}",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_method_contact_sheet(results, name, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = sorted((r for r in results if r["adapter"] == name),
                 key=lambda r: (r["telescope"], r["burst"]))
    fig, axes = plt.subplots(4, 6, figsize=(20, 11))
    for ax, r in zip(axes.ravel(), sel, strict=False):
        x = r["curve"].get("residual_dm")
        ykey = next((k for k in r["curve"] if k != "residual_dm"), None)
        if x is not None and ykey:
            ax.plot(x, r["curve"][ykey], lw=0.9)
        elif x is not None:
            ax.set_xlim(min(x), max(x))
            ax.set_yticks([])
            ax.text(0.5, 0.5, "package exposes\nno search curve",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color="0.5")
        if r.get("residual") is not None:
            ax.axvline(r["residual"], color="r", lw=0.8)
            note = f"{r['residual']:+.2f} $\\pm$ {r['sigma']:.2f}"
        else:
            note = f"unconstrained\n{r['reason'][:38]}"
        ax.set_title(f"{r['burst']}/{r['telescope']}", fontsize=9)
        ax.text(0.02, 0.95, note, transform=ax.transAxes, va="top", fontsize=8,
                color="r" if r.get("residual") is None else "k")
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(sel):]:
        ax.set_axis_off()
    fig.suptitle(f"{name}: residual-DM search on all 24 products "
                 "(red line = candidate residual vs file-stem reference DM)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_memo(results, cfg, path):
    """Per-product residuals for every adapter -- descriptive only, no
    promotion language (the primary-method choice is the owner's Phase-3
    adjudication, uniform across the sample)."""
    adapters = cfg["adapters"]
    keys = sorted({(r["burst"], r["telescope"]) for r in results})
    by = {(r["burst"], r["telescope"], r["adapter"]): r for r in results}
    lines = [
        "# DM battery memo (Phase 1)", "",
        "Residual DM vs the product file-stem reference DM, per adapter, on all",
        "24 co-detection products. One uniform config "
        f"(windows: {cfg['windows']}, {cfg['max_channels']} ch, "
        f"max {cfg['max_time']} samples). Entries: residual +- sigma [pc/cm3],",
        "or the recorded reason when unconstrained. Descriptive only -- no",
        "method is promoted here.", "",
        "| product | " + " | ".join(adapters) + " |",
        "|---| " + " | ".join("---" for _ in adapters) + " |",
    ]
    for burst, tel in keys:
        cells = []
        for a in adapters:
            r = by.get((burst, tel, a))
            if r is None:
                cells.append("--")
            elif r["dm"] is None:
                cells.append(f"unconstrained ({r['reason'][:40]})")
            else:
                cells.append(f"{r['residual']:+.3f} +- {r['sigma']:.3f}")
        lines.append(f"| {burst}/{tel} | " + " | ".join(cells) + " |")
    import numpy as _np

    lines += ["", "## Cross-method scatter (std of constrained residuals per product)", ""]
    for burst, tel in keys:
        res = [by[(burst, tel, a)]["residual"] for a in adapters
               if (burst, tel, a) in by and by[(burst, tel, a)]["residual"] is not None]
        if len(res) >= 2:
            lines.append(f"- {burst}/{tel}: {_np.std(res):.3f} (n={len(res)})")
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=_CFG_PATH)
    ap.add_argument("--replot", action="store_true",
                    help="re-render figures from existing battery_results.json")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    rows = load_manifest_rows(root)

    if args.replot:
        results = json.loads((out_dir / "battery_results.json").read_text())
    else:
        t0 = time.perf_counter()
        results = []
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
            for i, recs in enumerate(pool.map(run_product, [(r, cfg) for r in rows])):
                results.extend(recs)
                print(f"  {i + 1}/{len(rows)} products ({time.perf_counter() - t0:.0f}s)")
        (out_dir / "battery_results.json").write_text(json.dumps(results, indent=1))

    panel_dir = out_dir / "run_panels"
    panel_dir.mkdir(exist_ok=True)
    for row in rows:
        plot_run_panels(row, cfg, results, panel_dir)
    for r in results:
        if r.get("fit"):
            plot_regression_panel(
                r, panel_dir / f"{r['burst']}_{r['telescope']}_{r['adapter']}_fit.png")
    for name in cfg["adapters"]:
        plot_method_contact_sheet(results, name, out_dir / f"contact_sheet_{name}.png")
    write_memo(results, cfg, out_dir / "memo.md")
    n_ok = sum(r["dm"] is not None for r in results)
    print(f"battery: {n_ok}/{len(results)} runs constrained; wrote {out_dir}")


if __name__ == "__main__":
    main()
