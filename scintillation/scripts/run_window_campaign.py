"""CHIME per-subband scintillation campaign: objective windows + variant systematics.

Per burst: validity span -> S/N profile -> window_optimize.select_windows (fallback:
pipeline defaults, flagged) -> window_refit.refit at the chosen windows AND at every
deduplicated SCAN_GRID variant. The variant spread per subband is the window systematic:
    gamma +/- sigma_fit (curve_fit) +/- sigma_win (half-range across window variants).

Outputs (one JSON per burst so bursts parallelize as processes, + a campaign JSONL):
  <out>/<name>_campaign.json      full record: windows, variants, per-subband fits
  <out>/<name>_acf_fits.png       per-subband ACF + fit at the chosen windows
  <out>/campaign_results.jsonl    one summary line per burst (append)

Usage:
  FLITS_ROOT=<repo> python run_window_campaign.py <name> [outdir]     # one burst
  FLITS_ROOT=<repo> python run_window_campaign.py all [outdir]        # serial sweep
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.environ["FLITS_ROOT"]
sys.path.insert(0, R + "/scintillation")
from scint_analysis import window_refit as wr
from scint_analysis import window_optimize as wo
from scint_analysis import freya_scintillation as fs

BURSTS = ["casey", "chromatica", "freya", "hamilton", "isha", "johndoeII",
          "mahi", "oran", "phineas", "whitney", "wilhelm", "zach"]
OFF_SNR_FLAG = 3.0        # off-window matched response above this = contaminated de-scallop


def _windows_for(name):
    """(chosen, variants, source, off_snr, span) — objective rule with flagged fallback."""
    c = wr._base_config(name)
    an = c.setdefault("analysis", {})
    an.setdefault("bandpass_normalization", {})["enable"] = False
    an.setdefault("baseline_subtraction", {})["enable"] = False
    spec, bl_def, ol_def = fs.prepare_spectrum_from_config(c)
    span = wo.valid_span(spec.power) or (0, spec.power.shape[1])
    t0 = span[0]
    prof = wo.snr_profile(spec.power[:, t0:span[1]])
    sel = wo.select_windows(prof)
    if sel is None:
        default = dict(burst_lims=[int(bl_def[0]), int(bl_def[1])],
                       off_lims=[int(ol_def[0]), int(ol_def[1])] if ol_def else None)
        return default, [], "pipeline-default-fallback", None, span
    shift = lambda w: [int(w[0] + t0), int(w[1] + t0)]
    # matched CORE is the primary ACF window (tail inclusion inflates gamma 2-4x —
    # see window_optimize.select_windows); the tail-expanded window rides along as an
    # explicit variant so the systematic stays measured, not hidden
    chosen = dict(burst_lims=shift(sel["burst_core"]), off_lims=shift(sel["off_lims"]))
    variants = [dict(burst_lims=shift(v["burst_core"]), off_lims=shift(v["off_lims"]))
                for v in wo.window_variants(prof)]
    variants.append(dict(burst_lims=shift(sel["burst_lims"]), off_lims=shift(sel["off_lims"]),
                         label="tail-expanded"))
    return chosen, variants, "objective", float(sel["off_snr"]), span


def _fit_table(r):
    """Compact per-subband rows from a window_refit.refit result."""
    rows = []
    for i in r["order"]:
        f = r["fits"][int(i)]
        row = dict(center_mhz=round(float(r["center_freqs"][i]), 1), ok=bool(f.get("ok")))
        if f.get("ok"):
            row.update(gamma=f["gamma"], gamma_err=f["gamma_err"], m=f["m"],
                       amp_snr=f["amp_snr"], resolved=bool(f["resolved"]))
        else:
            row["reason"] = f.get("reason")
        rows.append(row)
    return rows


def run_burst(name, out):
    chosen, variants, source, off_snr, span = _windows_for(name)
    if chosen["off_lims"] is None:
        raise SystemExit(f"{name}: no off window available (source={source})")
    r0 = wr.refit(name, chosen["burst_lims"], chosen["off_lims"], [])
    var_tables = []
    for v in variants:
        key = (tuple(v["burst_lims"]), tuple(v["off_lims"]))
        if key == (tuple(chosen["burst_lims"]), tuple(chosen["off_lims"])):
            continue
        rv = wr.refit(name, v["burst_lims"], v["off_lims"], [])
        var_tables.append(dict(windows=v, fits=_fit_table(rv)))

    # window systematic per subband: half-range of gamma across chosen + CORE variants,
    # matched by subband rank (equal-S/N subbanding keeps ranks comparable). The
    # tail-expanded variant is a known ~2-4x biased configuration — report its gamma
    # separately (gamma_tail) instead of letting it blow up sigma_win.
    base = _fit_table(r0)
    for k, row in enumerate(base):
        gs = [row.get("gamma")]
        for t in var_tables:
            if k < len(t["fits"]) and t["fits"][k].get("ok"):
                if t["windows"].get("label") == "tail-expanded":
                    row["gamma_tail"] = t["fits"][k].get("gamma")
                else:
                    gs.append(t["fits"][k].get("gamma"))
        gs = [g for g in gs if g is not None]
        row["gamma_win_sys"] = float((max(gs) - min(gs)) / 2) if len(gs) > 1 else None
        row["n_variants"] = len(gs)

    # per-subband ACF figure at the chosen windows (visual vetting is the accept gate)
    order = r0["order"]
    fig, axes = plt.subplots(1, len(order), figsize=(3.4 * len(order), 3.4))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, order):
        f = r0["fits"][int(i)]
        if f.get("ok"):
            ax.plot(f["lp"], f["ap"], color="#c0392b", lw=0.9)
            col = "#1e8449" if f["resolved"] else "#7f8c8d"
            lab = (f"γ={f['gamma']:.3f}±{f['gamma_err']:.3f}\n"
                   f"m={f['m']:.2f} snr={f['amp_snr']:.1f}"
                   + ("" if base[list(order).index(i)]["gamma_win_sys"] is None else
                      f"\nσ_win={base[list(order).index(i)]['gamma_win_sys']:.3f}"))
            ax.plot(f["lp"], f["model"], color=col, lw=1.8, ls="--", label=lab)
            ax.legend(fontsize=6.5, loc="upper right")
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_title(f"{r0['center_freqs'][i]:.0f} MHz", fontsize=9)
        ax.set_xlabel("lag (MHz)")
    axes[0].set_ylabel("ACF (norm)")
    flagtxt = "" if (off_snr is None or off_snr <= OFF_SNR_FLAG) else \
        f"  [OFF CONTAMINATED off_snr={off_snr:.1f}]"
    fig.suptitle(f"{name}: per-subband fits, {source} windows "
                 f"{chosen['burst_lims']}/{chosen['off_lims']}, "
                 f"{len(var_tables)} variants{flagtxt}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/{name}_acf_fits.png", dpi=125, bbox_inches="tight")
    plt.close(fig)

    rec = dict(name=name, window_source=source, off_snr=off_snr, valid_span=list(span),
               windows=chosen, alpha=r0["alpha"], subbands=base,
               variants=var_tables, rfi_new=r0["rfi_new"], method=r0["method"])
    with open(f"{out}/{name}_campaign.json", "w") as fh:
        json.dump(rec, fh, indent=2, default=float)
    with open(f"{out}/campaign_results.jsonl", "a") as fh:
        slim = {k: rec[k] for k in ("name", "window_source", "off_snr", "windows", "alpha")}
        slim["subbands"] = base
        fh.write(json.dumps(slim, default=float) + "\n")
    nres = sum(1 for b in base if b.get("resolved"))
    print(f"{name}: {nres}/{len(base)} resolved, source={source}, "
          f"off_snr={off_snr if off_snr is None else round(off_snr, 1)}, "
          f"variants={len(var_tables)}")
    return rec


if __name__ == "__main__":
    target = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/Developer/scratch/window_campaign")
    os.makedirs(out, exist_ok=True)
    for n in (BURSTS if target == "all" else [target]):
        run_burst(n, out)
