"""Preview objective window selection vs pipeline defaults for all CHIME bursts.

Fast pass (no ACF fits): per burst, validity span -> standardized S/N profile ->
window_optimize.select_windows, overlaid against the pipeline's auto windows.
Output: one PNG per burst + windows.json.

Usage: FLITS_ROOT=<repo> python preview_windows.py [outdir]
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
from scint_analysis import figure_manifest as fm

BURSTS = ["casey", "chromatica", "freya", "hamilton", "isha", "johndoeII",
          "mahi", "oran", "phineas", "whitney", "wilhelm", "zach"]
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Developer/scratch/window_preview")
os.makedirs(OUT, exist_ok=True)


def _shift(win, t0):
    return [int(win[0] + t0), int(win[1] + t0)]


summary = {}
for name in BURSTS:
    c = wr._base_config(name)
    an = c.setdefault("analysis", {})
    an.setdefault("bandpass_normalization", {})["enable"] = False
    an.setdefault("baseline_subtraction", {})["enable"] = False
    spec, bl_def, ol_def = fs.prepare_spectrum_from_config(c)

    span = wo.valid_span(spec.power)
    t0, t1 = span if span else (0, spec.power.shape[1])
    prof = wo.snr_profile(spec.power[:, t0:t1])
    sel = wo.select_windows(prof)
    variants = wo.window_variants(prof)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    t = np.arange(t0, t1)
    ax.plot(t, prof, color="0.55", lw=0.6, label="S/N profile")
    ax.axvspan(0, t0, color="k", alpha=0.08)
    ax.axvspan(t1, spec.power.shape[1], color="k", alpha=0.08, label="invalid (pad/edge)")
    if sel:
        ax.plot(t, sel["smoothed"], color="k", lw=1.0, label="matched-scale smooth")
        m = sel["matched"]
        ax.axvspan(*_shift(sel["burst_lims"], t0), color="#2e86c1", alpha=0.25,
                   label=f"objective on {_shift(sel['burst_lims'], t0)} "
                         f"(w={m['width']}, snr={m['snr']:.1f})")
        off_lab = _shift(sel["off_lims"], t0)
        ax.axvspan(*off_lab, color="0.75", alpha=0.35,
                   label=f"objective off {off_lab} (off_snr={sel['off_snr']:.1f})")
        for v in variants:
            ax.axvspan(*_shift(v["burst_lims"], t0), color="#2e86c1", alpha=0.06)
    ax.axvspan(*bl_def, facecolor="none", edgecolor="#e67e22", hatch="//",
               label=f"pipeline default on {list(bl_def)}")
    if ol_def:
        ax.axvspan(*ol_def, facecolor="none", edgecolor="0.4", hatch="\\\\",
                   label=f"pipeline default off {list(ol_def)}")
    ax.set_xlim(0, spec.power.shape[1])
    ax.set_xlabel("time bin"); ax.set_ylabel("standardized S/N per bin")
    ax.set_title(f"{name}: objective window selection ({len(variants)} scan variants)", fontsize=10)
    ax.legend(fontsize=6.5, ncol=3, loc="upper right")
    fig.tight_layout()
    filename = f"{name}_window_preview.png"
    fig.savefig(f"{OUT}/{filename}", dpi=120, bbox_inches="tight")
    plt.close(fig)
    fm.register_figure(
        OUT,
        filename,
        "The objective on/off windows cover the burst and clean baseline respectively; "
        "invalid spans and pipeline-default overlays agree with windows.json.",
        campaign="CHIME objective-window selection previews",
    )

    summary[name] = dict(
        valid_span=[t0, t1],
        objective=None if sel is None else dict(
            burst_lims=_shift(sel["burst_lims"], t0),
            off_lims=_shift(sel["off_lims"], t0),
            matched=sel["matched"], off_snr=sel["off_snr"]),
        # objective rule below threshold -> documented fallback to the pipeline's
        # per-burst default windows (uniform ladder, flagged, never silent)
        window_source="objective" if sel else "pipeline-default-fallback",
        default=dict(burst_lims=[int(bl_def[0]), int(bl_def[1])],
                     off_lims=None if ol_def is None else [int(ol_def[0]), int(ol_def[1])]),
        n_variants=len(variants),
        variant_burst_lims=[_shift(v["burst_lims"], t0) for v in variants],
        science_status="diagnostic_only",
        figure_review_status="pending",
    )
    ob = summary[name]["objective"]
    print(f"{name}: objective={None if ob is None else ob['burst_lims']} "
          f"default={list(bl_def)} variants={len(variants)} span={[t0, t1]} "
          f"source={summary[name]['window_source']}")

with open(f"{OUT}/windows.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"wrote {OUT}/windows.json")
