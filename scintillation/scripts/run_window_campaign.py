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
# "<name>_hi" runs the same uniform rule on the _hi product (600-800 MHz,
# finer channels and/or time — the narrow-gamma reach is set by channel width,
# so bursts whose standard-product gamma sits near 2x24.4 kHz need these);
# "all_hi" sweeps every burst's _hi product.
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
    # PRIMARY = boxcar over the matched-significance CORE window (unweighted).
    # Decided by the committed injection suite (analysis/.../inject_discriminate.py):
    # when the scattering TAIL carries broad spectral structure distinct from the narrow
    # core scintle (the real chromatica/zach case), tail-expanded inflates recovered
    # gamma 2.6-3.6x and matched-weight-tail still inflates 1.3-1.9x, while the CORE
    # window recovers gamma unbiased (same modest offset as clean data). On clean data
    # all three agree to ~2%, so the core choice costs nothing where the tail is benign
    # and protects the measurement where it is not. The matched-weight tail-expanded
    # estimator (former primary) and the bare tail-expanded boxcar are kept as variants
    # so the systematic remains visible.
    weights = None
    if sel.get("weights") is not None:
        weights = np.zeros(spec.power.shape[1])
        weights[t0:t0 + sel["weights"].size] = sel["weights"]
    chosen = dict(burst_lims=shift(sel["burst_core"]), off_lims=shift(sel["off_lims"]),
                  weights=None, estimator="core-boxcar")
    # Variants measure the selection-rule + estimator systematic: SCAN_GRID cores, the
    # matched-weight tail estimator (former primary), and the bare tail-expanded boxcar.
    variants = [dict(burst_lims=shift(v["burst_core"]), off_lims=shift(v["off_lims"]))
                for v in wo.window_variants(prof)]
    variants.append(dict(burst_lims=shift(sel["burst_lims"]), off_lims=shift(sel["off_lims"]),
                         weights=weights, label="matched-weight-tail"))
    variants.append(dict(burst_lims=shift(sel["burst_lims"]), off_lims=shift(sel["off_lims"]),
                         label="tail-expanded"))
    return chosen, variants, sel.get("off_source", "objective"), float(sel["off_snr"]), span


def _fit_table(r):
    """Compact per-subband rows from a window_refit.refit result."""
    rows = []
    for i in r["order"]:
        f = r["fits"][int(i)]
        row = dict(center_mhz=round(float(r["center_freqs"][i]), 1), ok=bool(f.get("ok")))
        if f.get("ok"):
            row.update(gamma=f["gamma"], gamma_err=f["gamma_err"], m=f["m"],
                       amp_snr=f["amp_snr"], resolved=bool(f["resolved"]),
                       shape_ok=bool(f.get("shape_ok", True)),
                       dbic_line=f.get("dbic_line"),
                       gamma_scintle_err=f.get("gamma_scintle_err"),
                       subband_bw_mhz=f.get("subband_bw_mhz"),
                       model_sel=f.get("model_sel", "1L"),
                       gamma_b=f.get("gamma_b"), A_b=f.get("A_b"),
                       dbic_2comp=f.get("dbic_2comp"))
        else:
            row["reason"] = f.get("reason")
        rows.append(row)
    return rows


N_DETECT = 3              # >=3 resolved physical subbands => real slope (n=2 is 0-dof)

def _derive_status(base, alpha, off_null):
    """Earned per-burst status from stated rules, not hand-typed (B3).

    detection      : n_resolved_physical >= N_DETECT AND alpha physical
                     (window_refit.alpha_is_physical: ALPHA_BOUNDS = (1.5, 6.0), open)
                     AND the per-subband off-pulse null does not FAIL in a majority of the
                     fitted subbands.
    marginal       : 2 resolved physical subbands with physical alpha (0-dof slope,
                     provisional) — a candidate that needs more bands or a second epoch.
    diagnostic_only: resolved subbands exist but the slope is unphysical, or the off-pulse
                     null FAILS in a majority of subbands (scale is instrumental).
    non_detection  : fewer than 2 resolved physical subbands.

    off_null is window_refit's PER-SUBBAND off-pulse null (off_pulse_null=True): the
    reviewer-approved (R5) control that tiles the off-pulse window into on-duration noise
    slices and refits the SAME subbands that enter the alpha fit. A subband "fails" when
    its off-pulse noise reproduces the on-pulse gamma (null_pass is False); we require the
    failure to be a MAJORITY of judged subbands before it overrides a physical multi-band
    slope, so one noisy off-slice cannot demote a clean ladder.
    """
    res_phys = [b for b in base if b.get("resolved") and (b.get("m", 9) <= wr.M_PHYS)]
    n = len(res_phys)
    a = (alpha or {}).get("alpha")
    a_ok = wr.alpha_is_physical(alpha)
    fails = sum(1 for v in (off_null or {}).values()
                if isinstance(v, dict) and v.get("null_pass") is False)
    judged = sum(1 for v in (off_null or {}).values()
                 if isinstance(v, dict) and v.get("null_pass") is not None)
    null_majority_fail = judged > 0 and fails > judged / 2
    if n >= N_DETECT and a_ok and not null_majority_fail:
        return "detection", (f"n_resolved_physical={n}>={N_DETECT}, alpha={a:+.2f} in "
                             f"({wr.ALPHA_BOUNDS[0]},{wr.ALPHA_BOUNDS[1]}) (open), "
                             f"off_pulse_null not majority-fail ({fails}/{judged})")
    if n >= 2 and a_ok and not null_majority_fail:
        return "marginal", (f"n_resolved_physical={n} (>=2, <{N_DETECT}: 0-dof slope), "
                            f"alpha={a:+.2f} physical, provisional")
    if n >= 1 and (a is not None) and not a_ok:
        return "diagnostic_only", (f"n_resolved_physical={n} but alpha="
                                   f"{a:+.2f} outside ({wr.ALPHA_BOUNDS[0]},"
                                   f"{wr.ALPHA_BOUNDS[1]}) (unphysical slope)")
    if null_majority_fail:
        return "diagnostic_only", (f"off_pulse_null FAILED in {fails}/{judged} subbands "
                                   "(scale reproduced by off-pulse noise -> instrumental)")
    return "non_detection", f"n_resolved_physical={n} (<2: no measurable slope)"


def run_burst(name, out):
    chosen, variants, source, off_snr, span = _windows_for(name)
    if chosen["off_lims"] is None:
        raise SystemExit(f"{name}: no off window available (source={source})")
    # validate_artifacts=True records the committed reference-subband controls (B4 summary);
    # off_pulse_null=True runs the reviewer-approved (R5) per-subband off-pulse null on
    # exactly the subbands that enter the alpha fit. The per-subband null is the primary
    # verdict for science_status; artifact_controls is retained as the recorded summary.
    r0 = wr.refit(name, chosen["burst_lims"], chosen["off_lims"], [],
                  time_weights=chosen.get("weights"), validate_artifacts=True,
                  off_pulse_null=True)
    seen, var_tables = set(), []
    for v in variants:
        key = (tuple(v["burst_lims"]), tuple(v["off_lims"]), v.get("weights") is not None)
        if key in seen:            # same windows+weighting already fit
            continue
        seen.add(key)
        rv = wr.refit(name, v["burst_lims"], v["off_lims"], [],
                      time_weights=v.get("weights"))
        # Do not serialize the raw per-time weight array; keep only its provenance.
        vw = {k2: v2 for k2, v2 in v.items() if k2 != "weights"}
        if v.get("weights") is not None:
            vw["weighted"] = True
        var_tables.append(dict(windows=vw, fits=_fit_table(rv)))

    # window systematic per subband: half-range of gamma across the matched primary +
    # boxcar CORE variants, matched by subband rank (equal-S/N subbanding keeps ranks
    # comparable). The tail-expanded boxcar is a known-biased configuration (never
    # resolves under scattering in injection) — report its gamma separately
    # (gamma_tail) instead of letting it blow up sigma_win.
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
        # Physicality flags (both are envelope-contamination signatures, not fit
        # failures: m>1 cannot arise from point-source scintillation, and the
        # injection harness shows a smooth envelope both inflates m and fakes
        # resolved fits). Flag, do not delete — the owner vets by eye.
        if row.get("resolved") and row.get("m", 0) > 1.2:
            row["flag_m_unphysical"] = True

    # per-subband ACF figure at the chosen windows (visual vetting is the accept gate)
    order = r0["order"]
    fig, axes = plt.subplots(1, len(order), figsize=(3.4 * len(order), 3.4))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, order):
        f = r0["fits"][int(i)]
        if f.get("ok"):
            ax.plot(f["lp"], f["ap"], color="#c0392b", lw=0.9)
            col = "#1e8449" if f["resolved"] else "#7f8c8d"
            lab = (f"γ={f['gamma']:.3f}±{f['gamma_err']:.3f} [{f.get('model_sel','1L')}]"
                   + (f" γb={f['gamma_b']:.2f}" if f.get("gamma_b") else "") + "\n"
                   f"m={f['m']:.2f} snr={f['amp_snr']:.1f}"
                   + ("" if base[list(order).index(i)]["gamma_win_sys"] is None else
                      f"\nσ_win={base[list(order).index(i)]['gamma_win_sys']:.3f}"))
            ax.plot(f["lp"], f["model"], color=col, lw=1.8, ls="--", label=lab)
            ax.legend(fontsize=6.5, loc="upper right")
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_title(f"{r0['center_freqs'][i]:.0f} MHz", fontsize=9)
        ax.set_xlabel("lag (MHz)")
    axes[0].set_ylabel("ACF (norm)")
    # Pre-burst off-windows deliberately bypass the off_snr purity gate (the post-burst
    # scattering tail is what off_snr flags; a pre-burst run cannot contain the tail), so
    # do NOT brand them "contaminated" — off_snr is not a purity verdict for them (M1).
    if isinstance(source, str) and source.startswith("pre-burst"):
        flagtxt = "" if off_snr is None else f"  [pre-burst off, off_snr={off_snr:.1f} — purity gate n/a]"
    else:
        flagtxt = "" if (off_snr is None or off_snr <= OFF_SNR_FLAG) else \
            f"  [OFF CONTAMINATED off_snr={off_snr:.1f}]"
    alpha_flag = ""
    if r0["alpha"] and r0["alpha"]["alpha"] < 0:
        # gamma falling with frequency is backwards for scintillation (expect ~nu^+4):
        # an envelope/scattering-contamination signature at the sample level
        alpha_flag = f"  [ALPHA<0: {r0['alpha']['alpha']:+.2f}]"
    est = chosen.get("estimator", "boxcar")
    fig.suptitle(f"{name}: per-subband fits ({est}), {source} windows "
                 f"{chosen['burst_lims']}/{chosen['off_lims']}, "
                 f"{len(var_tables)} variants{flagtxt}{alpha_flag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}/{name}_acf_fits.png", dpi=125, bbox_inches="tight")
    plt.close(fig)

    # weights are provenance, not payload: store the nonzero span compactly
    wjson = None
    if chosen.get("weights") is not None:
        nz = np.flatnonzero(chosen["weights"])
        wjson = dict(t0=int(nz[0]), t1=int(nz[-1]) + 1,
                     values=[round(float(x), 6) for x in
                             chosen["weights"][nz[0]:nz[-1] + 1]])
    win_rec = dict(burst_lims=chosen["burst_lims"], off_lims=chosen["off_lims"],
                   estimator=chosen.get("estimator", "boxcar"), weights=wjson)
    # earned, machine-readable status (B3): the per-subband off-pulse null (R5) is the
    # primary null verdict; the reference-subband artifact_controls (B4) is recorded too.
    artifact_controls = r0.get("artifact_controls")
    science_status, status_basis = _derive_status(base, r0["alpha"], r0.get("off_pulse_null"))
    rec = dict(name=name, window_source=source, off_snr=off_snr, valid_span=list(span),
               windows=win_rec, alpha=r0["alpha"],
               alpha_unphysical=bool(r0["alpha"] and r0["alpha"]["alpha"] < 0),
               science_status=science_status, status_basis=status_basis,
               artifact_controls=artifact_controls,
               off_pulse_null={str(k): v for k, v in (r0.get("off_pulse_null") or {}).items()},
               subbands=base,
               variants=var_tables, rfi_new=r0["rfi_new"], method=r0["method"])
    with open(f"{out}/{name}_campaign.json", "w") as fh:
        json.dump(rec, fh, indent=2, default=float)
    with open(f"{out}/campaign_results.jsonl", "a") as fh:
        slim = {k: rec[k] for k in ("name", "window_source", "off_snr", "windows", "alpha",
                                    "science_status", "status_basis")}
        slim["subbands"] = base
        fh.write(json.dumps(slim, default=float) + "\n")
    nres = sum(1 for b in base if b.get("resolved"))
    nflag = sum(1 for b in base if b.get("flag_m_unphysical"))
    print(f"{name}: {nres}/{len(base)} resolved, source={source}, "
          f"est={win_rec['estimator']}, "
          f"off_snr={off_snr if off_snr is None else round(off_snr, 1)}, "
          f"variants={len(var_tables)}"
          + (f", m-flags={nflag}" if nflag else "")
          + ("  ALPHA<0" if rec["alpha_unphysical"] else ""))
    return rec


if __name__ == "__main__":
    target = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/Developer/scratch/window_campaign")
    os.makedirs(out, exist_ok=True)
    names = (BURSTS if target == "all"
             else [b + "_hi" for b in BURSTS] if target == "all_hi"
             else [target])
    # all/all_hi are sweeps: a missing product (e.g. a burst with no _hi npz) must SKIP,
    # not abort the whole sweep. Probe config availability first for the sweep targets (M5).
    is_sweep = target in ("all", "all_hi")
    for n in names:
        if is_sweep and not wr.product_available(n):
            print(f"{n}: SKIP (no product available)")
            continue
        run_burst(n, out)
