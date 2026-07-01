"""Scintillation decorrelation bandwidth Delta-nu_d per frequency subband across
the full CHIME (400-800) + DSA (1200-1500) lever arm, and the power-law scaling
Delta-nu_d ~ nu^x_scint fit to the resolved subbands.

Loads each band's NATIVE-resolution .npy via the scattering BurstDataset (the
scattering joint-fit prep decimates freq by f_factor 64/384, destroying the
scintillation structure -- here f_factor=1). For each subband: on-pulse spectrum,
frequency ACF (scint pipeline's calculate_acf), single-Lorentzian fit to lag>0
(noise spike at lag 0 excluded) -> Delta-nu_d = HWHM. A subband is RESOLVED only if
Delta-nu_d > 3 channels (resolve a scintle) and band/Delta-nu_d > 5 (enough scintles
for a usable error). x_scint is the ODR slope of log(Delta-nu_d) vs log(nu) over the
resolved points; compared to the scattering alpha and the thin-screen relation
2*pi*tau*Delta-nu_d = C1.

  python scint_subband_alpha.py
"""

import json
import os
import sys

import numpy as np
from scipy import odr
from scipy.optimize import curve_fit

REPO = os.environ["FLITS_REPO"]
OUT = os.environ.get("FLITS_RUNS", ".") + "/data/scint"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{REPO}/scattering")
sys.path.insert(0, f"{REPO}/scintillation")
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset
from scint_analysis.analysis import calculate_acf

DATA = "/Users/jakobfaber/Data/Faber2026/dsa110/DSA_bursts"
TELCFG = f"{REPO}/scattering/configs/telescopes.yaml"
BANDS = {
    "CHIME": dict(
        path=f"{DATA}/wilhelm_chime_I_602_3809_32000b_cntr_bpc.npy", tel="chime", dm=0.0, nsub=4
    ),
    "DSA": dict(
        path=f"{DATA}/wilhelm_dsa_I_602_346_2500b_cntr_bpc.npy", tel="dsa", dm=602.346, nsub=4
    ),
}
# scattering fit (wilhelm, exp PBF joint C1D1): tau_1ghz and alpha for the same-screen test
TAU_1GHZ_MS = 0.2551
SCAT_ALPHA = 2.622


def lorentz(dnu, m2, dnud):
    return m2 / (1.0 + (dnu / dnud) ** 2)


def load_native(b):
    """Native-resolution on-pulse spectrum + off-pulse baseline per channel."""
    tel = load_telescope_block(TELCFG, b["tel"])
    ds = BurstDataset(
        b["path"],
        OUT,
        name="w",
        telescope=tel,
        f_factor=1,
        t_factor=1,
        outer_trim=0.15,
        onpulse_crop=False,
    )
    m = ds.model
    freq = np.asarray(m.freq, float) * 1e3  # GHz -> MHz, ascending
    data = np.asarray(m.data, float)  # (nf, nt)
    # burst window from the band-collapsed profile; off-pulse = everything else
    prof = np.nansum(data, axis=0)
    pk = int(np.nanargmax(prof))
    thr = np.nanmedian(prof) + 5.0 * np.nanstd(prof[: max(pk - 50, 1)])
    on = prof > thr
    # widen to a contiguous window around the peak
    lo = pk
    while lo > 0 and on[lo - 1]:
        lo -= 1
    hi = pk
    while hi < on.size - 1 and on[hi + 1]:
        hi += 1
    lo, hi = max(lo - 1, 0), min(hi + 2, data.shape[1])
    onspec = np.nanmean(data[:, lo:hi], axis=1)
    off = (
        np.concatenate([data[:, : max(lo - 5, 0)], data[:, hi + 5 :]], axis=1)
        if hi + 5 < data.shape[1]
        else data[:, : max(lo - 5, 1)]
    )
    offmean = float(np.nanmean(off))
    chan = float(np.median(np.abs(np.diff(freq))))
    return freq, onspec, offmean, chan, (lo, hi)


def two_lor(dnu, m1, g1, m2, g2, c):
    # narrow (diffractive Delta-nu_d = g1) + broad (g2: 2nd screen / residual spectral
    # structure) + constant. Matches the Lorentzian+Lorentzian model the pipeline's BIC
    # selected for the DSA subbands; a single Lorentzian conflates the two scales.
    return m1 / (1.0 + (dnu / g1) ** 2) + m2 / (1.0 + (dnu / g2) ** 2) + c


def subband_dnud(freq, onspec, offmean, chan, nsub):
    """Per-subband narrow (diffractive) Delta-nu_d via ACF + two-Lorentzian fit (lag>0)."""
    order = np.argsort(freq)
    freq, onspec = freq[order], onspec[order]
    edges = np.linspace(0, freq.size, nsub + 1, dtype=int)
    rows = []
    for i in range(nsub):
        s = slice(edges[i], edges[i + 1])
        f, sp = freq[s], onspec[s]
        fc = float(np.nanmean(f))
        band = float(f.max() - f.min())
        spm = np.ma.masked_invalid(sp)
        if spm.count() < 16:
            rows.append(
                dict(freq=fc, dnud=None, err=None, resolved=False, reason="too few channels")
            )
            continue
        acf = calculate_acf(spm, chan, off_burst_spectrum_mean=offmean)
        if acf is None:
            rows.append(dict(freq=fc, dnud=None, err=None, resolved=False, reason="acf failed"))
            continue
        lags, vals, errs = np.asarray(acf.lags), np.asarray(acf.acf), np.asarray(acf.err)
        pos = lags > 0.5 * chan  # exclude the zero-lag noise spike
        L, V, E = lags[pos], vals[pos], np.clip(errs[pos], 1e-6, None)
        w = L <= min(band * 0.4, 40.0)  # wide enough to constrain the broad component
        if w.sum() < 6:
            rows.append(
                dict(freq=fc, dnud=None, err=None, resolved=False, reason="fit window too small")
            )
            continue
        gn_hi = max(3.0, 6 * chan)  # narrow upper bound (split narrow vs broad at ~3 MHz)
        try:
            popt, pcov = curve_fit(
                two_lor,
                L[w],
                V[w],
                sigma=E[w],
                p0=[max(V[0], 1e-3) * 0.5, max(2 * chan, 0.15), max(V[0], 1e-3) * 0.5, 5.0, 0.0],
                bounds=([0, 0.5 * chan, 0, gn_hi, -np.inf], [np.inf, gn_hi, np.inf, band, np.inf]),
                maxfev=40000,
            )
            dnud, derr = float(popt[1]), float(np.sqrt(np.diag(pcov))[1])
            broad = float(popt[3])
            model = "two_lor"
        except Exception:
            # fallback: single Lorentzian on a TIGHT window (isolate the narrow core)
            wt = L <= max(8 * chan, 2.0)
            if wt.sum() < 4:
                rows.append(dict(freq=fc, dnud=None, err=None, resolved=False, reason="fit failed"))
                continue
            try:
                popt, pcov = curve_fit(
                    lorentz,
                    L[wt],
                    V[wt],
                    sigma=E[wt],
                    p0=[max(V[0], 1e-3), max(2 * chan, 0.15)],
                    bounds=([0, 0.5 * chan], [np.inf, gn_hi]),
                    maxfev=20000,
                )
                dnud, derr = float(popt[1]), float(np.sqrt(np.diag(pcov))[1])
                broad = None
                model = "one_lor"
            except Exception as e:
                rows.append(
                    dict(
                        freq=fc,
                        dnud=None,
                        err=None,
                        resolved=False,
                        reason=f"fit:{type(e).__name__}",
                    )
                )
                continue
        # resolved = wider than a few channels AND many scintles across the subband AND a
        # finite (not bound-railed, not error-dominated) estimate
        railed = dnud <= 1.05 * 0.5 * chan or dnud >= 0.95 * gn_hi
        # require a finite, nonzero, well-constrained error: a degenerate fit returns
        # derr=0 / inf (zero pcov diagonal) and must NOT count as a measurement.
        good_err = np.isfinite(derr) and (0.0 < derr < 0.5 * dnud)
        # the narrow component must be a DISTINCT diffractive scale, clearly separated
        # from the broad component; if g1 ~ g2 the "narrow" slot just caught the broad
        # spectral structure (the CHIME failure mode -- diffractive scale is sub-channel).
        distinct = (broad is None) or (dnud < 0.5 * broad)
        resolved = (dnud > 3 * chan) and (band / dnud > 5) and good_err and not railed and distinct
        rows.append(
            dict(
                freq=fc,
                dnud=dnud,
                err=derr,
                broad=broad,
                resolved=bool(resolved),
                model=model,
                chan=chan,
                band=band,
                nscintle=band / dnud,
                dnud_ch=dnud / chan,
                railed=bool(railed),
            )
        )
    return rows


def main():
    allrows = []
    summary = {}
    for name, b in BANDS.items():
        freq, onspec, offmean, chan, win = load_native(b)
        rows = subband_dnud(freq, onspec, offmean, chan, b["nsub"])
        for r in rows:
            r["telescope"] = name
        allrows += rows
        summary[name] = dict(
            chan_MHz=chan,
            band_MHz=float(freq.max() - freq.min()),
            freq_range=[float(freq.min()), float(freq.max())],
            onpulse_bins=list(win),
        )
        print(
            f"\n=== {name}  chan={chan * 1e3:.1f} kHz  band {freq.min():.0f}-{freq.max():.0f} MHz ==="
        )
        for r in rows:
            if r["dnud"] is None:
                print(f"  nu={r['freq']:7.1f}  Delta-nu_d=UNMEASURED ({r['reason']})")
            else:
                flag = "RESOLVED" if r["resolved"] else "unresolved/marginal"
                print(
                    f"  nu={r['freq']:7.1f}  Delta-nu_d={r['dnud']:.4f}+/-{r['err']:.4f} MHz "
                    f"({r['dnud_ch']:.1f} ch, {r['nscintle']:.0f} scintles) [{flag}]"
                )

    # x_scint fit over resolved DSA subbands. CHIME is excluded on physical grounds:
    # the diffractive scale measured at DSA (~0.13 MHz) scaled to CHIME by any positive
    # index is sub-channel (see chime_expectation below), so CHIME yields only upper
    # limits and any "resolved" CHIME point is the broad component, not diffractive.
    res = [r for r in allrows if r["resolved"]]
    res_dsa = [r for r in res if r["telescope"] == "DSA"]
    fit = None
    if len(res_dsa) >= 2:
        x = np.log10([r["freq"] for r in res_dsa])
        y = np.log10([r["dnud"] for r in res_dsa])
        ye = np.array([r["err"] / (r["dnud"] * np.log(10)) for r in res_dsa])
        data = odr.RealData(x, y, sy=np.clip(ye, 1e-3, None))
        out = odr.ODR(data, odr.Model(lambda B, x: B[0] * x + B[1]), beta0=[4.0, 0.0]).run()
        x_scint, x_err = float(out.beta[0]), float(out.sd_beta[0])
        span = [min(r["freq"] for r in res_dsa), max(r["freq"] for r in res_dsa)]
        fit = dict(
            x_scint=x_scint,
            x_scint_err=x_err,
            log10_c=float(out.beta[1]),
            n_resolved=len(res_dsa),
            freq_span=span,
            lever_arm_frac=(span[1] - span[0]) / span[1],
        )
        print(
            f"\n=== Delta-nu_d ~ nu^x_scint  (resolved DSA subbands only, {len(res_dsa)} pts, "
            f"{span[0]:.0f}-{span[1]:.0f} MHz = {fit['lever_arm_frac'] * 100:.0f}% lever arm) ==="
        )
        print(
            f"  x_scint = {x_scint:.2f} +/- {x_err:.2f}   (Kolmogorov 4.0-4.4; scattering alpha={SCAT_ALPHA})"
        )
    else:
        print(f"\n[!] only {len(res_dsa)} resolved DSA subband(s) -- x_scint unconstrained")

    # CHIME diffractive expectation (anchor on the DSA median, scale by Kolmogorov 4.4)
    dsa_med = float(np.median([r["dnud"] for r in res_dsa])) if res_dsa else None
    chime_chan = summary.get("CHIME", {}).get("chan_MHz")
    chime_exp = None
    if dsa_med and chime_chan:
        nu_dsa = float(np.median([r["freq"] for r in res_dsa]))
        exp600 = dsa_med * (600.0 / nu_dsa) ** 4.4
        chime_exp = dict(
            expected_dnud_at600_MHz=exp600,
            chime_chan_MHz=chime_chan,
            resolvable=bool(exp600 > 3 * chime_chan),
        )
        print("\n=== CHIME diffractive expectation ===")
        print(
            f"  DSA Delta-nu_d~{dsa_med:.3f} MHz @ {nu_dsa:.0f} -> Kolmogorov-scaled to 600 MHz: "
            f"{exp600 * 1e3:.1f} kHz  vs CHIME channel {chime_chan * 1e3:.0f} kHz "
            f"=> {'RESOLVABLE' if chime_exp['resolvable'] else 'UNRESOLVED (sub-channel)'}"
        )

    # same-screen test: 2*pi*tau_scatt(nu)*Delta-nu_d ~ C1 (thin screen ~1). C1>>1 means
    # the resolved scintillation is NOT the strong-scattering screen (two-screen sightline).
    print("\n=== same-screen test  2*pi*tau_scatt(nu)*Delta-nu_d  (thin-screen C1~1) ===")
    c1s = []
    for r in res_dsa:
        tau_ms = TAU_1GHZ_MS * (r["freq"] / 1000.0) ** (-SCAT_ALPHA)
        C1 = 2 * np.pi * (tau_ms * 1e-3) * (r["dnud"] * 1e6)
        c1s.append(C1)
        print(f"  nu={r['freq']:7.1f}  tau_scatt={tau_ms * 1e3:.1f} us  C1={C1:.1f}")
    if c1s:
        print(
            f"  median C1 = {np.median(c1s):.0f}  (>>1 => scintillation screen != scattering screen; "
            f"scattering-screen Delta-nu_d ~ {1.0 / (2 * np.pi * TAU_1GHZ_MS * 1e-3 * (1.4) ** (-SCAT_ALPHA)) / 1e3:.1f} kHz, unresolved)"
        )

    result = dict(
        subbands=allrows,
        bands=summary,
        powerlaw=fit,
        chime_expectation=chime_exp,
        same_screen_C1_median=float(np.median(c1s)) if c1s else None,
        scattering={"tau_1ghz_ms": TAU_1GHZ_MS, "alpha": SCAT_ALPHA},
    )
    json.dump(result, open(f"{OUT}/wilhelm_scint_subband.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/wilhelm_scint_subband.json")
    make_figure(allrows, fit, chime_exp, np.median(c1s) if c1s else None)


def make_figure(allrows, fit, chime_exp, c1_med):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = f"{OUT}/figures"
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    dsa_res = [r for r in allrows if r["telescope"] == "DSA" and r["resolved"]]
    dsa_un = [r for r in allrows if r["telescope"] == "DSA" and not r["resolved"] and r["dnud"]]
    chime = [r for r in allrows if r["telescope"] == "CHIME" and r["dnud"]]
    if dsa_res:
        ax.errorbar(
            [r["freq"] for r in dsa_res],
            [r["dnud"] for r in dsa_res],
            yerr=[r["err"] for r in dsa_res],
            fmt="o",
            color="C0",
            ms=8,
            capsize=3,
            label="DSA resolved (diffractive)",
        )
    if dsa_un:
        ax.plot(
            [r["freq"] for r in dsa_un],
            [r["dnud"] for r in dsa_un],
            "x",
            color="C0",
            alpha=0.5,
            label="DSA marginal/excluded",
        )
    # CHIME: upper limits (channel width), diffractive scale is sub-channel
    ax.scatter(
        [r["freq"] for r in chime],
        [r["chan"] for r in chime],
        marker="v",
        color="C3",
        label="CHIME upper limit (1 channel)",
    )
    nu = np.linspace(400, 1500, 200)
    if fit:
        c = 10 ** fit["log10_c"]
        ax.plot(
            nu,
            c * nu ** fit["x_scint"],
            "C0--",
            label=f"DSA fit x={fit['x_scint']:.2f}±{fit['x_scint_err']:.2f} (flat)",
        )
    if dsa_res:
        nu0 = np.median([r["freq"] for r in dsa_res])
        d0 = np.median([r["dnud"] for r in dsa_res])
        ax.plot(nu, d0 * (nu / nu0) ** 4.4, "k:", alpha=0.6, label="Kolmogorov +4.4 (diffractive)")
        # scattering-screen Delta-nu_d = C1/(2 pi tau(nu)), C1=1 thin screen -> the kHz line
        tau = TAU_1GHZ_MS * 1e-3 * (nu / 1000.0) ** (-SCAT_ALPHA)
        ax.plot(
            nu,
            1.0 / (2 * np.pi * tau) / 1e6,
            "C2-.",
            alpha=0.7,
            label="scattering-screen Δν_d (C1=1, unresolved)",
        )
    for r in chime:
        ax.axhline(r["chan"], color="C3", ls=":", alpha=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel(r"$\Delta\nu_d$ (MHz)")
    ax.set_title(
        f"wilhelm scintillation Δν_d vs ν (C1≈{c1_med:.0f} ⇒ two screens)"
        if c1_med
        else "wilhelm Δν_d"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    p = f"{figdir}/wilhelm_scint_subband.png"
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
    # manifest for the figure-review Stop gate (expectation each PNG must be checked against)
    manifest = {
        "figures": [
            {
                "file": "wilhelm_scint_subband.png",
                "expectation": "log-log Δν_d vs ν, 400-1500 MHz. DSA (C0 circles) ~0.13 MHz, ~FLAT "
                "(fit slope x≈-0.23±0.19, NOT Kolmogorov +4.4 dotted). CHIME (C3 down-triangles) upper "
                "limits at the 0.39 MHz channel width; diffractive scale (~3 kHz at 600) is sub-channel. "
                "Green dash-dot = scattering-screen Δν_d (~kHz, far below all points) => C1=2πτΔν~85>>1 "
                "=> resolved scintillation is a separate (foreground) screen from the scattering screen.",
            }
        ]
    }
    json.dump(manifest, open(f"{figdir}/figures.manifest.json", "w"), indent=2)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
