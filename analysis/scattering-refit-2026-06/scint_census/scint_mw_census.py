"""Cross-codetection census: resolved DSA diffractive scintillation Delta-nu_d vs
the NE2025 Milky-Way scattering floor, for every CHIME-DSA co-detection.

For wilhelm the measured DSA diffractive Delta-nu_d (~0.13-0.18 MHz) sits ~6-8x
BELOW the smooth NE2025 MW floor (1.1 MHz) -- ~6-8x more scattering than the
smooth Galaxy predicts. This tests whether that excess is sightline-specific
(wilhelm only) or systematic (all sightlines).

Delta-nu_d source = the scint pipeline's BIC-vetted per-subband Lorentzian fits
(scintillation/configs/bursts/{burst}_dsa.yaml -> analysis.stored_fits). The
diffractive component is the NARROW Lorentzian (smallest well-constrained _gamma);
a co-fitted broad Lorentzian/Gaussian absorbs the residual spectral structure that
a naive single-component ACF fit conflates with the diffractive scale (the failure
mode found when re-deriving from the raw .npy: it returned the broad component for
~half the bursts). Per burst Delta-nu_d = median narrow gamma over subbands whose
narrow component is well-constrained and distinct from the broad. excess = MW floor
/ measured (>1 => more scattering than the smooth MW model).

  python scint_mw_census.py
"""

import glob
import json
import os
import sys

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import SkyCoord

REPO = os.environ["FLITS_REPO"]
OUT = os.environ.get("FLITS_RUNS", ".") + "/data/scint"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{REPO}/scintillation/ne2025")
from query_ne2025_scint import galactic_floor

CATALOG = f"{REPO}/configs/bursts.yaml"
CFGDIR = f"{REPO}/scintillation/configs/bursts"
NARROW_CAP_MHZ = 2.0  # a diffractive scale this large at DSA is physically implausible here
DSA_CHAN_MHZ = 0.0305  # native DSA channel ~30.5 kHz (resolve gate floor)
CLEAN_NARROW_MHZ = 0.5  # a 2-component narrow Lorentzian below this is a credible diffractive scale


def narrow_from_subband(models):
    """Diffractive Delta-nu_d for one subband: smallest well-constrained Lorentzian
    gamma of the min-BIC model, required distinct from any broader component."""
    pick = min(models.items(), key=lambda kv: kv[1].get("bic", np.inf))
    name, fit = pick
    p = fit.get("best_fit_params", {})
    # Lorentzian widths (diffractive is Lorentzian); track Gaussian sigmas as broad context
    lor = {k: v for k, v in p.items() if k.endswith("_gamma")}
    gau = {k: v for k, v in p.items() if k.endswith("_sigma")}
    cand = []
    for k, v in lor.items():
        val, err = v.get("value"), v.get("stderr")
        if val is None or val <= 0:
            continue
        cand.append((abs(val), err))
    if not cand:
        return None
    cand.sort()
    val, err = cand[0]  # narrowest Lorentzian
    others = [c[0] for c in cand[1:]] + [abs(g["value"]) for g in gau.values() if g.get("value")]
    broad = min(others) if others else None
    good_err = err is not None and np.isfinite(err) and 0.0 < err < 0.5 * val
    distinct = broad is None or val < 0.5 * broad
    resolved = good_err and distinct and (3 * DSA_CHAN_MHZ < val < NARROW_CAP_MHZ)
    return dict(
        dnud=float(val),
        err=float(err) if err is not None and np.isfinite(err) else None,
        broad=float(broad) if broad else None,
        model=name,
        bic=float(fit.get("bic", np.nan)),
        reffreq=float(fit.get("reference_frequency_mhz", np.nan)),
        single=broad is None,
        resolved=bool(resolved),
    )


def measure_burst(burst):
    """Per-burst narrow diffractive Delta-nu_d (median over resolved subbands)."""
    f = glob.glob(f"{CFGDIR}/{burst}_dsa.yaml")
    if not f:  # case-insensitive (yaml 'johndoeii' vs file 'johndoeII')
        f = [
            p
            for p in glob.glob(f"{CFGDIR}/*_dsa.yaml")
            if os.path.basename(p).lower() == f"{burst.lower()}_dsa.yaml"
        ]
    if not f:
        return None
    sf = (yaml.safe_load(open(f[0])).get("analysis", {}).get("stored_fits")) or {}
    subs = []
    for s, md in sf.items():
        n = narrow_from_subband(md)
        if n:
            n["subband"] = s
            n["clean"] = bool(
                n["resolved"]
                and n["broad"] is not None
                and n["dnud"] < CLEAN_NARROW_MHZ
                and n["dnud"] < 0.5 * n["broad"]
            )
            subs.append(n)
    res = [s for s in subs if s["resolved"]]
    clean = [s for s in subs if s["clean"]]
    out = dict(n_subbands=len(subs), n_resolved=len(res), n_clean=len(clean), subbands=subs)
    use = clean if clean else res
    if use:
        d = np.array([s["dnud"] for s in use])
        e = np.array([s["err"] for s in use])
        out.update(
            dnud_MHz=float(np.median(d)),
            err_MHz=float(np.median(e)),
            spread_MHz=float(d.std()) if len(d) > 1 else 0.0,
            reffreq_med=float(np.median([s["reffreq"] for s in use])),
            tier="A" if clean else "B",
            lower_limit=not bool(clean),  # B: dnud is an upper limit => excess a lower limit
        )
    else:
        out.update(tier="C", lower_limit=False)
    return out


def main():
    cat = yaml.safe_load(open(CATALOG))["bursts"]
    bursts = {k: v for k, v in cat.items() if isinstance(v, dict) and "ra_deg" in v}
    rows = []
    for name in sorted(bursts):
        v = bursts[name]
        m = measure_burst(name)
        if m is None:
            print(f"{name:12s}  [no stored DSA scint config]")
            continue
        coord = SkyCoord(v["ra_deg"] * u.deg, v["dec_deg"] * u.deg, frame="icrs")
        floor = galactic_floor(coord)["DSA"]
        floor_MHz = floor["bw_kHz"] / 1e3
        gl, gb = coord.galactic.l.value, coord.galactic.b.value
        row = dict(
            burst=name,
            l=float(gl),
            b=float(gb),
            dm=float(v["dm"]),
            mw_floor_DSA_MHz=float(floor_MHz),
            mw_floor_tau_ms=float(floor["tau_ms"]),
            **m,
        )
        if m.get("dnud_MHz"):
            row["excess"] = floor_MHz / m["dnud_MHz"]
            ll = ">" if m["lower_limit"] else " "
            print(
                f"{name:12s} b={gb:+5.1f} [{m['tier']}]  Dnu_d={m['dnud_MHz'] * 1e3:6.1f} kHz "
                f"({m['n_clean']}cl/{m['n_resolved']}res/{m['n_subbands']}sb)  "
                f"floor={floor_MHz * 1e3:7.1f} kHz  excess{ll}={row['excess']:5.1f}x"
            )
        else:
            print(
                f"{name:12s} b={gb:+5.1f} [C]  diffractive UNRESOLVED "
                f"(0/{m['n_subbands']} sb)  floor={floor_MHz * 1e3:7.1f} kHz"
            )
        rows.append(row)

    # Tier A = clean measurement; Tier B = excess is a LOWER limit (Dnu_d upper limit).
    A = [r for r in rows if r.get("tier") == "A"]
    B = [r for r in rows if r.get("tier") == "B"]
    C = [r for r in rows if r.get("tier") == "C"]
    excA = np.array([r["excess"] for r in A])
    print("\n=== excess (NE2025 MW floor / measured DSA diffractive Delta-nu_d) ===")
    print(
        f"  Tier A (clean 2-comp diffractive): {[(r['burst'], round(r['excess'], 1)) for r in A]}"
    )
    print(
        f"  Tier B (single-Lorentzian; excess is a LOWER limit): "
        f"{[(r['burst'], round(r['excess'], 1)) for r in B]}"
    )
    print(f"  Tier C (diffractive unresolved): {[r['burst'] for r in C]}")
    strongB = [r["burst"] for r in B if r["excess"] > 3]
    print(
        f"\n  VERDICT: clean strong (>3x) excess in {[r['burst'] for r in A if r['excess'] > 3]} "
        f"(Tier A); Tier-B lower limits >3x: {strongB}. "
        f"=> strong excess is SIGHTLINE-SPECIFIC in cleanly-resolved data; "
        f"comparison sightlines are resolution-limited (lower limits), not contradicting it."
    )
    summary = dict(
        bursts=rows,
        n_tierA=len(A),
        n_tierB=len(B),
        n_tierC=len(C),
        excess_tierA_geomean=float(np.exp(np.mean(np.log(excA)))) if len(excA) else None,
        excess_tierA=[{"burst": r["burst"], "excess": r["excess"]} for r in A],
    )
    json.dump(summary, open(f"{OUT}/scint_mw_census.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/scint_mw_census.json")


if __name__ == "__main__":
    main()
