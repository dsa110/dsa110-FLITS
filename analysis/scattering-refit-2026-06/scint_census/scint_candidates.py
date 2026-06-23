"""Dump every narrow-Lorentzian diffractive CANDIDATE across ALL stored scint fits
(not just the BIC-selected model) for each co-detection, with rail / error / distinct
flags, plus the NE2025 floor. Feeds the per-burst recovery judges. The diffractive
scale is the narrowest Lorentzian/Gen-Lorentz width; a co-fitted broader component
(broad) lets us tell a true narrow scale from the only scale. The fitter's narrow
lower bound is ~0.060 MHz, so a width pinned there (railed) is NOT a measurement.

  python scint_candidates.py            # prints a per-burst candidate menu + writes JSON
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
sys.path.insert(0, f"{REPO}/scintillation/ne2025")
from query_ne2025_scint import galactic_floor

CATALOG = f"{REPO}/configs/bursts.yaml"
CFGDIR = f"{REPO}/scintillation/configs/bursts"
CHAN = 0.0305  # DSA native channel (MHz)
BOUND = 0.060  # fitter narrow lower bound; widths within 5% are railed, not measured
DSA_BAND_MHZ = 187.5  # full DSA bandwidth; a "broad" wider than this is a flat baseline


def vw(v):
    return (v.get("value"), v.get("stderr")) if isinstance(v, dict) else (v, None)


def candidates(sf):
    rows = []
    for s, md in sf.items():
        for m, fit in md.items():
            p = fit.get("best_fit_params", {})
            w = {k: vw(v) for k, v in p.items() if k.endswith("_gamma") or k.endswith("_sigma")}
            lor = {k: val for k, (val, e) in w.items() if k.endswith("_gamma") and val and val > 0}
            if not lor:
                continue
            kmin = min(lor, key=lor.get)
            val, err = w[kmin]
            # a co-fitted "broad" wider than the full DSA band is a flat baseline, not a
            # resolved second scale -> it does NOT make the narrow a distinct 2-component
            # detection (e.g. freya's 259-MHz Lorentzian over a ~190-MHz band).
            others = [v for kk, (v, e) in w.items() if kk != kmin and v and 0 < v <= DSA_BAND_MHZ]
            broad = min(others) if others else None
            railed = abs(val - BOUND) < 0.05 * BOUND
            good_err = err is not None and np.isfinite(err) and 0 < err < 0.5 * val
            distinct = broad is None or val < 0.5 * broad
            rows.append(
                dict(
                    subband=s,
                    model=m,
                    comp=kmin,
                    dnud_MHz=round(float(val), 4),
                    err_MHz=round(float(err), 4) if err is not None and np.isfinite(err) else None,
                    broad_MHz=round(float(broad), 3) if broad else None,
                    n_chan=round(float(val / CHAN), 1),
                    bic=round(float(fit.get("bic", np.nan)), 1),
                    redchi=round(float(fit.get("redchi", np.nan)), 2),
                    railed=bool(railed),
                    good_err=bool(good_err),
                    distinct=bool(distinct),
                    valid=bool((not railed) and good_err and distinct and val > 2 * CHAN),
                )
            )
    return sorted(rows, key=lambda r: r["dnud_MHz"])


def main():
    cat = yaml.safe_load(open(CATALOG))["bursts"]
    bursts = {k: v for k, v in cat.items() if isinstance(v, dict) and "ra_deg" in v}
    out = {}
    for name in sorted(bursts):
        v = bursts[name]
        f = glob.glob(f"{CFGDIR}/{name}_dsa.yaml") or [
            p
            for p in glob.glob(f"{CFGDIR}/*_dsa.yaml")
            if os.path.basename(p).lower() == f"{name.lower()}_dsa.yaml"
        ]
        sf = (yaml.safe_load(open(f[0])).get("analysis", {}).get("stored_fits")) or {}
        coord = SkyCoord(v["ra_deg"] * u.deg, v["dec_deg"] * u.deg, frame="icrs")
        floor = galactic_floor(coord)["DSA"]["bw_kHz"] / 1e3
        cand = candidates(sf)
        out[name] = dict(
            b=round(float(coord.galactic.b.value), 1),
            floor_MHz=round(float(floor), 4),
            candidates=cand,
        )
        print(f"\n##### {name}  b={out[name]['b']:+.1f}  floor={floor * 1e3:.0f} kHz")
        for c in cand:
            vd = (
                "VALID"
                if c["valid"]
                else (
                    "rail"
                    if c["railed"]
                    else (
                        "err"
                        if not c["good_err"]
                        else ("notdistinct" if not c["distinct"] else "subch")
                    )
                )
            )
            br = f"{c['broad_MHz']}" if c["broad_MHz"] else "--"
            print(
                f"  {c['dnud_MHz']:.3f}±{c['err_MHz']} MHz ({c['n_chan']}ch) broad={br} "
                f"{c['model']:24s} {c['subband']} bic={c['bic']:.0f} rc={c['redchi']} [{vd}]"
            )
    json.dump(out, open(f"{OUT}/scint_candidates.json", "w"), indent=2)
    print(f"\nwrote {OUT}/scint_candidates.json")


if __name__ == "__main__":
    main()
