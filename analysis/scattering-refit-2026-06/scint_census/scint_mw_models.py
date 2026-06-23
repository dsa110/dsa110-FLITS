"""Break the NE2025 systematic: recompute the cross-codetection diffractive-excess
test against TWO more Galactic electron-density models (YMW16, NE2001 via pygedm)
and ask whether the mid-|b| 7-11x excess survives a different model.

The published census (scint_mw_final.py) takes the MW floor from NE2025's SBW. Here
all THREE models are put on a common footing: each model's scattering timescale tau
@1 GHz (NE2025 Dv["TAU"]; pygedm dist_to_dm tau_sc for YMW16/NE2001) is scaled to
the DSA band (nu^-alpha, alpha=4.4) and turned into a floor diffractive bandwidth via
the thin-screen relation Dnu_d = C1/(2*pi*tau), C1=1.16 (the value NE2025 carries).
Only the electron model differs between columns, so excess ratios between models are
C1-, 2pi- and measurement-independent: excess_YMW16/excess_NE2025 = tau_NE2025/tau_YMW16.

Measured Dnu_d per burst is reused from data/scint/scint_mw_final.json (the recovered
diffractive scales). pygedm needs a scipy>=1.14 shim (simps was renamed simpson).

  python scint_mw_models.py
"""

import json
import os

import astropy.units as u
import numpy as np

# scipy>=1.14 removed scipy.integrate.simps (renamed simpson); pygedm still imports it.
import scipy.integrate as _si  # noqa: E402
import yaml
from astropy.coordinates import SkyCoord

if not hasattr(_si, "simps") and hasattr(_si, "simpson"):
    _si.simps = _si.simpson

import pygedm  # noqa: E402
from mwprop.nemod.NE2025 import ne2025  # noqa: E402

REPO = os.environ.get("FLITS_REPO") or os.path.abspath(f"{os.path.dirname(__file__)}/../../..")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/data/scint"
CATALOG = f"{REPO}/configs/bursts.yaml"

EDGE_PC = 30_000.0  # integrate to the Galactic boundary (matches NE2025 EDGE_KPC=30)
DSA_GHZ = 1.405  # DSA band centre
ALPHA = 4.4  # tau ~ nu^-alpha (same scaling for every model -> cancels in ratios)
C1 = 1.16  # thin-screen 2*pi*tau*Dnu_d = C1 (the value NE2025 carries)
# mid-|b| sightlines flagged by the published census as the excess population
EXCESS = {"zach", "wilhelm", "hamilton", "chromatica", "casey", "oran"}


def dnud_floor_MHz(tau_1ghz_s):
    """Floor diffractive bandwidth at the DSA band from a 1-GHz scattering time."""
    if tau_1ghz_s is None or not np.isfinite(tau_1ghz_s) or tau_1ghz_s <= 0:
        return None
    tau_dsa = tau_1ghz_s * DSA_GHZ ** (-ALPHA)
    return C1 / (2 * np.pi * tau_dsa) / 1e6  # Hz -> MHz


def model_taus(gl, gb):
    """tau @1 GHz (s) at the Galactic edge for NE2025, YMW16, NE2001."""
    Dk, Dv, Du, Dd = ne2025(
        ldeg=gl, bdeg=gb, dmd=EDGE_PC / 1e3, ndir=-1, classic=False, dmd_only=False
    )
    tau_ne2025 = float(Dv["TAU"]) / 1e3  # NE2025 TAU is ms @1 GHz
    taus = {"NE2025": tau_ne2025}
    # pygedm.dist_to_dm returns tau_sc referenced to 1 GHz for both YMW16 and NE2001
    # (same reference as NE2025's TAU), so the single nu^-alpha scaling to the DSA band
    # below is consistent across all three models.
    for key, method in (("YMW16", "ymw16"), ("NE2001", "ne2001")):
        try:
            _dm, tau = pygedm.dist_to_dm(gl * u.deg, gb * u.deg, EDGE_PC, method=method)
            taus[key] = float(tau.to(u.s).value) if hasattr(tau, "to") else float(tau)
        except Exception as e:  # noqa: BLE001
            taus[key] = None
            print(f"    [{method} fail @ l={gl:.1f} b={gb:.1f}] {type(e).__name__}: {e}")
    return taus


def main():
    cat = yaml.safe_load(open(CATALOG))["bursts"]
    final = json.load(open(f"{OUT}/scint_mw_final.json"))
    meas = {r["burst"]: r for r in final["bursts"]}

    rows = []
    for name in sorted(meas):
        m = meas[name]
        dnud = m.get("dnud_MHz")
        if dnud is None:
            continue
        b = cat.get(name) or cat.get(name.lower()) or cat.get(name.capitalize())
        if not b or "ra_deg" not in b:
            continue
        coord = SkyCoord(b["ra_deg"] * u.deg, b["dec_deg"] * u.deg, frame="icrs")
        gl, gb = coord.galactic.l.value, coord.galactic.b.value
        taus = model_taus(gl, gb)
        floors = {k: dnud_floor_MHz(t) for k, t in taus.items()}
        excess = {k: (f / dnud if f else None) for k, f in floors.items()}
        rows.append(
            dict(
                burst=name,
                b=round(gb, 1),
                dnud_MHz=dnud,
                lower_limit=m.get("lower_limit", False),
                tau_1ghz_s=taus,
                floor_MHz={k: (round(v, 4) if v else None) for k, v in floors.items()},
                excess={k: (round(v, 2) if v else None) for k, v in excess.items()},
                excess_published_ne2025=round(m["excess"], 2),
            )
        )

    # consistency check: the tau-derived NE2025 floor must reproduce the published
    # SBW-based excess (NE2025 satisfies 2*pi*TAU*SBW~C1 by construction). If this
    # breaks, the tau->Dnu_d conversion (and the YMW16/NE2001 columns) are wrong.
    print("NE2025 SBW-floor vs tau-derived-floor (should agree if 2*pi*TAU*SBW~C1):")
    for r in rows:
        e0, e1 = r["excess_published_ne2025"], r["excess"]["NE2025"]
        if e1:
            assert abs(e0 / e1 - 1) < 0.05, (
                f"{r['burst']}: tau-derived NE2025 excess {e1} != published {e0}"
            )
            print(
                f"  {r['burst']:12s} published={e0:6.2f}  tau-derived={e1:6.2f}  ratio={e0 / e1:.2f}"
            )

    print(
        f"\n{'burst':12s} {'b':>5} {'meas_kHz':>9}  {'NE2025':>7} {'YMW16':>7} {'NE2001':>7}   excess (floor/meas)"
    )

    def cell(excess, k, ll):
        v = excess.get(k)
        return f"{ll}{v:6.2f}" if v else f"{'--':>7}"

    for r in sorted(rows, key=lambda x: -(x["excess"].get("NE2025") or 0)):
        ll = ">" if r["lower_limit"] else " "
        tag = " *" if r["burst"] in EXCESS else ""
        print(
            f"{r['burst']:12s} {r['b']:+5.1f} {r['dnud_MHz'] * 1e3:9.1f}  "
            f"{cell(r['excess'], 'NE2025', ll)} {cell(r['excess'], 'YMW16', ll)} "
            f"{cell(r['excess'], 'NE2001', ll)}{tag}"
        )

    # verdict: does the excess survive on the mid-|b| excess sightlines under ALL models?
    print("\n=== survival of the mid-|b| excess under YMW16 / NE2001 ===")
    survive = {"YMW16": [], "NE2001": []}
    for r in rows:
        if r["burst"] not in EXCESS:
            continue
        for mod in ("YMW16", "NE2001"):
            e = r["excess"].get(mod)
            mark = "survives" if (e and e > 2) else ("WEAKER" if e else "no-floor")
            if e and e > 2:
                survive[mod].append(r["burst"])
            print(f"  {r['burst']:12s} {mod:7s} excess={e}  -> {mark}")
    for mod in ("YMW16", "NE2001"):
        print(
            f"  {mod}: excess>2x survives on {len(survive[mod])}/{len(EXCESS)} flagged sightlines: {survive[mod]}"
        )

    json.dump(
        dict(c1=C1, alpha=ALPHA, dsa_ghz=DSA_GHZ, edge_pc=EDGE_PC, bursts=rows),
        open(f"{OUT}/scint_mw_models.json", "w"),
        indent=2,
        default=float,
    )
    print(f"\nwrote {OUT}/scint_mw_models.json")


if __name__ == "__main__":
    main()
