"""Final cross-codetection diffractive-scattering-excess census, with NE2025-floor
uncertainty propagated into a per-sightline significance.

Diffractive Delta-nu_d per burst = the best defensible narrow candidate from
scint_candidates.json (deterministic selector best_diffractive); this RECOVERS
sightlines the BIC-only census missed by scanning all stored fits, including
Gen-Lorentz narrow components, and rejecting 0.060-MHz rails. The recovery
classification (recovered / non-detection / rail-only) is cross-checked against the
per-burst adversarial judge agents (scint_recover_verdicts.json) when present.

Excess = NE2025 MW-floor Delta-nu_d / measured Delta-nu_d (>1 => more scattering than
the smooth MW model). NE2025 is accurate only to ~a factor of 2-3, so we attach a
log-normal floor uncertainty SIGMA_FLOOR_DEX and combine it with the measurement
error to get sigma_log10(excess) and z = log10(excess)/sigma (sigma above the MW
floor). z>2 => excess significant at >~factor level.

  python scint_mw_final.py
"""

import json
import os

import numpy as np

NL = os.path.dirname(os.path.abspath(__file__))
OUT = f"{NL}/data/scint"
CAND = json.load(open(f"{OUT}/scint_candidates.json"))
VERD_PATH = f"{OUT}/scint_recover_verdicts.json"
VERD = json.load(open(VERD_PATH)) if os.path.exists(VERD_PATH) else {}

SIGMA_FLOOR_DEX = 0.4  # NE2025 floor uncertainty: ~x2.5 (1-sigma log-normal). factor 2-3 lit.
LN10 = np.log(10)


def best_diffractive(cands):
    """Pick the best defensible diffractive Delta-nu_d from a burst's candidate menu.

    Returns (dnud, err, tier, lower_limit, chosen, n_support). Tier:
      A  = >=1 valid candidate with a DISTINCT broad co-fit (clean 2-component) -> measurement
      B  = valid single-Lorentzian only -> Delta-nu_d an upper limit, excess a lower limit
      C  = no valid narrow (only rails / broad-only) -> non-detection
    """
    valid = [c for c in cands if c["valid"]]
    clean = [c for c in valid if c["broad_MHz"] and c["dnud_MHz"] < 0.5 * c["broad_MHz"]]
    # exclude overfit-noise subbands (redchi<0.2) from the clean set when alternatives exist
    clean_ok = [c for c in clean if c["redchi"] is None or c["redchi"] >= 0.2] or clean
    if clean_ok:
        # the diffractive scale is the NARROWEST consistent cluster; a "clean" candidate
        # that is really a broad scale with an even-broader 2nd component (e.g. a 6.7 MHz
        # Lorentzian + 36 MHz Lorentzian) passes distinct but is NOT narrow diffractive --
        # keep only candidates within a factor 3 of the narrowest clean one.
        clean_ok.sort(key=lambda c: c["dnud_MHz"])
        nn = clean_ok[0]["dnud_MHz"]
        cluster = [c for c in clean_ok if c["dnud_MHz"] <= 3 * nn]
        d = np.array([c["dnud_MHz"] for c in cluster])
        e = np.array([c["err_MHz"] for c in cluster])
        return float(np.median(d)), float(np.median(e)), "A", False, cluster[0], len(cluster)
    if valid:
        d = np.array([c["dnud_MHz"] for c in valid])
        e = np.array([c["err_MHz"] for c in valid])
        return float(np.median(d)), float(np.median(e)), "B", True, valid[0], len(valid)
    return None, None, "C", False, None, 0


def significance(floor, dnud, err):
    """excess + log-normal sigma (NE2025 floor + measurement) + z above the MW floor."""
    excess = floor / dnud
    sig = np.sqrt(SIGMA_FLOOR_DEX**2 + (err / (dnud * LN10)) ** 2)
    z = np.log10(excess) / sig
    return float(excess), float(sig), float(z)


def main():
    rows = []
    for name in sorted(CAND):
        e = CAND[name]
        dnud, err, tier, ll, chosen, nsup = best_diffractive(e["candidates"])
        row = dict(
            burst=name,
            b=e["b"],
            floor_MHz=e["floor_MHz"],
            dnud_MHz=dnud,
            err_MHz=err,
            tier=tier,
            lower_limit=ll,
            n_support=nsup,
        )
        if VERD.get(name):
            row["judge"] = {
                k: VERD[name].get(k)
                for k in (
                    "classification",
                    "dnud_MHz",
                    "confidence",
                    "is_lower_limit",
                    "adversarial_note",
                )
            }
        if dnud:
            exc, sig, z = significance(e["floor_MHz"], dnud, err)
            row.update(excess=exc, sigma_log10=sig, z_sigma=z, chosen=chosen)
        rows.append(row)

    print(f"NE2025 floor uncertainty: {SIGMA_FLOOR_DEX} dex (~x{10**SIGMA_FLOOR_DEX:.1f})\n")
    print(
        f"{'burst':12s} {'b':>5} {'tier':>4} {'Dnu_d_kHz':>10} {'floor_kHz':>9} {'excess':>7} {'z':>5}  judge"
    )
    for r in sorted(rows, key=lambda x: -(x.get("excess") or 0)):
        if r.get("excess"):
            j = r.get("judge", {}).get("classification", "-")
            ll = ">" if r["lower_limit"] else "="
            print(
                f"{r['burst']:12s} {r['b']:+5.1f} {r['tier']:>4} {r['dnud_MHz'] * 1e3:10.1f} "
                f"{r['floor_MHz'] * 1e3:9.0f} {ll}{r['excess']:6.1f} {r['z_sigma']:5.1f}  {j}"
            )
        else:
            j = r.get("judge", {}).get("classification", "-")
            print(
                f"{r['burst']:12s} {r['b']:+5.1f} {r['tier']:>4} {'(non-detection)':>10} "
                f"{r['floor_MHz'] * 1e3:9.0f} {'':>7} {'':>5}  {j}"
            )

    det = [r for r in rows if r.get("excess")]
    sig_hi = [r for r in det if r["z_sigma"] > 2]
    print(
        f"\n  {len(det)}/12 with a diffractive measurement; "
        f"{len(sig_hi)} have excess significant at z>2 (factor) above the MW floor:"
    )
    for r in sorted(sig_hi, key=lambda x: -x["z_sigma"]):
        print(f"    {r['burst']:12s} excess {r['excess']:.1f}x  ({r['z_sigma']:.1f} sigma)")
    json.dump(
        dict(sigma_floor_dex=SIGMA_FLOOR_DEX, bursts=rows),
        open(f"{OUT}/scint_mw_final.json", "w"),
        indent=2,
        default=float,
    )
    print(f"\nwrote {OUT}/scint_mw_final.json")


if __name__ == "__main__":
    main()
