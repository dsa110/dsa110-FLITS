"""CHIME-DSA co-detection association significance (pillars 1-4).

Adds the rigorous apparatus the bare temporal-consistency test lacks, as pure functions
with explicit inputs (mirrors crossmatching/toa_crossmatch.py). Assembled by
``build_association_report`` into ``association_report.json`` — the golden
``toa_crossmatch_results.json`` is never touched. See
``.agents/research-codetection-validation-rigor.md`` and
``.agents/experiment-chance-coincidence-falsealarm.md``.

Pillar 1 — chance-coincidence probability (analytic Poisson; experiment-validated):
the expected number of unrelated CHIME FRBs falling in a burst's (position x time x DM)
window, ``mu = R_sr_s * Omega_win * 2*dt * f_DM``; ``P = 1 - exp(-mu)``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# CHIME/FRB Catalogue 1 (Amiri et al. 2021, ApJS 257, 59): ~525 FRBs sky^-1 day^-1 above 5 Jy ms.
R_SKY_PER_DAY_CENTRAL = 525.0
FULL_SKY_SR = 4.0 * math.pi
SECONDS_PER_DAY = 86400.0
DEG2_PER_SR = (180.0 / math.pi) ** 2

# CHIME extragalactic-DM distribution, modelled log-normal (median 500, sigma_ln 0.7).
# Assumption (catalogue file not on h17); shared by analytic + MC so it cancels in their ratio.
DM_MEDIAN, DM_SIGMA_LN = 500.0, 0.7

# Baseline coincidence windows: deliberately generous (chance-maximising) -> conservative P.
OMEGA_WIN_BASELINE_DEG2 = math.pi * 0.5**2  # 0.5 deg radius disk ~ 0.785 deg^2
DT_BASELINE_S, DDM_BASELINE = 1.0, 5.0


def _r_sr_s(rate_per_day: float) -> float:
    """FRB rate per steradian per second from an all-sky per-day rate."""
    return rate_per_day / FULL_SKY_SR / SECONDS_PER_DAY


def f_dm(
    dm: float, half_width: float, *, dm_median: float = DM_MEDIAN, dm_sigma_ln: float = DM_SIGMA_LN
) -> float:
    """P(random CHIME DM within +/- half_width of ``dm``), local-density approximation."""
    z = (math.log(dm) - math.log(dm_median)) / dm_sigma_ln
    pdf = math.exp(-0.5 * z * z) / (dm * dm_sigma_ln * math.sqrt(2.0 * math.pi))
    return min(1.0, pdf * 2.0 * half_width)


def chance_mu(
    dm: float, *, rate_per_day: float, omega_win_deg2: float, dt_s: float, ddm: float
) -> float:
    """Expected number of unrelated CHIME FRBs in the (position x time x DM) window."""
    return _r_sr_s(rate_per_day) * (omega_win_deg2 / DEG2_PER_SR) * (2.0 * dt_s) * f_dm(dm, ddm)


def chance_probability(dm: float, **kw) -> float:
    """Poisson P(>=1 chance association) = 1 - exp(-mu)."""
    return 1.0 - math.exp(-chance_mu(dm, **kw))


def expected_chance_associations(dms, **kw) -> float:
    """Sample-level expected chance count = sum of per-burst mu."""
    return sum(chance_mu(d, **kw) for d in dms)


# --- Pillar 2: independent DM agreement ---------------------------------------
def dm_agreement(
    *,
    dm_chime,
    dm_chime_err,
    dm_dsa,
    dm_dsa_err,
    n_sigma_thresh: float = 3.0,
    dm_floor: float = 1.0,
) -> dict:
    """CHIME-vs-DSA DM consistency, each with its own error. Null+reason when CHIME DM absent.

    ``dm_floor`` (pc/cm^3) is a PHYSICAL tolerance floor on the combined sigma: the CHIME arrival
    regression returns a statistical sigma that can sit far below the ~1 pc/cm^3 scale at which a DM
    difference is physically meaningful (e.g. casey +/-0.0009), so a sub-pc offset would otherwise read
    as many-sigma. The floor (expert verdict, .agents/audit-chime-side-dm.md) keeps the test honest:
    sigma_eff = max(quadrature errors, dm_floor).
    """
    if dm_chime is None or dm_dsa is None:
        return {
            "delta": None,
            "sigma": None,
            "n_sigma": None,
            "consistent": None,
            "reason": "no CHIME DM available",
        }
    delta = abs(dm_chime - dm_dsa)
    sigma = max(math.hypot(dm_chime_err or 0.0, dm_dsa_err or 0.0), dm_floor)
    n = delta / sigma if sigma > 0 else float("inf")
    return {
        "delta": delta,
        "sigma": sigma,
        "n_sigma": n,
        "consistent": bool(n <= n_sigma_thresh),
        "reason": None,
    }


# --- Pillar 3: timing error budget + residual-pedestal significance ------------
def timing_budget_ms(
    *,
    dm_unc_ms: float,
    fwhm_ms: float,
    clock_ms: float = 0.0,
    baseline_ms: float = 0.0,
    intrachannel_ms: float = 0.0,
) -> float:
    """Full quadrature timing error: DM-uncertainty (+) pulse width (+) clock/baseline/intra-channel."""
    return math.sqrt(dm_unc_ms**2 + fwhm_ms**2 + clock_ms**2 + baseline_ms**2 + intrachannel_ms**2)


def residual_pedestal(residuals_ms, errors_ms) -> dict:
    """Inverse-variance-weighted mean residual and its significance (tests the +2.4 ms pedestal)."""
    w = [1.0 / e**2 for e in errors_ms]
    wm = sum(wi * r for wi, r in zip(w, residuals_ms, strict=True)) / sum(w)
    err = math.sqrt(1.0 / sum(w))
    return {"weighted_mean_ms": wm, "error_ms": err, "n_sigma": abs(wm) / err}


# --- Pillar 4: positional coincidence -----------------------------------------
def omega_disk_deg2(radius_deg: float) -> float:
    """Solid angle (deg^2) of a CHIME localization disk of the given radius."""
    return math.pi * radius_deg**2


def position_agreement(dsa_coord: str, chime_ra_deg, chime_dec_deg, radius_deg: float) -> dict:
    """CHIME tied-beam point vs DSA position; consistent within a stated CHIME radius.

    ``tiedbeam_locations`` is the CHIME position the singlebeam was formed at (independent
    of DSA) but carries no error ellipse, so consistency is judged against a stated CHIME
    localization radius (assumption; Michilli et al. 2021 baseband localizations sub-arcmin).
    Null+reason when no CHIME position is available.
    """
    if chime_ra_deg is None or chime_dec_deg is None:
        return {
            "separation_deg": None,
            "radius_deg": radius_deg,
            "consistent": None,
            "reason": "no CHIME position available",
        }
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    a = SkyCoord(dsa_coord, unit=(u.hourangle, u.deg), frame="icrs")
    b = SkyCoord(chime_ra_deg, chime_dec_deg, unit=u.deg, frame="icrs")
    sep = float(a.separation(b).deg)
    return {
        "separation_deg": sep,
        "radius_deg": radius_deg,
        "consistent": bool(sep <= radius_deg),
        "reason": None,
    }


# --- Assemble the association report (golden artifact never touched) -----------
def _load_chime_inputs(chime_inputs_path) -> dict:
    """Index CHIME-side extraction rows by chime_id (empty dict if path absent/None)."""
    if not chime_inputs_path or not Path(chime_inputs_path).exists():
        return {}
    rows = json.loads(Path(chime_inputs_path).read_text())
    return {str(r["chime_id"]): r for r in rows if "dm_chime" in r}


def build_association_report(
    fixture_path,
    *,
    rate_per_day: float = 1000.0,
    omega_win_deg2: float = OMEGA_WIN_BASELINE_DEG2,
    dt_s: float = DT_BASELINE_S,
    ddm: float = DDM_BASELINE,
    chime_inputs_path=None,
    chime_radius_deg: float = 0.1,
    toa_results_path=None,
) -> dict:
    """Run all four pillars over the fixture and return the report dict. Read-only on disk.

    Pillars 2 (CHIME DM) and 4 (CHIME position) activate per burst when a CHIME-side inputs
    file is supplied (``chime_inputs_path``, keyed by chime_id); bursts absent from it emit
    explicit null+reason rather than fabricated values.
    """
    fx = json.loads(Path(fixture_path).read_text())
    chime = _load_chime_inputs(chime_inputs_path)
    bursts, dms = [], []
    for row in fx["bursts"]:
        dm = row["dm"]
        dms.append(dm)
        ci = chime.get(str(row["chime_id"]), {})
        bursts.append(
            {
                "name": row["name"],
                "chime_id": row["chime_id"],
                "dm": dm,
                "chance_coincidence_P": chance_probability(
                    dm, rate_per_day=rate_per_day, omega_win_deg2=omega_win_deg2, dt_s=dt_s, ddm=ddm
                ),
                "dm_agreement": dm_agreement(
                    dm_chime=ci.get("dm_chime"),
                    dm_chime_err=ci.get("dm_chime_err"),
                    dm_dsa=dm,
                    dm_dsa_err=row.get("dm_uncertainty"),
                ),
                "dm_confidence": ci.get("dm_confidence"),  # figure-review: real/marginal/noise
                "position": position_agreement(
                    row.get("source_coord"),
                    ci.get("chime_ra_deg"),
                    ci.get("chime_dec_deg"),
                    chime_radius_deg,
                ),
            }
        )

    # Pillar 3, sample level: the inter-site geometric delay is a near-constant
    # ~-2.2 ms pedestal (CHIME leads DSA) shared by every pair because the sources
    # cluster in declination near CHIME transit. A common pedestal is a positive
    # association signature -- unrelated triggers would not sit on one. We test the
    # residual (measured 400-MHz offset minus the predicted geometric delay) for a
    # significant nonzero sample mean, weighted by each pair's full timing budget.
    pedestal = None
    if toa_results_path and Path(toa_results_path).exists():
        toa = json.loads(Path(toa_results_path).read_text())
        resid, errs, geos = [], [], []
        for b in fx["bursts"]:
            tr = toa.get(b["name"])
            if tr is None:
                continue
            off = tr.get("measured_offset_ms")
            geo = tr.get("geometric_delay_ms")
            err = tr.get("combined_error_full_ms") or tr.get("combined_error_ms")
            if off is None or geo is None or not err:
                continue
            resid.append(off - geo)
            errs.append(err)
            geos.append(geo)
        if len(resid) >= 2:
            rp = residual_pedestal(resid, errs)
            pedestal = {
                "geometric_delay_mean_ms": sum(geos) / len(geos),
                "geometric_delay_spread_ms": max(geos) - min(geos),
                "residual_weighted_mean_ms": rp["weighted_mean_ms"],
                "residual_error_ms": rp["error_ms"],
                "residual_n_sigma": rp["n_sigma"],
                "n_pairs": len(resid),
                "note": (
                    "residual = measured_offset_ms - geometric_delay_ms, weighted by "
                    "combined_error_full_ms; tests the shared geometric pedestal. "
                    "Arrival order / offset sign is a DM-referral + intrinsic-band "
                    "diagnostic and is NOT an input to the chance-coincidence gate."
                ),
            }

    return {
        "inputs": {
            "rate_per_day": rate_per_day,
            "omega_win_deg2": omega_win_deg2,
            "dt_s": dt_s,
            "ddm": ddm,
            "dm_model": "lognormal(500,0.7) [assumption]",
            "chime_dm_method": "arrival-time regression (scatter-deconvolved EMG sub-band t0 vs "
            "nu^-2; coherent_dedisp at DSA DM; uniform TDS=32/N_SB=6) on library coherent_dedisp; "
            "8/12 bursts constrained, 4 unconstrained (<3 sub-bands above S/N 4). Supersedes the "
            "retracted DM-phase extraction (1e-3*K_DM inter-channel unit bug). Per-burst DMs and "
            "CHIME-DSA agreement in dm_provenance.csv; see .agents/audit-chime-side-dm.md.",
            "chime_localization_radius_deg": chime_radius_deg,
            "chime_localization_note": "tiedbeam pointing; no multi-beam error ellipse "
            "(Michilli+2021 sub-arcmin assumed)",
        },
        "expected_chance_associations": expected_chance_associations(
            dms, rate_per_day=rate_per_day, omega_win_deg2=omega_win_deg2, dt_s=dt_s, ddm=ddm
        ),
        "geometric_pedestal": pedestal,
        "bursts": bursts,
    }


def main() -> None:
    here = Path(__file__).resolve().parent
    chime_path = here / "chime_side_inputs.json"
    toa_path = here / "toa_crossmatch_results.json"
    rep = build_association_report(
        here / "notebook_reproduction_fixture.json",
        chime_inputs_path=chime_path if chime_path.exists() else None,
        toa_results_path=toa_path if toa_path.exists() else None,
    )
    out = here / "association_report.json"
    out.write_text(json.dumps(rep, indent=2))
    n_dm = sum(1 for b in rep["bursts"] if b["dm_agreement"]["consistent"] is not None)
    print(f"wrote {out}  (sum_mu={rep['expected_chance_associations']:.3e}, dm_active={n_dm}/12)")


if __name__ == "__main__":
    main()
