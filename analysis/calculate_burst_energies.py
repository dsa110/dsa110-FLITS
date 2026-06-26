#!/usr/bin/env python
"""Isotropic-equivalent energy E_iso for the co-detected FRB sample.

For each sightline with a spectroscopic host redshift and a joint CHIME+DSA
scattering fit, integrate the fitted fluence spectrum F(nu) = c0 (nu/nu_ref)^gamma
over each telescope's OWN observed band, convert each band to an ABSOLUTE Jy
scale, k-correct, and sum:

    E_iso = (4 pi D_L(z)^2 / (1+z)) [ s_C int_CHIME F_C dnu + s_D int_DSA F_D dnu ]

where s_C, s_D are the per-band flux scales (`flux_jy_per_unit` in
configs/telescopes.yaml). The live setting is "fluxcal": each band's absolute Jy
integral is computed per channel by the radiometer conversion in
analysis/flux_cal.joint_band_fluence_jy_ms_hz (sigma_S(nu) folded in), not a
single Jy-per-native scalar. A float scale (the legacy native-units * scalar
path) and None (uncalibrated) are still accepted.

Why the flux scale is mandatory (see analysis/burst_energies/CALIBRATION_REVIEW.md):
the joint fit recovers c0 in the input dynamic spectrum's NATIVE units, and CHIME
and DSA are not on a common flux standard. c0_C and c0_D are therefore in
independent, arbitrary per-telescope units. Summing two such bands is summing
dollars and yen: it is not an energy, and the per-burst CHIME:DSA mix varies, so
even the relative ORDERING of the total is unreliable. So this script refuses to
emit an energy until BOTH bands carry a real `flux_jy_per_unit` (from each
telescope's SEFD + beam response at the burst position). Until then it reports the
band fluence integrals in native units, flagged uncalibrated, and writes a
"pending calibration" LaTeX stub rather than a publishable table of fake erg.

A second trust boundary gates the energy's OWN inputs, NOT scattering quality:
E_iso is alpha-independent, so a joint fit the committed-fit quality gate marks
FAIL (prior-railed / unphysical shared alpha; see the *_joint_gate.json sidecars
from gate_joint_committed.py) is still used IF its per-band c0/gamma are physical
(finite, c0>0) -- the FAIL verdict is the shared-alpha judgement, which the energy
does not depend on. Only fits with missing or non-physical c0/gamma are dropped;
the quality_flag rides along as an informational column (ADR-0003/0004; 3-expert
panel 2026-06-24).

The (1+z) bandwidth k-correction (Zhang 2018) is applied by default; the
no-k-correction value is tabulated alongside so the manuscript can state the
convention explicitly.

ref_freq is the band centre (f_min+f_max)/2; the fitter uses median(freq), which
equals the centre for a symmetric, uniformly-flagged band (small <~few% offset
under asymmetric channel flagging -> minor vs the calibration systematic).

Usage:  python analysis/calculate_burst_energies.py [--check]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import astropy.units as u  # noqa: E402

from analysis.flux_cal import joint_band_fluence_jy_ms_hz, joint_c0_gamma  # noqa: E402
from galaxies.foreground.config import COSMO, TARGETS  # noqa: E402

JOINT_DIR = REPO / "analysis" / "scattering-refit-2026-06" / "joint_json"
TEL_CFG = REPO / "configs" / "telescopes.yaml"
OUT_DIR = REPO / "analysis" / "burst_energies"

JY_MS_HZ_TO_SI = 1e-29  # 1 Jy*ms*Hz = 1e-29 J*m^-2
J_TO_ERG = 1e7
PLACEHOLDER_Z = 1.0  # Freya/Mahi/Johndoeii carry z=1.0000 as a "redshift unknown" flag

# Per-band absolute-scale systematic [dex], folded into the energy error. DSA: measured beam +
# coherent-beam SEFD (N_ant=48) assumption; CHIME: documented cylinder beam + derived SEFD
# (chime_sefd.csv). Lognormal: fractional sigma = ln(10)*dex. Dominates the c0 statistical error.
BAND_SYS_DEX = {"C": 0.25, "D": 0.20}

# Host-redshift provenance (nick -> (quality, source)). All 8 E_iso hosts are spectroscopic;
# hamilton/chromatica have no published host paper yet (value repo-internal, provenance TBD).
# Sharma+2024 (arXiv:2409.16964) Gold sample; Connor+2024 (arXiv:2409.16952) Keck/MOSFIRE for
# wilhelm. See docs/rse/specs/research-energetics-followups.md. Surfaced as row["z_src"].
Z_PROVENANCE = {
    "zach": ("spec", "Sharma+2024 Keck/LRIS"),
    "whitney": ("spec", "Sharma+2024 Keck/LRIS"),
    "oran": ("spec", "Sharma+2024 Keck/LRIS"),
    "isha": ("spec", "Sharma+2024 P200/DBSP"),
    "phineas": ("spec", "Sharma+2024 Keck/DEIMOS"),
    "wilhelm": ("spec", "Connor+2024 Keck/MOSFIRE"),
    "hamilton": ("spec-provisional", "unpublished host; provenance TBD"),
    "chromatica": ("spec-provisional", "unpublished host; provenance TBD"),
}


def band_edges_hz() -> dict[str, tuple[float, float, float]]:
    """(nu1, nu2, nu_ref) in Hz per band from the instrument config (band centre = ref)."""
    cfg = yaml.safe_load(TEL_CFG.read_text())
    out = {}
    for key, tag in (("chime", "C"), ("dsa", "D")):
        f1 = cfg[key]["f_min_GHz"] * 1e9
        f2 = cfg[key]["f_max_GHz"] * 1e9
        out[tag] = (f1, f2, 0.5 * (f1 + f2))
    return out


def flux_scales() -> dict[str, float | None]:
    """tag -> flux_jy_per_unit (Jy per native fluence unit), or None if uncalibrated."""
    cfg = yaml.safe_load(TEL_CFG.read_text())
    return {tag: cfg[key].get("flux_jy_per_unit") for key, tag in (("chime", "C"), ("dsa", "D"))}


def band_integral(c0: float, gamma: float, nu_ref: float, nu1: float, nu2: float) -> float:
    """int_{nu1}^{nu2} c0 (nu/nu_ref)^gamma dnu  [native*ms*Hz], freqs in Hz."""
    if abs(gamma + 1.0) < 1e-9:  # power law -> 1/nu, integral is a log
        return c0 * nu_ref * np.log(nu2 / nu1)
    g = gamma + 1.0
    return c0 * nu_ref / g * ((nu2 / nu_ref) ** g - (nu1 / nu_ref) ** g)


def band_energy_erg(
    i_raw: float, flux_scale: float, d_l_m: float, z: float, kcorr: bool = True
) -> float:
    """Band fluence integral [native*ms*Hz] x flux scale [Jy/native] -> E [erg].

    (1+z) bandwidth k-correction (Zhang 2018, ApJ 867 L21) applied when kcorr=True.
    """
    factor = 1.0 / (1.0 + z) if kcorr else 1.0
    return 4.0 * np.pi * d_l_m**2 * i_raw * flux_scale * JY_MS_HZ_TO_SI * J_TO_ERG * factor


def load_redshifts() -> dict[str, float]:
    """nickname (lowercased) -> z_frb, from the galaxy-search TARGETS list."""
    return {name.lower(): z for name, _ra, _dec, z in TARGETS}


def load_gate_flags() -> dict[str, str]:
    """nickname (lowercased) -> joint-fit quality_flag (PASS/MARGINAL/FAIL).

    Source: the committed-fit quality gate `*_joint_gate.json` sidecars produced by
    analysis/scattering-refit-2026-06/gate_joint_committed.py (the authoritative
    runtime FLITS contract applied to the joint fits). One file per burst.
    """
    out = {}
    for p in sorted(JOINT_DIR.glob("*_joint_gate.json")):
        d = json.loads(p.read_text())
        out[d["burst"].lower()] = d["quality_flag"]
    return out


def load_joint_params() -> dict[str, dict]:
    """nickname (lowercased) -> {c0_C, gamma_C, c0_D, gamma_D, quality_flag, pbf} medians.

    Trust boundary (ADR-0003/0004; 3-expert panel 2026-06-24). E_iso is
    alpha-INDEPENDENT: ``band_integral`` uses only c0/gamma and the band edges, and
    the absolute scale is a per-channel radiometer integral -- alpha appears nowhere
    in the energy. So energy citability must NOT gate on the scattering joint-fit
    FAIL, which is the shared-alpha L1 verdict; doing so deletes a valid energy over
    a scattering-physics judgement that the energy does not depend on. (oran/whitney
    FAIL only on the *superseded* mixed-PBF alpha + the *retired* 1.5 floor, yet have
    physical, rail-free c0/gamma.) We therefore gate on the energy's OWN inputs: a
    fit is dropped iff its per-band c0/gamma are missing or non-physical (non-finite,
    or c0 <= 0). The scattering quality_flag is carried through as an INFORMATIONAL
    column, not an exclusion.

    c0/gamma come from the committed joint fits (mixed-legacy PBF: pbf_C/pbf_D
    absent). The all-exp campaign fits scattering only and produces NO c0/gamma
    (verified 2026-06-24, local + HPCC), and since E_iso is alpha-independent the
    mixed-PBF c0/gamma are the correct amplitude inputs; the PBF family is stamped
    per fit + in the provenance sidecar.
    """
    flags = load_gate_flags()
    out, dropped, skipped = {}, [], []
    for p in sorted(JOINT_DIR.glob("*_joint_fit.json")):
        d = json.loads(p.read_text())
        pct = d["percentiles"]
        nick = d["burst"].lower()
        try:
            params = {k: pct[k]["median"] for k in ("c0_C", "gamma_C", "c0_D", "gamma_D")}
        except KeyError:
            skipped.append(nick)  # no per-band amplitude (all-exp/scattering-only or schema shift)
            continue
        physical = (
            all(np.isfinite(v) for v in params.values())
            and params["c0_C"] > 0
            and params["c0_D"] > 0
        )
        if not physical:
            dropped.append(nick)
            continue
        params["quality_flag"] = flags.get(nick)
        pc, pd = d.get("pbf_C"), d.get("pbf_D")
        params["pbf"] = "all-exp" if (pc == "exp" and pd == "exp") else "mixed-legacy"
        out[nick] = params
    if dropped:
        print(
            f"[gate] dropped {len(dropped)} fit(s) with non-physical c0/gamma "
            f"(non-finite or c0<=0) from E_iso: {', '.join(sorted(dropped))}",
            file=sys.stderr,
        )
    if skipped:
        print(
            f"[gate] skipped {len(skipped)} fit(s) with no per-band c0/gamma "
            f"(all-exp/scattering-only or schema shift): {', '.join(sorted(skipped))}",
            file=sys.stderr,
        )
    return out


def _band_jy(scale, i_native, nick, fluence_fn):
    """(I_jy_ms_hz or None, is_calibrated) for one band given its scale sentinel.

    scale is None -> uncalibrated; the string "fluxcal" -> data-driven per-channel radiometer
    integral (fluence_fn(nick), already in Jy*ms*Hz); a float -> legacy native-units power-law
    integral times a single Jy-per-native scalar.
    """
    if scale is None:
        return None, False
    if scale == "fluxcal":
        if fluence_fn is None:
            raise NotImplementedError("fluxcal selected but no fluence_fn supplied for this band")
        return float(fluence_fn(nick)), True
    return i_native * float(scale), True


def compute(scales: dict[str, float | str | None] | None = None) -> list[dict]:
    bands = band_edges_hz()
    if scales is None:
        scales = flux_scales()
    s_C, s_D = scales["C"], scales["D"]
    zs = load_redshifts()
    fits = load_joint_params()
    nu1_C, nu2_C, ref_C = bands["C"]
    nu1_D, nu2_D, ref_D = bands["D"]

    rows = []
    for nick in sorted(fits):
        z = zs.get(nick)
        if z is None or abs(z - PLACEHOLDER_Z) < 1e-6:
            continue  # no real host redshift -> cannot place a distance
        fp = fits[nick]
        d_l_m = COSMO.luminosity_distance(z).to(u.m).value

        I_C = band_integral(fp["c0_C"], fp["gamma_C"], ref_C, nu1_C, nu2_C)
        I_D = band_integral(fp["c0_D"], fp["gamma_D"], ref_D, nu1_D, nu2_D)

        I_C_jy, cal_C = _band_jy(s_C, I_C, nick, lambda n: joint_band_fluence_jy_ms_hz(n, "C"))
        I_D_jy, cal_D = _band_jy(s_D, I_D, nick, lambda n: joint_band_fluence_jy_ms_hz(n, "D"))
        calibrated = cal_C and cal_D

        row = {
            "burst": nick,
            "z": z,
            "D_L_Mpc": COSMO.luminosity_distance(z).to(u.Mpc).value,
            "c0_C": fp["c0_C"],
            "gamma_C": fp["gamma_C"],
            "c0_D": fp["c0_D"],
            "gamma_D": fp["gamma_D"],
            "I_CHIME_native_ms_Hz": I_C,  # band fluence integral, NATIVE units
            "I_DSA_native_ms_Hz": I_D,
            "calibrated": calibrated,
            "quality_flag": fp.get("quality_flag"),
            "c0gamma_pbf": fp.get("pbf"),
            "z_src": Z_PROVENANCE.get(nick, ("unknown", ""))[0],
        }
        if I_C_jy is not None:
            row["I_CHIME_jy_ms_hz"] = I_C_jy
        if I_D_jy is not None:
            row["I_DSA_jy_ms_hz"] = I_D_jy
        if calibrated:
            e_C = band_energy_erg(I_C_jy, 1.0, d_l_m, z)  # scale already folded into I_*_jy
            e_D = band_energy_erg(I_D_jy, 1.0, d_l_m, z)
            e_C0 = band_energy_erg(I_C_jy, 1.0, d_l_m, z, kcorr=False)
            e_D0 = band_energy_erg(I_D_jy, 1.0, d_l_m, z, kcorr=False)
            # Per-band fractional error: c0 posterior width (statistical) (+) absolute-scale
            # systematic (SEFD+beam, lognormal sigma = ln10*dex). Bands independent -> quadrature.
            err = {}
            for tag, e_band in (("C", e_C), ("D", e_D)):
                f_stat = joint_c0_gamma(nick, tag)[2]  # c0 fractional posterior width
                f_sys = np.log(10.0) * BAND_SYS_DEX[tag]
                err[tag] = e_band * np.hypot(f_stat, f_sys)
            row.update(
                flux_jy_per_unit_C=s_C,
                flux_jy_per_unit_D=s_D,
                E_iso_CHIME_erg=e_C,
                E_iso_DSA_erg=e_D,
                E_iso_erg=e_C + e_D,  # legitimate: both bands now in Jy
                E_iso_erg_no_kcorr=e_C0 + e_D0,
                E_iso_erg_err=float(np.hypot(err["C"], err["D"])),
            )
        rows.append(row)
    return rows


def markdown_table(rows: list[dict]) -> str:
    if rows and rows[0]["calibrated"]:
        head = (
            "| Burst | z | D_L (Mpc) | gamma_C | gamma_D | "
            "E_CHIME (erg) | E_DSA (erg) | E_iso (erg) | +/- E_iso (erg) | E_iso no-(1+z) (erg) |"
        )
        sep = "|" + "|".join(["---"] * 10) + "|"
        lines = [head, sep]
        for r in rows:
            lines.append(
                f"| {r['burst']} | {r['z']:.4f} | {r['D_L_Mpc']:.1f} | {r['gamma_C']:.2f} | "
                f"{r['gamma_D']:.2f} | {r['E_iso_CHIME_erg']:.3e} | {r['E_iso_DSA_erg']:.3e} | "
                f"{r['E_iso_erg']:.3e} | {r['E_iso_erg_err']:.3e} | {r['E_iso_erg_no_kcorr']:.3e} |"
            )
        return "\n".join(lines)
    # uncalibrated: band fluence integrals in NATIVE units, flagged not-an-energy. When a band is
    # flux-calibrated but the other is not (gate closed), also surface its Jy*ms*Hz integral.
    has_dsa_jy = any("I_DSA_jy_ms_hz" in r for r in rows)
    head = (
        "| Burst | z | D_L (Mpc) | gamma_C | gamma_D | "
        "int F_C (native*ms*Hz) | int F_D (native*ms*Hz) |"
    )
    if has_dsa_jy:
        head += " int F_D (Jy*ms*Hz) |"
    sep = "|" + "|".join(["---"] * (8 if has_dsa_jy else 7)) + "|"
    lines = [head, sep]
    for r in rows:
        line = (
            f"| {r['burst']} | {r['z']:.4f} | {r['D_L_Mpc']:.1f} | {r['gamma_C']:.2f} | "
            f"{r['gamma_D']:.2f} | {r['I_CHIME_native_ms_Hz']:.3e} | {r['I_DSA_native_ms_Hz']:.3e} |"
        )
        if has_dsa_jy:
            jy = r.get("I_DSA_jy_ms_hz")
            line += f" {jy:.3e} |" if jy is not None else " -- |"
        lines.append(line)
    return "\n".join(lines)


def _tex_pow(x: float) -> str:
    r"""Render a float as a\times10^{b} for LaTeX."""
    if x <= 0 or not np.isfinite(x):
        return r"\mathrm{n/a}"
    exp = int(np.floor(np.log10(x)))
    mant = x / 10**exp
    return rf"{mant:.2f}\times10^{{{exp}}}"


def _tex_val_err(x: float, dx: float) -> str:
    r"""Render x +/- dx sharing one mantissa exponent: (a \pm b)\times10^{c}."""
    if x <= 0 or not np.isfinite(x):
        return r"\mathrm{n/a}"
    exp = int(np.floor(np.log10(x)))
    return rf"({x / 10**exp:.2f} \pm {dx / 10**exp:.2f})\times10^{{{exp}}}"


def latex_pending() -> str:
    """LaTeX stub written while the flux calibration is missing (no fake energies)."""
    return r"""% Auto-generated by analysis/calculate_burst_energies.py -- do not hand-edit.
% STATUS: flux calibration pending -- no energy table emitted (see below).
\subsection{Isotropic-equivalent energies}
\label{sec:burst-energies}

Isotropic-equivalent burst energies are pending absolute flux calibration. The
joint CHIME--DSA fit constrains each band's fluence spectrum only in the native
units of its input dynamic spectrum, and CHIME and DSA are not placed on a common
flux standard, so the per-band amplitudes are not directly comparable and may not
be summed into an energy. Converting to physical $\mathrm{Jy\,ms}$ requires each
telescope's system-equivalent flux density and primary-beam response at the burst
position (Andersen et~al.\ 2023 and the baseband CHIME/FRB catalog update for
CHIME; Law et~al.\ 2024 for DSA-110). Once those scales are supplied, the energy
follows $E_{\mathrm{iso}} = (4\pi D_L^2(z)/(1+z))\,[\,s_C\!\int F_C\,d\nu +
s_D\!\int F_D\,d\nu\,]$ with the $(1+z)$ bandwidth k-correction.
"""


def latex_section(rows: list[dict]) -> str:
    body = []
    for r in rows:
        body.append(
            f"  {r['burst'].capitalize()} & ${r['z']:.4f}$ & ${r['D_L_Mpc']:.0f}$ & "
            f"${r['gamma_C']:+.2f}$ & ${r['gamma_D']:+.2f}$ & "
            f"${_tex_val_err(r['E_iso_erg'], r['E_iso_erg_err'])}$ \\\\"
        )
    table_body = "\n".join(body)
    return rf"""% Auto-generated by analysis/calculate_burst_energies.py -- do not hand-edit.
\subsection{{Isotropic-equivalent energies}}
\label{{sec:burst-energies}}

For each sightline with a spectroscopic host redshift we estimate the
isotropic-equivalent burst energy directly from the joint CHIME--DSA fit,
without extrapolating either band's spectrum beyond where it is constrained.
The joint fit returns a per-band spectral amplitude and index,
$F_X(\nu) = c_{{0,X}}\,(\nu/\nu_{{\mathrm{{ref}},X}})^{{\gamma_X}}$, which we put on
an absolute scale with each telescope's flux calibration $s_X$
($\mathrm{{Jy}}$ per native unit) and integrate over its own observing
band---CHIME over $0.400$--$0.800\,\mathrm{{GHz}}$ and DSA over
$1.311$--$1.499\,\mathrm{{GHz}}$,
%
\begin{{equation}}
E_{{\mathrm{{iso}}}} = \frac{{4\pi D_L^2(z)}}{{1+z}}\left[
  s_C\!\int_{{\nu_1^{{C}}}}^{{\nu_2^{{C}}}} F_{{\mathrm{{CHIME}}}}(\nu)\,d\nu
  + s_D\!\int_{{\nu_1^{{D}}}}^{{\nu_2^{{D}}}} F_{{\mathrm{{DSA}}}}(\nu)\,d\nu
\right],
\label{{eq:eiso}}
\end{{equation}}
%
with $D_L(z)$ from the fiducial \textit{{Planck}}\,2018 cosmology, the $(1+z)$
bandwidth k-correction applied, and
$1\,\mathrm{{Jy\,ms\,Hz}} = 10^{{-29}}\,\mathrm{{J\,m^{{-2}}}}$. Bursts whose hosts
lack a spectroscopic redshift are omitted. This band-restricted integral
constrains the energy released \emph{{within the observed spectral envelope}}: it
avoids the large, model-dependent extrapolation that a single power law fit
across the full $0.4$--$1.5\,\mathrm{{GHz}}$ span would require, at the cost of not
counting flux in the unobserved $0.8$--$1.3\,\mathrm{{GHz}}$ gap (so the values are
lower limits on the $0.4$--$1.5\,\mathrm{{GHz}}$ energy).

\begin{{table}}
\centering
\caption{{Isotropic-equivalent energies of the co-detected FRBs, integrated over
the CHIME ($0.4$--$0.8\,\mathrm{{GHz}}$) and DSA ($1.31$--$1.50\,\mathrm{{GHz}}$) bands
from the joint scattering fit and put on an absolute scale with each band's flux
calibration. $\gamma_C,\gamma_D$ are the per-band spectral indices;
$E_{{\mathrm{{iso}}}}$ includes the $(1+z)$ k-correction. The quoted $1\sigma$
uncertainty combines the per-band $c_0$ posterior width with the absolute-scale
systematic (SEFD and primary-beam, $\sim\!0.25$ dex CHIME, $\sim\!0.20$ dex DSA),
added in quadrature across the two bands.}}
\label{{tab:burst-energies}}
\begin{{tabular}}{{lccccc}}
\hline\hline
Burst & $z$ & $D_L$ [Mpc] & $\gamma_{{C}}$ & $\gamma_{{D}}$ & $E_{{\mathrm{{iso}}}}$ [erg] \\
\hline
{table_body}
\hline
\end{{tabular}}
\end{{table}}
"""


def _check() -> None:
    # 1. closed-form band integral must match a dense numeric quadrature
    nu1, nu2, ref = 0.4e9, 0.8e9, 0.6e9
    for c0, g in ((2.0, -3.4), (0.18, -4.95), (1.0, -1.0), (3.0, 0.5)):
        nu = np.linspace(nu1, nu2, 200_001)
        num = np.trapezoid(c0 * (nu / ref) ** g, nu)
        ana = band_integral(c0, g, ref, nu1, nu2)
        assert abs(ana - num) / abs(num) < 1e-6, (c0, g, ana, num)

    # 2. oracle: flat spectrum (gamma=0, c0=1) over a known band, scale=1, gives the
    #    analytic energy; (1+z) divides it; doubling the flux scale doubles it.
    i_flat = band_integral(1.0, 0.0, ref, nu1, nu2)
    assert abs(i_flat - (nu2 - nu1)) < 1e-3, i_flat  # int 1 dnu = bandwidth
    e0 = band_energy_erg(i_flat, 1.0, d_l_m=1.0, z=0.0)
    assert abs(e0 - 4.0 * np.pi * (nu2 - nu1) * JY_MS_HZ_TO_SI * J_TO_ERG) < 1e-6 * e0
    assert abs(band_energy_erg(i_flat, 1.0, 1.0, 1.0) - e0 / 2.0) < 1e-9 * e0  # (1+z)=2
    assert abs(band_energy_erg(i_flat, 2.0, 1.0, 0.0) - 2.0 * e0) < 1e-9 * e0  # linear in scale

    # 3. the gate itself, both directions (inject scales; independent of the live config)
    assert not any("E_iso_erg" in r for r in compute(scales={"C": None, "D": 1.0})), (
        "partial calibration leaked an energy"
    )
    cal_rows = compute(scales={"C": 1.0, "D": 1.0})
    assert cal_rows and all("E_iso_erg" in r for r in cal_rows), "calibrated run emitted no energy"
    r = cal_rows[0]
    assert abs(r["E_iso_erg"] - r["E_iso_erg_no_kcorr"] / (1.0 + r["z"])) < 1e-9 * r["E_iso_erg"]
    # and the live config must not be half-set (would silently drop one band's flux)
    live = flux_scales()
    assert (live["C"] is None) == (live["D"] is None), (
        "telescopes.yaml: calibrate both bands or neither"
    )

    # 4. energy trust boundary (ADR-0003/0004; 3-expert panel 2026-06-24): E_iso is
    #    alpha-independent, so a scattering-FAIL fit with PHYSICAL c0/gamma is now
    #    RETAINED (energy gates on its own inputs, not the shared-alpha verdict);
    #    only missing/non-physical c0/gamma are refused.
    flags = load_gate_flags()
    fits = load_joint_params()
    live_fail = {b for b, f in flags.items() if f == "FAIL"}
    assert live_fail, "no live FAIL verdict to exercise the boundary"
    assert live_fail & set(fits), (
        "alpha-FAIL fits with valid c0/gamma must now be retained, not gated on alpha"
    )
    assert all(
        np.isfinite([f["c0_C"], f["gamma_C"], f["c0_D"], f["gamma_D"]]).all()
        and f["c0_C"] > 0
        and f["c0_D"] > 0
        for f in fits.values()
    ), "a kept fit has non-physical c0/gamma"
    print(
        "self-check OK: integral matches quadrature; energy oracle, k-correction, "
        "calibration gate, and the alpha-independent c0/gamma boundary all hold"
    )


def _provenance(rows: list[dict]) -> dict:
    """Audit record of WHAT produced this energy table, so a stale/wrong input can be
    caught (the sci-python review found the table previously stamped none). git_sha is
    best-effort HEAD at generation; git_dirty flags an uncommitted producing tree, and
    the per-input sha256 census pins reproducibility independent of the commit."""
    import hashlib
    import subprocess

    def _git(*args) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return ""

    sha = _git("rev-parse", "--short", "HEAD") or "unknown"
    inputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        for p in sorted(JOINT_DIR.glob("*_joint_fit.json"))
        + sorted(JOINT_DIR.glob("*_joint_gate.json"))
    }
    return {
        "git_sha": sha,
        "git_dirty": bool(_git("status", "--porcelain")),
        "n_bursts": len(rows),
        "gate_policy": (
            "E_iso is alpha-independent; energy gates on spec-z + calibrated fluence + physical "
            "(finite, c0>0) per-band c0/gamma. The scattering joint-fit FAIL verdict is "
            "INFORMATIONAL, not an energy exclusion (ADR-0003/0004, 3-expert panel 2026-06-24)."
        ),
        "c0gamma_pbf": {r["burst"]: r.get("c0gamma_pbf") for r in rows},
        "quality_flag": {r["burst"]: r.get("quality_flag") for r in rows},
        "inputs_sha256": inputs,
        "note": (
            "git_sha is HEAD at generation; if git_dirty, the producing tree had uncommitted "
            "changes -- trust inputs_sha256 (the consumed joint_fit/gate JSONs) for exact "
            "reproducibility. all-exp campaign fits scattering only (no c0/gamma; verified "
            "local+HPCC 2026-06-24); c0/gamma are from the committed mixed-legacy-PBF joint fits, "
            "which is correct because the energy does not use the PBF alpha."
        ),
    }


def main() -> None:
    if "--check" in sys.argv:
        _check()
        return

    rows = compute()
    if not rows:
        sys.exit("no bursts with both a real redshift and a joint c0/gamma fit")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "burst_energies.json").write_text(json.dumps(rows, indent=2))
    (OUT_DIR / "burst_energies.provenance.json").write_text(json.dumps(_provenance(rows), indent=2))
    calibrated = rows[0]["calibrated"]
    tex = latex_section(rows) if calibrated else latex_pending()
    (OUT_DIR / "burst_energies.tex").write_text(tex)

    print(markdown_table(rows))
    print(
        f"\nN = {len(rows)} bursts.  "
        f"Skipped: placeholder z=1.0 (freya/mahi/johndoeii) and any sightline "
        f"without a joint c0/gamma fit (e.g. casey)."
    )
    if not calibrated:
        print(
            "\n*** UNCALIBRATED: configs/telescopes.yaml has no flux_jy_per_unit for "
            "one or both bands. ***\n"
            "The columns above are band fluence integrals in NATIVE .npy units -- NOT "
            "energies, NOT comparable across bands, NOT to be quoted. Supply "
            "flux_jy_per_unit (Jy per native unit, from each telescope's SEFD + beam "
            "response at the burst position) for BOTH bands to emit E_iso. See "
            "analysis/burst_energies/CALIBRATION_REVIEW.md."
        )
    print(f"\nWrote {OUT_DIR / 'burst_energies.json'}")
    print(
        f"Wrote {OUT_DIR / 'burst_energies.tex'} ({'energy table' if calibrated else 'calibration-pending stub'})"
    )


if __name__ == "__main__":
    main()
