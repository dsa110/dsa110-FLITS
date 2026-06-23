#!/usr/bin/env python
"""CHIME primary-beam gain + radiometer noise (documented cylinder approximation).

The full CHIME beam model lives in the private `ch_util`/CHIME beam-model package on the
baseband container (h17), which is not reachable from this analysis env (not pip-installable;
`import ch_util` fails on h17's default env; the local clones are partial). Per
docs/rse/specs/plan-radiometer-flux-cal.md Phase 6, we fall back to a DOCUMENTED separable-Gaussian
cylinder beam anchored to the CHIME/FRB system paper (Amiri et al. 2018, ApJ 863:48, Table 1), with
the approximation error stated.

Geometry (CHIME is a meridian transit instrument at latitude 49.32 N):
  - E-W (across the 20 m cylinder): primary-beam FWHM = 2.5 deg at 400 MHz, 1.3 deg at 800 MHz
    (Table 1 "E-W FoV"), i.e. ~ 1/nu diffraction scaling. At meridian transit (HA=0) the source
    sits at the E-W beam centre, so this term is ~1 for a baseband-detected burst.
  - N-S (along the cylinder focal line): the 1024 FFT-formed beams tile the sky; the synthesized
    beam FWHM = 40' at 400 MHz, 20' at 800 MHz (Table 1 "Beam width"), again ~ 1/nu. A baseband
    burst is beamformed AT its localized position (the singlebeam product is the tied-array beam at
    tiedbeam_locations), so this term is ~1 too.

Hence for our baseband-localized bursts G_CHIME ~ 1 at the source: unlike DSA (fixed transit
pointing, source offset up to ~2.6 deg), CHIME re-points its formed beam onto the burst. The
function still models the chromatic falloff for any off-centre query (and the boresight=1 /
falls-off-axis test). The residual primary-envelope term we do NOT model (the broad feed-element
pattern vs zenith angle, and Tsys/A_eff vs declination and frequency) is folded into the stated
absolute-scale systematic (~0.25 dex), consistent with CHIME/FRB Catalog 1 treating beam-model
fluences as accurate only to a factor of a few.
"""

from __future__ import annotations

import numpy as np

from analysis.flux_cal import radiometer_sigma_jy

CHIME_LAT_DEG = 49.3207  # 49 deg 19' 14.52" N (Amiri+2018 Table 1)
FWHM_EW_400 = 2.5  # E-W primary-beam FWHM at 400 MHz [deg] (Table 1 "E-W FoV" 2.5->1.3)
FWHM_NS_400 = 40.0 / 60.0  # N-S formed-beam FWHM at 400 MHz [deg] (Table 1 "Beam width" 40'->20')
_K = 4.0 * np.log(2.0)  # Gaussian: G = exp(-K (offset/FWHM)^2) -> 0.5 at offset = FWHM/2


def _fwhm_deg(fwhm_400, freq_mhz):
    """Chromatic FWHM [deg]: diffraction ~1/nu, pinned to the 400 MHz value (Table 1)."""
    return fwhm_400 * 400.0 / freq_mhz


def beam_gain(ra_deg, dec_deg, freq_mhz, *, ra0_deg=None, dec0_deg=None):
    """Normalized CHIME beam gain (boresight=1) at (ra,dec) for a beam formed at (ra0,dec0).

    Separable Gaussian: E-W in (ra-ra0) cos(dec0), N-S in (dec-dec0), with the chromatic FWHMs
    above. Defaults ra0/dec0 to (ra,dec) -> the source sits at its own formed-beam centre -> G=1
    (the baseband case). For a transit source the E-W offset is the hour-angle term; pass ra0=ra to
    drop it. Error: a Gaussian ignores sidelobes and the broad N-S element envelope; good near the
    main lobe (the regime that matters for a beam-centred burst), degrading past ~1 FWHM.
    """
    ra0 = ra_deg if ra0_deg is None else ra0_deg
    dec0 = dec_deg if dec0_deg is None else dec0_deg
    d_ew = (ra_deg - ra0) * np.cos(np.radians(dec0))  # E-W angular offset [deg]
    d_ns = dec_deg - dec0  # N-S angular offset [deg]
    g_ew = np.exp(-_K * (d_ew / _fwhm_deg(FWHM_EW_400, freq_mhz)) ** 2)
    g_ns = np.exp(-_K * (d_ns / _fwhm_deg(FWHM_NS_400, freq_mhz)) ** 2)
    return float(g_ew * g_ns)


def chime_sigma_jy(freq_hz, dnu_hz, sefd_jy, dt_s, g=1.0):
    """Per-channel radiometer noise sigma_S(nu) [Jy] for CHIME (n_pol=2).

    g is the beam gain at the source (default 1.0: baseband burst formed at its position). Pass a
    scalar or per-channel array from beam_gain(...) for an off-centre source. sefd_jy may be a
    scalar (frequency-flat documented SEFD) or per-channel.
    """
    return radiometer_sigma_jy(sefd_jy, 2, dnu_hz, dt_s, g)


# Documented zenith SEFD from Amiri+2018 Table 1: SEFD = 2 k_B Tsys / A_eff.
#   2 k_B = 2761.3 Jy m^2 / K; Tsys = 50 K (receiver noise, Table 1); A_eff = eta * 8000 m^2
#   (physical collecting area, Table 1) with aperture efficiency eta = 0.5 (typical cylinder).
#   -> SEFD_zenith ~ 2761.3 * 50 / (0.5 * 8000) = 34.5 Jy.
# Systematic: eta in [0.4,0.6] and real Tsys (sky+ground) up to ~80 K give SEFD ~29-69 Jy (~0.25
# dex); the declination (zenith-angle, |dec-49.32|<~25 deg for these bursts) and band-edge Tsys/A_eff
# dependence are also inside this band -- hence the frequency-flat constant.
TWO_KB_JY_M2_PER_K = 2.0 * 1.380649e-23 / 1e-26  # = 2761.298 Jy m^2 / K


def sefd_zenith_jy(tsys_k=50.0, a_phys_m2=8000.0, eta=0.5):
    """CHIME zenith SEFD [Jy] = 2 k_B Tsys / (eta * A_phys). See the note above for provenance."""
    return TWO_KB_JY_M2_PER_K * tsys_k / (eta * a_phys_m2)


def load_chime_sefd(nick=None):
    """Documented CHIME SEFD [Jy] from analysis/burst_energies/chime_sefd.csv (single row).

    nick is accepted for call-site symmetry with load_dsa_sefd but ignored: CHIME re-points onto
    each burst, so a single zenith value (with the systematic above) is used for all.
    """
    import csv
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "analysis" / "burst_energies" / "chime_sefd.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run the Phase 6 derivation")
    return float(next(csv.DictReader(p.open()))["sefd_jy"])


def _check() -> None:
    # 1. boresight (source at its own beam centre) = 1 at any frequency
    g0 = beam_gain(120.0, 45.0, 600.0)
    assert abs(g0 - 1.0) < 1e-12, g0
    # 2. half power at N-S offset = FWHM/2 (exp(-K*(1/2)^2) = exp(-ln2) = 0.5)
    fwhm_ns = _fwhm_deg(FWHM_NS_400, 600.0)
    g_hp = beam_gain(120.0, 45.0 + fwhm_ns / 2, 600.0, ra0_deg=120.0, dec0_deg=45.0)
    assert abs(g_hp - 0.5) < 1e-6, (g_hp, fwhm_ns)
    # 3. falls further off-axis, and is chromatic (narrower -> lower gain at higher freq)
    g_far = beam_gain(120.0, 45.0 + fwhm_ns, 600.0, ra0_deg=120.0, dec0_deg=45.0)
    assert g_far < g_hp, (g_far, g_hp)
    g_hi = beam_gain(120.0, 45.3, 800.0, ra0_deg=120.0, dec0_deg=45.0)
    g_lo = beam_gain(120.0, 45.3, 400.0, ra0_deg=120.0, dec0_deg=45.0)
    assert g_hi < g_lo, ("beam not chromatic", g_hi, g_lo)
    # 4. SEFD derivation matches the documented ~34.5 Jy
    s = sefd_zenith_jy()
    assert abs(s - 34.5) < 0.5, s
    print(
        f"self-check OK: boresight=1, half-power at {fwhm_ns / 2:.3f} deg (600 MHz), "
        f"chromatic (g_800={g_hi:.2f} < g_400={g_lo:.2f}); SEFD_zenith={s:.1f} Jy"
    )


if __name__ == "__main__":
    _check()
