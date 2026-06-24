"""NE2025 Galactic scattering floor wiring for the scintillation pipeline.

Thin wrapper over ``scintillation.ne2025.query_ne2025_scint.galactic_floor`` that
attaches the Milky-Way scattering floor + an extragalactic-excess flag to each
scintillation component. The floor is the Galactic-vs-extragalactic discriminator:
a measured decorrelation bandwidth *below* the MW floor means more scattering than
the Galaxy provides, i.e. a host/intervening (extragalactic) screen.

``query_ne2025_scint`` imports the optional ``mwprop`` package at module load
(``model="ne2025"``) and ``galactic_floor`` falls back to ``pygedm`` for
ne2001/ymw16; both are optional macOS-manual-build deps. So the import is lazy and
every failure path is a clean no-op (``galactic_floor=None``), never a hard error.
"""

import importlib
import logging

import numpy as np

log = logging.getLogger(__name__)


def extragalactic_excess(measured_bw_mhz, floor_bw_khz):
    """Flag a sub-Galactic decorrelation bandwidth.

    True when the measured Δν (MHz) is below the MW floor Δν (kHz) -> excess
    scattering -> extragalactic screen. None if either input is not a usable
    positive number (so the caller omits the flag rather than asserting False).
    """
    try:
        mb_khz = float(measured_bw_mhz) * 1e3  # MHz -> kHz to match the floor
        fb_khz = float(floor_bw_khz)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(mb_khz) and np.isfinite(fb_khz) and mb_khz > 0 and fb_khz > 0):
        return None
    return bool(mb_khz < fb_khz)


def attach_galactic_floor(comp, coord, bands=None, alpha=4.4, model="ne2025"):
    """Attach the MW floor + extragalactic-excess flag to one component, in place.

    ``coord`` is an astropy ``SkyCoord``. The measured Δν is the median of the
    component's ``subband_measurements`` ``bw`` (MHz), compared against the floor at
    the band whose centre is nearest the median measurement frequency. On any floor
    failure (optional dep absent, out-of-Galaxy sightline) sets
    ``comp['galactic_floor'] = None`` and omits the flag.
    """
    if not isinstance(comp, dict):
        return comp
    try:
        q = importlib.import_module("scintillation.ne2025.query_ne2025_scint")
    except Exception as e:  # optional mwprop/pygedm dep absent
        log.warning(f"NE2025 floor unavailable ({e}); skipping galactic_floor.")
        comp["galactic_floor"] = None
        return comp

    bands = q.BAND_CENTERS_MHZ if bands is None else bands
    try:
        floor = q.galactic_floor(coord, bands, alpha=alpha, model=model)
    except Exception as e:  # e.g. out-of-Galaxy sightline, pygedm failure
        log.warning(f"galactic_floor() failed: {e}")
        comp["galactic_floor"] = None
        return comp

    comp["galactic_floor"] = floor

    def _finite(v):
        try:
            return np.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    meas = comp.get("subband_measurements") or []
    pairs = [
        (float(sm["freq_mhz"]), float(sm["bw"]))
        for sm in meas
        if _finite(sm.get("freq_mhz")) and _finite(sm.get("bw"))
    ]
    if not pairs:
        return comp
    freqs, bws = zip(*pairs, strict=True)
    med_freq = float(np.nanmedian(np.array(freqs, dtype=float)))
    med_bw = float(np.nanmedian(np.array(bws, dtype=float)))
    band = min(bands, key=lambda b: abs(float(bands[b]) - med_freq))  # nearest band
    flag = extragalactic_excess(med_bw, floor[band]["bw_kHz"])
    if flag is not None:
        comp["extragalactic_excess"] = flag
    return comp


def attach_galactic_floor_all(final_results, ra_deg, dec_deg, **kw):
    """Pipeline helper: build the burst SkyCoord once and attach the floor to every
    component of an ``analyze_scintillation_from_acfs`` result. No-op if astropy is
    unavailable or coords are not finite."""
    if not isinstance(final_results, dict):
        return final_results
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        coord = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    except Exception as e:
        log.warning(f"Could not build burst SkyCoord ({e}); skipping galactic_floor.")
        return final_results
    for comp in final_results.get("components", {}).values():
        attach_galactic_floor(comp, coord, **kw)
    return final_results
