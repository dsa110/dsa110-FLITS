"""
io_adapters.py
=============

Map **committed dsa110-FLITS result files** into the figure dataclasses in this
package, so the paper figures can be driven by real pipeline outputs rather than
synthetic demos.

Currently wired (files that live in the repo):
- ``sightline_from_results(burst)`` : ``results/search_summary.csv`` +
  ``results/<burst>_galaxies.csv``  ->  :class:`sightline_plots.Sightline`.
- ``scattering_table(...)``         : ``scattering/dsa_diagnostics/dsa_fitting_summary.json``
  ->  ``{burst: {tau_1ghz, alpha, quality, ...}}``.

Documented assumptions (override as needed)
------------------------------------------
- Foreground filter keeps galaxies with ``0 < z_gal < z_frb`` (set
  ``foreground_only=False`` to keep all; useful when z_frb is a placeholder).
- Halo mass: from ``M_star`` via a Moster+2010 (z=0) stellar-to-halo relation
  when available; else estimated from the i-band magnitude (M/L=1, no
  k-correction); else a fiducial value. Each halo records how it was derived.
- R_200 and the per-halo DM are then filled by :mod:`dm_models` inside the plot.
"""
from __future__ import annotations

import csv
import json
import os
import re
from typing import Optional

import numpy as np

try:
    from .dm_models import Cosmology, ModifiedNFW
    from .sightline_plots import ForegroundHalo, Sightline
except ImportError:  # plain-script execution
    from dm_models import Cosmology, ModifiedNFW
    from sightline_plots import ForegroundHalo, Sightline

__all__ = ["sightline_from_results", "scattering_table", "halo_mass_from_stellar"]

_I_SUN_ABS = 4.58  # i-band absolute magnitude of the Sun


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_ra_hms(s):
    m = re.search(r"(\d+)h(\d+)m([\d.]+)s", s or "")
    if not m:
        return None
    h, mi, se = map(float, m.groups())
    return 15.0 * (h + mi / 60 + se / 3600)


def _parse_dec_dms(s):
    m = re.search(r"([+-]?)(\d+)d(\d+)m([\d.]+)s", s or "")
    if not m:
        return None
    sgn = -1.0 if m.group(1) == "-" else 1.0
    d, mi, se = float(m.group(2)), float(m.group(3)), float(m.group(4))
    return sgn * (d + mi / 60 + se / 3600)


# --- Moster, Naab & White (2010) z=0 stellar-to-halo mass, numerically inverted ---
def _mstar_of_mhalo(Mh):
    M1, ratio, beta, gamma = 10 ** 11.884, 0.02820, 1.057, 0.556
    return Mh * 2 * ratio / ((Mh / M1) ** -beta + (Mh / M1) ** gamma)


_MH_GRID = np.logspace(10.0, 15.3, 500)
_MS_GRID = _mstar_of_mhalo(_MH_GRID)


def halo_mass_from_stellar(mstar_msun: float) -> float:
    """Halo mass [Msun] for a given stellar mass via Moster+2010 (z=0), inverted."""
    return float(10 ** np.interp(np.log10(mstar_msun), np.log10(_MS_GRID), np.log10(_MH_GRID)))


def _galaxy_halo_mass(g: dict, z: float, cosmo: Cosmology, fiducial: float):
    """(M_halo, provenance) for one galaxy row, trying M_star -> i-mag -> fiducial."""
    ms = _to_float(g.get("M_star"))
    if ms is not None:
        mstar = 10 ** ms if ms < 20 else ms      # column is log10 in GLADE export
        return halo_mass_from_stellar(mstar), "M_star"
    imag = _to_float(g.get("imag"))
    if imag is not None and z > 0:
        dL_mpc = _to_float(g.get("d_L")) or cosmo.luminosity_distance_cm(z) / 3.0857e24
        M_i = imag - 5 * np.log10(dL_mpc * 1e6 / 10.0)
        log_mstar = 0.4 * (_I_SUN_ABS - M_i)     # M/L_i = 1, no k-correction
        return halo_mass_from_stellar(10 ** log_mstar), "i-mag"
    return fiducial, "fiducial"


def sightline_from_results(burst: str, results_dir: str = "results",
                           cosmo: Optional[Cosmology] = None,
                           dm_mw: float = 55.0, dm_host_obs: float = 50.0,
                           fiducial_halo_msun: float = 3e11,
                           foreground_only: bool = True, verbose: bool = False) -> Sightline:
    """Build a :class:`Sightline` for ``burst`` from the committed result CSVs."""
    cosmo = cosmo or Cosmology()
    key = burst.strip().lower()
    summary = os.path.join(results_dir, "search_summary.csv")
    row = None
    with open(summary) as f:
        for r in csv.DictReader(f):
            if (r.get("name") or "").strip().lower() == key:
                row = r; break
    if row is None:
        raise KeyError(f"burst '{burst}' not found in {summary}")
    z_frb = float(row["z_frb"])
    ra_frb, dec_frb = _parse_ra_hms(row.get("ra")), _parse_dec_dms(row.get("dec"))

    halos = []
    gpath = os.path.join(results_dir, f"{key}_galaxies.csv")
    if os.path.exists(gpath):
        with open(gpath) as f:
            for g in csv.DictReader(f):
                z = _to_float(g.get("z") or g.get("z_cmb") or g.get("z_helio"))
                b = _to_float(g.get("impact_kpc"))
                if z is None or b is None:
                    continue
                if foreground_only and not (0 < z < z_frb):
                    continue
                Mh, prov = _galaxy_halo_mass(g, z, cosmo, fiducial_halo_msun)
                ra, dec = _to_float(g.get("ra")), _to_float(g.get("dec"))
                dra = ddec = None
                if None not in (ra, dec, ra_frb, dec_frb):
                    dra = (ra - ra_frb) * np.cos(np.radians(dec_frb)) * 60.0
                    ddec = (dec - dec_frb) * 60.0
                halos.append(ForegroundHalo(z=z, mass_msun=Mh, impact_kpc=b,
                             dra_arcmin=dra, ddec_arcmin=ddec, is_cluster=Mh > 1e14,
                             label=str(g.get("id", ""))[:4] or None))
                if verbose:
                    print(f"  {key}: z={z:.3f} b={b:.1f}kpc logMh={np.log10(Mh):.2f} ({prov})")
    return Sightline(z_frb=z_frb, halos=halos, dm_mw=dm_mw, dm_host_obs=dm_host_obs)


def scattering_table(summary_path: str = "scattering/dsa_diagnostics/dsa_fitting_summary.json") -> dict:
    """Read the per-burst DSA scattering fit summary into {burst_name: {...}}."""
    with open(summary_path) as f:
        data = json.load(f)
    return {d.get("burst_name", str(i)): d for i, d in enumerate(data)}


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    rdir = os.path.join(repo, "results")
    # rank bursts by foreground-halo count
    best, best_n = None, -1
    import glob as _glob
    for gp in _glob.glob(os.path.join(rdir, "*_galaxies.csv")):
        name = os.path.basename(gp).replace("_galaxies.csv", "")
        try:
            sl = sightline_from_results(name, rdir)
        except Exception:
            continue
        if len(sl.halos) > best_n:
            best, best_n = name, len(sl.halos)
    print(f"richest real foreground sightline: {best}  ({best_n} halos)")
    sl = sightline_from_results(best, rdir, verbose=True)
    try:
        from .sightline_plots import plot_sightline
    except ImportError:
        from sightline_plots import plot_sightline
    fig = plot_sightline(sl)
    fig.suptitle(f"Real sightline: {best} (committed results)  z_frb={sl.z_frb}", fontsize=10)
    fig.savefig(f"sightline_real_{best}.png", bbox_inches="tight", dpi=150)
    print(f"wrote sightline_real_{best}.png")
    st = scattering_table(os.path.join(repo, "scattering/dsa_diagnostics/dsa_fitting_summary.json"))
    print("scattering_table: %d bursts; e.g. whitney tau_1ghz=%.2f ms (%s)"
          % (len(st), st["whitney"]["tau_1ghz"], st["whitney"]["quality_highres"]))
