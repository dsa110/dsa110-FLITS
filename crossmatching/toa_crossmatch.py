from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from flits.common.constants import K_DM
from flits.common.utils import calculate_dm_timing_error

# Assume these are defined elsewhere in your script
# from baseband_analysis.core.bbdata import BBData
# from baseband_analysis.core.dedispersion import delay_across_the_band


@dataclass(frozen=True)
class ChimeTimingProvenance:
    """Notebook-derived CHIME timing facts used for reproduction."""

    toa_utc_400: str
    toa_unix_400: float | None = None
    baseband_path: str | None = None
    peak_index: int | None = None
    delta_time_s: float | None = None
    center_frequency_mhz: float | None = None

    @property
    def toa_time_400(self) -> Time:
        if self.toa_unix_400 is not None:
            return Time(self.toa_unix_400, format="unix", scale="utc")
        return Time(self.toa_utc_400, format="iso", scale="utc")


@dataclass(frozen=True)
class DsaTimingProvenance:
    """Curated DSA timing plus filterbank facts kept as provenance."""

    dsa_mjd: float
    native_frequency_mhz: float = 1530.0
    reference_frequency_mhz: float = 400.0
    filterbank_path: str | None = None
    filterbank_tstart_mjd: float | None = None
    tsamp_s: float | None = None
    nchans: int | None = None
    fch1_mhz: float | None = None
    foff_mhz: float | None = None

    @property
    def curated_time(self) -> Time:
        return Time(self.dsa_mjd, format="mjd", scale="utc")


@dataclass(frozen=True)
class CrossmatchInput:
    """Minimal notebook facts needed to reproduce one legacy result row."""

    name: str
    chime_id: str
    dm: float
    source_coord: str
    chime: ChimeTimingProvenance
    dsa: DsaTimingProvenance
    dm_uncertainty: float = 0.1
    error_chime_ms: float | None = None
    error_dsa_ms: float | None = None
    fwhm_ms: float | None = None


@dataclass(frozen=True)
class CrossmatchResult:
    """Legacy JSON-shaped result used by existing plotting code."""

    chime_id: str
    dm: float
    fwhm_ms: float | None
    toa_chime_unix_400: float | None
    toa_chime_utc_400: str
    dm_mjd: float
    toa_dsa_utc_400: str
    dm_uncertainty: float
    error_chime_ms: float | None
    error_dsa_ms: float | None
    measured_offset_ms: float
    combined_dm_uncertainty_ms: float | None
    geometric_delay_ms: float

    def to_legacy_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_toa(t0, offset, f_center, DM, f_ref):
    """Compute a time of arrival referenced to a frequency.

    Parameters
    ----------
    t0 : astropy.units.Quantity or astropy.time.Time
        Reference time. If a Quantity, assumed to be seconds since the Unix
        epoch.
    offset : astropy.units.Quantity
        Instrumental or processing offset to add.
    f_center : astropy.units.Quantity
        Central observing frequency in MHz.
    DM : astropy.units.Quantity
        Dispersion measure.
    f_ref : astropy.units.Quantity
        Reference frequency in MHz.

    Returns
    -------
    astropy.time.Time
        Time of arrival referred to ``f_ref``.
    """
    shift = K_DM * DM.value * (1 / f_ref.value**2 - 1 / f_center.value**2) * u.s
    if isinstance(t0, Time):
        return t0 + offset + shift
    toa = t0 + offset + shift
    return Time(toa.to_value(u.s), format="unix", scale="utc")


def compute_geometric_delay(t, src, loc1, loc2):
    """Compute the geometric delay between two observatories.

    Parameters
    ----------
    t : astropy.time.Time
        Time of arrival.
    src : astropy.coordinates.SkyCoord
        Source coordinates.
    loc1, loc2 : astropy.coordinates.EarthLocation
        Observatory locations.

    Returns
    -------
    astropy.units.Quantity
        Geometric delay in milliseconds.
    """
    p1 = loc1.get_gcrs(t).cartesian.xyz
    p2 = loc2.get_gcrs(t).cartesian.xyz
    proj = (p2 - p1).dot(src.cartesian.xyz)
    return (proj / const.c).to(u.ms)


def reproduce_notebook_result(crossmatch: CrossmatchInput) -> CrossmatchResult:
    """Reproduce the current notebook-derived legacy crossmatch result."""

    dm = crossmatch.dm * (u.pc / u.cm**3)
    chime_toa = crossmatch.chime.toa_time_400
    dsa_toa = compute_toa(
        crossmatch.dsa.curated_time,
        0.0 * u.s,
        crossmatch.dsa.native_frequency_mhz * u.MHz,
        dm,
        crossmatch.dsa.reference_frequency_mhz * u.MHz,
    )
    measured_offset_ms = (chime_toa - dsa_toa).to_value(u.ms)

    src = SkyCoord(crossmatch.source_coord, unit=(u.hourangle, u.deg), frame="icrs")
    geometric_delay_ms = compute_geometric_delay(
        chime_toa,
        src,
        EarthLocation.of_site("DRAO"),
        EarthLocation.of_site("OVRO"),
    ).to_value(u.ms)

    combined_dm_uncertainty_ms = None
    if crossmatch.error_chime_ms is not None and crossmatch.error_dsa_ms is not None:
        combined_dm_uncertainty_ms = float(
            np.hypot(crossmatch.error_chime_ms, crossmatch.error_dsa_ms)
        )

    return CrossmatchResult(
        chime_id=crossmatch.chime_id,
        dm=crossmatch.dm,
        fwhm_ms=crossmatch.fwhm_ms,
        toa_chime_unix_400=crossmatch.chime.toa_unix_400,
        toa_chime_utc_400=chime_toa.iso,
        dm_mjd=crossmatch.dsa.dsa_mjd,
        toa_dsa_utc_400=dsa_toa.iso,
        dm_uncertainty=crossmatch.dm_uncertainty,
        error_chime_ms=crossmatch.error_chime_ms,
        error_dsa_ms=crossmatch.error_dsa_ms,
        measured_offset_ms=measured_offset_ms,
        combined_dm_uncertainty_ms=combined_dm_uncertainty_ms,
        geometric_delay_ms=geometric_delay_ms,
    )


def crossmatch_input_from_dict(row: dict[str, Any]) -> CrossmatchInput:
    """Build a reproduction input from the compact notebook fixture."""

    chime = ChimeTimingProvenance(**row["chime"])
    dsa = DsaTimingProvenance(**row["dsa"])
    values = {key: value for key, value in row.items() if key not in {"chime", "dsa"}}
    return CrossmatchInput(chime=chime, dsa=dsa, **values)


def main():
    logging.basicConfig(level=logging.INFO)

    # --- Input Parameters for the Single Burst ---
    dm_opt = 550.0  # pc/cm^3
    dm_uncertainty = 0.2  # pc/cm^3
    dsa_mjd = 59000.1
    chime_unix_timestamp = 1598882400.0  # Example Unix time
    source_coord = "12:00:00 +20:00:00"

    logging.info("--- Analyzing Single Burst ---")

    # ==================================================================
    # This section would contain your CHIME data processing code
    # to derive peak_idx_chime, etc.
    # For this example, we'll use placeholder values.
    # ==================================================================
    DM = dm_opt * (u.pc) / (u.cm**3)
    # Mocking CHIME results
    t0_unix_chime = chime_unix_timestamp * u.s
    offset_chime = 0.01 * u.s

    # CHIME frequency setup
    # Common reference frequency for all TOAs
    F_REF = 400.0 * u.MHz
    # Representative central frequency for CHIME's band (400.39 - 800.39 MHz)
    f_center_chime = 600.39 * u.MHz

    # Your TOA calculation for CHIME
    toa_400_utc_chime = compute_toa(t0_unix_chime, offset_chime, f_center_chime, DM, F_REF)

    # ==================================================================
    # This section would contain your DSA-110 data processing code
    # ==================================================================
    # Mocking DSA-110 results
    t0_utc_dsa = Time(dsa_mjd, format="mjd", scale="utc")
    offset_dsa = 0.005 * u.s

    # DSA-110 frequency setup
    # Representative central frequency for DSA-110's band (1311.25 - 1498.75 MHz)
    f_center_dsa = 1405.0 * u.MHz

    # Your TOA calculation for DSA-110
    toa_400_utc_dsa = compute_toa(t0_utc_dsa, offset_dsa, f_center_dsa, DM, F_REF)

    # --- UNCERTAINTY CALCULATION ---
    logging.info("Assumed DM Uncertainty: %.2f pc/cm^3", dm_uncertainty)

    # Calculate timing error for each observatory relative to the 400 MHz reference
    error_chime = calculate_dm_timing_error(dm_uncertainty, f_center_chime, F_REF)
    error_dsa = calculate_dm_timing_error(dm_uncertainty, f_center_dsa, F_REF)

    # The total uncertainty on the offset is the sum in quadrature
    delta_t_uncertainty = np.sqrt(error_chime**2 + error_dsa**2)

    logging.info("CHIME TOA Error due to DM uncertainty: %s", error_chime)
    logging.info("DSA-110 TOA Error due to DM uncertainty: %s", error_dsa)

    # --- Final Results ---
    dt = toa_400_utc_chime - toa_400_utc_dsa
    logging.info("Measured TOA Offset (Δt): %s", dt.to(u.ms))
    logging.info("Combined Uncertainty on Δt from DM: ±%s", delta_t_uncertainty)

    # Geometric delay calculation
    src = SkyCoord(source_coord, unit=(u.hourangle, u.deg), frame="icrs")
    chime_loc = EarthLocation(lat=49.3206 * u.deg, lon=-119.6236 * u.deg, height=545 * u.m)
    dsa_loc = EarthLocation(lat=37.2333 * u.deg, lon=-118.2834 * u.deg, height=1222 * u.m)
    geom_delay = compute_geometric_delay(toa_400_utc_chime, src, chime_loc, dsa_loc)
    logging.info("Geometric Delay: %s", geom_delay)


if __name__ == "__main__":
    main()
