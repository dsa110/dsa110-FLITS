from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from flits.common.constants import K_DM
from flits.common.utils import calculate_dm_timing_error

# Observatory locations for the CHIME (DRAO) and DSA-110 (OVRO) sites. These
# are held as explicit literals rather than EarthLocation.of_site(...) so the
# crossmatch and its tests are deterministic and do not depend on astropy's
# remote site registry (data.astropy.org). The values match the canonical
# coordinates used throughout the pipeline; a site-position uncertainty of even
# 1 km projects to < 3e-3 ms of geometric delay, far below any reported
# precision and below the 1e-2 ms tolerance of the reproduction test.
DRAO_LOCATION = EarthLocation(lat=49.3206 * u.deg, lon=-119.6236 * u.deg, height=545 * u.m)
OVRO_LOCATION = EarthLocation(lat=37.2333 * u.deg, lon=-118.2834 * u.deg, height=1222 * u.m)

# Assume these are defined elsewhere in your script
# from baseband_analysis.core.bbdata import BBData
# from baseband_analysis.core.dedispersion import delay_across_the_band


@dataclass(frozen=True)
class ChimeTimingProvenance:
    """Notebook-derived CHIME timing facts used for reproduction."""

    toa_utc_400: str
    toa_unix_400: float | None = None
    baseband_path: str | None = None
    baseband_vospace_uri: str | None = None
    baseband_verified_exists: bool | None = None
    baseband_vls_listing: str | None = None
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
    # Joint-fit model scatter corrections (peak-shift, ms, >= 0) per band. When
    # both are supplied, reproduce_model_result() subtracts them from the
    # peak-based ToA to yield the intrinsic (de-scattered) model arrival.
    scatter_corr_chime_ms: float | None = None
    scatter_corr_dsa_ms: float | None = None
    # Posterior localization error (t0 half-width, ms) per band, folded into
    # the model error budget in quadrature with the DM-referral term.
    loc_err_chime_ms: float | None = None
    loc_err_dsa_ms: float | None = None
    # Uncertainty ON the scatter correction, propagated from the tau/beta
    # posteriors (ms, per band). Folded into the per-band arrival error.
    scatter_corr_unc_chime_ms: float | None = None
    scatter_corr_unc_dsa_ms: float | None = None
    # Multi-component ToA ambiguity (ms, per band): flux-weighted spread of the
    # candidate component-t0 arrivals, weighted by the observed profile
    # amplitude at each component. Zero for single-component bursts. Folded into
    # the per-band arrival error.
    comp_ambig_chime_ms: float | None = None
    comp_ambig_dsa_ms: float | None = None
    # Inter-site geometric-delay error from source localization (ms). Projects
    # the position uncertainty onto the DRAO-OVRO baseline; folded once into the
    # combined offset error (it is not a per-band term).
    geo_pos_err_ms: float | None = None


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
    # Model (scatter-corrected) extension. Populated by reproduce_model_result;
    # left None by the legacy reproduce_notebook_result so to_legacy_dict is
    # byte-for-byte unchanged.
    scatter_corr_chime_ms: float | None = None
    scatter_corr_dsa_ms: float | None = None
    differential_scatter_shift_ms: float | None = None
    peak_measured_offset_ms: float | None = None
    loc_err_chime_ms: float | None = None
    loc_err_dsa_ms: float | None = None
    combined_error_ms: float | None = None
    # Extended error budget. error_chime_ms/error_dsa_ms above carry the
    # DM-referral + t0-localization terms; these add the scatter-correction
    # uncertainty and the multi-component ambiguity per band, plus the
    # inter-site geometric-position term, into combined_error_full_ms.
    scatter_corr_unc_chime_ms: float | None = None
    scatter_corr_unc_dsa_ms: float | None = None
    comp_ambig_chime_ms: float | None = None
    comp_ambig_dsa_ms: float | None = None
    geo_pos_err_ms: float | None = None
    error_chime_full_ms: float | None = None
    error_dsa_full_ms: float | None = None
    combined_error_full_ms: float | None = None

    _LEGACY_KEYS: ClassVar[tuple[str, ...]] = (
        "chime_id", "dm", "fwhm_ms", "toa_chime_unix_400", "toa_chime_utc_400",
        "dm_mjd", "toa_dsa_utc_400", "dm_uncertainty", "error_chime_ms",
        "error_dsa_ms", "measured_offset_ms", "combined_dm_uncertainty_ms",
        "geometric_delay_ms",
    )

    def to_legacy_dict(self) -> dict[str, Any]:
        """Legacy 13-key row; identical to the pre-model schema."""
        d = asdict(self)
        return {k: d[k] for k in self._LEGACY_KEYS}

    def to_model_dict(self) -> dict[str, Any]:
        """Full row including the model scatter-correction extension."""
        return asdict(self)


def compute_toa(t0, offset, f_center, DM, f_ref, scatter_correction=0.0 * u.s):
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
    scatter_correction : astropy.units.Quantity, optional
        Scattering peak-shift for this band (>= 0). The peak of a scattered
        profile lags the intrinsic arrival by this amount, so it is
        *subtracted* to recover the model (de-scattered) time of arrival. The
        default of zero reproduces the legacy peak-based ToA. See
        :func:`reproduce_model_result` and the joint-fit scatter-correction
        table for how the per-band value is derived.

    Returns
    -------
    astropy.time.Time
        Time of arrival referred to ``f_ref``.
    """
    shift = K_DM * DM.value * (1 / f_ref.value**2 - 1 / f_center.value**2) * u.s
    if isinstance(t0, Time):
        return t0 + offset + shift - scatter_correction
    toa = t0 + offset + shift - scatter_correction
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
        DRAO_LOCATION,
        OVRO_LOCATION,
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


def reproduce_model_result(crossmatch: CrossmatchInput) -> CrossmatchResult:
    """Model (scatter-corrected) crossmatch result.

    Identical to :func:`reproduce_notebook_result` except that the per-band
    joint-fit scattering peak-shift is subtracted from each band's ToA, so the
    reported ``measured_offset_ms`` is the intrinsic (de-scattered) CHIME-DSA
    offset at 400 MHz. The legacy peak-based offset is retained as
    ``peak_measured_offset_ms``. The error budget folds the posterior
    localization error into the DM-referral term in quadrature per band.

    Requires ``scatter_corr_chime_ms``/``scatter_corr_dsa_ms`` on the input; if
    absent it falls back to the legacy (zero-correction) result so callers can
    use it uniformly.
    """
    if (
        crossmatch.scatter_corr_chime_ms is None
        or crossmatch.scatter_corr_dsa_ms is None
    ):
        return reproduce_notebook_result(crossmatch)

    dm = crossmatch.dm * (u.pc / u.cm**3)
    scat_chime = crossmatch.scatter_corr_chime_ms * u.ms
    scat_dsa = crossmatch.scatter_corr_dsa_ms * u.ms

    # CHIME ToA is stored provenance already referred to 400 MHz; subtract its
    # scattering peak-shift directly to recover the intrinsic arrival.
    chime_toa = crossmatch.chime.toa_time_400 - scat_chime
    dsa_toa = compute_toa(
        crossmatch.dsa.curated_time,
        0.0 * u.s,
        crossmatch.dsa.native_frequency_mhz * u.MHz,
        dm,
        crossmatch.dsa.reference_frequency_mhz * u.MHz,
        scatter_correction=scat_dsa,
    )
    measured_offset_ms = (chime_toa - dsa_toa).to_value(u.ms)
    peak_measured_offset_ms = reproduce_notebook_result(crossmatch).measured_offset_ms
    differential_scatter_shift_ms = float(
        crossmatch.scatter_corr_chime_ms - crossmatch.scatter_corr_dsa_ms
    )

    src = SkyCoord(crossmatch.source_coord, unit=(u.hourangle, u.deg), frame="icrs")
    geometric_delay_ms = compute_geometric_delay(
        chime_toa,
        src,
        DRAO_LOCATION,
        OVRO_LOCATION,
    ).to_value(u.ms)

    # Error budget: fold posterior localization into the DM-referral term.
    err_chime = crossmatch.error_chime_ms
    err_dsa = crossmatch.error_dsa_ms
    loc_chime = crossmatch.loc_err_chime_ms
    loc_dsa = crossmatch.loc_err_dsa_ms
    err_chime_model = (
        float(np.hypot(err_chime, loc_chime))
        if err_chime is not None and loc_chime is not None
        else err_chime
    )
    err_dsa_model = (
        float(np.hypot(err_dsa, loc_dsa))
        if err_dsa is not None and loc_dsa is not None
        else err_dsa
    )
    combined_error_ms = None
    if err_chime_model is not None and err_dsa_model is not None:
        combined_error_ms = float(np.hypot(err_chime_model, err_dsa_model))
    combined_dm_uncertainty_ms = None
    if err_chime is not None and err_dsa is not None:
        combined_dm_uncertainty_ms = float(np.hypot(err_chime, err_dsa))

    # Extended per-band budget: add the scatter-correction uncertainty and the
    # multi-component ToA ambiguity in quadrature onto the model per-band error.
    def _extend(base, *terms):
        if base is None:
            return None
        acc = base**2
        for term in terms:
            if term is not None:
                acc += term**2
        return float(acc**0.5)

    err_chime_full = _extend(
        err_chime_model,
        crossmatch.scatter_corr_unc_chime_ms,
        crossmatch.comp_ambig_chime_ms,
    )
    err_dsa_full = _extend(
        err_dsa_model,
        crossmatch.scatter_corr_unc_dsa_ms,
        crossmatch.comp_ambig_dsa_ms,
    )
    # Combined offset error folds both full per-band errors and the single
    # inter-site geometric-position term.
    combined_error_full_ms = None
    if err_chime_full is not None and err_dsa_full is not None:
        combined_error_full_ms = _extend(
            float(np.hypot(err_chime_full, err_dsa_full)),
            crossmatch.geo_pos_err_ms,
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
        error_chime_ms=err_chime_model,
        error_dsa_ms=err_dsa_model,
        measured_offset_ms=measured_offset_ms,
        combined_dm_uncertainty_ms=combined_dm_uncertainty_ms,
        geometric_delay_ms=geometric_delay_ms,
        scatter_corr_chime_ms=crossmatch.scatter_corr_chime_ms,
        scatter_corr_dsa_ms=crossmatch.scatter_corr_dsa_ms,
        differential_scatter_shift_ms=differential_scatter_shift_ms,
        peak_measured_offset_ms=peak_measured_offset_ms,
        loc_err_chime_ms=loc_chime,
        loc_err_dsa_ms=loc_dsa,
        combined_error_ms=combined_error_ms,
        scatter_corr_unc_chime_ms=crossmatch.scatter_corr_unc_chime_ms,
        scatter_corr_unc_dsa_ms=crossmatch.scatter_corr_unc_dsa_ms,
        comp_ambig_chime_ms=crossmatch.comp_ambig_chime_ms,
        comp_ambig_dsa_ms=crossmatch.comp_ambig_dsa_ms,
        geo_pos_err_ms=crossmatch.geo_pos_err_ms,
        error_chime_full_ms=err_chime_full,
        error_dsa_full_ms=err_dsa_full,
        combined_error_full_ms=combined_error_full_ms,
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
