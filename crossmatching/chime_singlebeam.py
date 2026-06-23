from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.time import Time
from scipy.signal import savgol_filter

from crossmatching.toa_crossmatch import compute_toa

CHIME_BASEBAND_SAMPLE_TIME_S = 2.56e-6
CHIME_DEFAULT_REFERENCE_FREQUENCY_MHZ = 400.0
_SAVGOL_WINDOW, _SAVGOL_POLYORDER = 9, 3


class SinglebeamExtractionError(RuntimeError):
    """Raised when a singlebeam file lacks the metadata needed for a TOA."""


@dataclass(frozen=True)
class ChimeSinglebeamToa:
    """CHIME 400 MHz TOA extracted from a singlebeam HDF5 file.

    Reproduces ``crossmatching/toa_crossmatch.ipynb`` cell 11: coherent + incoherent
    dedispersion, Stokes-I, frequency-collapse, Savitzky-Golay peak. The 400 MHz
    reference and bottom-channel ``time0``/``f_center`` convention match the notebook,
    so ``toa_unix_400`` is directly comparable to the notebook-derived value.
    """

    path: str
    method: str
    dm: float
    peak_index: int
    peak_offset_s: float
    native_frequency_mhz: float
    reference_frequency_mhz: float
    native_toa_unix: float
    native_toa_utc: str
    toa_unix_400: float
    toa_utc_400: str
    sample_time_s: float
    time0_ctime: float
    time0_ctime_offset: float
    data_shape: tuple[int, ...]
    n_rfi_channels_masked: int
    noise_window_offpulse_frac: float
    peak_snr_like: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def toa_400_from_peak(
    *,
    peak_index: int,
    sample_time_s: float,
    time0_ctime: float,
    time0_ctime_offset: float,
    native_frequency_mhz: float,
    dm: float,
    reference_frequency_mhz: float = CHIME_DEFAULT_REFERENCE_FREQUENCY_MHZ,
) -> tuple[Time, Time]:
    """Convert a peak sample to native and 400 MHz TOAs (notebook convention).

    Pure timing math (no baseband_analysis); shares ``compute_toa`` with the
    crossmatch so the dispersive shift uses the same ``K_DM``.
    """
    native = (
        Time(time0_ctime, val2=time0_ctime_offset, format="unix", scale="utc")
        + (peak_index * sample_time_s) * u.s
    )
    toa_400 = compute_toa(
        native,
        0.0 * u.s,
        native_frequency_mhz * u.MHz,
        dm * (u.pc / u.cm**3),
        reference_frequency_mhz * u.MHz,
    )
    native.precision = 9
    toa_400.precision = 9
    return native, toa_400


def extract_singlebeam_toa(
    path: str | Path,
    *,
    dm: float,
    reference_frequency_mhz: float = CHIME_DEFAULT_REFERENCE_FREQUENCY_MHZ,
    sample_time_s: float | None = None,
) -> ChimeSinglebeamToa:
    """Extract the CHIME 400 MHz TOA from a singlebeam HDF5 file.

    Requires the CANFAR ``baseband-analysis`` runtime (imported lazily). The
    per-burst noise window and RFI channel mask used by the notebook live in
    ``bursts.fits``; absent that, an off-pulse noise window and a mild
    zero-variance RFI mask are derived here. Bright bursts are insensitive to
    this substitution and reproduce the notebook TOA exactly.
    """
    BBData, coherent_dedisp, incoherent_dedisp = _require_baseband_analysis()

    bb = BBData.from_file(str(path))
    if "time0" not in bb:
        raise SinglebeamExtractionError("missing required time0 dataset: time0")
    delta_time = float(sample_time_s if sample_time_s is not None else bb.attrs["delta_time"])
    freq = np.asarray(bb.index_map["freq"]["centre"], dtype=float)

    bb["tiedbeam_baseband"][:] = coherent_dedisp(bb, float(dm), time_shift=False)
    bb_inc, _freq, _freq_id = incoherent_dedisp(bb, float(dm), fill_wfall=True)

    intensity = np.abs(bb_inc[:, 0, :]) ** 2 + np.abs(bb_inc[:, 1, :]) ** 2
    profile, peak_index, n_rfi, offpulse_frac = _collapse_and_peak(intensity)

    # Notebook convention: bottom channel (index -1, ~400 MHz) sets time0 / f_center.
    t0 = bb["time0"]
    ctime, ctime_offset = float(t0["ctime"][-1]), float(t0["ctime_offset"][-1])
    native_frequency = float(freq[-1])
    native_toa, toa_400 = toa_400_from_peak(
        peak_index=peak_index,
        sample_time_s=delta_time,
        time0_ctime=ctime,
        time0_ctime_offset=ctime_offset,
        native_frequency_mhz=native_frequency,
        dm=dm,
        reference_frequency_mhz=reference_frequency_mhz,
    )

    return ChimeSinglebeamToa(
        path=str(path),
        method="notebook_dedispersed_savgol_peak",
        dm=float(dm),
        peak_index=peak_index,
        peak_offset_s=float(peak_index * delta_time),
        native_frequency_mhz=native_frequency,
        reference_frequency_mhz=float(reference_frequency_mhz),
        native_toa_unix=float(native_toa.to_value("unix")),
        native_toa_utc=native_toa.iso,
        toa_unix_400=float(toa_400.to_value("unix")),
        toa_utc_400=toa_400.iso,
        sample_time_s=delta_time,
        time0_ctime=ctime,
        time0_ctime_offset=ctime_offset,
        data_shape=tuple(int(v) for v in bb["tiedbeam_baseband"].shape),
        n_rfi_channels_masked=n_rfi,
        noise_window_offpulse_frac=offpulse_frac,
        peak_snr_like=_snr_like(profile, peak_index),
    )


def _collapse_and_peak(intensity: np.ndarray) -> tuple[np.ndarray, int, int, float]:
    """Noise-normalise per channel, mask dead channels, collapse, Savgol peak-pick."""
    ntime = intensity.shape[-1]
    coarse = np.nansum(np.nan_to_num(intensity), axis=0)
    coarse_peak = int(np.nanargmax(coarse))

    guard = 4000  # ~10 ms either side of the coarse peak at 2.56 us
    idx = np.arange(ntime)
    noise = (idx < coarse_peak - guard) | (idx > coarse_peak + guard)
    if noise.sum() < ntime * 0.1:
        noise = (idx < int(0.2 * ntime)) | (idx > int(0.8 * ntime))

    mu = np.nanmean(intensity[:, noise], axis=-1)[:, None]
    sd = np.nanstd(intensity[:, noise], axis=-1)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = (intensity - mu) / sd
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

    chan_ok = np.isfinite(sd[:, 0]) & (sd[:, 0] > 0)
    norm = norm * chan_ok[:, None]

    profile = np.nansum(norm, axis=0)
    peak_index = int(np.nanargmax(savgol_filter(profile, _SAVGOL_WINDOW, _SAVGOL_POLYORDER)))
    return profile, peak_index, int((~chan_ok).sum()), float(noise.mean())


def _require_baseband_analysis():
    try:
        from baseband_analysis.core.bbdata import BBData
        from baseband_analysis.core.dedispersion import coherent_dedisp, incoherent_dedisp
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SinglebeamExtractionError(
            "baseband_analysis is required (CANFAR image); not importable in this environment"
        ) from exc
    return BBData, coherent_dedisp, incoherent_dedisp


def _snr_like(timeseries: np.ndarray, peak_index: int) -> float:
    baseline = np.nanmedian(timeseries)
    mad = np.nanmedian(np.abs(timeseries - baseline))
    if not np.isfinite(mad) or mad == 0:
        return float("inf")
    return float((timeseries[peak_index] - baseline) / (1.4826 * mad))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract the CHIME 400 MHz TOA from a singlebeam HDF5 file."
    )
    parser.add_argument("path", help="Path to singlebeam_*.h5")
    parser.add_argument("--dm", type=float, required=True, help="Dispersion measure in pc cm^-3")
    parser.add_argument("--reference-frequency-mhz", type=float, default=400.0)
    args = parser.parse_args(argv)

    result = extract_singlebeam_toa(
        args.path,
        dm=args.dm,
        reference_frequency_mhz=args.reference_frequency_mhz,
    )
    for key, value in result.to_dict().items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
