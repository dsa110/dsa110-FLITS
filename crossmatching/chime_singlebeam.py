from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.time import Time

from crossmatching.toa_crossmatch import compute_toa

CHIME_BASEBAND_SAMPLE_TIME_S = 2.56e-6
CHIME_DEFAULT_REFERENCE_FREQUENCY_MHZ = 400.0


class SinglebeamExtractionError(RuntimeError):
    """Raised when a singlebeam file lacks the metadata needed for a TOA."""


@dataclass(frozen=True)
class SinglebeamLayout:
    """Minimal HDF5 layout facts used by the CHIME singlebeam extractor."""

    data_path: str
    time0_path: str
    freq_path: str | None
    data_shape: tuple[int, ...]
    time_axis: int
    frequency_axis: int | None
    polarization_axis: int | None
    sample_time_s: float
    ntime: int
    nfreq: int | None


@dataclass(frozen=True)
class ChimeSinglebeamToa:
    """Candidate CHIME TOA derived directly from a singlebeam HDF5 file."""

    path: str
    method: str
    data_path: str
    time0_path: str
    frequency_path: str | None
    peak_index: int
    peak_offset_s: float
    native_frequency_mhz: float
    reference_frequency_mhz: float
    native_toa_unix: float
    native_toa_utc: str
    toa_unix_400: float
    toa_utc_400: str
    sample_time_s: float
    time0_frequency_index: int
    time0_ctime: float
    time0_ctime_offset: float
    time0_fpga_count: int | None
    data_shape: tuple[int, ...]
    peak_snr_like: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_singlebeam_layout(
    path: str | Path,
    *,
    data_path: str | None = None,
    time0_path: str = "time0",
    freq_path: str | None = None,
    sample_time_s: float = CHIME_BASEBAND_SAMPLE_TIME_S,
) -> SinglebeamLayout:
    """Inspect a CHIME singlebeam HDF5 file without loading the full payload."""

    h5py = _require_h5py()
    with h5py.File(path, "r") as h5:
        resolved_data_path = data_path or _find_first_dataset(
            h5,
            (
                "tiedbeam_baseband",
                "tiedbeam_power",
                "intensity",
                "waterfall",
                "data",
            ),
        )
        if resolved_data_path is None:
            raise SinglebeamExtractionError("no singlebeam data dataset found")
        if time0_path not in h5:
            raise SinglebeamExtractionError(f"missing required time0 dataset: {time0_path}")

        data = h5[resolved_data_path]
        shape = tuple(int(value) for value in data.shape)
        axes = _infer_axes(shape)
        resolved_freq_path = freq_path or _find_first_dataset(
            h5,
            (
                "index_map/freq",
                "freq",
                "frequency",
                "frequencies",
            ),
            required=False,
        )

    return SinglebeamLayout(
        data_path=resolved_data_path,
        time0_path=time0_path,
        freq_path=resolved_freq_path,
        data_shape=shape,
        time_axis=axes["time"],
        frequency_axis=axes["frequency"],
        polarization_axis=axes["polarization"],
        sample_time_s=float(sample_time_s),
        ntime=shape[axes["time"]],
        nfreq=shape[axes["frequency"]] if axes["frequency"] is not None else None,
    )


def extract_singlebeam_toa(
    path: str | Path,
    *,
    dm: float,
    data_path: str | None = None,
    time0_path: str = "time0",
    freq_path: str | None = None,
    native_frequency_mhz: float | None = None,
    reference_frequency_mhz: float = CHIME_DEFAULT_REFERENCE_FREQUENCY_MHZ,
    sample_time_s: float = CHIME_BASEBAND_SAMPLE_TIME_S,
    chunk_time: int = 16384,
) -> ChimeSinglebeamToa:
    """Extract a candidate CHIME TOA from a singlebeam HDF5 file.

    This is intentionally conservative: it derives a power-summed time series
    and peak-picks it, then shifts the native-frequency TOA to 400 MHz. It does
    not perform CHIME-native coherent dedispersion. The returned ``method``
    records that distinction.
    """

    h5py = _require_h5py()
    layout = inspect_singlebeam_layout(
        path,
        data_path=data_path,
        time0_path=time0_path,
        freq_path=freq_path,
        sample_time_s=sample_time_s,
    )

    with h5py.File(path, "r") as h5:
        data = h5[layout.data_path]
        timeseries = _collapse_power_timeseries(
            data,
            time_axis=layout.time_axis,
            chunk_time=chunk_time,
        )
        if not np.isfinite(timeseries).any():
            raise SinglebeamExtractionError("collapsed time series has no finite samples")

        peak_index = int(np.nanargmax(timeseries))
        peak_snr_like = _snr_like(timeseries, peak_index)
        frequencies = _read_frequency_centres_mhz(h5, layout.freq_path)
        frequency_index, native_frequency = _select_frequency(
            frequencies,
            layout.nfreq,
            native_frequency_mhz,
        )
        ctime, ctime_offset, fpga_count = _read_time0(h5[layout.time0_path], frequency_index)
        native_toa = (
            Time(ctime, val2=ctime_offset, format="unix", scale="utc")
            + (peak_index * layout.sample_time_s) * u.s
        )
        toa_400 = compute_toa(
            native_toa,
            0.0 * u.s,
            native_frequency * u.MHz,
            dm * (u.pc / u.cm**3),
            reference_frequency_mhz * u.MHz,
        )
        native_toa.precision = 9
        toa_400.precision = 9
        method = "power_sum_peak_no_coherent_dedispersion"
        if _dataset_dm_matches(data, dm):
            method = "power_sum_peak_pre_dedispersed_dataset"

    return ChimeSinglebeamToa(
        path=str(path),
        method=method,
        data_path=layout.data_path,
        time0_path=layout.time0_path,
        frequency_path=layout.freq_path,
        peak_index=peak_index,
        peak_offset_s=float(peak_index * layout.sample_time_s),
        native_frequency_mhz=float(native_frequency),
        reference_frequency_mhz=float(reference_frequency_mhz),
        native_toa_unix=float(native_toa.to_value("unix")),
        native_toa_utc=native_toa.iso,
        toa_unix_400=float(toa_400.to_value("unix")),
        toa_utc_400=toa_400.iso,
        sample_time_s=layout.sample_time_s,
        time0_frequency_index=int(frequency_index),
        time0_ctime=float(ctime),
        time0_ctime_offset=float(ctime_offset),
        time0_fpga_count=None if fpga_count is None else int(fpga_count),
        data_shape=layout.data_shape,
        peak_snr_like=peak_snr_like,
    )


def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SinglebeamExtractionError("h5py is required to read singlebeam HDF5 files") from exc
    return h5py


def _find_first_dataset(h5, candidates: tuple[str, ...], *, required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in h5:
            obj = h5[candidate]
            if hasattr(obj, "shape"):
                return candidate
    if required:
        raise SinglebeamExtractionError(f"none of these datasets were found: {candidates}")
    return None


def _infer_axes(shape: tuple[int, ...]) -> dict[str, int | None]:
    if len(shape) == 1:
        return {"time": 0, "frequency": None, "polarization": None}

    polarization_axis = next((idx for idx, size in enumerate(shape) if size in {1, 2}), None)
    if polarization_axis is None:
        polarization_axis = next((idx for idx, size in enumerate(shape) if size == 4), None)
    remaining = [idx for idx in range(len(shape)) if idx != polarization_axis]
    if not remaining:
        raise SinglebeamExtractionError(f"cannot infer time axis from shape {shape}")

    time_axis = max(remaining, key=lambda idx: shape[idx])
    frequency_candidates = [idx for idx in remaining if idx != time_axis]
    frequency_axis = frequency_candidates[0] if frequency_candidates else None
    return {"time": time_axis, "frequency": frequency_axis, "polarization": polarization_axis}


def _collapse_power_timeseries(data, *, time_axis: int, chunk_time: int) -> np.ndarray:
    ntime = int(data.shape[time_axis])
    out = np.zeros(ntime, dtype=np.float64)
    for start in range(0, ntime, chunk_time):
        stop = min(start + chunk_time, ntime)
        selection = [slice(None)] * data.ndim
        selection[time_axis] = slice(start, stop)
        chunk = np.asarray(data[tuple(selection)])
        if np.iscomplexobj(chunk):
            chunk = np.abs(chunk) ** 2
        else:
            chunk = np.asarray(chunk, dtype=np.float64)
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        local_time_axis = time_axis
        reduce_axes = tuple(idx for idx in range(chunk.ndim) if idx != local_time_axis)
        out[start:stop] = np.sum(chunk, axis=reduce_axes)
    return out


def _read_frequency_centres_mhz(h5, freq_path: str | None) -> np.ndarray | None:
    if freq_path is None:
        return None
    arr = np.asarray(h5[freq_path])
    if arr.dtype.fields and "centre" in arr.dtype.fields:
        arr = arr["centre"]
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    if np.nanmedian(np.abs(arr)) < 10:
        arr = arr * 1000.0
    return arr


def _select_frequency(
    frequencies_mhz: np.ndarray | None,
    nfreq: int | None,
    requested_mhz: float | None,
) -> tuple[int, float]:
    if frequencies_mhz is not None:
        if requested_mhz is None:
            index = len(frequencies_mhz) // 2
        else:
            index = int(np.nanargmin(np.abs(frequencies_mhz - requested_mhz)))
        return index, float(frequencies_mhz[index])

    if requested_mhz is not None:
        if nfreq is None:
            return 0, float(requested_mhz)
        return nfreq // 2, float(requested_mhz)

    raise SinglebeamExtractionError(
        "native_frequency_mhz is required when the file has no frequency dataset"
    )


def _read_time0(time0, frequency_index: int) -> tuple[float, float, int | None]:
    row = np.asarray(time0)[frequency_index]
    if row.dtype.fields is None:
        raise SinglebeamExtractionError("time0 must be a structured dataset")
    fields = row.dtype.fields
    if "ctime" not in fields:
        raise SinglebeamExtractionError("time0 is missing required ctime field")
    ctime = float(row["ctime"])
    ctime_offset = float(row["ctime_offset"]) if "ctime_offset" in fields else 0.0
    fpga_count = int(row["fpga_count"]) if "fpga_count" in fields else None
    return ctime, ctime_offset, fpga_count


def _dataset_dm_matches(data, dm: float) -> bool:
    if "DM" not in data.attrs:
        return False
    try:
        return bool(np.isclose(float(data.attrs["DM"]), float(dm)))
    except (TypeError, ValueError):
        return False


def _snr_like(timeseries: np.ndarray, peak_index: int) -> float:
    baseline = np.nanmedian(timeseries)
    mad = np.nanmedian(np.abs(timeseries - baseline))
    if not np.isfinite(mad) or mad == 0:
        return float("inf")
    return float((timeseries[peak_index] - baseline) / (1.4826 * mad))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a candidate TOA from a CHIME singlebeam HDF5 file."
    )
    parser.add_argument("path", help="Path to singlebeam_*.h5")
    parser.add_argument("--dm", type=float, required=True, help="Dispersion measure in pc cm^-3")
    parser.add_argument("--native-frequency-mhz", type=float, default=None)
    parser.add_argument("--reference-frequency-mhz", type=float, default=400.0)
    args = parser.parse_args(argv)

    result = extract_singlebeam_toa(
        args.path,
        dm=args.dm,
        native_frequency_mhz=args.native_frequency_mhz,
        reference_frequency_mhz=args.reference_frequency_mhz,
    )
    for key, value in result.to_dict().items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
