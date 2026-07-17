"""Load raw CHIME+DSA cubes into ``BandSpectrum`` for codetection figures.

TOA alignment uses ``crossmatching/toa_crossmatch_results.json`` where
``measured_offset_ms`` is CHIME_TOA_400 - DSA_TOA_400. Each instrument cube
carries its own on-pulse crop time origin, so CHIME is shifted by
``t_DSA_peak + measured_offset_ms - t_CHIME_peak`` so the band-integrated peaks
match the crossmatch offset. The display axis is then centered on the midpoint
between those aligned sub-burst peaks.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from .codetection_plots import BandSpectrum

__all__ = [
    "BURST_ORDER",
    "band_onpulse_span",
    "band_peak_time",
    "burst_cube_paths",
    "center_bands_at_peak",
    "chime_toa_shift_ms",
    "crop_bands_to_subburst_window",
    "load_codetection_bands",
    "on_pulse_window",
    "subburst_time_window",
    "SUBBURST_PAD_MS",
    "tight_time_window",
    "toa_offset_ms",
]

BURST_ORDER: tuple[str, ...] = (
    "casey",
    "chromatica",
    "freya",
    "hamilton",
    "isha",
    "johndoeII",
    "mahi",
    "oran",
    "phineas",
    "whitney",
    "wilhelm",
    "zach",
)

_REPO = Path(__file__).resolve().parents[2]
_TEL_CFG = _REPO / "scattering/configs/telescopes.yaml"
_TOA_JSON = _REPO / "crossmatching/toa_crossmatch_results.json"
_TARGET_DT_MS = float(yaml.safe_load(_TEL_CFG.read_text())["dsa"]["dt_ms_raw"])
SUBBURST_PAD_MS = 3.0
SUBBURST_TRAIL_CAP_MS = 6.0


def _t_factor(telescope: str) -> int:
    dt_raw = float(yaml.safe_load(_TEL_CFG.read_text())[telescope]["dt_ms_raw"])
    return max(1, int(round(_TARGET_DT_MS / dt_raw)))


def default_data_dir() -> Path:
    env = os.environ.get("FLITS_DATA")
    if env:
        p = Path(env)
        return p if p.name == "DSA_bursts" else p / "DSA_bursts"
    home = Path.home()
    for candidate in (
        home / "Data/Faber2026/dsa110/DSA_bursts",
        home / "Developer/dsa110-local-data/DSA_bursts",
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Set FLITS_DATA to the DSA_bursts directory (CHIME+DSA .npy cubes)."
    )


def burst_cube_paths(data_dir: Path, burst: str) -> tuple[Path, Path]:
    """Return (chime_npy, dsa_npy) for a nickname."""
    chime = sorted(data_dir.glob(f"{burst}_chime_I_*.npy"))
    dsa = sorted(data_dir.glob(f"{burst}_dsa_I_*.npy"))
    if not chime:
        raise FileNotFoundError(f"no CHIME cube for {burst} under {data_dir}")
    if not dsa:
        raise FileNotFoundError(f"no DSA cube for {burst} under {data_dir}")
    return chime[0], dsa[0]


def toa_offset_ms(burst: str, toa_json: Path | None = None) -> float | None:
    """Offset used to align the two bands' *observed* cube peaks for display.

    This is the peak-based CHIME-DSA offset, not the scatter-corrected model
    offset: the codetection gallery overlays the observed (scattered) profiles,
    so their peaks must be aligned. Since the model switch, the golden JSON's
    ``measured_offset_ms`` is the intrinsic (de-scattered) offset and the peak
    offset lives in ``peak_measured_offset_ms``. Prefer the peak key; fall back
    to ``measured_offset_ms`` for older JSONs that predate the model extension.
    """
    fp = toa_json or _TOA_JSON
    if not fp.exists():
        return None
    rows = json.loads(fp.read_text())
    row = rows.get(burst) or rows.get(burst.lower())
    if not row:
        return None
    offset = row.get("peak_measured_offset_ms", row.get("measured_offset_ms"))
    return float(offset) if offset is not None else None


def chime_toa_shift_ms(
    dsa: BandSpectrum, chime: BandSpectrum, measured_offset_ms: float
) -> float:
    """Shift CHIME cube times so its peak is ``measured_offset_ms`` after DSA peak."""
    return band_peak_time(dsa) + measured_offset_ms - band_peak_time(chime)


def _load_dataset(path: Path, *, telescope: str, name: str, outdir: Path):
    from scat_analysis.config_utils import load_telescope_block
    from scat_analysis.pipeline.io import BurstDataset

    tel = load_telescope_block(str(_TEL_CFG), telescope)
    return BurstDataset(
        path,
        outdir,
        name=name,
        telescope=tel,
        f_factor=1,
        t_factor=_t_factor(telescope),
        outer_trim=0.15,
        onpulse_crop=True,
        onpulse_pad_factor=0.5,
    )


def _band_from_dataset(ds, *, label: str, t_shift_ms: float = 0.0) -> BandSpectrum:
    data = np.asarray(ds.data, float)
    t = np.asarray(ds.time, float) + t_shift_ms
    f_mhz = np.asarray(ds.freq, float) * 1e3
    noise = np.asarray(ds.model.noise_std, float).reshape(-1)
    valid = None
    if ds.model.valid is not None:
        valid = np.asarray(ds.model.valid, bool).reshape(-1)
    return BandSpectrum(
        freq_mhz=f_mhz,
        time_ms=t,
        data=data,
        model=data.copy(),
        sigma=noise,
        label=label,
        channel_valid=valid,
    )


def band_peak_time(b: BandSpectrum) -> float:
    """Band-integrated profile maximum (sub-burst TOA in the current time frame)."""
    t = np.asarray(b.time_ms, float)
    pd = np.nansum(b.data, axis=0)
    if t.size == 0 or pd.size == 0:
        return float(t[0]) if t.size else 0.0
    return float(t[int(np.argmax(pd))])


def band_onpulse_span(b: BandSpectrum) -> tuple[float, float]:
    """Half-max bounds of the band-integrated profile."""
    t = np.asarray(b.time_ms, float)
    pd = np.nansum(b.data, axis=0)
    if t.size == 0 or pd.size == 0:
        pk = band_peak_time(b)
        return pk, pk
    peak = float(np.max(pd))
    if peak <= 0:
        pk = band_peak_time(b)
        return pk, pk
    on = pd >= 0.5 * peak
    if not np.any(on):
        pk = int(np.argmax(pd))
        return float(t[pk]), float(t[pk])
    idx = np.where(on)[0]
    return float(t[idx[0]]), float(t[idx[-1]])


def _subburst_trailing_reference(
    b: BandSpectrum, *, trail_cap_ms: float = SUBBURST_TRAIL_CAP_MS
) -> float:
    peak = band_peak_time(b)
    _, halfmax_right = band_onpulse_span(b)
    return min(float(halfmax_right), peak + trail_cap_ms)


def subburst_time_window(
    bands: Sequence[BandSpectrum], *, pad_ms: float = SUBBURST_PAD_MS
) -> tuple[float, float]:
    """Pad first peak and capped last trailing edge by ``pad_ms``."""
    active = [b for b in bands if b.time_ms.size]
    if not active:
        return 0.0, 1.0
    ordered = sorted(active, key=band_peak_time)
    first, last = ordered[0], ordered[-1]
    return band_peak_time(first) - pad_ms, _subburst_trailing_reference(last) + pad_ms


def tight_time_window(
    bands: Sequence[BandSpectrum], *, pad_ms: float = SUBBURST_PAD_MS
) -> tuple[float, float]:
    """Alias for :func:`subburst_time_window` (legacy name)."""
    return subburst_time_window(bands, pad_ms=pad_ms)


def on_pulse_window(bands: Sequence[BandSpectrum]) -> tuple[float, float]:
    """Time bounds where any band's band-integrated profile exceeds threshold."""
    spans: list[tuple[float, float]] = []
    for b in bands:
        t = np.asarray(b.time_ms, float)
        pd = np.nansum(b.data, axis=0)
        if t.size == 0 or pd.size == 0:
            continue
        peak = float(np.max(pd))
        thr = max(0.08 * peak, float(np.median(pd)))
        on = pd >= thr
        if not np.any(on):
            pk = int(np.argmax(pd))
            spans.append((float(t[max(0, pk - 8)]), float(t[min(len(t) - 1, pk + 8)])))
            continue
        idx = np.where(on)[0]
        spans.append((float(t[idx[0]]), float(t[idx[-1]])))
    if not spans:
        return 0.0, 1.0
    return min(s[0] for s in spans), max(s[1] for s in spans)


def _interp_time_rows(t_out: np.ndarray, t_in: np.ndarray, arr: np.ndarray) -> np.ndarray:
    if t_in.size == 0:
        return np.full((arr.shape[0], t_out.size), np.nan)
    out = np.empty((arr.shape[0], t_out.size), dtype=float)
    for i, row in enumerate(np.asarray(arr, float)):
        out[i] = np.interp(t_out, t_in, row, left=np.nan, right=np.nan)
    return out


def _crop_band_to_xlim(b: BandSpectrum, xlim: tuple[float, float], *, t_ref: float) -> BandSpectrum:
    shifted_t = np.asarray(b.time_ms, float) - t_ref
    x0, x1 = map(float, xlim)
    inside = shifted_t[(shifted_t >= x0) & (shifted_t <= x1)]
    t_out = np.unique(np.concatenate(([x0], inside, [x1])))
    return BandSpectrum(
        freq_mhz=b.freq_mhz,
        time_ms=t_out,
        data=_interp_time_rows(t_out, shifted_t, np.asarray(b.data, float)),
        model=_interp_time_rows(t_out, shifted_t, np.asarray(b.model, float)),
        sigma=b.sigma,
        label=b.label,
        channel_valid=b.channel_valid,
    )


def crop_bands_to_subburst_window(
    bands: Sequence[BandSpectrum],
    *,
    pad_ms: float = SUBBURST_PAD_MS,
    center: bool = True,
) -> list[BandSpectrum]:
    """Crop: pad first peak and capped last trailing edge, preserving missing data."""
    active = [b for b in bands if b.time_ms.size]
    if not active:
        return list(bands)
    ordered = sorted(active, key=band_peak_time)
    first, last = ordered[0], ordered[-1]
    t_trail = _subburst_trailing_reference(last)
    t_first_peak = band_peak_time(first)
    t_last_peak = band_peak_time(last)
    t_ref = 0.5 * (t_first_peak + t_last_peak) if center else 0.0
    xlim = (
        (t_first_peak - pad_ms - t_ref, t_trail + pad_ms - t_ref)
        if center
        else (t_first_peak - pad_ms, t_trail + pad_ms)
    )
    return [_crop_band_to_xlim(b, xlim, t_ref=t_ref) for b in bands]


def center_bands_at_peak(
    bands: Sequence[BandSpectrum],
    *,
    offpulse_pad_ms: float = SUBBURST_PAD_MS,
    offpulse_pad_frac: float = 0.25,
) -> list[BandSpectrum]:
    """Center on the midpoint between sub-burst TOAs and crop with fixed padding."""
    _ = offpulse_pad_frac  # legacy kwarg; padding is TOA-based only
    return crop_bands_to_subburst_window(bands, pad_ms=offpulse_pad_ms, center=True)


def load_codetection_bands(
    burst: str,
    data_dir: Path | None = None,
    *,
    cache_dir: Path | None = None,
    align_toa: bool = True,
    center_time: bool = True,
    xlim: tuple[float, float] | None = None,
) -> list[BandSpectrum]:
    """Raw CHIME + DSA bands on a shared frequency axis (TOA-aligned)."""
    from .codetection_joint import crop_band_dict

    data_dir = data_dir or default_data_dir()
    cache_dir = cache_dir or (data_dir.parent / ".codetection_cache" / burst)
    cache_dir.mkdir(parents=True, exist_ok=True)

    chime_fp, dsa_fp = burst_cube_paths(data_dir, burst)
    dsa = _load_dataset(dsa_fp, telescope="dsa", name=f"{burst}_dsa", outdir=cache_dir)
    chime = _load_dataset(chime_fp, telescope="chime", name=f"{burst}_chime", outdir=cache_dir)

    dsa_band = _band_from_dataset(dsa, label="DSA-110")
    chime_raw = _band_from_dataset(chime, label="CHIME/FRB")
    shift_c = 0.0
    if align_toa:
        offset = toa_offset_ms(burst)
        if offset is not None:
            shift_c = chime_toa_shift_ms(dsa_band, chime_raw, offset)

    bands_dict = {
        "C": _band_from_dataset(chime, label="CHIME/FRB", t_shift_ms=shift_c),
        "D": dsa_band,
    }
    bands = list(bands_dict.values())
    if xlim is not None:
        cropped = {
            key: crop_band_dict(
                {
                    "t": b.time_ms,
                    "d": b.data,
                    "m": b.model,
                    "f": b.freq_mhz / 1e3,
                    "noise": b.sigma,
                    "valid": b.channel_valid,
                },
                xlim,
            )
            for key, b in bands_dict.items()
        }
        bands = [
            BandSpectrum(
                freq_mhz=cropped["C"]["f"] * 1e3,
                time_ms=cropped["C"]["t"],
                data=cropped["C"]["d"],
                model=cropped["C"]["m"],
                sigma=cropped["C"]["noise"],
                label="CHIME/FRB",
                channel_valid=cropped["C"].get("valid"),
            ),
            BandSpectrum(
                freq_mhz=cropped["D"]["f"] * 1e3,
                time_ms=cropped["D"]["t"],
                data=cropped["D"]["d"],
                model=cropped["D"]["m"],
                sigma=cropped["D"]["noise"],
                label="DSA-110",
                channel_valid=cropped["D"].get("valid"),
            ),
        ]
    if center_time:
        return crop_bands_to_subburst_window(bands, pad_ms=SUBBURST_PAD_MS, center=True)
    return bands
