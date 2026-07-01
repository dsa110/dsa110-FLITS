"""Adapter: joint-fit band cubes → ``BandSpectrum`` for ``plot_codetection``."""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .codetection_plots import BandSpectrum

__all__ = ["band_dict_to_spectrum", "crop_band_dict", "spectra_from_joint_bands"]


def crop_band_dict(b: Mapping, xlim: tuple[float, float]) -> dict:
    """Slice band cube to ``xlim`` (ms)."""
    t = np.asarray(b["t"], float)
    i0 = int(np.searchsorted(t, xlim[0], side="left"))
    i1 = int(np.searchsorted(t, xlim[1], side="right"))
    sl = slice(max(0, i0), min(len(t), i1))
    out = dict(b)
    out["t"] = t[sl]
    for key in ("d", "m", "resid", "pd", "pm"):
        if key not in b:
            continue
        arr = np.asarray(b[key])
        out[key] = arr[..., sl] if arr.ndim == 2 else arr[sl]
    return out


def band_dict_to_spectrum(b: Mapping, *, label: str) -> BandSpectrum:
    """Convert GHz-axis joint band dict to ``BandSpectrum`` (MHz frequencies)."""
    return BandSpectrum(
        freq_mhz=np.asarray(b["f"], float) * 1e3,
        time_ms=np.asarray(b["t"], float),
        data=np.asarray(b["d"], float),
        model=np.asarray(b["m"], float),
        sigma=np.asarray(b["noise"], float).reshape(-1) if "noise" in b else None,
        label=label,
    )


def spectra_from_joint_bands(
    bands: Mapping[str, Mapping],
    *,
    xlim: tuple[float, float] | None = None,
    order: tuple[str, ...] = ("C", "D"),
    labels: Mapping[str, str] | None = None,
) -> list[BandSpectrum]:
    """Build ascending-frequency ``BandSpectrum`` list from joint band dicts."""
    labels = labels or {"C": "CHIME", "D": "DSA-110"}
    out: list[BandSpectrum] = []
    for key in order:
        b = bands[key]
        if xlim is not None:
            b = crop_band_dict(b, xlim)
        out.append(band_dict_to_spectrum(b, label=labels[key]))
    return sorted(out, key=lambda s: s.frange[0])
