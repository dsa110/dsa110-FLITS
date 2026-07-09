"""
codetection_plots.py
====================

Unified multi-band dynamic-spectrum figure for FRB co-detections (e.g. CHIME +
DSA-110), showing the observed data, the best-fit 2-D model, and the residuals
side by side.

Each column (data | model | residual) carries a frequency-averaged **time
series** along the top and a time-averaged **spectrum** down the right side.
Bands are drawn to scale on a shared frequency axis, and any *unobserved gap*
between bands (e.g. the 0.80-1.31 GHz gap between the CHIME and DSA bands) is
hatched so the missing coverage is explicit.

The function is data-source agnostic: for each frequency band you supply the
observed dynamic spectrum, the best-fit 2-D model (same shape), and optionally
the noise level. It is intended to consume ``burstfit`` outputs directly.

Example
-------
>>> bands = [
...     BandSpectrum(freq_mhz=chime_freq, time_ms=t, data=chime_data,
...                  model=chime_model, label="CHIME"),
...     BandSpectrum(freq_mhz=dsa_freq, time_ms=t, data=dsa_data,
...                  model=dsa_model, label="DSA"),
... ]
>>> fig = plot_codetection(bands, title="FRB 20231019")
>>> fig.savefig("frb_codetection.pdf", bbox_inches="tight")

Notes
-----
Bands may use different time grids; ``plot_codetection`` regrids internally to
the finest shared step before building marginals. Frequency channel counts may
also differ.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

__all__ = ["BandSpectrum", "plot_codetection", "plot_codetection_observations"]

# Match the hatch-gap fill (unobserved band / flagged channel).
_GAP_FILL = "0.93"
_BAND_LINE_COLORS = ("#262626", "#b2182b", "#2166ac", "#1b7837")
_HATCH_ZORDER = 0.5
_DATA_ZORDER = 2.0
_TICK_ZORDER = 10.0
_OUTBOARD_GRID_LEFT = 0.18
_OUTBOARD_BRACKET_X = -0.010
_OUTBOARD_TEXT_X = -0.034
_DEFAULT_GRID_LEFT = 0.08
_DEFAULT_GRID_RIGHT = 0.985
_SPECTRUM_LABEL_GRID_RIGHT = 0.90
_INTEXT_LABEL_X = 0.025  # axes-fraction inset from left spine for in-band tags
_SPECTRUM_LABEL_X = 1.04  # axes-fraction, right edge of spectrum marginal
_BAND_LABEL_STYLES = frozenset({"strip", "outboard", "tag", "gap", "keyed"})


@dataclass(frozen=True)
class _LabelPlan:
    style: str | None

    @property
    def is_keyed(self) -> bool:
        return self.style == "keyed"

    @property
    def draw_on_waterfall(self) -> bool:
        return self.style in {"outboard", "tag", "gap"}

    @property
    def draw_on_spectrum(self) -> bool:
        return self.style == "strip"


@dataclass(frozen=True)
class _LayoutPlan:
    data_xlim: tuple[float, float]
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    gaps: tuple[tuple[float, float], ...]
    grid_left: float
    grid_right: float
    labels: _LabelPlan


@dataclass
class BandSpectrum:
    """A single frequency band of a (co-)detection.

    Parameters
    ----------
    freq_mhz : (nchan,) ndarray
        Channel centre frequencies in MHz (ascending).
    time_ms : (nsamp,) ndarray
        Time-sample centres in ms.
    data : (nchan, nsamp) ndarray
        Observed dynamic spectrum.
    model : (nchan, nsamp) ndarray
        Best-fit 2-D model, same shape as ``data``.
    sigma : float or (nchan,) ndarray, optional
        Per-channel noise (``burstfit`` ``noise_std``). Scalar applies to all
        channels. If ``None``, per-channel MAD of ``data - model`` is used.
    label : str, optional
        Short band label, e.g. ``"CHIME"`` / ``"DSA"``.
    channel_valid : (nchan,) bool ndarray, optional
        Per-channel validity (``burstfit`` ``valid`` mask). ``False`` marks
        dead/RFI channels for grey overlay on the waterfall.
    """

    freq_mhz: np.ndarray
    time_ms: np.ndarray
    data: np.ndarray
    model: np.ndarray
    sigma: float | np.ndarray | None = None
    label: str | None = None
    channel_valid: np.ndarray | None = None

    @property
    def frange(self) -> tuple[float, float]:
        return float(np.min(self.freq_mhz)), float(np.max(self.freq_mhz))

    def noise_per_channel(self) -> np.ndarray:
        """(nchan,) noise std for whitened residuals."""
        nchan = self.data.shape[0]
        if self.sigma is not None:
            s = np.asarray(self.sigma, float).reshape(-1)
            if s.size == 1:
                return np.full(nchan, float(s[0]))
            if s.size != nchan:
                raise ValueError(f"sigma length {s.size} != nchan {nchan}")
            return s
        resid = self.data - self.model
        mad = np.median(np.abs(resid - np.median(resid, axis=1, keepdims=True)), axis=1)
        s = 1.4826 * mad
        fallback = float(resid.std() or 1.0)
        return np.where(s > 0, s, fallback)


def _interp_along_time(t_out, t_in, arr: np.ndarray) -> np.ndarray:
    t_in = np.asarray(t_in, float)
    t_out = np.asarray(t_out, float)
    out = np.empty((arr.shape[0], t_out.size))
    for i in range(arr.shape[0]):
        row = arr[i]
        out[i] = np.interp(t_out, t_in, row, left=np.nan, right=np.nan)
    return out


def _common_time_ms(bands: Sequence[BandSpectrum]) -> np.ndarray:
    dt = min(float(np.median(np.diff(b.time_ms))) for b in bands)
    t0 = min(float(b.time_ms[0]) for b in bands)
    t1 = max(float(b.time_ms[-1]) for b in bands)
    n = max(2, int(round((t1 - t0) / dt)) + 1)
    t = t0 + np.arange(n) * dt
    return t[t <= t1 + 1e-9]


def _regrid_band(b: BandSpectrum, t_common: np.ndarray) -> BandSpectrum:
    if b.time_ms.shape == t_common.shape and np.allclose(b.time_ms, t_common):
        return b
    return replace(
        b,
        time_ms=t_common,
        data=_interp_along_time(t_common, b.time_ms, b.data),
        model=_interp_along_time(t_common, b.time_ms, b.model),
    )


def _regrid_bands(bands: Sequence[BandSpectrum]) -> tuple[list[BandSpectrum], np.ndarray]:
    t_common = _common_time_ms(bands)
    return [_regrid_band(b, t_common) for b in bands], t_common


def _plan_band_labels(enabled: bool, style: str) -> _LabelPlan:
    if not enabled:
        return _LabelPlan(None)
    if style not in _BAND_LABEL_STYLES:
        allowed = ", ".join(sorted(_BAND_LABEL_STYLES))
        raise ValueError(f"band_label_style must be one of {allowed}; got {style!r}")
    return _LabelPlan(style)


def _data_xlim(
    bands: Sequence[BandSpectrum], tref: np.ndarray, *, symmetric_time_axis: bool
) -> tuple[float, float]:
    if symmetric_time_axis:
        half = max(abs(float(tref[0])), abs(float(tref[-1])))
        return -half, half
    return (
        min(float(b.time_ms[0]) for b in bands),
        max(float(b.time_ms[-1]) for b in bands),
    )


def _band_gaps(bands: Sequence[BandSpectrum]) -> tuple[tuple[float, float], ...]:
    return tuple(
        (lo.frange[1], hi.frange[0])
        for lo, hi in zip(bands[:-1], bands[1:], strict=False)
        if hi.frange[0] > lo.frange[1] + 1e-6
    )


def _plan_layout(
    bands: Sequence[BandSpectrum],
    tref: np.ndarray,
    *,
    symmetric_time_axis: bool,
    band_labels: bool,
    band_label_style: str,
) -> _LayoutPlan:
    labels = _plan_band_labels(band_labels, band_label_style)
    xlim = _data_xlim(bands, tref, symmetric_time_axis=symmetric_time_axis)
    ylim = (bands[0].frange[0], bands[-1].frange[1])
    return _LayoutPlan(
        data_xlim=xlim,
        xlim=xlim,
        ylim=ylim,
        gaps=_band_gaps(bands),
        grid_left=_OUTBOARD_GRID_LEFT if labels.style == "outboard" else _DEFAULT_GRID_LEFT,
        grid_right=_SPECTRUM_LABEL_GRID_RIGHT
        if labels.draw_on_spectrum
        else _DEFAULT_GRID_RIGHT,
        labels=labels,
    )


def _finite_percentile(arr: np.ndarray, pct: float, default: float = 1.0) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.percentile(finite, pct))


def _cmap_with_bad(name: str):
    c = cm.get_cmap(name).copy()
    c.set_bad(color="white", alpha=0.0)
    return c


def _band_array(b: BandSpectrum, key: str) -> np.ndarray:
    if key == "resid":
        sig = b.noise_per_channel()[:, None]
        return (b.data - b.model) / np.clip(sig, 1e-9, None)
    return getattr(b, key)


def _nanmean(arr: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(arr, float)
    valid = np.isfinite(arr)
    count = valid.sum(axis=axis)
    total = np.where(valid, arr, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def _hatch_rect(ax, x0, x1, y0, y1):
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="white",
            edgecolor="0.55",
            hatch="////",
            lw=0.0,
            zorder=_HATCH_ZORDER,
        )
    )


def _time_edges_ms(time_ms: np.ndarray) -> np.ndarray:
    t = np.asarray(time_ms, float)
    if t.size == 0:
        return t
    if t.size == 1:
        return np.array([t[0] - 0.5, t[0] + 0.5])
    edges = np.empty(t.size + 1)
    edges[1:-1] = 0.5 * (t[:-1] + t[1:])
    edges[0] = t[0] - 0.5 * (t[1] - t[0])
    edges[-1] = t[-1] + 0.5 * (t[-1] - t[-2])
    return edges


def _no_data_time_spans(band: BandSpectrum, key: str) -> list[tuple[float, float]]:
    arr = _band_array(band, key)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return []
    missing = np.all(~np.isfinite(arr), axis=0)
    if not np.any(missing):
        return []
    edges = _time_edges_ms(band.time_ms)
    spans: list[tuple[float, float]] = []
    start: int | None = None
    for i, is_missing in enumerate(missing):
        if is_missing and start is None:
            start = i
        if start is not None and (not is_missing or i == missing.size - 1):
            stop = i + 1 if is_missing else i
            spans.append((float(edges[start]), float(edges[stop])))
            start = None
    return spans


def _channel_edges_mhz(freq_mhz: np.ndarray) -> np.ndarray:
    """Channel boundary frequencies (length nchan + 1)."""
    f = np.asarray(freq_mhz, float)
    if f.size == 0:
        return f
    if f.size == 1:
        half = 0.5
        return np.array([f[0] - half, f[0] + half])
    edges = np.empty(f.size + 1)
    edges[1:-1] = 0.5 * (f[:-1] + f[1:])
    edges[0] = f[0] - 0.5 * (f[1] - f[0])
    edges[-1] = f[-1] + 0.5 * (f[-1] - f[-2])
    return edges


def _rfi_channel_markers(ax, band: BandSpectrum, t0: float, t1: float) -> None:
    """Grey horizontal stripes on dead/RFI channels (``~channel_valid``)."""
    if band.channel_valid is None:
        return
    valid = np.asarray(band.channel_valid, bool).reshape(-1)
    if valid.size != band.freq_mhz.size or valid.all():
        return
    edges = _channel_edges_mhz(band.freq_mhz)
    for i in np.where(~valid)[0]:
        ax.add_patch(
            Rectangle(
                (t0, edges[i]),
                t1 - t0,
                edges[i + 1] - edges[i],
                facecolor=_GAP_FILL,
                edgecolor="none",
                lw=0.0,
                zorder=4,
            )
        )


def _resolve_rc_fontsize(key: str) -> float:
    """Resolve an rcParams font-size entry to points."""
    val = plt.rcParams[key]
    if isinstance(val, (int, float)):
        return float(val)
    scales = {
        "xx-small": 0.6,
        "x-small": 0.75,
        "small": 0.9,
        "medium": 1.0,
        "large": 1.2,
        "x-large": 1.4,
        "xx-large": 1.8,
    }
    base = float(plt.rcParams["font.size"])
    return base * scales.get(str(val), 1.0)


def _raise_axis_ticks(ax) -> None:
    """Draw spines and inward ticks above hatched / data patches."""
    ax.set_axisbelow(False)
    for spine in ax.spines.values():
        spine.set_zorder(_TICK_ZORDER)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_zorder(_TICK_ZORDER)
        for line in axis.get_ticklines():
            line.set_zorder(_TICK_ZORDER)
        for label in axis.get_ticklabels():
            label.set_zorder(_TICK_ZORDER)


def _band_spectrum_labels(ax, bands: Sequence[BandSpectrum], fmin: float, fmax: float) -> None:
    """Horizontal telescope names on the right edge of the spectrum marginal."""
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    label_fs = max(6.0, _resolve_rc_fontsize("axes.labelsize") - 2.0)
    fspan = max(fmax - fmin, 1e-6)
    for b in bands:
        if not b.label:
            continue
        f_lo, f_hi = b.frange
        edge_pad = min(0.30 * (f_hi - f_lo), 0.045 * fspan)
        f_mid = float(np.clip(0.5 * (f_lo + f_hi), f_lo + edge_pad, f_hi - edge_pad))
        ax.text(
            _SPECTRUM_LABEL_X,
            f_mid,
            b.label,
            transform=trans,
            ha="left",
            va="center",
            fontsize=label_fs,
            fontweight="bold",
            color="black",
            clip_on=False,
            zorder=8,
        )


def _band_outboard_labels(ax, bands: Sequence[BandSpectrum]) -> None:
    """Horizontal telescope names in the left margin, bracketed to each band."""
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    label_fs = max(6.0, _resolve_rc_fontsize("axes.labelsize") - 0.5)
    cap_dx = 0.004

    for b in bands:
        if not b.label:
            continue
        f_lo, f_hi = b.frange
        f_mid = 0.5 * (f_lo + f_hi)
        band_h = max(f_hi - f_lo, 1e-6)
        cap = min(0.04 * band_h, 0.12 * band_h + 3.0)
        y0, y1 = f_lo + cap, f_hi - cap
        if y1 <= y0:
            y0, y1 = f_lo, f_hi
            cap = 0.0

        bracket_kw = dict(transform=trans, color="0.25", lw=0.9, clip_on=False, zorder=7)
        ax.plot([_OUTBOARD_BRACKET_X, _OUTBOARD_BRACKET_X], [y0, y1], **bracket_kw)
        if cap > 0:
            for y in (y0, y1):
                ax.plot(
                    [_OUTBOARD_BRACKET_X - cap_dx, _OUTBOARD_BRACKET_X + cap_dx],
                    [y, y],
                    **bracket_kw,
                )

        ax.text(
            _OUTBOARD_TEXT_X,
            f_mid,
            b.label,
            transform=trans,
            ha="right",
            va="center",
            fontsize=label_fs,
            fontweight="bold",
            color="black",
            clip_on=False,
            zorder=7,
        )


def _band_intext_labels(ax, bands: Sequence[BandSpectrum], fmin: float, fmax: float) -> None:
    """Horizontal telescope names inside each band at the left (off-pulse) edge.

    A semi-opaque white box keeps them legible over the dark magma noise floor
    (or grey RFI stripes) without stealing horizontal data width or colliding
    with the y-axis tick numerals / title. One constant font for every band, so
    a thin band's height never shrinks the other band's label.
    """
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    label_fs = max(6.5, _resolve_rc_fontsize("axes.labelsize") - 1.0)
    fspan = max(fmax - fmin, 1e-6)
    for b in bands:
        if not b.label:
            continue
        f_lo, f_hi = b.frange
        # Keep the tag off the band edges so a thin band (mahi, oran) never
        # bleeds the box into the gap; clamp toward the band centre.
        edge_pad = min(0.30 * (f_hi - f_lo), 0.045 * fspan)
        f_mid = float(np.clip(0.5 * (f_lo + f_hi), f_lo + edge_pad, f_hi - edge_pad))
        ax.text(
            _INTEXT_LABEL_X,
            f_mid,
            b.label,
            transform=trans,
            ha="left",
            va="center",
            fontsize=label_fs,
            fontweight="bold",
            color="black",
            zorder=8,
            clip_on=True,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="0.6",
                alpha=0.78,
                linewidth=0.5,
            ),
        )


def _band_gap_labels(
    ax,
    bands: Sequence[BandSpectrum],
    gaps: Sequence[tuple[float, float]],
    data_t0: float,
    data_t1: float,
) -> None:
    """Telescope names centered in the hatched inter-band gap (no burst data there)."""
    if len(bands) < 2 or not gaps:
        return
    label_fs = max(6.5, _resolve_rc_fontsize("axes.labelsize") - 0.5)
    x = 0.5 * (data_t0 + data_t1)
    lo_band, hi_band = bands[0], bands[-1]
    for g0, g1 in gaps:
        gap_h = max(g1 - g0, 1e-6)
        if lo_band.label:
            ax.text(
                x,
                g0 + 0.24 * gap_h,
                lo_band.label,
                ha="center",
                va="center",
                fontsize=label_fs,
                fontweight="bold",
                color="0.30",
                zorder=5,
                clip_on=True,
            )
        if hi_band.label:
            ax.text(
                x,
                g1 - 0.24 * gap_h,
                hi_band.label,
                ha="center",
                va="center",
                fontsize=label_fs,
                fontweight="bold",
                color="0.30",
                zorder=5,
                clip_on=True,
            )


def _draw_waterfall_band_labels(
    ax,
    labels: _LabelPlan,
    bands: Sequence[BandSpectrum],
    gaps: Sequence[tuple[float, float]],
    data_xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    if labels.style == "outboard":
        _band_outboard_labels(ax, bands)
    elif labels.style == "tag":
        _band_intext_labels(ax, bands, *ylim)
    elif labels.style == "gap":
        _band_gap_labels(ax, bands, gaps, *data_xlim)


def _draw_spectrum_band_labels(
    ax, labels: _LabelPlan, bands: Sequence[BandSpectrum], ylim: tuple[float, float]
) -> None:
    if labels.draw_on_spectrum:
        _band_spectrum_labels(ax, bands, *ylim)


def _band_line_color(index: int) -> str:
    return _BAND_LINE_COLORS[index % len(_BAND_LINE_COLORS)]


def _profile_band_legend(ax, bands: Sequence[BandSpectrum]) -> None:
    handles = [
        Line2D([0], [0], color=_band_line_color(i), lw=1.4, label=b.label)
        for i, b in enumerate(bands)
        if b.label
    ]
    if not handles:
        return
    fs = max(6.0, _resolve_rc_fontsize("axes.labelsize") - 2.0)
    ax.legend(
        handles=handles,
        loc="center left",
        fontsize=fs,
        framealpha=0.92,
        edgecolor="0.75",
        handlelength=1.6,
        borderpad=0.35,
        labelspacing=0.35,
    )


def _unit_peak(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    if not np.any(np.isfinite(y)):
        return np.zeros_like(y)
    peak = float(np.nanmax(y))
    if peak > 0:
        return y / peak
    scale = float(np.nanmax(np.abs(y)))
    return y / scale if scale > 0 else np.zeros_like(y)


def plot_codetection(
    bands: Sequence[BandSpectrum],
    *,
    water_cmap: str = "magma",
    resid_cmap: str = "RdBu_r",
    resid_clip: float = 5.0,
    per_band_scale: bool = False,
    show_model_on_data: bool = True,
    columns: Sequence[str] = ("data", "model", "resid"),
    col_wspace: float = 0.15,
    marg_pad: float = 0.05,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    gap_label: bool = True,
    symmetric_time_axis: bool = False,
    band_labels: bool = False,
    band_label_style: str = "strip",
    show_column_titles: bool = True,
    per_band_marginals: bool = False,
):
    """Render the data | model | residual triptych with marginals.

    Parameters
    ----------
    bands : sequence of BandSpectrum
        Any number of bands; sorted internally by frequency. Gaps between
        adjacent bands are hatched as unobserved.
    water_cmap, resid_cmap : str
        Colormaps for the data/model waterfalls and the residual panel.
    resid_clip : float
        Symmetric colour/axis limit for residuals, in units of sigma.
    per_band_scale : bool
        If True each band is normalised to its own 99.5th percentile (better
        visibility when bands differ greatly in S/N); otherwise a single shared
        intensity scale is used (honest cross-band comparison).
    show_model_on_data : bool
        Overlay the model (red) on the data column's time series and spectrum.
    columns : sequence of str
        Subset of ``("data", "model", "resid")`` to render. Use ``("data",)``
        for a single connected observations panel.
    col_wspace, marg_pad : float
        Spacing between the three columns, and between a waterfall and its
        marginal panels, respectively.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not bands:
        raise ValueError("`bands` must contain at least one BandSpectrum")
    bands = sorted(bands, key=lambda b: b.frange[0])
    bands, tref = _regrid_bands(bands)
    layout = _plan_layout(
        bands,
        tref,
        symmetric_time_axis=symmetric_time_axis,
        band_labels=band_labels,
        band_label_style=band_label_style,
    )
    data_t0, data_t1 = layout.data_xlim
    t0, t1 = layout.xlim
    fmin, fmax = layout.ylim
    gaps = layout.gaps
    keyed_labels = layout.labels.is_keyed

    # intensity scales (ignore NaN padding from unequal snippet lengths)
    norm = Normalize(0, _finite_percentile(np.concatenate([b.model for b in bands]), 99.5))
    bnorm = (
        {id(b): Normalize(0, _finite_percentile(b.model, 99.5)) for b in bands}
        if per_band_scale
        else None
    )
    water_cmap_bad = _cmap_with_bad(water_cmap)
    rnorm = Normalize(-resid_clip, resid_clip)

    # marginal projections (channel-averaged profile; time-averaged spectrum)
    prof = {k: _nanmean(np.concatenate([_band_array(b, k) for b in bands], axis=0), axis=0)
            for k in ("data", "model", "resid")}
    pmax = float(np.nanmax([np.nanmax(prof["data"]), np.nanmax(prof["model"])]))
    if not np.isfinite(pmax) or pmax <= 0:
        pmax = 1.0
    if per_band_scale:
        smax = 1.0
    else:
        smax = max(np.nanmax(_nanmean(b.data, axis=1)) for b in bands)
        smax = max(smax, max(np.nanmax(_nanmean(b.model, axis=1)) for b in bands))

    all_cols = [("data", "Data", water_cmap, norm),
                ("model", "Model (2-D fit)", water_cmap, norm),
                ("resid", r"Residual / $\sigma$", resid_cmap, rnorm)]
    cols = [c for c in all_cols if c[0] in columns]
    if not cols:
        raise ValueError(f"`columns` must include at least one of data/model/resid; got {columns!r}")

    if figsize is None:
        w = 3.6 * len(cols) + 0.6
        figsize = (w, 4.7)

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        1,
        len(cols),
        wspace=col_wspace,
        left=layout.grid_left,
        right=layout.grid_right,
        top=0.94 if title else 0.96,
        bottom=0.12,
    )

    for j, (key, ctitle, cmap, nrm) in enumerate(cols):
        inner = outer[0, j].subgridspec(2, 2, width_ratios=[4, 1.05],
                                        height_ratios=[1, 4],
                                        wspace=marg_pad, hspace=marg_pad)
        ax_ts = fig.add_subplot(inner[0, 0])
        ax_wf = fig.add_subplot(inner[1, 0], sharex=ax_ts)
        ax_sp = fig.add_subplot(inner[1, 1], sharey=ax_wf)
        ax_leg = None
        if keyed_labels and j == 0:
            ax_leg = fig.add_subplot(inner[0, 1])
            ax_leg.axis("off")

        # --- waterfall (each band drawn to scale) ---
        for b in bands:
            use = bnorm[id(b)] if (bnorm is not None and key != "resid") else nrm
            arr = np.ma.masked_invalid(_band_array(b, key))
            ax_wf.imshow(
                arr,
                origin="lower",
                aspect="auto",
                cmap=water_cmap_bad if key != "resid" else cmap,
                norm=use,
                extent=[b.time_ms[0], b.time_ms[-1], b.frange[0], b.frange[1]],
                zorder=_DATA_ZORDER,
            )
            for nt0, nt1 in _no_data_time_spans(b, key):
                _hatch_rect(ax_wf, nt0, nt1, b.frange[0], b.frange[1])
            _rfi_channel_markers(ax_wf, b, data_t0, data_t1)
        for g0, g1 in gaps:
            _hatch_rect(ax_wf, data_t0, data_t1, g0, g1)
            if gap_label and j == 0 and layout.labels.style != "gap" and not keyed_labels:
                ax_wf.text((t0 + t1) / 2, (g0 + g1) / 2, "no coverage",
                           ha="center", va="center", fontsize=6.5,
                           style="italic", color="0.4", zorder=4)
        ax_wf.set_ylim(fmin, fmax)
        ax_wf.set_xlim(t0, t1)
        if j == 0 and layout.labels.draw_on_waterfall:
            _draw_waterfall_band_labels(
                ax_wf, layout.labels, bands, gaps, layout.data_xlim, layout.ylim
            )
        _raise_axis_ticks(ax_wf)
        ax_wf.set_xlabel("Time (ms)")
        if j == 0:
            ax_wf.set_ylabel("Frequency (MHz)")
        else:
            ax_wf.tick_params(labelleft=False)

        # --- top time series ---
        if key == "resid":
            ax_ts.axhline(0, color="0.6", lw=0.6)
            ax_ts.plot(tref, prof["resid"], color="#34495e", lw=0.8)
            ax_ts.set_ylim(-0.6 * resid_clip, 0.6 * resid_clip)
        else:
            if per_band_marginals:
                for i, b in enumerate(bands):
                    color = _BAND_LINE_COLORS[i % len(_BAND_LINE_COLORS)]
                    ax_ts.plot(
                        b.time_ms,
                        _unit_peak(_nanmean(_band_array(b, key), axis=0)),
                        color=color,
                        lw=0.8,
                    )
                    if key == "data" and show_model_on_data:
                        ax_ts.plot(
                            b.time_ms,
                            _unit_peak(_nanmean(b.model, axis=0)),
                            color=color,
                            lw=0.8,
                            alpha=0.55,
                            linestyle="--",
                        )
                ax_ts.set_ylim(-0.05, 1.15)
                if keyed_labels and j == 0 and ax_leg is not None:
                    _profile_band_legend(ax_leg, bands)
            else:
                ax_ts.plot(tref, prof[key], color="black", lw=0.8)
                if key == "data" and show_model_on_data:
                    ax_ts.plot(tref, prof["model"], color="crimson", lw=0.9, alpha=0.85)
                ax_ts.set_ylim(-0.05 * pmax, 1.15 * pmax)
        ax_ts.set_xlim(t0, t1)
        ax_ts.tick_params(labelbottom=False, labelleft=False)
        if show_column_titles:
            ax_ts.set_title(ctitle, fontsize=10, pad=4)

        # --- right spectrum ---
        if key == "resid":
            ax_sp.axvline(0, color="0.6", lw=0.6)
            for b in bands:
                ax_sp.plot(_band_array(b, "resid").mean(1), b.freq_mhz,
                           color="#34495e", lw=0.7)
            ax_sp.set_xlim(-0.6 * resid_clip, 0.6 * resid_clip)
        else:
            for i, b in enumerate(bands):
                spec = _nanmean(getattr(b, key), axis=1)
                if per_band_scale:
                    spec = _unit_peak(spec)
                line_color = _band_line_color(i) if keyed_labels else "black"
                ax_sp.plot(spec, b.freq_mhz, color=line_color, lw=0.7)
            if key == "data" and show_model_on_data:
                for b in bands:
                    spec = _nanmean(b.model, axis=1)
                    if per_band_scale:
                        spec = _unit_peak(spec)
                    ax_sp.plot(spec, b.freq_mhz, color="crimson", lw=0.8)
            ax_sp.set_xlim(-0.05 * smax, 1.2 * smax)
        for g0, g1 in gaps:
            ax_sp.axhspan(
                g0,
                g1,
                facecolor="white",
                edgecolor="0.55",
                hatch="////",
                lw=0,
                zorder=_HATCH_ZORDER,
            )
        ax_sp.set_ylim(fmin, fmax)
        ax_sp.tick_params(labelleft=False, labelbottom=False)
        _raise_axis_ticks(ax_sp)
        if j == 0:
            _draw_spectrum_band_labels(ax_sp, layout.labels, bands, layout.ylim)

    if title:
        fig.suptitle(title, fontsize=10, y=0.985)
    return fig


def plot_codetection_observations(
    bands: Sequence[BandSpectrum],
    *,
    title: str | None = None,
    per_band_scale: bool = True,
    **kwargs,
):
    """Single connected CHIME+gap+DSA data panel (waterfall + marginals)."""
    return plot_codetection(
        bands,
        columns=("data",),
        show_model_on_data=False,
        per_band_scale=per_band_scale,
        symmetric_time_axis=False,
        band_labels=True,
        show_column_titles=False,
        per_band_marginals=True,
        title=None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Self-contained demo (synthetic co-detection) so the module can be run as
# `python -m flits.batch.codetection_plots` to preview the layout without data.
# ---------------------------------------------------------------------------
def _demo_bands(seed: int = 7):
    rng = np.random.default_rng(seed)
    t = np.linspace(-3, 18, 360)

    def _profile(t, t0, sig, tau):
        dt = t[1] - t[0]
        g = np.exp(-0.5 * ((t - t0) / sig) ** 2)
        if tau <= 1e-3:
            return g
        tk = np.arange(0.0, 8 * tau + dt, dt)
        h = np.exp(-tk / tau)
        h /= h.sum()
        return np.convolve(g, h, "full")[: len(t)]

    def _band(frange, nf, tau1, dnu, contrast, amp, snr):
        f = np.linspace(*frange, nf)
        env = np.zeros(nf)
        for _ in range(3):
            c = rng.uniform(*frange)
            w = rng.uniform(0.15, 0.4) * (frange[1] - frange[0])
            env += rng.uniform(0.5, 1.0) * np.exp(-0.5 * ((f - c) / w) ** 2)
        env = 0.25 + 0.75 * env / env.max()
        k = np.exp(-0.5 * (np.arange(-40, 41) / max(1, dnu / ((frange[1] - frange[0]) / nf))) ** 2)
        k /= k.sum()
        scint = np.clip(1 + contrast * np.convolve(rng.standard_normal(nf), k, "same"), 0.05, None)
        m = np.array([_profile(t, 0.0, 0.35, tau1 * (fi / 1000.0) ** -4.0) * env[i] * scint[i]
                      for i, fi in enumerate(f)])
        m *= amp / m.max()
        sigma = m.max() / snr
        return BandSpectrum(f, t, m + rng.normal(0, sigma, m.shape), m, sigma=sigma)

    chime = _band((400.0, 800.0), 256, 0.7, 2.0, 0.9, 1.0, 24)
    chime.label = "CHIME"
    dsa = _band((1311.0, 1498.75), 160, 0.7, 18.0, 0.55, 0.9, 24)
    dsa.label = "DSA"
    return [chime, dsa]


if __name__ == "__main__":
    fig = plot_codetection(_demo_bands(), title="Co-detection layout demo (synthetic)")
    out = "codetection_demo.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")
