"""
codetection_plots.py
====================

Unified multi-band dynamic-spectrum figure for FRB co-detections (e.g. CHIME +
DSA-110), showing the observed data, the best-fit 2-D model, and the residuals
side by side.

Each column (data | model | residual) carries a frequency-averaged **time
series** along the top and a time-averaged **spectrum** down the right side.
Bands are drawn to scale on a shared frequency axis, and any *unobserved gap*
between bands (e.g. the 0.80-1.28 GHz gap between the CHIME and DSA bands) is
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
All bands are assumed to share a common (dedispersed) time axis, i.e. the same
number of time samples; their frequency channel counts may differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

__all__ = ["BandSpectrum", "plot_codetection"]


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
    sigma : float, optional
        Per-sample noise level. If ``None`` it is estimated from the robust
        (MAD-based) scatter of ``data - model``.
    label : str, optional
        Short band label, e.g. ``"CHIME"`` / ``"DSA"``.
    """

    freq_mhz: np.ndarray
    time_ms: np.ndarray
    data: np.ndarray
    model: np.ndarray
    sigma: Optional[float] = None
    label: Optional[str] = None

    @property
    def frange(self) -> tuple[float, float]:
        return float(np.min(self.freq_mhz)), float(np.max(self.freq_mhz))

    def noise(self) -> float:
        """Per-sample noise: ``sigma`` if given, else a robust MAD estimate."""
        if self.sigma is not None:
            return float(self.sigma)
        resid = self.data - self.model
        mad = np.median(np.abs(resid - np.median(resid)))
        s = 1.4826 * mad
        return float(s if s > 0 else (resid.std() or 1.0))


def _band_array(b: BandSpectrum, key: str) -> np.ndarray:
    if key == "resid":
        return (b.data - b.model) / b.noise()
    return getattr(b, key)


def _hatch_rect(ax, x0, x1, y0, y1):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="0.93",
                           edgecolor="0.55", hatch="////", lw=0.0, zorder=3))


def plot_codetection(
    bands: Sequence[BandSpectrum],
    *,
    water_cmap: str = "magma",
    resid_cmap: str = "RdBu_r",
    resid_clip: float = 5.0,
    per_band_scale: bool = False,
    show_model_on_data: bool = True,
    col_wspace: float = 0.15,
    marg_pad: float = 0.05,
    figsize: tuple[float, float] = (10.2, 4.7),
    title: Optional[str] = None,
    gap_label: bool = True,
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
    t0 = min(b.time_ms[0] for b in bands)
    t1 = max(b.time_ms[-1] for b in bands)
    fmin, fmax = bands[0].frange[0], bands[-1].frange[1]
    gaps = [(lo.frange[1], hi.frange[0]) for lo, hi in zip(bands[:-1], bands[1:])
            if hi.frange[0] > lo.frange[1] + 1e-6]

    # intensity scales
    all_model = np.concatenate([b.model.ravel() for b in bands])
    norm = Normalize(0, np.percentile(all_model, 99.5))
    bnorm = ({id(b): Normalize(0, np.percentile(b.model, 99.5)) for b in bands}
             if per_band_scale else None)
    rnorm = Normalize(-resid_clip, resid_clip)

    # marginal projections (channel-averaged profile; time-averaged spectrum)
    prof = {k: np.concatenate([_band_array(b, k) for b in bands], axis=0).mean(axis=0)
            for k in ("data", "model", "resid")}
    pmax = max(prof["data"].max(), prof["model"].max())
    smax = max(np.max(b.data.mean(1)) for b in bands)
    smax = max(smax, max(np.max(b.model.mean(1)) for b in bands))
    tref = bands[0].time_ms

    cols = [("data", "Data", water_cmap, norm),
            ("model", "Model (2-D fit)", water_cmap, norm),
            ("resid", r"Residual / $\sigma$", resid_cmap, rnorm)]

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(1, 3, wspace=col_wspace, left=0.06, right=0.985,
                             top=0.90, bottom=0.12)

    for j, (key, ctitle, cmap, nrm) in enumerate(cols):
        inner = outer[0, j].subgridspec(2, 2, width_ratios=[4, 1.05],
                                        height_ratios=[1, 4],
                                        wspace=marg_pad, hspace=marg_pad)
        ax_ts = fig.add_subplot(inner[0, 0])
        ax_wf = fig.add_subplot(inner[1, 0], sharex=ax_ts)
        ax_sp = fig.add_subplot(inner[1, 1], sharey=ax_wf)

        # --- waterfall (each band drawn to scale) ---
        for b in bands:
            use = bnorm[id(b)] if (bnorm is not None and key != "resid") else nrm
            ax_wf.imshow(_band_array(b, key), origin="lower", aspect="auto",
                         cmap=cmap, norm=use,
                         extent=[b.time_ms[0], b.time_ms[-1], b.frange[0], b.frange[1]])
        for g0, g1 in gaps:
            _hatch_rect(ax_wf, t0, t1, g0, g1)
            if gap_label and j == 0:
                ax_wf.text((t0 + t1) / 2, (g0 + g1) / 2, "no coverage",
                           ha="center", va="center", fontsize=6.5,
                           style="italic", color="0.4", zorder=4)
        ax_wf.set_ylim(fmin, fmax)
        ax_wf.set_xlim(t0, t1)
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
            ax_ts.plot(tref, prof[key], color="black", lw=0.8)
            if key == "data" and show_model_on_data:
                ax_ts.plot(tref, prof["model"], color="crimson", lw=0.9, alpha=0.85)
            ax_ts.set_ylim(-0.05 * pmax, 1.15 * pmax)
        ax_ts.set_xlim(t0, t1)
        ax_ts.tick_params(labelbottom=False, labelleft=False)
        ax_ts.set_title(ctitle, fontsize=10, pad=4)

        # --- right spectrum ---
        if key == "resid":
            ax_sp.axvline(0, color="0.6", lw=0.6)
            for b in bands:
                ax_sp.plot(_band_array(b, "resid").mean(1), b.freq_mhz,
                           color="#34495e", lw=0.7)
            ax_sp.set_xlim(-0.6 * resid_clip, 0.6 * resid_clip)
        else:
            for b in bands:
                ax_sp.plot(getattr(b, key).mean(1), b.freq_mhz, color="black", lw=0.7)
            if key == "data" and show_model_on_data:
                for b in bands:
                    ax_sp.plot(b.model.mean(1), b.freq_mhz, color="crimson", lw=0.8)
            ax_sp.set_xlim(-0.05 * smax, 1.2 * smax)
        for g0, g1 in gaps:
            ax_sp.axhspan(g0, g1, facecolor="0.93", edgecolor="0.55", hatch="////", lw=0)
        ax_sp.set_ylim(fmin, fmax)
        ax_sp.tick_params(labelleft=False, labelbottom=False)
        ax_sp.set_xlabel("spec.", fontsize=7)

    if title:
        fig.suptitle(title, fontsize=10, y=0.985)
    return fig


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
    dsa = _band((1280.0, 1530.0), 160, 0.7, 18.0, 0.55, 0.9, 24)
    dsa.label = "DSA"
    return [chime, dsa]


if __name__ == "__main__":
    fig = plot_codetection(_demo_bands(), title="Co-detection layout demo (synthetic)")
    out = "codetection_demo.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"wrote {out}")
