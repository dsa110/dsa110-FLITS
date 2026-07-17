#!/usr/bin/env python
"""S/N-driven resolution + robust tail-aware on-pulse window for joint CHIME+DSA fits.

Motivation (2026-07 manuscript review). The manuscript joint-fit figures exposed
two undocumented, unstable heuristics in the joint-fit preprocessing:

1. **Window.** ``BurstDataset._crop_on_pulse`` (pipeline/io.py) set the on-pulse
   span to the global ``[min, max]`` of *every* time bin clearing 3-sigma, then
   padded by ``0.5 * span``. A single far off-pulse noise sample above 3-sigma
   blew the window open (DSA/whitney -> 57 ms, 875 samples) while a clean sharp
   burst collapsed it (DSA/isha -> 0.6 ms, 9 samples). Run per band, CHIME and
   DSA landed on wildly different time extents, so the joint figure hatched one
   band across times where the other plainly showed burst signal, and the fit
   likelihood was dominated either by off-pulse baseline (over-wide) or clipped
   the scattering tail (over-narrow). The clipped CHIME window is why e.g.
   whitney's CHIME ``t0`` came back at -5.2 +/- 6.9 ms (unconstrained, outside
   its own 5.5 ms window) while the DSA ``t0`` was +/- 0.0016 ms.

2. **Resolution.** ``f_factor`` / ``t_factor`` were hard-coded per band (CHIME
   64/24, DSA 384/2) regardless of burst brightness, giving peak per-pixel S/N
   from ~5 (oran CHIME: burst lost in noise, over-resolved) to ~95 (freya CHIME:
   structure averaged away, under-resolved). No S/N justification, and casey
   silently diverged at 32/4.

This module replaces both with stated, S/N-driven rules applied identically to
every burst and both bands (no per-burst hand-tuning):

* :func:`robust_onpulse_bounds` -- a peak-anchored, contiguous, hysteresis-tail
  window. The core is grown outward from the profile peak while it stays above a
  high threshold; the trailing edge is then extended down a low threshold to
  capture the scattering tail, with a bounded gap tolerance and an absolute cap
  so an isolated off-pulse spike can neither open the window nor collapse it.

* :func:`choose_resolution` -- the finest ``(t_factor, f_factor)`` whose on-pulse
  S/N clears a fixed floor: the finest binning the burst brightness honestly
  supports. Time binning is refined until the band-integrated profile peak drops
  to the S/N floor; frequency binning keeps the most channels (within a band-count
  cap) whose median on-pulse channel still clears the floor.

Both are consumed by :func:`prepare_band`, a drop-in replacement for the
``prepare`` in ``run_joint_fit.py`` / ``dump_jointmodel.py`` so the FIT and the
FIGURE share one preprocessing. The chosen factors and window are returned as
metadata for the figure captions and the fit log.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import yaml
from scat_analysis.burstfit import FRBModel, downsample
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset
from scipy.ndimage import gaussian_filter1d

# --- S/N-driven resolution knobs (stated, not tuned per burst) --------------
# Peak S/N floor the on-pulse must clear at the chosen binning. The rule picks
# the FINEST binning still meeting this, so a higher floor -> coarser bins. 10 is
# a conservative "structure is real, not noise" bar for the manuscript figures.
SNR_TARGET = 10.0
# Frequency channels kept per band are bounded: >= MIN so the intra-band
# scattering slope (and the CHIME<->DSA alpha lever) is constrained; <= MAX so a
# bright burst is not split into per-channel-starved bins. Native counts are
# 1024 (CHIME) and 6144 (DSA); the block factor is chosen from these bounds.
MIN_CHANNELS = 8
MAX_CHANNELS = 64
# Candidate block factors are powers of two (native counts are powers-of-two
# multiples of these), so binning stays an exact block average.
_POW2 = tuple(1 << k for k in range(0, 14))  # 1, 2, 4, ..., 8192
# Cap on time bins kept in the fit window. The scattering/burst features live at
# ~50 us-1 ms, so finer than this buys no physics but multiplies the per-likelihood
# cost (CHIME native is 2.56 us -> ~8000 bins over a 21 ms window). 512 keeps the
# fit tractable and matches the sample counts the legacy fixed factors produced
# (~90-875 bins); the S/N floor still coarsens below this for faint bursts.
MAX_TIME_BINS = 512

# --- Robust-window knobs ----------------------------------------------------
WIN_K_HI = 5.0          # core threshold (sigma over baseline): bright burst body
WIN_K_LO = 1.5          # tail threshold (sigma): follow the scattering tail down
WIN_MAX_GAP_MS = 1.0    # tolerated sub-threshold gap before an edge stops growing
WIN_MARGIN_FRAC = 0.4   # off-pulse margin added each side, as a fraction of span
WIN_TRAIL_CAP_MS = 30.0 # absolute cap on the trailing extension past the core
WIN_MIN_OFFPULSE_FRAC = 0.15  # keep at least this fraction of the window off-pulse


@dataclass
class BandPrep:
    """Preprocessing outcome for one band (returned beside the FRBModel)."""

    f_factor: int
    t_factor: int
    n_chan: int
    df_MHz: float
    dt_ms: float
    n_time: int
    window_ms: float
    win_lo_native: int
    win_hi_native: int
    peak_pixel_snr: float
    peak_profile_snr: float

    def caption(self) -> str:
        return (
            f"{self.n_chan} ch x {self.df_MHz:.1f} MHz, "
            f"dt={self.dt_ms * 1e3:.1f} us (f{self.f_factor}/t{self.t_factor}); "
            f"window {self.window_ms:.1f} ms; peak S/N {self.peak_pixel_snr:.0f}/px"
        )


def _band_profile(data: np.ndarray, dt_ms: float, smooth_ms: float = 0.1) -> np.ndarray:
    """Band-integrated, lightly smoothed time profile."""
    prof = np.nansum(np.asarray(data, float), axis=0)
    if smooth_ms > 0 and dt_ms > 0:
        prof = gaussian_filter1d(prof, sigma=(smooth_ms / 2.355) / dt_ms)
    return prof


def _baseline(prof: np.ndarray) -> tuple[float, float]:
    """Robust (median, MAD-sigma) baseline from the outer quarters."""
    n = prof.size
    q = max(n // 4, 1)
    base = np.r_[prof[:q], prof[-q:]]
    mu = float(np.median(base))
    sig = float(1.4826 * np.median(np.abs(base - mu)))
    return mu, sig


def _grow_edge(
    excess: np.ndarray, start: int, direction: int, thr: float, max_gap: int, cap: int
) -> int:
    """Grow a contiguous edge from ``start`` while ``excess`` stays above ``thr``.

    ``max_gap`` consecutive sub-threshold samples are tolerated (so a scintillation
    null or an inter-component dip does not truncate the span); ``cap`` bounds how
    far past ``start`` the edge may travel (guards against a runaway off-pulse
    spike). Returns the last in-span index (inclusive)."""
    n = excess.size
    edge = start
    gap = 0
    i = start
    while 0 <= i < n and abs(i - start) <= cap:
        if excess[i] > thr:
            edge = i
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
        i += direction
    return edge


def robust_onpulse_bounds(
    prof: np.ndarray,
    dt_ms: float,
    *,
    k_hi: float = WIN_K_HI,
    k_lo: float = WIN_K_LO,
    max_gap_ms: float = WIN_MAX_GAP_MS,
    margin_frac: float = WIN_MARGIN_FRAC,
    trail_cap_ms: float = WIN_TRAIL_CAP_MS,
    min_offpulse_frac: float = WIN_MIN_OFFPULSE_FRAC,
) -> tuple[int, int]:
    """Peak-anchored, contiguous, tail-aware on-pulse window.

    Returns ``(lo, hi)`` inclusive-exclusive sample bounds. The window is anchored
    at the profile peak and grown outward: the leading edge follows the high
    threshold ``k_hi`` (bursts rise fast, no leading tail), the trailing edge is
    extended down to the low threshold ``k_lo`` to follow the exponential
    scattering tail. Contiguity + a bounded gap keep an isolated off-pulse sample
    from opening the window; the peak anchor keeps a sharp burst from collapsing
    it. A symmetric off-pulse margin is added for the noise estimate."""
    n = prof.size
    mu, sig = _baseline(prof)
    if sig <= 0 or n < 4:
        return 0, n
    excess = prof - mu
    peak = int(np.argmax(excess))
    if excess[peak] <= k_hi * sig:
        # Faint burst: fall back to the low threshold around the peak.
        if excess[peak] <= k_lo * sig:
            return 0, n
        k_hi = k_lo
    max_gap = max(int(round(max_gap_ms / dt_ms)), 1)
    cap = max(int(round(trail_cap_ms / dt_ms)), 1)
    # Core: contiguous body above the high threshold, anchored at the peak.
    core_lo = _grow_edge(excess, peak, -1, k_hi * sig, max_gap, cap)
    core_hi = _grow_edge(excess, peak, +1, k_hi * sig, max_gap, cap)
    # Leading edge to the low threshold; trailing edge follows the tail down.
    lo = _grow_edge(excess, core_lo, -1, k_lo * sig, max_gap, cap)
    hi = _grow_edge(excess, core_hi, +1, k_lo * sig, max_gap, cap)
    span = hi - lo + 1
    margin = int(round(margin_frac * span))
    lo2 = max(0, lo - margin)
    hi2 = min(n, hi + margin + 1)
    # Guarantee some off-pulse baseline for the noise estimate.
    win = hi2 - lo2
    need = int(np.ceil(min_offpulse_frac * win))
    on = hi - lo + 1
    if win - on < need:
        extra = int(np.ceil((need - (win - on)) / 2))
        lo2 = max(0, lo2 - extra)
        hi2 = min(n, hi2 + extra)
    return lo2, hi2


def _channel_noise(data: np.ndarray, win: tuple[int, int]) -> np.ndarray:
    """Per-channel MAD noise from the off-pulse (outside ``win``), full array."""
    lo, hi = win
    n = data.shape[1]
    off = np.r_[0:lo, hi:n]
    if off.size < max(8, n // 8):  # too little off-pulse -> use outer quarters
        q = max(n // 4, 1)
        off = np.r_[0:q, n - q : n]
    seg = data[:, off]
    med = np.median(seg, axis=1, keepdims=True)
    return 1.4826 * np.median(np.abs(seg - med), axis=1)


def _peak_pixel_snr(data_ds: np.ndarray, win_ds: tuple[int, int]) -> float:
    """Brightest on-pulse pixel over its channel's off-pulse noise."""
    noise = _channel_noise(data_ds, win_ds)
    lo, hi = win_ds
    on = data_ds[:, lo:hi] / np.clip(noise[:, None], 1e-9, None)
    return float(np.nanmax(on))


def _median_channel_snr(data_ds: np.ndarray, win_ds: tuple[int, int]) -> float:
    """Median over on-pulse channels of each channel's peak-pixel S/N."""
    noise = _channel_noise(data_ds, win_ds)
    lo, hi = win_ds
    on = data_ds[:, lo:hi] / np.clip(noise[:, None], 1e-9, None)
    chan_peak = np.nanmax(on, axis=1)
    # Restrict to channels that actually carry signal (peak above a low bar), so a
    # dead/edge channel does not drag the median under the floor.
    live = chan_peak[chan_peak > 3.0]
    return float(np.median(live)) if live.size else float(np.median(chan_peak))


def _profile_peak_snr(data: np.ndarray, win: tuple[int, int]) -> float:
    """Band-integrated profile peak over the profile's off-pulse noise."""
    prof = np.nansum(np.asarray(data, float), axis=0)
    lo, hi = win
    n = prof.size
    off = np.r_[0:lo, hi:n]
    if off.size < max(8, n // 8):
        q = max(n // 4, 1)
        off = np.r_[0:q, n - q : n]
    mu = np.median(prof[off])
    sig = 1.4826 * np.median(np.abs(prof[off] - mu))
    return float((np.max(prof[lo:hi]) - mu) / max(sig, 1e-9))


def _downsample_window(win: tuple[int, int], t_factor: int, n_ds: int) -> tuple[int, int]:
    lo, hi = win
    return max(0, lo // t_factor), min(n_ds, int(np.ceil(hi / t_factor)))


def choose_resolution(
    data_native: np.ndarray,
    win_native: tuple[int, int],
    n_ch_raw: int,
    *,
    snr_target: float = SNR_TARGET,
    min_channels: int = MIN_CHANNELS,
    max_channels: int = MAX_CHANNELS,
) -> tuple[int, int]:
    """Finest ``(f_factor, t_factor)`` whose on-pulse S/N clears ``snr_target``.

    Time: pick the finest ``t_factor`` (power of two) whose band-integrated profile
    peak per-bin S/N still clears ``snr_target`` -- the finest time binning the
    burst brightness supports (bright -> fine bins resolve the tail; faint ->
    coarser bins respect the S/N floor). Frequency: at that ``t_factor``, keep the
    MOST channels (finest ``f_factor``) within ``[min_channels, max_channels]``
    whose median on-pulse channel still clears ``snr_target``; if none do, fall
    back to the coarsest allowed (fewest, brightest channels)."""
    nf, nt = data_native.shape

    # --- time factor: finest (smallest t) that clears the profile S/N floor ---
    # Per-bin profile S/N rises with coarser binning (more averaging) until the
    # bin outgrows the burst, so the smallest passing t is the finest honest
    # binning. If none pass (faint burst), take the t that maximizes S/N. A floor
    # keeps the window under MAX_TIME_BINS (tractability + no sub-feature over-res).
    win_native_span = max(1, win_native[1] - win_native[0])
    t_floor = max(1, int(np.ceil(win_native_span / MAX_TIME_BINS)))
    t_cands = [t for t in _POW2 if t >= t_floor and t <= max(t_floor, nt // 8)]
    if not t_cands:
        t_cands = [next(t for t in _POW2 if t >= t_floor)]
    t_snr: list[tuple[int, float]] = []
    chosen_t = None
    for t in t_cands:
        d_t = downsample(data_native, 1, t)
        win_t = _downsample_window(win_native, t, d_t.shape[1])
        if win_t[1] - win_t[0] < 3:
            continue
        snr = _profile_peak_snr(d_t, win_t)
        t_snr.append((t, snr))
        if chosen_t is None and snr >= snr_target:
            chosen_t = t
    if chosen_t is None:
        chosen_t = max(t_snr, key=lambda x: x[1])[0] if t_snr else t_cands[0]

    # --- freq factor at the chosen time binning: most channels (smallest f)
    # whose median on-pulse channel still clears the floor; else fewest (coarsest).
    f_cands = sorted(
        f for f in _POW2 if nf % f == 0 and min_channels <= nf // f <= max_channels
    )
    if not f_cands:  # native count not divisible into the band bounds; nearest
        f_cands = [max(1, int(round(nf / np.sqrt(min_channels * max_channels))))]
    d_full_t = downsample(data_native, 1, chosen_t)
    win_ct = _downsample_window(win_native, chosen_t, d_full_t.shape[1])
    chosen_f = f_cands[-1]  # coarsest allowed = fewest, brightest channels
    for f in f_cands:  # smallest f first = most channels
        d_ft = downsample(d_full_t, f, 1)
        if _median_channel_snr(d_ft, win_ct) >= snr_target:
            chosen_f = f
            break
    return int(chosen_f), int(chosen_t)


@dataclass
class _Probe:
    """Native-resolution load + resolution/window decision for one band."""

    native: np.ndarray
    dt_native: float
    tel: object
    peak: int
    win: tuple[int, int]
    f_factor: int
    t_factor: int


def _probe_band(
    cfg_path: str, name: str, outdir: str, *, auto: bool, snr_target: float
) -> _Probe:
    """Native load (bandpass-corrected, trimmed, centered, uncropped) + decide the
    resolution factors and the robust on-pulse window. One decode per band."""
    cfg = yaml.safe_load(open(cfg_path))
    tel = load_telescope_block(cfg["telcfg_path"], cfg["telescope"])
    ds = BurstDataset(
        cfg["path"],
        outdir,
        name=name,
        telescope=tel,
        f_factor=1,
        t_factor=1,
        outer_trim=float(cfg.get("outer_trim", 0.15)),
        onpulse_crop=False,
    )
    native = np.asarray(ds.data, float)
    dt_native = float(ds.dt_ms)
    prof = _band_profile(native, dt_native)
    peak = int(np.argmax(prof - _baseline(prof)[0]))
    win = robust_onpulse_bounds(prof, dt_native)
    if auto:
        f_factor, t_factor = choose_resolution(native, win, native.shape[0], snr_target=snr_target)
    else:
        f_factor, t_factor = int(cfg["f_factor"]), int(cfg["t_factor"])
    return _Probe(native, dt_native, tel, peak, win, f_factor, t_factor)


def _build_model(p: _Probe, win_native: tuple[int, int]) -> tuple[FRBModel, BandPrep]:
    """Bin the native array to the chosen resolution, crop to ``win_native``, and
    build the ``FRBModel`` (native df_MHz for intra-channel smearing, robust
    full-window off-pulse noise)."""
    # Enforce the time-bin cap against the FINAL (possibly common) window: the
    # per-band resolution choice used each band's own window, but the reconciled
    # common window can be wider, so re-check tractability here.
    t_factor = p.t_factor
    span = max(1, win_native[1] - win_native[0])
    while span // t_factor > MAX_TIME_BINS:
        t_factor *= 2
    binned = downsample(p.native, p.f_factor, t_factor)
    win_ds = _downsample_window(win_native, t_factor, binned.shape[1])
    if win_ds[1] - win_ds[0] < 3:  # degenerate crop guard
        win_ds = (0, binned.shape[1])
    noise_ds = _channel_noise(binned, win_ds)
    lo, hi = win_ds
    data_c = binned[:, lo:hi]
    dt_ms = p.dt_native * t_factor
    freq = np.linspace(p.tel.f_min_GHz, p.tel.f_max_GHz, binned.shape[0])
    time = np.arange(data_c.shape[1]) * dt_ms
    model = FRBModel(
        time=time, freq=freq, data=data_c, df_MHz=p.tel.df_MHz_raw, noise_std=noise_ds
    )
    meta = BandPrep(
        f_factor=p.f_factor,
        t_factor=t_factor,
        n_chan=binned.shape[0],
        df_MHz=p.tel.df_MHz_raw * p.f_factor,
        dt_ms=dt_ms,
        n_time=data_c.shape[1],
        window_ms=data_c.shape[1] * dt_ms,
        win_lo_native=int(win_native[0]),
        win_hi_native=int(win_native[1]),
        peak_pixel_snr=_peak_pixel_snr(binned, win_ds),
        peak_profile_snr=_profile_peak_snr(binned, win_ds),
    )
    return model, meta


def prepare_band(
    cfg_path: str,
    name: str,
    outdir: str,
    *,
    auto: bool = True,
    snr_target: float = SNR_TARGET,
) -> tuple[FRBModel, BandPrep]:
    """Single-band drop-in for the legacy ``prepare``: S/N resolution + robust
    per-band window. (The joint path uses :func:`prepare_pair`, which additionally
    reconciles the two bands onto a common display window.)"""
    p = _probe_band(cfg_path, name, outdir, auto=auto, snr_target=snr_target)
    return _build_model(p, p.win)


def _common_peak_relative_window(probes: list[_Probe]) -> list[tuple[int, int]]:
    """Union each band's on-pulse window in peak-relative milliseconds, then map
    the common span back to each band's native samples.

    This is what removes the spurious cross-band hatching: both bands end up
    covering the identical peak-relative time span, so on the peak/TOA-aligned
    figure neither band is blank where the other shows signal. The band with the
    shorter intrinsic burst simply shows real off-pulse baseline over the other
    band's scattering tail (correct), instead of a hatched "no data" patch."""
    left = min((pb.win[0] - pb.peak) * pb.dt_native for pb in probes)
    right = max((pb.win[1] - pb.peak) * pb.dt_native for pb in probes)
    out: list[tuple[int, int]] = []
    for pb in probes:
        lo = pb.peak + int(np.floor(left / pb.dt_native))
        hi = pb.peak + int(np.ceil(right / pb.dt_native))
        lo = max(0, lo)
        hi = min(pb.native.shape[1], hi)
        out.append((lo, hi))
    return out


def prepare_pair(
    cfg_C: str,
    cfg_D: str,
    name: str,
    outdir: str,
    *,
    auto: bool = True,
    snr_target: float = SNR_TARGET,
    common_window: bool = True,
) -> tuple[tuple[FRBModel, BandPrep], tuple[FRBModel, BandPrep]]:
    """Prepare CHIME and DSA together for a joint fit/figure.

    Each band gets its own S/N-driven resolution and robust on-pulse window; the
    two windows are then unioned into a common peak-relative span so the joint
    figure never hatches one band where the other has signal (owner complaint #1),
    and the joint likelihood sees each band's full burst + scattering tail (no
    clipping). Set ``common_window=False`` to keep strictly per-band windows."""
    pC = _probe_band(cfg_C, f"{name}_chime", outdir, auto=auto, snr_target=snr_target)
    pD = _probe_band(cfg_D, f"{name}_dsa", outdir, auto=auto, snr_target=snr_target)
    if common_window:
        winC, winD = _common_peak_relative_window([pC, pD])
    else:
        winC, winD = pC.win, pD.win
    return _build_model(pC, winC), _build_model(pD, winD)


def _env_auto() -> bool:
    """Whether callers should use the S/N-driven path (default ON)."""
    return os.environ.get("FLITS_JOINT_AUTO_TF", "1") == "1"
