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
from scat_analysis.burstfit import FRBModel
from scat_analysis.config_utils import load_telescope_block
from scat_analysis.pipeline.io import BurstDataset
from scipy.ndimage import gaussian_filter1d

# --- S/N-driven resolution knobs (stated, not tuned per burst) --------------
# Peak S/N floor the on-pulse must clear at the chosen binning. The rule picks
# the FINEST binning still meeting this, so a higher floor -> coarser bins. 10 is
# a conservative "structure is real, not noise" bar for the manuscript figures.
# FLITS_SNR_TARGET overrides the floor (opt-in): LOWER it to let a finer f clear
# the per-channel S/N gate, so a bright/clean burst keeps more channels. Used by
# the finest-vs-coarsest f A/B and, if that wins, the fleet mass-refit.
SNR_TARGET = float(os.environ.get("FLITS_SNR_TARGET", 10.0))
# Frequency channels kept per band are bounded: >= MIN so the intra-band
# scattering slope (and the CHIME<->DSA alpha lever) is constrained; <= MAX so a
# bright burst is not split into per-channel-starved bins. Native counts are
# 1024 (CHIME) and 6144 (DSA); the block factor is chosen from these bounds.
# FLITS_MAX_CHANNELS overrides the cap (opt-in): RAISE it to allow a finer f
# (more, narrower channels -> less intra-channel scattering smearing) on a burst
# already pinned at the default cap.
MIN_CHANNELS = 8
MAX_CHANNELS = int(os.environ.get("FLITS_MAX_CHANNELS", 64))
# Candidate block factors are powers of two (native counts are powers-of-two
# multiples of these), so binning stays an exact block average.
_POW2 = tuple(1 << k for k in range(0, 14))  # 1, 2, 4, ..., 8192
# Cap on time bins kept in the fit window. The scattering/burst features live at
# ~50 us-1 ms, so finer than this buys no physics but multiplies the per-likelihood
# cost (CHIME native is 2.56 us -> ~8000 bins over a 21 ms window). 512 keeps the
# fit tractable and matches the sample counts the legacy fixed factors produced
# (~90-875 bins); the S/N floor still coarsens below this for faint bursts.
MAX_TIME_BINS = 512
# Fraction of a coarse channel's fine channels that must survive upstream RFI
# flagging for the channel to be kept. RFI-flagged fine channels arrive here as
# exact all-zero rows (io.py off-pulse z-scores each fine channel and
# nan_to_num's an all-NaN flagged channel to 0). The frequency block-average is
# taken over the SURVIVING fine channels only (mask-aware), so a partially
# flagged coarse channel keeps its true amplitude instead of being diluted by
# the flagged fraction -- the plain block-mean's amplitude bias, worst at low
# frequency where RFI is heaviest and the scattering lever arm lives, which
# railed alpha/beta at the thin-screen prior corner. A coarse channel below this
# floor is zeroed so FRBModel's noise>1e-9 valid mask drops it; between the floor
# and 1.0 it keeps the mask-corrected amplitude and is down-weighted by its
# (higher, fewer-fine-channel) measured off-pulse noise.
MIN_VALID_FRAC = 0.25

# --- Robust-window knobs ----------------------------------------------------
WIN_K_HI = 5.0          # core threshold (sigma over baseline): bright burst body
WIN_K_LO = 1.5          # tail threshold (sigma): follow the scattering tail down
WIN_MAX_GAP_MS = 0.02   # tight tail contiguity; native CHIME noise must not chain to the cap
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
    median_channel_snr: float
    snr_target: float
    snr_qualified: bool

    def caption(self) -> str:
        return (
            f"{self.n_chan} ch x {self.df_MHz:.1f} MHz, "
            f"dt={self.dt_ms * 1e3:.1f} us (f{self.f_factor}/t{self.t_factor}); "
            f"window {self.window_ms:.1f} ms; peak S/N {self.peak_pixel_snr:.0f}/px; "
            f"S/N-qualified={self.snr_qualified} (target {self.snr_target:g})"
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
    # Bursts rise quickly: keep the leading edge at the high-threshold core.
    lo = core_lo
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


def _masked_downsample(
    data: np.ndarray, f_factor: int, t_factor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mask-aware block-average mirroring :func:`downsample`, but excluding
    upstream-RFI-flagged fine channels from the FREQUENCY average.

    Flagged fine channels reach the joint prep as exact all-zero rows (io.py
    off-pulse z-scores each fine channel and nan_to_num's an all-NaN flagged
    channel to 0). The stock ``downsample`` block-means over all fine channels
    including those zeros, so a coarse channel's amplitude is diluted by its
    flagged fraction -- worst at low frequency where RFI is heaviest, exactly
    where the scattering lever arm lives, which drove alpha/beta to the
    thin-screen prior corner. Here the time block-mean stays plain (a flagged
    fine channel is zero at every time sample, so it only biases the frequency
    direction) while the frequency average runs over the surviving fine channels
    only. For a fully clean coarse block this reduces EXACTLY to ``downsample``.

    Returns ``(binned, valid_frac)``: the coarse array (freq axis length
    unchanged; fully-flagged blocks left as zero rows) and, per coarse channel,
    the fraction of its ``f_factor`` fine channels that survived flagging.
    """
    if f_factor == 1 and t_factor == 1:
        arr = np.asarray(data, float)
        return arr, np.ones(arr.shape[0])
    nf, nt = data.shape
    nf_new = nf - (nf % f_factor)  # mirror downsample: drop the tail remainder
    nt_new = nt - (nt % t_factor)
    d = (
        data[:nf_new, :nt_new]
        .reshape(nf_new // f_factor, f_factor, nt_new // t_factor, t_factor)
        .mean(axis=3)
    )  # plain time block-mean -> (ncoarse, f_factor, ntime)
    good = np.any(d != 0.0, axis=2)  # surviving fine channels per (coarse, fine-in-block)
    cnt = good.sum(axis=1).astype(float)  # surviving fine channels per coarse channel
    dsum = d.sum(axis=1)  # sum over fine channels (flagged rows contribute 0)
    binned = np.where(cnt[:, None] > 0.0, dsum / np.clip(cnt[:, None], 1.0, None), 0.0)
    valid_frac = cnt / float(f_factor)
    return binned, valid_frac


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


def resolution_snr_status(
    data: np.ndarray, win: tuple[int, int], snr_target: float
) -> tuple[float, float, bool]:
    """Return profile/channel S/N and whether both clear the publication floor."""
    profile_snr = _profile_peak_snr(data, win)
    channel_snr = _median_channel_snr(data, win)
    return profile_snr, channel_snr, bool(
        profile_snr >= snr_target and channel_snr >= snr_target
    )


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

    Search time factors from finest to coarsest. At each time factor, keep the
    MOST channels (finest ``f_factor``) within ``[min_channels, max_channels]``
    for which BOTH the band-integrated profile and median on-pulse channel clear
    ``snr_target``. This joint search matters for faint bursts: a time factor can
    clear the profile floor while every frequency choice still fails the channel
    floor. In that case, continue to coarser time bins instead of returning a
    combination that the publication gate must reject.

    The qualification calculation uses the same mask-aware downsampling as the
    final model. If no candidate clears both floors, return the candidate whose
    weaker statistic is strongest; the caller records it as explicitly
    unqualified rather than silently publishing it.
    """
    nf, nt = data_native.shape

    # A floor keeps the window under MAX_TIME_BINS (tractability + no
    # sub-feature over-resolution).
    win_native_span = max(1, win_native[1] - win_native[0])
    t_floor = max(1, int(np.ceil(win_native_span / MAX_TIME_BINS)))
    t_cands = [t for t in _POW2 if t >= t_floor and t <= max(t_floor, nt // 8)]
    if not t_cands:
        t_cands = [next(t for t in _POW2 if t >= t_floor)]
    f_cands = sorted(
        f for f in _POW2 if nf % f == 0 and min_channels <= nf // f <= max_channels
    )
    if not f_cands:  # native count not divisible into the band bounds; nearest
        f_cands = [max(1, int(round(nf / np.sqrt(min_channels * max_channels))))]

    candidates: list[tuple[float, int, int]] = []
    for t in t_cands:  # smallest first = finest time resolution
        for f in f_cands:  # smallest first = most frequency channels
            binned, valid_frac = _masked_downsample(data_native, f, t)
            binned[valid_frac < MIN_VALID_FRAC] = 0.0
            win_ds = _downsample_window(win_native, t, binned.shape[1])
            if win_ds[1] - win_ds[0] < 3:
                continue
            profile_snr, channel_snr, qualified = resolution_snr_status(
                binned, win_ds, snr_target
            )
            candidates.append((min(profile_snr, channel_snr), f, t))
            if qualified:
                return int(f), int(t)

    if not candidates:
        return int(f_cands[-1]), int(t_cands[0])
    _, best_f, best_t = max(candidates, key=lambda item: (item[0], -item[2], -item[1]))
    return int(best_f), int(best_t)


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
    snr_target: float


def _probe_band(cfg_path: str, name: str, outdir: str, *, snr_target: float) -> _Probe:
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
    f_factor, t_factor = choose_resolution(native, win, native.shape[0], snr_target=snr_target)
    return _Probe(native, dt_native, tel, peak, win, f_factor, t_factor, snr_target)


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
    binned, valid_frac = _masked_downsample(p.native, p.f_factor, t_factor)
    # Zero (drop) coarse channels too heavily flagged to trust; FRBModel's
    # noise>1e-9 valid mask then excludes them. The freq axis keeps its length so
    # the linspace below stays a valid uniform f_min->f_max map for the survivors.
    binned[valid_frac < MIN_VALID_FRAC] = 0.0
    win_ds = _downsample_window(win_native, t_factor, binned.shape[1])
    crop_qualified = win_ds[1] - win_ds[0] >= 3
    if not crop_qualified:  # retain a debuggable model, but fail the science gate
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
    peak_profile_snr, median_channel_snr, snr_qualified = resolution_snr_status(
        binned, win_ds, p.snr_target
    )
    snr_qualified = bool(crop_qualified and snr_qualified)
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
        peak_profile_snr=peak_profile_snr,
        median_channel_snr=median_channel_snr,
        snr_target=p.snr_target,
        snr_qualified=snr_qualified,
    )
    return model, meta


def prepare_band(
    cfg_path: str,
    name: str,
    outdir: str,
    *,
    snr_target: float = SNR_TARGET,
) -> tuple[FRBModel, BandPrep]:
    """Single-band drop-in for the legacy ``prepare``: S/N resolution + robust
    per-band window. (The joint path uses :func:`prepare_pair`, which additionally
    reconciles the two bands onto a common display window.)"""
    p = _probe_band(cfg_path, name, outdir, snr_target=snr_target)
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
    snr_target: float = SNR_TARGET,
) -> tuple[tuple[FRBModel, BandPrep], tuple[FRBModel, BandPrep]]:
    """Prepare CHIME and DSA together for a joint fit/figure.

    Each band gets its own S/N-driven resolution and robust on-pulse window; the
    two windows are then unioned into a common peak-relative span so the joint
    figure never hatches one band where the other has signal (owner complaint #1),
    and the joint likelihood sees each band's full burst + scattering tail (no
    clipping)."""
    pC = _probe_band(cfg_C, f"{name}_chime", outdir, snr_target=snr_target)
    pD = _probe_band(cfg_D, f"{name}_dsa", outdir, snr_target=snr_target)
    winC, winD = _common_peak_relative_window([pC, pD])
    # The common span can contain more off-pulse baseline than either band's
    # own window. Re-select against the exact final windows so a choice that only
    # qualified before reconciliation cannot slip into the fit contract.
    pC.f_factor, pC.t_factor = choose_resolution(
        pC.native, winC, pC.native.shape[0], snr_target=snr_target
    )
    pD.f_factor, pD.t_factor = choose_resolution(
        pD.native, winD, pD.native.shape[0], snr_target=snr_target
    )
    return _build_model(pC, winC), _build_model(pD, winD)


def _env_auto() -> bool:
    """Whether callers should use the S/N-driven path (default ON)."""
    return os.environ.get("FLITS_JOINT_AUTO_TF", "1") == "1"
