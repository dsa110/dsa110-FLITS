"""Automatic, off-pulse-defined RFI channel flagger for CHIME upchannelized spectra.

TRUST BASIS
-----------
1. Every flagging statistic is computed from the OFF-PULSE (burst-free) time range only, so the
   flagger structurally cannot remove burst signal. We verify this per burst: the overlap between
   flagged channels and the burst-bright channels (|z|>8 in the on-pulse spectrum) is reported and
   should be ~0.
2. Success is OBJECTIVE, not by eye: the off-pulse region is pure noise, so its frequency-ACF must
   be flat (white). We report the off-pulse ACF low-lag power BEFORE and AFTER flagging; a good flag
   drives it toward zero. RFI is exactly the excess correlation the flag removes.

METHOD (iterative, MAD-robust, off-pulse only)
  For each fine channel compute, over the off-pulse window:
     mu   = time-mean            (persistent bright/dead channels)
     sd   = time-std             (variable-gain / bursty RFI)
     tac1 = lag-1 temporal autocorr of the off-pulse time series (RFI is temporally correlated;
            thermal noise is not) — catches RFI that a mean/std cut misses.
  Robust z = (x - median)/(1.4826*MAD) across surviving channels; flag |z|>SIGMA on ANY statistic,
  iterate to convergence (<=ITERS). Dead channels (mu<=0 or non-finite) are flagged too.

This is a channel (frequency) flag. It is applied ON TOP of any existing pipeline mask.
"""
from __future__ import annotations
import numpy as np


def _mad_z(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x), med, 0.0
    return (x - med) / mad, med, mad


def offpulse_channel_stats(power, off_lims):
    """Per-channel off-pulse statistics. power: (nchan, ntime) masked array."""
    o0, o1 = int(off_lims[0]), int(off_lims[1])
    seg = power[:, o0:o1]
    mu = np.ma.filled(np.ma.mean(seg, axis=1), np.nan)
    sd = np.ma.filled(np.ma.std(seg, axis=1), np.nan)
    # lag-1 temporal autocorrelation per channel (normalized), off-pulse only
    x = seg - np.ma.mean(seg, axis=1, keepdims=True)
    x = np.ma.filled(x, 0.0)
    num = np.sum(x[:, 1:] * x[:, :-1], axis=1)
    den = np.sum(x * x, axis=1)
    tac1 = np.where(den > 0, num / den, 0.0)
    return mu, sd, tac1


def auto_flag(power, off_lims, sigma=5.0, iters=6):
    """Return (chan_mask, info). chan_mask True = flagged/RFI.

    Note: statistics are computed on the de-scalloped/baseline-subtracted spectrum, where a channel
    mean of ~0 is NORMAL (not 'dead'). Dead/gain-starved channels are those ALREADY fully masked by
    the pipeline (or with non-finite std), NOT those with small mean — so we do not cut on mu<=0.
    """
    nchan = power.shape[0]
    mu, sd, tac1 = offpulse_channel_stats(power, off_lims)
    # dead = already fully masked by the pipeline over the off-pulse window, or non-finite std
    pmask = power.mask if power.mask is not np.ma.nomask else np.zeros(power.shape, bool)
    o0, o1 = int(off_lims[0]), int(off_lims[1])
    already_dead = pmask[:, o0:o1].all(axis=1)
    flagged = already_dead | ~np.isfinite(sd) | (sd <= 0)
    for _ in range(iters):
        keep = ~flagged
        if keep.sum() < 10:
            break
        newly = np.zeros(nchan, bool)
        for stat in (sd, tac1):   # mu is ~0 post-normalization -> not a usable RFI statistic
            z = np.full(nchan, np.nan)
            zk, _, mad = _mad_z(stat[keep])
            if mad == 0:
                continue
            z[keep] = zk
            newly |= keep & (np.abs(z) > sigma)
        if not newly.any():
            break
        flagged |= newly
    info = dict(n_flagged=int(flagged.sum()), n_chan=int(nchan),
                frac=round(float(flagged.mean()), 4))
    return flagged, info


def offpulse_acf_lowlag(power, off_lims, burst_lims, chan_mask=None, nlag=6):
    """Mean low-lag (1..nlag-1) frequency-ACF of the off-pulse region, on non-overlapping
    burst-width chunks (ensemble median). White noise -> ~0; RFI -> >0."""
    o0, o1 = int(off_lims[0]), int(off_lims[1])
    onw = int(burst_lims[1]) - int(burst_lims[0])
    if onw <= 0 or (o1 - o0) <= onw:
        return float("nan")
    P = power
    if chan_mask is not None:
        m = P.mask if P.mask is not np.ma.nomask else np.zeros(P.shape, bool)
        P = np.ma.MaskedArray(P.data, mask=m | chan_mask[:, None])
    vals = []
    for s in range(o0, o1 - onw, onw):
        spec1d = np.ma.mean(P[:, s:s + onw], axis=1)
        x = spec1d - np.ma.mean(spec1d)
        x = np.ma.filled(x, 0.0)
        n = x.size
        ac = np.correlate(x, x, mode="full")[n - 1:]
        if ac[0] != 0:
            ac = ac / ac[0]
            vals.append(ac[1:nlag].mean())
    return float(np.median(vals)) if vals else float("nan")


def burst_bright_channels(power, burst_lims, z_thresh=8.0):
    b0, b1 = int(burst_lims[0]), int(burst_lims[1])
    on = np.ma.filled(np.ma.mean(power[:, b0:b1], axis=1), np.nan)
    z, _, _ = _mad_z(on)
    return np.isfinite(z) & (np.abs(z) > z_thresh)
