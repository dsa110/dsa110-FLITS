"""
Data loading and preprocessing for the BurstFit pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..burstfit import FRBModel, downsample
from ..config_utils import SamplerConfig, TelescopeConfig

log = logging.getLogger(__name__)

class BurstDataset:
    """Loads and preprocesses a burst from a .npy file."""

    def __init__(
        self,
        inpath: str | Path,
        outpath: str | Path,
        *,
        name: str = "FRB",
        telescope: TelescopeConfig | None = None,
        sampler: SamplerConfig | None = None,
        f_factor: int = 1,
        t_factor: int = 1,
        outer_trim: float = 0.45,
        smooth_ms: float = 0.1,
        center_burst: bool = True,
        flip_freq: bool = False,  # Data is now pre-standardized to ascending
        onpulse_crop: bool = False,
        onpulse_pad_factor: float = 0.5,
        onpulse_thresh: float = 3.0,
        lazy: bool = False,
    ):
        self.inpath = Path(inpath)
        self.outpath = Path(outpath)
        self.name = name
        if telescope is None:
            raise ValueError("telescope configuration must be provided")
        self.telescope = telescope
        self.sampler = sampler
        self.f_factor, self.t_factor = f_factor, t_factor
        self.outer_trim = outer_trim if outer_trim is not None else 0.45
        self.smooth_ms = smooth_ms
        self.center_burst, self.flip_freq = center_burst, flip_freq
        # On-pulse crop: after centering, restrict the time axis to the burst +
        # scattering tail + a noise margin, so the likelihood (summed over every
        # time sample, burstfit.py:601) is not dominated by far off-pulse pixels
        # whose residual baseline structure drives zeta/alpha runaway.
        self.onpulse_crop = onpulse_crop
        self.onpulse_pad_factor = onpulse_pad_factor
        self.onpulse_thresh = onpulse_thresh
        self.data = self.freq = self.time = self.df_MHz = self.dt_ms = self.model = None
        if not lazy:
            self.load()

    def load(self):
        if self.data is not None:
            return
        raw = self._load_raw()
        # Standardize to ascending frequency (row 0 = f_min), which the freq axis,
        # model and plots all assume. CHIME .npy arrive descending (row 0 = f_max);
        # the telescope declares this via freq_descending. flip_freq is a manual
        # override for one-off cases.
        if self.flip_freq or getattr(self.telescope, "freq_descending", False):
            raw = np.flipud(raw)

        # Build axes for the raw data to use in preprocessing
        raw_freq, raw_time, _, _ = self._build_axes(raw.shape, f_factor=1, t_factor=1)
        ds = self._bandpass_correct(raw, raw_time)
        ds = self._trim_buffer(ds)
        self.data = self._downsample_and_renormalize(ds)

        # Re-build axes for final downsampled data shape
        self.freq, self.time, self.df_MHz, self.dt_ms = self._build_axes(
            self.data.shape
        )

        if self.center_burst:
            self._centre_burst()

        # Per-channel noise is an instrument property (time-independent), so estimate
        # it from the FULL window's off-pulse BEFORE cropping. Cropping to a narrow
        # on-pulse window leaves too few clean off-pulse samples for a robust MAD
        # (DSA bursts crop to ~20 samples), so re-estimating post-crop corrupts the
        # likelihood. Pass the full-window noise through to the cropped FRBModel.
        noise_full = self._estimate_noise_full() if self.onpulse_crop else None

        if self.onpulse_crop:
            self._crop_on_pulse()

        self.model = FRBModel(
            time=self.time, freq=self.freq, data=self.data,
            # NATIVE channel width: intra-channel DM smearing is set at the native
            # dedispersion resolution, not the downsampled width (df_MHz_raw*f_factor).
            df_MHz=self.telescope.df_MHz_raw,
            noise_std=noise_full,
        )

    def _load_raw(self):
        if not self.inpath.exists():
            raise FileNotFoundError(f"Data not found: {self.inpath}")
        try:
            data = np.load(self.inpath)
            return np.nan_to_num(data.astype(np.float64))
        except Exception as e:
            raise IOError(f"Failed to load {self.inpath}: {e}")

    def _build_axes(self, shape, f_factor=None, t_factor=None):
        f_factor = f_factor if f_factor is not None else self.f_factor
        t_factor = t_factor if t_factor is not None else self.t_factor

        # Get raw shape from config, not from current array shape
        p = self.telescope
        n_ch_raw = p.n_ch_raw if p.n_ch_raw is not None else shape[0] * f_factor

        df_MHz = p.df_MHz_raw * f_factor
        dt_ms = p.dt_ms_raw * t_factor

        final_n_ch = shape[0]
        final_n_t = shape[1]

        # All data is now standardized to ascending frequency order (data[0] = f_min)
        freq = np.linspace(p.f_min_GHz, p.f_max_GHz, final_n_ch)
        time = np.arange(final_n_t) * dt_ms
        return freq, time, df_MHz, dt_ms

    def _bandpass_correct(self, arr, time_axis):
        q = time_axis.size // 4
        off_pulse_idx = np.r_[0:q, -q:0]
        mu = np.nanmean(arr[:, off_pulse_idx], axis=1, keepdims=True)
        sig = np.nanstd(arr[:, off_pulse_idx], axis=1, keepdims=True)
        sig[sig < 1e-9] = np.nan
        return np.nan_to_num((arr - mu) / sig, nan=0.0)

    def _trim_buffer(self, arr):
        n_trim = int(self.outer_trim * arr.shape[1])
        return arr[:, n_trim:-n_trim] if n_trim > 0 else arr

    def _downsample_and_renormalize(self, arr):
        ds_arr = downsample(arr, self.f_factor, self.t_factor)
        # Do NOT normalize by peak. Keep units as S/N (z-score from bandpass_correct).
        return ds_arr

    def _centre_burst(self):
        prof = np.nansum(self.data, axis=0)
        if self.smooth_ms > 0 and self.dt_ms > 0:
            sigma_samps = (self.smooth_ms / 2.355) / self.dt_ms
            prof = gaussian_filter1d(prof, sigma=sigma_samps)
        shift = self.data.shape[1] // 2 - np.argmax(prof)
        self.data = np.roll(self.data, shift, axis=1)

    def _estimate_noise_full(self):
        """Per-channel MAD noise from the full window's outer quarters (off-pulse).

        Mirrors FRBModel._estimate_noise but is run pre-crop, so the noise estimate
        is anchored on the largest clean off-pulse region rather than the narrow
        cropped window. Returns shape (nchan,)."""
        n = self.data.shape[1]
        q = max(n // 4, 1)
        idx = np.r_[0:q, n - q:n]
        seg = self.data[:, idx]
        mad = np.median(np.abs(seg - np.median(seg, axis=1, keepdims=True)), axis=1)
        return 1.4826 * mad

    def _crop_on_pulse(self):
        """Crop the time axis to the burst + scattering tail + a noise margin.

        Detects the on-pulse span as the contiguous-ish region where the smoothed
        band-integrated profile exceeds onpulse_thresh * sigma_offpulse (sigma from
        the outer-quarter MAD), then pads it by onpulse_pad_factor * span on each
        side. The pad leaves clean off-pulse for FRBModel._estimate_noise (which
        uses the cropped window's outer quarters) and absorbs the steep tail's
        approach to baseline. No-op (with a warning) if nothing clears threshold.
        """
        prof = np.nansum(self.data, axis=0)
        if self.smooth_ms > 0 and self.dt_ms > 0:
            prof = gaussian_filter1d(prof, sigma=(self.smooth_ms / 2.355) / self.dt_ms)
        n = prof.size
        q = max(n // 4, 1)
        base = np.r_[prof[:q], prof[-q:]]
        mu = np.median(base)
        sig = 1.4826 * np.median(np.abs(base - mu))
        if sig <= 0:
            log.warning(f"[{self.name}] on-pulse crop skipped: zero off-pulse spread")
            return
        on = np.where((prof - mu) > self.onpulse_thresh * sig)[0]
        if on.size == 0:
            log.warning(f"[{self.name}] on-pulse crop skipped: no samples above "
                        f"{self.onpulse_thresh}-sigma")
            return
        lo, hi = int(on.min()), int(on.max())
        span = hi - lo + 1
        pad = int(self.onpulse_pad_factor * span)
        lo2, hi2 = max(0, lo - pad), min(n, hi + pad + 1)
        self.data = self.data[:, lo2:hi2]
        self.time = np.arange(self.data.shape[1]) * self.dt_ms
        log.info(f"[{self.name}] on-pulse crop: {n} -> {self.data.shape[1]} samples "
                 f"(span {span}, pad {pad}); off-pulse fraction "
                 f"{1 - span / self.data.shape[1]:.2f}")
