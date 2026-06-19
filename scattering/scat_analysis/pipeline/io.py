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
        self.data = self.freq = self.time = self.df_MHz = self.dt_ms = self.model = None
        if not lazy:
            self.load()

    def load(self):
        if self.data is not None:
            return
        raw = self._load_raw()
        if self.flip_freq:
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

        self.model = FRBModel(
            time=self.time, freq=self.freq, data=self.data, df_MHz=self.df_MHz
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
