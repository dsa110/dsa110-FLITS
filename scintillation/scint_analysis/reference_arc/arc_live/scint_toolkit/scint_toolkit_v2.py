#!/usr/bin/env python3
"""
File: scint_toolkit.py
Author: Jakob T Faber
Version: 06.12.25

General‑purpose scintillation analysis toolkit (concise)

Flow chart
──────────
raw voltages / filterbank / ndarray
    └─► Channelizer (skipped if data already channelised)
         └─► RippleCorrector → burst spectrum (auto‑detected)
              └─► ACF (pair default) → Lorentzian fit
"""
from __future__ import annotations

import os
import sys
import abc
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as _fft
import scipy.optimize as _opt

try:
    from lmfit import Model, Parameters
except ImportError:
    print("Installing lmfit...")
    os.system("pip install lmfit")
    from lmfit import Model, Parameters
    #raise ImportError("Installed lmfit")


###############################################################################
# 1.  Telescope configuration                                                 #
###############################################################################


@dataclass
class TelescopeConfig(abc.ABC):
    """Abstract instrument description."""
    coarse_channel_hz: float
    sample_time_s: float
    centre_freq_hz: float
    fft_size: int
    downmix_factor: int
    pre_channelised: bool = False
    standing_wave_hz: Optional[float] = None

    @abc.abstractmethod
    def read_data(self, path: pathlib.Path) -> np.ndarray:  # pragma: no cover
        """Return an ndarray shaped (time, channel)."""


@dataclass
class CHIMEConfig(TelescopeConfig):
    """Configuration for CHIME/FRB (requires complex voltage data)."""
    coarse_channel_hz: float = 390_625.0
    sample_time_s: float = 2.56e-6
    centre_freq_hz: float = 600e6
    fft_size: int = 512
    downmix_factor: int = 2
    standing_wave_hz: Optional[float] = 29e6

    def read_data(self, path: pathlib.Path) -> np.ndarray:  # pragma: no cover
        # Placeholder for a CHIME complex voltage reader
        raise NotImplementedError("Bind to CHIME bbdata reader here")


@dataclass
class DSA110Config(TelescopeConfig):
    """Configuration for DSA-110 (assumes pre-channelised power data)."""
    coarse_channel_hz: float = 30_517.5781
    sample_time_s: float = 32.768e-6
    centre_freq_hz: float = 1_400e6
    fft_size: int = 1
    downmix_factor: int = 1
    pre_channelised: bool = True
    standing_wave_hz: Optional[float] = None

    def read_data(self, path: pathlib.Path) -> np.ndarray:
        """Reads pre-channelised dynamic spectrum from .npy or .fil files."""
        path = pathlib.Path(path)
        if path.suffix == ".fil":
            try:
                from sigpyproc.Readers import FilReader
            except ImportError:
                raise ImportError("Please install sigpyproc (`pip install sigpyproc-python`) to read .fil files.")
            
            print(f"Reading full filterbank file: {path}...")
            rdr = FilReader(str(path))
            dyn = rdr.readBlock(0, rdr.header.nsamples).astype(np.float32)
            return dyn.T  # Returns as (time, chan)
        
        elif path.suffix == ".npy":
            print(f"Reading dynamic spectrum from .npy file: {path}...")
            arr = np.load(path)
            if arr.ndim != 2:
                raise ValueError("Input .npy file must be a 2D array.")
            print(f"Input shape {arr.shape} appears to be (freq, time). Transposing to (time, freq)...")
            arr = arr.T
            print(f"New shape: {arr.shape}")
            return arr.astype(np.float32)
        
        else:
            raise ValueError(f"Unsupported file type for DSA-110: {path.suffix}")



###############################################################################
# 2.  DSP helpers                                                             #
###############################################################################

class Channelizer:
    """Coarse→fine FFT returning **power** dynamic spectrum from **voltages**."""
    def __init__(self, cfg: TelescopeConfig):
        self.cfg = cfg

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if self.cfg.pre_channelised:
            return data

        if not np.iscomplexobj(data):
            raise TypeError("Channelizer input must be complex voltage data.")
        
        bb_c = data
        n_time, n_coarse = bb_c.shape
        new_len = (n_time // self.cfg.fft_size) * self.cfg.fft_size
        bb_c = bb_c[:new_len]
        resh = bb_c.reshape(-1, self.cfg.fft_size, n_coarse)
        fine_fft = _fft.fftshift(_fft.fft(resh, axis=1), axes=1)
        fine_fft = fine_fft.transpose(0, 2, 1)
        dyn = fine_fft.reshape(-1, n_coarse * self.cfg.fft_size)
        if self.cfg.downmix_factor > 1:
            dyn = dyn.reshape(dyn.shape[0], -1, self.cfg.downmix_factor).mean(axis=2)
        return np.abs(dyn) ** 2

class RippleCorrector:
    """Band-pass flattening via off-pulse template."""
    def estimate(self, off_dyn: np.ndarray) -> np.ndarray:
        return np.nanmean(off_dyn, axis=0)

    def apply(self, dyn: np.ndarray, template: np.ndarray) -> np.ndarray:
        template_safe = template.copy()
        template_safe[template_safe == 0] = 1e-9
        return dyn / template_safe[np.newaxis, :]


###############################################################################
# 3.  Autocorrelation                                                         #
###############################################################################

def _next_pow_two(n: int) -> int:
    return 1 << (n - 1).bit_length()

class PairCountACF:
    """Mask-aware O(N^2) autocorrelation with proper normalization."""
    def __init__(self, *, skip_zero: bool = True, maxlag: Optional[int] = None):
        self.skip_zero = skip_zero
        self.maxlag = maxlag

    def __call__(self, spec: np.ndarray, noise_spec: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if mask is None:
            mask = np.ones(spec.shape, dtype=bool)
        
        # Using proper normalization from Nimmo et al. (2025)
        mean_on = np.nanmean(spec[mask])
        mean_off = np.nanmean(noise_spec[mask])
        denom = (mean_on - mean_off) ** 2
        denom = denom if denom > 1e-9 else 1.0 # Avoid division by zero/small numbers

        y = spec - mean_on
        n = y.size
        L = self.maxlag or n
        acf = np.zeros(L, dtype=float)

        for lag in range(L):
            if lag == 0 and self.skip_zero:
                acf[lag] = 1.0; continue
            
            # Corrected indexing logic
            y1, y2 = y[:n-lag], y[lag:]
            m1, m2 = mask[:n-lag], mask[lag:]
            valid_mask = m1 & m2
            
            if not np.any(valid_mask):
                acf[lag] = np.nan; continue
            
            prod = y1[valid_mask] * y2[valid_mask]
            acf[lag] = np.nanmean(prod) / denom

        return np.arange(L), acf

class FFTACF:
    """FFT autocorrelation (with mask correction)."""
    def __init__(self, *, mask_correct: bool = True):
        self.mask_correct = mask_correct

    def __call__(self, spec: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if mask is None:
            mask = np.ones_like(spec, dtype=bool)
        y = spec.copy()
        mean = np.nanmean(y[mask])
        y[~mask] = mean # Set masked values to mean for FFT
        y -= mean
        
        n = _next_pow_two(2 * y.size - 1)
        fft = _fft.rfft(y, n)
        acf = _fft.irfft(fft * np.conj(fft))[: y.size]
        
        if self.mask_correct:
            m = mask.astype(float)
            mfft = _fft.rfft(m, n)
            norm = _fft.irfft(mfft * np.conj(mfft))[: y.size]
            acf = np.divide(acf, norm, out=np.full_like(acf, np.nan), where=norm != 0)

        # Normalize to peak=1, can be adjusted for physical mod index later
        acf /= acf[0] if acf[0] != 0 else 1.0
        return np.arange(acf.size), acf


###############################################################################
# 4.  Lorentzian fitting                                                      #
###############################################################################

def lorentzian(x, amplitude, hwhm):
    """A single Lorentzian function for use with lmfit."""
    return amplitude / (1.0 + (x / hwhm) ** 2)

class LorentzianFitter:
    """Least-squares multi-Lorentzian fit using the lmfit library."""
    def __init__(self, *, n_components: int):
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        self.n_components = n_components
        
        # Build the composite model from individual Lorentzian components
        full_model = None
        for i in range(1, n_components + 1):
            prefix = f'l{i}_'
            model = Model(lorentzian, prefix=prefix)
            if full_model is None:
                full_model = model
            else:
                full_model += model
        self.model = full_model

    def __call__(self, lag: np.ndarray, acf: np.ndarray) -> Any: # Returns lmfit ModelResult
        """Prepares parameters and runs the fit."""
        params = self.model.make_params()
        
        # Data-driven initial guesses
        acf_peak = acf[1] if len(acf) > 1 else acf[0] # Peak ignoring lag-0
        
        # Distribute initial amplitudes and widths
        initial_amps = np.geomspace(acf_peak, max(acf_peak / 10, 0.1), self.n_components)
        initial_widths = np.logspace(np.log10(2), np.log10(len(lag) / 8), self.n_components)
        
        for i in range(1, self.n_components + 1):
            prefix = f'l{i}_'
            # Sort by width to assign narrowest component first
            amp_guess = initial_amps[i-1]
            hwhm_guess = initial_widths[i-1]
            
            params[f'{prefix}amplitude'].set(value=amp_guess, min=0)
            params[f'{prefix}hwhm'].set(value=hwhm_guess, min=1e-6) # HWHM must be > 0

        print("Fitting with lmfit...")
        result = self.model.fit(acf, params, x=lag)
        print(result.fit_report())
        return result

###############################################################################
# 5.  Diagnostic plots                                                        #
###############################################################################

def _plot_dynamic(dyn: np.ndarray, *, vmax_pct: float = 99.0, title: str = "Dynamic Spectrum") -> None:
    plt.figure(figsize=(8, 5))
    vmax = np.percentile(dyn[np.isfinite(dyn)], vmax_pct)
    plt.imshow(dyn.T, aspect="auto", origin="lower", vmax=vmax, interpolation="none")
    plt.colorbar(label="Power (arb. units)")
    plt.xlabel("Time bin")
    plt.ylabel("Channel")
    plt.title(title)
    plt.tight_layout()

def _plot_acf_fit(self, fit_result):
    """Helper to plot the lmfit result."""
    plt.figure(figsize=(10, 7))
    fit_result.plot_components(ax=plt.gca())
    plt.title(f"ACF Fit for {self.cfg.__class__.__name__} Data")
    plt.xlabel("Lag (channels)")
    plt.ylabel("ACF Power")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
###############################################################################
# 6.  Pipeline                                                                #
###############################################################################

class ScintillationPipeline:
    """End-to-end analysis pipeline, now using lmfit."""
    def __init__(self, cfg: TelescopeConfig, plot: bool = False):
        self.cfg = cfg
        self.channelizer = Channelizer(cfg)
        self.ripple = RippleCorrector()
        self.acf = PairCountACF() # Defaulting to the most accurate method
        self.plot = plot
        self.n_components = 2  # Assuming 2 scint components by default
        if cfg.standing_wave_hz:
            self.n_components += 1
        self.fitter = LorentzianFitter(n_components=self.n_components)

    def _load(self, src: str | pathlib.Path | np.ndarray) -> np.ndarray:
        if isinstance(src, np.ndarray):
            return src
        return self.cfg.read_data(pathlib.Path(src))

    @staticmethod
    def detect_burst(ts: np.ndarray, *, snr_thresh: float = 7.0) -> Tuple[int, int]:
        """Robust MAD-based on-burst window detection."""
        med = np.nanmedian(ts)
        mad = np.nanmedian(np.abs(ts - med))
        if mad < 1e-9: raise RuntimeError("No variance in time series, cannot detect burst.")
        sigma = 1.4826 * mad
        mask = ts > med + snr_thresh * sigma
        if not np.any(mask): raise RuntimeError(f"No burst detected above threshold S/N > {snr_thresh}")
        
        edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
        starts = np.where(edges == 1)[0]
        stops = np.where(edges == -1)[0]
        
        if len(starts) == 0: raise RuntimeError("Could not find burst edges.")
        seg_idx = np.argmax(stops - starts)
        return int(starts[seg_idx]), int(stops[seg_idx])

    def run_full_analysis(self, src: str | pathlib.Path | np.ndarray, *,
                          burst_slice: Optional[Tuple[int, int]] = None,
                          snr_thresh: float = 7.0) -> Dict[str, Any]:
        
        dyn = self._load(src)
        dyn = self.channelizer(dyn)

        if burst_slice is None:
            ts = np.nansum(dyn, axis=1)
            burst_slice = self.detect_burst(ts, snr_thresh=snr_thresh)
        t0, t1 = burst_slice

        # Robust handling of off-burst window
        off_dyn_parts = []
        if t0 > 5:
            off_dyn_parts.append(dyn[:t0])
        if t1 < len(dyn) - 5:
            off_dyn_parts.append(dyn[t1:])
        
        if not off_dyn_parts:
            raise ValueError("No off-burst data available for bandpass correction.")
        
        off_dyn = np.concatenate(off_dyn_parts, axis=0)
        template = self.ripple.estimate(off_dyn)
        fdyn = self.ripple.apply(dyn, template)

        on_spectrum = np.nansum(fdyn[t0:t1], axis=0)
        # Process the off-burst data in the same way for correct normalization
        off_spectrum = np.nansum(self.ripple.apply(off_dyn, template), axis=0)
        
        mask = np.isfinite(on_spectrum) & (on_spectrum > 0)
        
        # Pass the off_spectrum to the ACF function
        lag, acf = self.acf(on_spectrum, off_spectrum, mask=mask)
        
        valid_lags = ~np.isnan(acf)
        lag_fit, acf_fit = lag[valid_lags], acf[valid_lags]

        fit_result = self.fitter(lag_fit, acf_fit)

        if self.plot:
            _plot_dynamic(fdyn, title="Processed Dynamic Spectrum")
            _plot_acf_fit(fit_result)
            plt.show()

        return {
            "burst_slice": burst_slice,
            "on_spectrum": on_spectrum,
            "off_spectrum": off_spectrum,
            "lag": lag,
            "acf": acf,
            "fit": fit_result,
            "config": self.cfg,
        }
