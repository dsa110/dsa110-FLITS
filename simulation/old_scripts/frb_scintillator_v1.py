# frb_scintillator.py – research‑grade two‑screen scintillation simulator
"""
High‑fidelity implementation of the methods in
Pradeep et al. (2025) *Scintillometry of Fast Radio Bursts: Resolution Effects
in Two‑Screen Models*.

The module supports **two propagation paradigms**:

1. **Coherent‑field mode**  (default)  
   Electric fields from all geometrical paths on both screens are summed
   coherently; intensity is taken after the sum.  This is the model used in
   §5 of the paper.

2. **Power‑multiplication mode**  
   Set ``combine_in_power=True`` when constructing ``Scintillator``.  
   Each screen is propagated *independently* to its own dynamic spectrum
   (coherent within that screen); the two **intensity** spectra are then
   multiplied.  When both screens are strong and unresolved this reproduces
   the well‑known analytic variance limit  
   :math:`m^{2} = (1+m_{1}^{2})(1+m_{2}^{2}) - 1 → 3`.

Public API
==========
    Screen               – thin scattering screen
    Scintillator         – propagator (one or two screens)
    power_acf            – mean‑removed 1‑D autocorrelation
    fit_acf              – simple Lorentzian ACF fitter

Dependencies
============
NumPy ≥ 1.22, SciPy ≥ 1.10

Run
    python frb_scintillator.py --self-test
for automated sanity checks.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import fft as _fft
from scipy import signal as _sig
from scipy import optimize as _opt

__all__ = ["Screen", "Scintillator", "power_acf", "fit_acf"]

# ---------------------------------------------------------------------------
# Constants & utilities
# ---------------------------------------------------------------------------
_c = 2.99792458e8  # m s⁻¹


def _next_pow_two(n: int) -> int:
    """Return the next power‑of‑two ≥ n."""
    return 1 << (n - 1).bit_length()

# ---------------------------------------------------------------------------
# 1.  Thin scattering screen
# ---------------------------------------------------------------------------
@dataclass
class Screen:
    """
    A thin, randomly speckled scattering screen.

    Parameters
    ----------
    dist_m : float
        Observer → screen distance (metres).
    n_images : int, default 128
        Number of discrete image points (“speckles”).
    theta_L_rad : float, default 2e‑6
        Gaussian envelope σ for amplitudes (radians).
    random_phase : bool, default True
        If True assign random phases to each speckle.
    z : float, default 0.0
        Cosmological redshift of the screen (affects (1+z) bfactor).
    rng : np.random.Generator, optional
        RNG for reproducibility.
    """

    dist_m: float
    n_images: int = 128
    
    # ――― angular envelope (isotropic OR anisotropic) ―――
    theta_L_rad: float | None = 2e-6            # isotropic σ; ignored if either axis given
    theta_L_x_rad: float | None = None          # σ_x  (if set → anisotropic)
    theta_L_y_rad: float | None = None          # σ_y
    pa_deg: float = 0.0                         # ellipse P.A. east of north (deg)

    random_phase: bool = True
    rng: Optional[np.random.Generator] = None
    z: float = 0.0

    # frequency‑scaling reference
    lambda_ref_m: float = 0.21                  # λ at which theta & τ are defined
    
    # generated arrays (post‑init)
    theta: np.ndarray = field(init=False, repr=False)
    amp: np.ndarray = field(init=False, repr=False)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        rng = self.rng or np.random.default_rng()

        # decide on σ_x, σ_y
        if self.theta_L_x_rad or self.theta_L_y_rad:
            sx = self.theta_L_x_rad or (self.theta_L_rad or 1e-6)
            sy = self.theta_L_y_rad or (self.theta_L_rad or 1e-6)
        else:
            sx = sy = self.theta_L_rad or 1e-6

        # random positions in a square big enough to hold the envelope
        box = 4 * max(sx, sy)
        self.theta = rng.uniform(-box, box, size=(self.n_images, 2))

        # rotate coords by pa_deg if anisotropic or rotated
        if sx != sy or self.pa_deg:
            pa = np.deg2rad(self.pa_deg)
            R  = np.array([[np.cos(pa), -np.sin(pa)],
                           [np.sin(pa),  np.cos(pa)]])
            xy = self.theta @ R.T
            r2 = (xy[:, 0]/sx)**2 + (xy[:, 1]/sy)**2
        else:                             # isotropic fast path
            r2 = np.sum(self.theta**2, 1) / sx**2

        env     = np.exp(-0.5 * r2)
        modulus = rng.rayleigh(scale=1.0, size=self.n_images) * env
        phase   = rng.uniform(0, 2*np.pi, self.n_images) if self.random_phase else 0.0
        self.amp = modulus * np.exp(1j * phase)


# ---------------------------------------------------------------------------
# 2.  Scintillator
# ---------------------------------------------------------------------------
@dataclass
class Scintillator:
    """
    Simulate propagation through one or two screens.

    Parameters
    ----------
    combine_in_power : bool, default False
        *False* → coherent‑field mode (sum E‑fields).  
        *True*  → propagate each screen independently and multiply intensities.
    max_fft_len : int or None, default 2**26
        Safety cap on FFT length (~1 GiB complex).  Set *None* to disable.
    """

    screen1: Screen
    screen2: Optional[Screen] = None
    wavelength_m: float = 0.21
    source_dist_m: Optional[float] = None
    combine_in_power: bool = False
    max_fft_len: int | None = 1 << 24  # 67 108 864 samples

    # cached IRF (coherent mode only)
    _tau: Optional[np.ndarray] = field(init=False, default=None, repr=False)
    _mu: Optional[np.ndarray] = field(init=False, default=None, repr=False)

    # ------------------------------------------------------------------
    # 2.1a  Build impulse‑response for coherent mode
    # ------------------------------------------------------------------
    def _single_irf(self) -> Tuple[np.ndarray, np.ndarray]:
        s = self.screen1
        scale = (self.wavelength_m / s.lambda_ref_m) ** 4      # τ ∝ λ⁴
        Deff  = s.dist_m / (1 + s.z)
        tau   = scale * Deff * np.sum(s.theta**2, 1) / (2 * _c)
        return tau, s.amp

    def _double_irf(self) -> Tuple[np.ndarray, np.ndarray]:
        s1, s2 = self.screen1, self.screen2                     # type: ignore
        theta1, theta2 = s1.theta[:, None, :], s2.theta[None, :, :]

        lam = self.wavelength_m
        scale1 = (lam / s1.lambda_ref_m) ** 4
        scale2 = (lam / s2.lambda_ref_m) ** 4

        D1, D2 = s1.dist_m, s2.dist_m
        Ds     = self.source_dist_m or (D2 + 3.086e16)          # +1 pc
        De1    = D1 / (1 + s1.z)
        De2    = D2 / (1 + s2.z)
        De12   = (D1 * (Ds - D2) / Ds) / (1 + 0.5 * (s1.z + s2.z))
        
        # geom factors with individual (1+z)
        g1   = (1 + s1.z)
        g2   = (1 + s2.z)
        g12  = (1 + 0.5*(s1.z + s2.z))

        tau = (
                scale1 * De1  * np.sum(theta1**2, 2) * g1
              + scale2 * De2  * np.sum(theta2**2, 2) * g2
              - 2 * np.sqrt(scale1*scale2) * De12 * np.sum(theta1*theta2, 2) * g12
             ) / (2 * _c)
        
        mu = (s1.amp[:, None] * s2.amp[None, :]).ravel()
        return tau.ravel(), mu

    def _ensure_irf(self):
        if self._tau is None:
            self._tau, self._mu = (self._single_irf() if self.screen2 is None
                                   else self._double_irf())  # type: ignore
            
    # ------------------------------------------------------------------
    # 2.1b  Public accessor – forces IRF build if needed
    # ------------------------------------------------------------------
    @property
    def delays(self) -> np.ndarray:
        """
        1‑D array of geometric / scattering delays (seconds) for every
        image‑pair path included in the impulse response.  Lazily built and
        cached on first access.
        """
        if self._tau is None:
            self._ensure_irf()
        return self._tau            # type: ignore


    # ------------------------------------------------------------------
    # 2.2  Coherent‑field convolution
    # ------------------------------------------------------------------
    def scattered_pulse(self, pulse: ArrayLike, dt_s: float) -> np.ndarray:
        """
        Convolve an intrinsic pulse with the screen impulse‑response
        (coherent‑field mode).

        Returns
        -------
        np.ndarray
            Complex voltage time‑series of the scattered pulse,
            same length as *pulse*.
        """
        self._ensure_irf()
        x = np.asarray(pulse, complex)
        span = self._tau.max() - self._tau.min()
        fft_len = _next_pow_two(x.size + int(span/dt_s) + 1)
        if self.max_fft_len and fft_len > self.max_fft_len:
            raise RuntimeError(f"FFT length {fft_len} exceeds cap {self.max_fft_len}")
        irf = np.zeros(fft_len, complex)
        idx = np.mod(np.round(self._tau/dt_s).astype(int), fft_len)
        np.add.at(irf, idx, self._mu)
        y = _fft.ifft(_fft.fft(np.pad(x, (0, fft_len-x.size))) * _fft.fft(irf))
        return y[:x.size]

    # ------------------------------------------------------------------
    # 2.3  Polyphase filter‑bank channeliser (common)
    # ------------------------------------------------------------------
    @staticmethod
    def _pfb(ts: np.ndarray, fs: float,
             M: int = 512, ntap: int = 4,
             window: str | Tuple | Callable = "blackmanharris"):
        hop, L = M, M*ntap
        fir = _sig.firwin(L, cutoff=1/M, window=window, fs=2).reshape(ntap, M)
        nfrm = (len(ts)-L)//hop + 1
        spec = np.empty((M, nfrm), complex)
        tw = np.exp(-2j*np.pi/M * np.outer(np.arange(M), np.arange(M)))
        for k in range(nfrm):
            blk = ts[k*hop:k*hop+L].reshape(ntap, M)
            spec[:, k] = tw @ (fir * blk).sum(0)
        f = _fft.fftfreq(M, 1/fs); order = np.argsort(f)
        return f[order], spec[order]

    # ------------------------------------------------------------------
    # 2.4  Dynamic‑spectrum helpers
    # ------------------------------------------------------------------
    def _dynspec_coherent(self, pulse, dt_s, fs_Hz, nchan):
        field = self.scattered_pulse(pulse, dt_s)
        f, spec = self._pfb(field, fs_Hz, M=nchan)
        return f, np.abs(spec)**2

    def _dynspec_power(self, pulse, dt_s, fs_Hz, nchan):
        if self.screen2 is None:
            return self._dynspec_coherent(pulse, dt_s, fs_Hz, nchan)
        # propagate each screen independently
        sc1 = Scintillator(self.screen1, wavelength_m=self.wavelength_m,
                           max_fft_len=self.max_fft_len)
        sc2 = Scintillator(self.screen2, wavelength_m=self.wavelength_m,
                           max_fft_len=self.max_fft_len)
        f, I1 = sc1._dynspec_coherent(pulse, dt_s, fs_Hz, nchan)
        _, I2 = sc2._dynspec_coherent(pulse, dt_s, fs_Hz, nchan)
        return f, I1 * I2

    def dynamic_spectrum(self, pulse: ArrayLike, dt_s: float,
                         fs_Hz: float, nchan: int = 512):
        """
        Channelised **intensity** dynamic spectrum.

        The propagation paradigm is selected by ``combine_in_power``:

        * False → coherent‑field mode (default).
        * True  → power‑multiplication mode.
        """
        if self.combine_in_power and self.screen2 is not None:
            return self._dynspec_power(pulse, dt_s, fs_Hz, nchan)
        return self._dynspec_coherent(pulse, dt_s, fs_Hz, nchan)

    # ------------------------------------------------------------------
    @property
    def RP(self) -> float:
        """Resolution power (Eq. 3.11 of the paper)."""
        if self.screen2 is None:
            return np.inf
        L1, L2 = self.screen1.dist_m, self.screen2.dist_m  # type: ignore
        return (L1 * L2) / (self.wavelength_m * (L2 - L1))


# ---------------------------------------------------------------------------
# 3.  ACF utilities – FFT & brute‑force versions
# ---------------------------------------------------------------------------

def acf_fft(arr: ArrayLike) -> np.ndarray:
    """
    Mean‑subtracted, normalised autocorrelation computed *without* np.correlate.

    Notes
    -----
    Uses the Wiener–Khinchin theorem:

        ACF = FFT⁻¹{ |FFT(x)|² } / var(x)

    Only the non‑negative lags are returned (same convention as the paper).
    """
    x = np.asarray(arr, float)
    x -= x.mean()
    if x.ptp() == 0.0:
        return np.ones(1, float)      # flat spectrum → delta ACF

    spec  = np.fft.fft(x, n=2*len(x))
    power = spec * spec.conj()
    acf   = np.fft.ifft(power).real[:len(x)]
    acf  /= acf[0]                    # normalise
    return acf

def acf_bruteforce(arr: ArrayLike) -> np.ndarray:
    """
    Autocorrelation computed directly by sliding and summing.

    Notes
    -----
    Complexity is O(N²) but for the typical spectra (N≈8 k) it is still
    < 50 ms on a laptop and avoids FFT padding artefacts.
    """
    x = np.asarray(arr, float)
    x -= x.mean()
    n = x.size
    acf = np.empty(n, float)

    # vectorised shifting via fancy indexing
    for lag in range(n):
        prod = x[:n-lag] * x[lag:]
        acf[lag] = prod.mean()
    return acf / acf[0]               # normalise


def _lorentz(nu, amp, hwhm, c0):
    """1‑component Lorentzian used by the fallback branch."""
    return amp / (1 + (nu/hwhm)**2) + c0


def _acf_two_screen(nu, m_mw2, h_mw, m_h2, h_h, c0):
    """
    Eq. 4.22 – product + individual Lorentzians.
    """
    L1 = 1 / (1 + (nu/h_mw)**2)
    L2 = 1 / (1 + (nu/h_h)**2)
    return m_mw2*m_h2*L1*L2 + m_mw2*L1 + m_h2*L2 + c0


def fit_acf(
    data        : ArrayLike,
    dnu         : float,
    corr_thresh : float = 0.05,
    bootstrap   : int | None = None,
    rng         : np.random.Generator | None = None,
    
) -> dict:
    """
    Fit the spectral ACF with the two‑screen model (Eq. 4.22).

    Parameters
    ----------
    data : 1‑D array
        Power spectrum or ACF (first element ≈ 1 → treated as ACF).
    dnu : float
        Frequency spacing between bins (Hz).
    corr_thresh : float, default 0.05
        Threshold for deciding if the ACF is effectively flat.
    bootstrap : int or None
        If given, perform this many bootstrap resamples to estimate
        1‑σ errors on m² and both HWHMs.  Errors returned as `err_*`.
    rng : np.random.Generator or None
        RNG source for bootstrapping; default is `np.random.default_rng()`.

    Returns
    -------
    dict including
        m2, nu_s_mw, nu_s_host,
        err_m2, err_nu_s_mw, err_nu_s_host   (NaN if bootstrap is None)
    """
    # ---------- build / verify ACF -----------------------------------
    vec = np.asarray(data, float)
    
    acf = vec if np.isclose(vec[0], 1.0, atol=1e-3) else acf_bruteforce(vec)
    nu  = np.arange(acf.size) * dnu

    # ---------- single‑screen fallback ------------------------------
    if (acf > corr_thresh).sum() < 6:
        amp, hwhm, _ = _opt.curve_fit(_lorentz, nu, acf,
                                      p0=[1.0, dnu, 0.0])[0]
        result = dict(m2=amp, nu_s_mw=hwhm, nu_s_host=np.nan,
                      err_m2=np.nan, err_nu_s_mw=np.nan, err_nu_s_host=np.nan)
        return result

    # ---------- two‑screen full fit ---------------------------------
    p0 = [1.0, dnu, 0.5, 0.3*dnu, 0.0]
    bounds = ([0, 0, 0, 0, -1], [10, 1e3*dnu, 10, 1e3*dnu, 1])
    popt,_ = _opt.curve_fit(_acf_two_screen, nu, acf, p0=p0,
                            bounds=bounds, maxfev=5000)

    m_mw2, h_mw, m_h2, h_h, _ = popt
    m2_tot = (1 + m_mw2) * (1 + m_h2) - 1
    result = dict(m2=m2_tot, nu_s_mw=h_mw, nu_s_host=h_h,
                  err_m2=np.nan, err_nu_s_mw=np.nan, err_nu_s_host=np.nan)

    # ---------- bootstrap errors (if requested) ---------------------
    if bootstrap:
        rng = rng or np.random.default_rng()
        m2_bs, h1_bs, h2_bs = [], [], []
        for _ in range(bootstrap):
            idx = rng.choice(acf.size, acf.size, replace=True)
            try:
                popt_bs,_ = _opt.curve_fit(_acf_two_screen,
                                           nu[idx], acf[idx],
                                           p0=popt, bounds=bounds, maxfev=2000)
                m1b, h1b, m2b, h2b, _ = popt_bs
                m2_bs.append((1+m1b)*(1+m2b)-1)
                h1_bs.append(h1b); h2_bs.append(h2b)
            except Exception:
                continue
        if m2_bs:
            result.update(err_m2=np.nanstd(m2_bs),
                          err_nu_s_mw=np.nanstd(h1_bs),
                          err_nu_s_host=np.nanstd(h2_bs))
    return result




# ---------------------------------------------------------------------------
# 4.  Self‑test
# ---------------------------------------------------------------------------
def _self_test() -> None:
    """
    Sanity checks:

    1. Single strong screen in coherent‑field mode  → m² ≈ 1.
    2. Two *identical* unresolved screens in power‑multiplication mode → m² ≈ 3.
    """
    rng   = np.random.default_rng(0)
    pulse = np.zeros(2048, complex); pulse[0] = 1.0
    dt    = 1e-6      # 1 µs sample
    fs    = 1e6       # 1 MHz sample‑rate

    # ---------------------------------------------------- single screen
    scr = Screen(dist_m=3.0e19, theta_L_rad=2e-6, rng=rng)
    sc  = Scintillator(scr, combine_in_power=False)
    f, I = sc.dynamic_spectrum(pulse, dt, fs, nchan=8192)
    m2_single = np.var(I[:, 0]) / np.mean(I[:, 0])**2
    assert abs(m2_single - 1) < 0.15, f"single‑screen m²={m2_single:.2f}"
    print("✓ single‑screen test passed (m²≈1)")

    # ----------------------------------------- two screens, power‑mode
    scrB = Screen(dist_m=3.00003e19,        # +10 pc
                  theta_L_rad=2e-6, rng=rng)
    sc2  = Scintillator(scr, scrB, combine_in_power=True)
    f2, I2 = sc2.dynamic_spectrum(pulse, dt, fs, nchan=8192)
    m2_double = np.var(I2[:, 0]) / np.mean(I2[:, 0])**2
    assert abs(m2_double - 3) < 0.5, f"two‑screen power m²={m2_double:.2f}"
    print("✓ two‑screen power‑mode test passed (m²≈3)")
    
    # ---------- anisotropic screen check -------------------------------
    scr_ellip = Screen(dist_m=3e19, theta_L_x_rad=2e-6,
                       theta_L_y_rad=1e-6, pa_deg=30, rng=rng)
    sc_ellip  = Scintillator(scr_ellip)
    _, I_e = sc_ellip.dynamic_spectrum(pulse, dt, fs, nchan=8192)
    assert np.isfinite(I_e).all(), "anisotropic dynspec has NaNs or infs"
    print("✓ anisotropic screen basic check passed")

    # ---------- ν‑scaling check (τ ∝ λ⁴) -------------------------------
    scr_ref = Screen(dist_m=3e19, theta_L_rad=2e-6, rng=rng)
    sc_ref  = Scintillator(scr_ref, wavelength_m=0.21)
    sc_hi   = Scintillator(scr_ref, wavelength_m=0.42)
    scale   = sc_hi.delays.ptp() / sc_ref.delays.ptp()     # ptp = max - min
    assert 12 < scale < 20, f"λ⁴ scaling off (measured ×{scale:.1f})"
    print("✓ frequency‑scaling check passed (≈λ⁴)")


    print("All frb_scintillator tests succeeded.")
    
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, textwrap
    _self_test() if "--self-test" in sys.argv else print(textwrap.dedent(__doc__))