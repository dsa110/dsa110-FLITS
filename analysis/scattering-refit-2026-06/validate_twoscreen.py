"""Kernel validation for the two-screen forward model (Stage-0 prerequisite).

The generic-r branch is EXACT by construction: the two-exponential impulse is the
exact linear combination [tau2 h2 - tau1 h1]/(tau2-tau1) of single exponentials, so
K = [tau2 EMG2 - tau1 EMG1]/(tau2-tau1) is an exact linear combination of two
analytic EMGs. Nesting (r->0 == production EMG, machine zero) confirms the EMG reuse.

The ONLY new hand-derived math is the r=1 derivative branch f'(tau)=d/dtau[tau EMG].
Checks:
  1. NEST      : two_screen_perchan(r->0) == analytic EMG(tau1)             (machine 0)
  2. DERIV     : _dtau_tau_emg(tau1) vs Richardson finite-diff of tau*EMG   (< 1e-7)
  3. GT-shape  : closed form vs OVERSAMPLED FFT convolution, r in {0.3,1.5} (< 2e-3)
  4. SWITCH    : closed vs midpoint-derivative at |r-1|=R_UNIT_EPS boundary (< 1e-5)
  5. AREA      : unit area per channel across r                            (< 2e-3)
"""
import sys

import numpy as np

from scat_analysis.burstfit import _next_fast_len, analytic_gaussian_exp_convolution
from twoscreen import two_screen_perchan, _dtau_tau_emg, R_UNIT_EPS

T, dt = 2048, 0.02
time = (np.arange(T) * dt)[None, :]
mu = np.array([[8.0]])
sig = np.array([[0.12]])
tau1 = np.array([[0.45]])


def relerr(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)


def f_tauemg(tau):
    return tau * analytic_gaussian_exp_convolution(time, mu, sig, tau)


def fft_gt(r, os=8):
    """Gaussian (x) exact two-exp impulse on an OS-times-finer grid, sampled back."""
    Tf, dtf = T * os, dt / os
    tf = (np.arange(Tf) * dtf)[None, :]
    tau2 = r * tau1
    g = (1.0 / (np.sqrt(2.0 * np.pi) * sig)) * np.exp(-0.5 * ((tf - mu) / sig) ** 2)
    lag = (np.arange(Tf) * dtf)[None, :]
    h = (np.exp(-lag / tau2) - np.exp(-lag / tau1)) / (tau2 - tau1)
    h = h / np.clip(h.sum(axis=1, keepdims=True) * dtf, 1e-30, None)
    L = _next_fast_len(2 * Tf)
    conv = np.fft.irfft(np.fft.rfft(g, L, axis=1) * np.fft.rfft(h, L, axis=1), L, axis=1)
    return (conv[:, :Tf] * dtf)[:, ::os]  # sample at original grid points


fail = 0

emg = analytic_gaussian_exp_convolution(time, mu, sig, tau1)
e = relerr(two_screen_perchan(time, mu, sig, tau1, 1e-9), emg)
fail += not (e < 1e-12)
print(f"[NEST]   r->0 vs EMG(tau1)                : relerr={e:.2e}  {'OK' if e<1e-12 else 'FAIL'}")

# Richardson central diff of f(tau)=tau*EMG at tau1: D=(4 D(h/2)-D(h))/3, O(h^4)
def cdiff(h):
    return (f_tauemg(tau1 + h) - f_tauemg(tau1 - h)) / (2.0 * h)
hh = 1e-2 * tau1
fd = (4.0 * cdiff(hh / 2.0) - cdiff(hh)) / 3.0
an = _dtau_tau_emg(time, mu, sig, tau1)
e = relerr(an, fd)
fail += not (e < 1e-7)
print(f"[DERIV]  f'(tau1) analytic vs Richardson  : relerr={e:.2e}  {'OK' if e<1e-7 else 'FAIL'}")

for r in [0.3, 1.5]:
    k = two_screen_perchan(time, mu, sig, tau1, r)
    e = relerr(k, fft_gt(r))
    fail += not (e < 2e-3)
    print(f"[GT]     r={r} closed vs oversampled FFT   : relerr={e:.2e}  {'OK' if e<2e-3 else 'FAIL'}")

r_out = 1.0 - R_UNIT_EPS * 1.0001   # just outside -> closed branch
r_in = 1.0 - R_UNIT_EPS * 0.9999    # just inside  -> midpoint-derivative branch
e = relerr(two_screen_perchan(time, mu, sig, tau1, r_out),
           two_screen_perchan(time, mu, sig, tau1, r_in))
fail += not (e < 1e-5)
print(f"[SWITCH] closed vs deriv at boundary      : relerr={e:.2e}  {'OK' if e<1e-5 else 'FAIL'}")

bad_area = []
for r in [0.01, 0.1, 0.3, 0.9, 1.0, 1.5, 3.0]:
    a = float((two_screen_perchan(time, mu, sig, tau1, r).sum(axis=1) * dt)[0])
    if abs(a - 1.0) > 2e-3:
        bad_area.append((r, a))
fail += bool(bad_area)
print(f"[AREA]   unit area across r                : {'OK' if not bad_area else 'FAIL '+str(bad_area)}")

print("RESULT:", "ALL PASS" if fail == 0 else f"{fail} FAILURE(S)")
sys.exit(1 if fail else 0)
