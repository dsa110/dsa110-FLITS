#!/usr/bin/env python3
"""Fit 1/2/3-Lorentzian models to DSA gains, BIC-ranked + synthetic recovery."""
import numpy as np
import json
from scipy.optimize import least_squares, minimize
from scipy.special import erfcinv

freya_dsa = np.load('freya_ladder.npz')
freq_dsa = freya_dsa['freq_dsa_mhz']
gains_ladder = freya_dsa['gains_dsa_ladder']
var_ladder = freya_dsa['var_dsa_ladder']

print("="*70)
print("1/2/3-LORENTZIAN FITS ON DSA LADDER")
print("="*70)

def lorentzian_acf(lag, *params):
    """Sum of Lorentzians. params = [A1, W1, A2, W2, ...] (amps, widths in MHz)."""
    result = np.zeros_like(lag, dtype=float)
    for i in range(0, len(params), 2):
        if i+1 >= len(params): break
        A, W = params[i], params[i+1]
        result += A / (1 + (lag / W)**2)
    return result

def fit_lorentzian(acf_data, lag, n_comp, var_lag=None):
    """Fit n_comp Lorentzians to ACF. Returns (params, chi2, bic)."""
    # Initial guess: equal-amplitude, logarithmically-spaced widths
    x0 = []
    for i in range(n_comp):
        w_init = 10 ** (np.log10(1.0) + i * np.log10(10.0) / max(1, n_comp-1)) if n_comp > 1 else 10
        x0.extend([1.0/n_comp, w_init])
    
    # Fit with bounds
    bounds = ([0.01]*n_comp + [0.01]*n_comp,  # lower
              [10.0]*n_comp + [100.0]*n_comp)  # upper (A, W)
    bounds = (bounds[0], bounds[1])
    
    def residual(p):
        pred = lorentzian_acf(lag, *p)
        diff = (acf_data - pred)
        if var_lag is not None:
            return diff / np.sqrt(var_lag + 1e-6)
        else:
            return diff
    
    res = least_squares(residual, x0, bounds=bounds, max_nfev=10000)
    
    chi2 = np.sum(res.fun**2)
    k = len(res.x)
    n_data = len(acf_data)
    bic = k * np.log(n_data) + chi2
    
    return res.x, chi2, bic

# Process each f_factor in the ladder
for ff_name in ['ff384', 'ff192', 'ff96', 'ff48', 'ff24']:
    if ff_name not in freya_dsa:
        continue
    
    gains = freya_dsa[ff_name]
    var = freya_dsa[ff_name.replace('ff', 'var_')]
    chan_width = freya_dsa[ff_name.replace('ff', 'chan_')]
    
    # Compute ACF
    acf_data = np.correlate(gains - np.mean(gains), gains - np.mean(gains), mode='full')
    acf_data = acf_data[len(acf_data)//2:]  # keep positive lags
    acf_data = acf_data[:min(20, len(acf_data))]  # truncate to lag 20
    acf_data /= acf_data[0]  # normalize
    
    lag = np.arange(len(acf_data))
    var_acf = np.correlate(var, var, mode='full')[len(var)//2:][:len(acf_data)]
    
    # Fit 1, 2, 3 components
    results = {}
    for n_comp in [1, 2, 3]:
        p, chi2, bic = fit_lorentzian(acf_data, lag, n_comp, var_lag=var_acf)
        results[n_comp] = {'params': p, 'chi2': chi2, 'bic': bic, 'k': len(p)}
        
        widths = [p[i] for i in range(1, len(p), 2)]
        amps = [p[i] for i in range(0, len(p), 2)]
        print(f"{ff_name} n={n_comp}: widths={widths} amps={amps} BIC={bic:.1f}")
    
    # BIC ranking
    bic_1 = results[1]['bic']
    bic_2 = results[2]['bic']
    bic_3 = results[3]['bic']
    
    if bic_1 < min(bic_2, bic_3):
        winner = 1
    elif bic_2 < bic_3:
        winner = 2
    else:
        winner = 3
    
    print(f"  → WINNER: {winner}-component (BIC={results[winner]['bic']:.1f})")
    print()

print("="*70)
print("SYNTHETIC RECOVERY TEST (DSA, ff24 = 0.74 MHz/ch)")
print("="*70)

# Create synthetic 2-component gain spectrum
np.random.seed(42)
gains_syn = freya_dsa['ff24'].copy()
nn = len(gains_syn)
lag_syn = np.arange(nn)

# True: 0.4 MHz narrow + 15 MHz broad
acf_true = 0.3 * lorentzian_acf(lag_syn, 0.3, 0.4, 0.7, 15.0)
acf_true[0] = 1.0  # normalize lag-0
noise = np.random.normal(0, 0.05, nn)
acf_measured = acf_true + noise
acf_measured[:100] /= acf_measured[0]  # renorm

# Fit 1, 2, 3
for n_comp in [1, 2, 3]:
    p, chi2, bic = fit_lorentzian(acf_measured[:50], np.arange(50), n_comp)
    print(f"n={n_comp}: fitted_widths={[p[i] for i in range(1, len(p), 2)]} BIC={bic:.1f}")

print("✓ 2-component should win; should recover ~0.4 and ~15 MHz")
