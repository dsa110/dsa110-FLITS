"""Component-leakage bias harness (green-lit after harsh-tail ruled out single-component
tail grossness as casey's sole mechanism: max |alpha bias| = 0.21 at W/tau=0.3, vs the
observed free-alpha shift of ~-1.6).

Question: does an UNMODELED weak secondary component bias a single-component free-alpha
EMG fit low in alpha (the fit stretching its tail / lowering alpha to cover the leaked
flux)? This is the gross-tail-vs-tail+leakage discriminant. Inject casey's fitted C1D1
primary (tau~0.019 ms, beta~3.99 -> alpha~4, essentially exponential) PLUS a weak secondary
EMG (amplitude 5-20% of primary, scanned; a few * tau to a few ms offset, scanned), then
refit the SAME single-component free-alpha EMG the real fit used and measure alpha_apparent.

Reuses the validated embias machinery (kernel, per-channel gain-marginal _band_ll,
single-component free-alpha _LLEMG, dynesty `fit`) so the only change vs the harsh-tail
control is a 2-component injection instead of a 1-component PL-PBF one. Isolated module;
does not touch plpbf_inject.py.

CLI: python plpbf_leakage.py <beta> <tau_1ghz> <W_over_tau> <amp_frac> <dt_offset_ms> \
                             [nlive=300] [nproc=4] [seed=0] [sig2_frac=1.0]
"""
import json
import os
import sys

import numpy as np

REFIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REFIT)

from plpbf_inject import (  # noqa: E402  reuse the validated pieces
    DT,
    EMG_HI,
    EMG_LO,
    FREQ_C,
    FREQ_D,
    NT,
    NU0,
    OUTDIR,
    _LLEMG,
    _kernel,
    alpha_from_beta,
    fit,
)


def simulate_leakage(tau_1ghz, beta, W_over_tau, amp_frac, dt_offset_ms,
                     snr=40.0, seed=0, sig2_frac=1.0):
    """Joint 2-component EMG injection: casey-like primary + a weak secondary at
    +dt_offset_ms, amplitude amp_frac * primary. beta ~ 4 so the PBF is the exponential
    limit (s_i=inf), matching casey's railed production fit. Per-channel gain x scint +
    white noise at peak S/N ~ snr, identical to plpbf_inject.simulate."""
    rng = np.random.default_rng(seed)
    alpha = alpha_from_beta(beta)               # injection truth (casey beta~4 -> alpha~4)
    sig = W_over_tau * tau_1ghz
    t0 = 0.30 * NT * DT
    out = {}
    for band, freq in (("C", FREQ_C), ("D", FREQ_D)):
        K1 = _kernel(freq, tau_1ghz, alpha, sig, np.inf, beta, t0)                       # primary
        K2 = _kernel(freq, tau_1ghz, alpha, sig * sig2_frac, np.inf, beta, t0 + dt_offset_ms)  # secondary
        K = K1 + amp_frac * K2
        gain = (freq / NU0) ** (-1.5) * (1.0 + 0.3 * rng.standard_normal(len(freq)))
        clean = gain[:, None] * K
        noise = clean.max() / snr
        data = clean + rng.normal(0.0, noise, clean.shape)
        out[band] = dict(freq=freq, data=data, noise=noise)
    out["truth"] = dict(tau_1ghz=tau_1ghz, beta=beta, alpha=alpha, sig=sig, t0=t0,
                        W_over_tau=W_over_tau, amp_frac=amp_frac, dt_offset_ms=dt_offset_ms,
                        sig2_frac=sig2_frac, snr=snr)
    return out


def main():
    beta = float(sys.argv[1])
    tau_1ghz = float(sys.argv[2])
    W = float(sys.argv[3])
    amp_frac = float(sys.argv[4])
    dt_off = float(sys.argv[5])
    nlive = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    nproc = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    sig2_frac = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0

    inj = simulate_leakage(tau_1ghz, beta, W, amp_frac, dt_off, seed=seed, sig2_frac=sig2_frac)
    tr = inj["truth"]
    print(f"INJECT-LEAKAGE tau_1ghz={tau_1ghz} beta={beta} alpha_true={tr['alpha']:.3f} "
          f"W/tau={W} amp_frac={amp_frac} dt_offset={dt_off}ms sig2_frac={sig2_frac}", flush=True)

    # refit the SAME single-component free-alpha EMG (theta=[tau,alpha,sig,t0], beta->exp)
    eq, lnz, lnze = fit(_LLEMG(inj), EMG_LO, EMG_HI, nlive, nproc, seed)
    tauP = np.percentile(eq[:, 0], [5, 50, 95])
    aP = np.percentile(eq[:, 1], [5, 50, 95])
    bias = aP[1] - tr["alpha"]
    tau_ratio = tauP[1] / tr["tau_1ghz"]
    print(f"  1-COMP EMG fit alpha_apparent={aP[1]:.3f}[{aP[0]:.3f},{aP[2]:.3f}]  "
          f"(alpha_true={tr['alpha']:.3f})  bias={bias:+.3f}  lnZ={lnz:.1f}", flush=True)
    print(f"  tau_apparent={tauP[1]:.4f} (true={tr['tau_1ghz']:.4f}) ratio={tau_ratio:.3f}", flush=True)
    # the discriminant: does leakage buy a LARGE negative bias (toward the observed -1.6)?
    print(f"  LEAKAGE-BIAS {bias:+.3f}  (harsh-tail single-comp max was -0.21; need ~-1.6)", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(dict(mode="leakage", truth=tr, alpha_apparent=aP.tolist(), alpha_true=tr["alpha"],
                   bias=bias, tau_apparent=tauP.tolist(), tau_ratio=tau_ratio, lnZ=lnz),
              open(f"{OUTDIR}/plpbf_leakage_b{beta}_tau{tau_1ghz}_W{W}_a{amp_frac}_dt{dt_off}_s{seed}.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
