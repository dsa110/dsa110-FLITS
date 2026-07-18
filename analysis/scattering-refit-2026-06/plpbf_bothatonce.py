"""Both-at-once bias harness: a heavy PL-PBF-tail PRIMARY (interior s_i, beta<4 so the
per-frequency pulse has a genuine power-law tail) PLUS a close unfit secondary, refit
with the SAME single-component free-alpha EMG the real fits use. Tests whether the two
negative-bias mechanisms STACK (plausibly super-additively, since both attack the same
near-tail region t/tau ~ 3-15) toward the observed casey wedge (~-1.6). Neither alone
reaches it: heavy-tail single-comp maxed -0.21, close leakage maxed -0.43.

Reuses the validated plpbf_inject machinery (kernel, single-comp free-alpha _LLEMG, fit).
Only difference vs plpbf_leakage: the primary AND secondary carry a finite inner-scale s_i
(heavy tail) instead of s_i=inf (exponential).

CLI: python plpbf_bothatonce.py <beta_prim> <tau_1ghz> <s_i> <W_over_tau> <amp_frac> \
                                <dt_offset_ms> [nlive=300] [nproc=4] [seed=0] [sig2_frac=1.0]
"""
import json
import os
import sys

import numpy as np

REFIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REFIT)

from plpbf_inject import (  # noqa: E402
    DT, EMG_HI, EMG_LO, FREQ_C, FREQ_D, NT, NU0, OUTDIR,
    _LLEMG, _kernel, alpha_from_beta, fit,
)


def simulate_both(tau_1ghz, beta_prim, s_i, W_over_tau, amp_frac, dt_offset_ms,
                  snr=40.0, seed=0, sig2_frac=1.0):
    """Heavy PL-tail primary (finite s_i, beta_prim<4) + close secondary at +dt_offset_ms,
    amplitude amp_frac * primary, same screen (same s_i/beta). Per-channel gain x white
    noise at peak S/N ~ snr, identical to plpbf_leakage.simulate_leakage."""
    rng = np.random.default_rng(seed)
    alpha = alpha_from_beta(beta_prim)  # beta<4 -> alpha>4, unclamped region (clamp bites beta>=3.98)
    sig = W_over_tau * tau_1ghz
    t0 = 0.30 * NT * DT
    out = {}
    for band, freq in (("C", FREQ_C), ("D", FREQ_D)):
        K1 = _kernel(freq, tau_1ghz, alpha, sig, s_i, beta_prim, t0)
        K2 = _kernel(freq, tau_1ghz, alpha, sig * sig2_frac, s_i, beta_prim, t0 + dt_offset_ms)
        K = K1 + amp_frac * K2
        gain = (freq / NU0) ** (-1.5) * (1.0 + 0.3 * rng.standard_normal(len(freq)))
        clean = gain[:, None] * K
        noise = clean.max() / snr
        out[band] = dict(freq=freq, data=clean + rng.normal(0.0, noise, clean.shape), noise=noise)
    out["truth"] = dict(tau_1ghz=tau_1ghz, beta=beta_prim, alpha=alpha, s_i=s_i, sig=sig, t0=t0,
                        W_over_tau=W_over_tau, amp_frac=amp_frac, dt_offset_ms=dt_offset_ms,
                        sig2_frac=sig2_frac, snr=snr)
    return out


def main():
    beta = float(sys.argv[1])
    tau = float(sys.argv[2])
    s_i = float(sys.argv[3])
    W = float(sys.argv[4])
    amp = float(sys.argv[5])
    dt = float(sys.argv[6])
    nlive = int(sys.argv[7]) if len(sys.argv) > 7 else 300
    nproc = int(sys.argv[8]) if len(sys.argv) > 8 else 4
    seed = int(sys.argv[9]) if len(sys.argv) > 9 else 0
    sig2 = float(sys.argv[10]) if len(sys.argv) > 10 else 1.0

    inj = simulate_both(tau, beta, s_i, W, amp, dt, seed=seed, sig2_frac=sig2)
    tr = inj["truth"]
    print(f"INJECT-BOTH tau={tau} beta_prim={beta} s_i={s_i} alpha_true={tr['alpha']:.3f} "
          f"W/tau={W} amp={amp} dt={dt}ms", flush=True)

    eq, lnz, lnze = fit(_LLEMG(inj), EMG_LO, EMG_HI, nlive, nproc, seed)
    aP = np.percentile(eq[:, 1], [5, 50, 95])
    tauP = np.percentile(eq[:, 0], [5, 50, 95])
    bias = aP[1] - tr["alpha"]
    print(f"  1-COMP EMG fit alpha_apparent={aP[1]:.3f}[{aP[0]:.3f},{aP[2]:.3f}]  "
          f"(alpha_true={tr['alpha']:.3f})  bias={bias:+.3f}  lnZ={lnz:.1f}", flush=True)
    print(f"  BOTH-BIAS {bias:+.3f}  (heavy-tail-alone max -0.21; close-leakage-alone max -0.43; "
          f"need ~-1.6)", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(dict(mode="both", truth=tr, alpha_apparent=aP.tolist(), alpha_true=tr["alpha"],
                   bias=bias, tau_apparent=tauP.tolist(), lnZ=lnz),
              open(f"{OUTDIR}/plpbf_both_b{beta}_si{s_i}_W{W}_a{amp}_dt{dt}_s{seed}.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
