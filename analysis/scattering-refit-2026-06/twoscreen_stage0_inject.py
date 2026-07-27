"""Stage 0 -- two-screen wedge-reproduction falsifier (charter sec 3).

Pre-registered question: with BOTH screens at the same thin-screen alpha (=4 here,
pure exponential screens), the composite pulse-broadening width scales exactly nu^-4,
so a free-alpha SINGLE-screen fit "should" return alpha~4. The charter's hypothesis is
that shape NON-self-similarity -- the intrinsic Gaussian sigma and the time bin do NOT
scale with nu, while tau_1(nu),tau_2(nu) do -- makes the two-tail composite shape change
across the CHIME/DSA lever arm, so a single-screen EMG mis-estimates alpha. Whether that
mis-estimation reaches the observed wedge (casey alpha=2.43 / wilhelm 2.57, bias ~ -1.6)
is testable HERE, before any real-data two-screen fit.

Design:
  - Inject a two-screen burst (twoscreen.two_screen_perchan) with alpha_true=4, both
    screens exponential, ratio r=tau_2/tau_1 at 1 GHz in {0.1,0.3,1.0}. Screen-1
    tau_1ghz = tau_real/(1+r) so the COMPOSITE mean delay tau_1(1+r) is held at the
    burst's real production tau_1ghz -- r sweeps SHAPE at fixed total scattering.
  - casey-like / wilhelm-like tau/sigma/band configs from the real production fits
    (tau_1ghz, DSA-anchored intrinsic sigma, real channelization).
  - Refit with the FREE-ALPHA single-screen EMG diagnostic (plpbf_inject._LLEMG, the
    exact fitter that measured the real wedges).

PASS (per config) = recovered alpha <= alpha_true - 1 (bias <= -1). Stage-0 PASS =
ANY config passes. FAIL = max reachable |bias| << 1.6 across the grid (+ the W/tau
envelope sweep) => rung-1 two-screen joins the elimination table, no real-data fits.

CLI: python twoscreen_stage0_inject.py <burst casey|wilhelm> <r> [alpha_true=4]
        [w_over_tau=auto] [snr=40] [nlive=400] [nproc=4] [seed=0] [--smoke]
"""
import json
import os
import sys

import numpy as np

REFIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REFIT)

from plpbf_inject import DT, EMG_HI, EMG_LO, NT, NU0, OUTDIR, TIME, _LLEMG, fit  # noqa: E402
from twoscreen import two_screen_perchan  # noqa: E402

# casey-like / wilhelm-like configs: real production tau_1ghz, DSA-anchored intrinsic
# sigma (ms), and real channelization (nchan_C, nchan_D). alpha_true=4 (both screens exp).
CONFIGS = {
    "casey":   dict(tau_real=0.019, sigma_ms=0.055, nC=64, nD=48),
    "wilhelm": dict(tau_real=0.330, sigma_ms=0.100, nC=8,  nD=48),
}
BAND_EDGES = {"C": (0.40, 0.80), "D": (1.31, 1.50)}  # GHz


def _twoscreen_band(freq, tau1_1ghz, r, alpha, sigma_ms, t0, snr, rng):
    """One band's (nf, NT) dynamic spectrum: two-screen composite scattered burst.

    tau1_1ghz is SCREEN-1's tau at 1 GHz; per channel tau_1(nu)=tau1_1ghz*nu^-alpha and
    tau_2=r*tau_1. Smooth spectrum x mild per-channel gain scatter (absorbed by the gain
    marginal); white noise to a per-channel peak S/N ~ snr.
    """
    nf = freq.size
    tau1 = np.clip(tau1_1ghz * (freq / NU0) ** (-alpha), 1e-6, None)[:, None]  # (nf,1)
    mu = np.full((nf, 1), float(t0))
    sig = np.full((nf, 1), float(sigma_ms))
    K = two_screen_perchan(TIME[None, :], mu, sig, tau1, r)                    # (nf,NT)
    gain = (freq / NU0) ** (-1.5) * (1.0 + 0.3 * rng.standard_normal(nf))
    clean = gain[:, None] * K
    noise = clean.max() / snr
    data = clean + rng.normal(0.0, noise, clean.shape)
    return dict(freq=freq, data=data, noise=noise, K_true=K, gain=gain)


def simulate(cfg, r, alpha_true, w_over_tau, snr, seed):
    rng = np.random.default_rng(seed)
    tau_real = cfg["tau_real"]
    tau1_1ghz = tau_real / (1.0 + r)                       # composite mean held at tau_real
    sigma_ms = cfg["sigma_ms"] if w_over_tau is None else w_over_tau * tau_real
    t0 = 0.30 * NT * DT
    out = {}
    for band, nch in (("C", cfg["nC"]), ("D", cfg["nD"])):
        lo, hi = BAND_EDGES[band]
        freq = np.linspace(lo, hi, nch)
        out[band] = _twoscreen_band(freq, tau1_1ghz, r, alpha_true, sigma_ms, t0, snr, rng)
    out["truth"] = dict(tau_real=tau_real, tau1_screen1=tau1_1ghz, r=r, alpha_true=alpha_true,
                        sigma_ms=sigma_ms, t0=t0, w_over_tau=(sigma_ms / tau_real))
    return out


def main():
    burst = sys.argv[1]
    r = float(sys.argv[2])
    alpha_true = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    w_over_tau = None
    if len(sys.argv) > 4 and sys.argv[4] not in ("auto", "-"):
        w_over_tau = float(sys.argv[4])
    snr = float(sys.argv[5]) if len(sys.argv) > 5 else 40.0
    nlive = int(sys.argv[6]) if len(sys.argv) > 6 else 400
    nproc = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    if "--smoke" in sys.argv:
        nlive, nproc = 60, 2

    cfg = CONFIGS[burst]
    inj = simulate(cfg, r, alpha_true, w_over_tau, snr, seed)
    tr = inj["truth"]
    print(f"INJECT-TWOSCREEN burst={burst} r={r} alpha_true={alpha_true} "
          f"tau_real={tr['tau_real']} tau1_screen1={tr['tau1_screen1']:.4f} "
          f"sigma={tr['sigma_ms']:.4f} W/tau={tr['w_over_tau']:.2f} nC={cfg['nC']} nD={cfg['nD']} "
          f"snr={snr} seed={seed}", flush=True)

    # free-alpha SINGLE-screen EMG refit (the real-wedge diagnostic)
    eq, lnz, lnze = fit(_LLEMG(inj), EMG_LO, EMG_HI, nlive, nproc, seed)
    aP = np.percentile(eq[:, 1], [5, 50, 95])   # alpha_apparent
    tauP = np.percentile(eq[:, 0], [5, 50, 95])
    bias = aP[1] - alpha_true
    verdict = "REPRODUCES" if bias <= -1.0 else ("partial" if bias <= -0.3 else "no-wedge")
    print(f"  EMG-refit alpha_apparent={aP[1]:.3f}[{aP[0]:.3f},{aP[2]:.3f}] "
          f"(alpha_true={alpha_true})  bias={bias:+.3f}  tau_app={tauP[1]:.4f}  lnZ={lnz:.1f}", flush=True)
    print(f"  VERDICT r={r} {burst}: {verdict} (PASS if bias<=-1)", flush=True)
    rec = dict(burst=burst, r=r, truth=tr, alpha_apparent=aP.tolist(), alpha_true=alpha_true,
               bias=float(bias), tau_apparent=tauP.tolist(), lnZ=lnz, verdict=verdict,
               nC=cfg["nC"], nD=cfg["nD"], snr=snr, seed=seed)
    tag = f"{burst}_r{r}_W{tr['w_over_tau']:.2f}_s{seed}"
    json.dump(rec, open(f"{OUTDIR}/twoscreen_stage0_{tag}.json", "w"), indent=2)
    print(f"DONE TWOSCREEN-STAGE0 {tag} bias={bias:+.3f} verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()
