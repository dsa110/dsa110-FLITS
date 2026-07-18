"""Injection harness for the power-law-PBF (inner-scale) model.

Standing rule: validate a new model on known-truth injections BEFORE any real-data
fit. Two arms:

  RECOVERY  -- simulate a joint CHIME+DSA burst scattered by the inner-scale PL-PBF
               at known (tau_1ghz, beta, s_i), fit it back with the SAME model, and
               check the truth lands in the posterior.
  EMG BIAS  -- fit the same PL-PBF injections with the exponential PBF + FREE alpha
               (the EMG), and confirm the Cordes Fig-58 result: the apparent alpha is
               biased LOW relative to the true alpha=2 beta/(beta-2), worsening as the
               unscattered width W_u/tau grows. This positive control also calibrates
               how to read the relaxed-alpha EMG A/B.

Likelihood = per-channel amplitude (gain) marginalized matched filter, identical in
form to FRBModel.log_likelihood_gain_marginal (S_dd - S_dk^2/S_kk), so recovery here
mirrors the real gain-marginal joint fit. Single temporal component per band.

  python plpbf_inject.py recover  <beta> <s_i> [W_over_tau] [nlive] [nproc] [seed]
  python plpbf_inject.py embias    <beta> <s_i> [W_over_tau] [nlive] [nproc] [seed]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from dynesty import NestedSampler

OUTDIR = os.environ.get("PLPBF_OUTDIR", "/tmp")

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from plpbf import gaussian_pbf_innerscale_convolution

# alpha tie (thin-screen); clamps to 4 for beta>=3.98 exactly like the production map
from scat_analysis.turbulence import alpha_from_beta

# Joint band setup mirroring the real fit (GHz); coarse channelization for speed.
FREQ_C = np.linspace(0.40, 0.80, 32)   # CHIME
FREQ_D = np.linspace(1.31, 1.50, 24)   # DSA
NU0 = 1.0                               # tau_1ghz reference (GHz)
DT = 0.02                               # ms/bin
NT = 1024                               # time bins = 20.5 ms window (covers the PL tail
                                        # for tau_1ghz~0.05 ms: tau_CHIME~0.5 ms, s_i<=~30
                                        # tail < 15 ms; injection+recovery share NT so any
                                        # truncation is consistent and unbiased)
TIME = np.arange(NT) * DT


def tau_nu(tau_1ghz, alpha, freq):
    return tau_1ghz * (freq / NU0) ** (-alpha)


def _kernel(freq, tau_1ghz, alpha, sig_ms, s_i, beta, t0):
    """Unit-amplitude per-channel model K[f,t] for the inner-scale PL-PBF (s_i finite,
    the value at NU0) or the exponential PBF (s_i=inf; beta>=3.98 -> pure exp). Three
    regimes in s=lag/tau (Cordes Fig 40), matching plpbf.pbf_innerscale exactly:
      core exp(-s) for s<=s_c;  CLEAN power-law exp(-s_c)(s/s_c)^(-beta/2) for s_c<s<s_i;
      inner-scale cutoff [pl(s_i)] exp(-(s-s_i)/s_i) for s>=s_i.
    (The cutoff runs from s_i, NOT s_c -- an earlier cutoff-from-s_c form softened the
    power-law and biased beta high.) The inner scale is CHROMATIC: per Cordes Fig 58,
    zeta(nu) propto nu^(-2/(beta-2)) at fixed l_i gives s_i(nu) = s_i0 (nu/nu0)^(+4/(beta-2))
    -- tail LONGER at high nu (DSA), shorter at low nu (CHIME). Fully vectorized: one
    batched rfft over the (nf, nt) kernel, not nf separate transforms."""
    from plpbf import BETA_MIN, BETA_MAX
    taus = np.clip(tau_nu(tau_1ghz, alpha, freq), 1e-4, None)     # (nf,)
    beta = float(np.clip(beta, BETA_MIN, BETA_MAX))
    s_c = max(2.0 * np.log(2.0 / (4.0 - beta)), 1e-3)
    lag = (np.arange(NT) * DT)[None, :]                           # (1, nt)
    s = lag / taus[:, None]                                       # (nf, nt)
    core = np.exp(-s)
    with np.errstate(over="ignore", invalid="ignore"):
        pl = np.exp(-s_c) * (np.maximum(s, s_c) / s_c) ** (-0.5 * beta)   # (nf, nt)
        if np.isfinite(s_i):
            s_i_ch = (float(s_i) * (freq / NU0) ** (4.0 / (beta - 2.0)))[:, None]  # (nf,1) chromatic
            pl_si = np.exp(-s_c) * (np.maximum(s_i_ch, s_c) / s_c) ** (-0.5 * beta)  # value at s_i
            cut = pl_si * np.exp(-(s - s_i_ch) / s_i_ch)
            tail = np.where(s < s_i_ch, pl, cut)                 # clean PL then exp cutoff
            tail = np.where(s <= s_c, core, tail)
            h = np.where(s_i_ch <= s_c, core, tail)              # window closed per-channel -> exp
        else:
            h = np.where(s <= s_c, core, pl)                     # production PL-PBF (unbounded tail)
    h = np.where(np.isfinite(h), h, 0.0)
    h = h / np.clip(h.sum(axis=1, keepdims=True) * DT, 1e-30, None)
    g = (1.0 / (np.sqrt(2.0 * np.pi) * sig_ms)) * np.exp(-0.5 * ((TIME[None, :] - t0) / sig_ms) ** 2)
    L = 2 * NT
    conv = np.fft.irfft(np.fft.rfft(g, L, axis=1) * np.fft.rfft(h, L, axis=1), L, axis=1)
    return conv[:, :NT] * DT                                      # (nf, nt)


def simulate(tau_1ghz, beta, s_i, W_over_tau, snr=40.0, seed=0):
    """Inner-scale PL-PBF joint injection. sig (intrinsic Gaussian width) set from
    W_over_tau * tau at 1 GHz; per-channel gains ~ smooth spectrum * scintillation;
    white noise scaled to a per-channel peak S/N ~ snr."""
    rng = np.random.default_rng(seed)
    alpha = alpha_from_beta(beta)
    sig = W_over_tau * tau_1ghz
    t0 = 0.30 * NT * DT
    out = {}
    for band, freq in (("C", FREQ_C), ("D", FREQ_D)):
        K = _kernel(freq, tau_1ghz, alpha, sig, s_i, beta, t0)
        gain = (freq / NU0) ** (-1.5) * (1.0 + 0.3 * rng.standard_normal(len(freq)))  # spectrum x scint
        clean = gain[:, None] * K
        noise = clean.max() / snr
        data = clean + rng.normal(0.0, noise, clean.shape)
        out[band] = dict(freq=freq, data=data, noise=noise, K_true=K, gain=gain)
    out["truth"] = dict(tau_1ghz=tau_1ghz, beta=beta, s_i=s_i, alpha=alpha, sig=sig, t0=t0,
                        W_over_tau=W_over_tau)
    return out


def _band_ll(data, noise, K):
    """Per-channel gain-marginal matched-filter logL (flat gain prior), summed over
    channels: -0.5 (S_dd - S_dk^2/S_kk)/var - 0.5 ln S_kk + 0.5 ln(2 pi var)."""
    var = noise ** 2
    S_dd = np.einsum("ij,ij->i", data, data)
    S_dk = np.einsum("ij,ij->i", data, K)
    S_kk = np.einsum("ij,ij->i", K, K)
    ok = S_kk > 1e-30
    S_kk_safe = np.where(ok, S_kk, 1.0)
    chi2 = np.where(ok, (S_dd - S_dk ** 2 / S_kk_safe) / var, S_dd / var)
    occam = np.where(ok, -0.5 * np.log(S_kk_safe), 0.0)
    const = 0.5 * np.log(2.0 * np.pi * var)
    ll = float(np.sum(-0.5 * chi2 + occam + const))
    return ll if np.isfinite(ll) else -1e100


class _LLPL:
    """PL-PBF joint logL. theta = [tau_1ghz, beta, log10 s_i, sig, t0]; alpha tied."""

    def __init__(self, inj):
        self.inj = inj

    def __call__(self, th):
        tau, beta, log_si, sig, t0 = (float(th[i]) for i in range(5))
        s_i = 10.0 ** log_si
        a = alpha_from_beta(beta)
        ll = 0.0
        for band in ("C", "D"):
            b = self.inj[band]
            K = _kernel(b["freq"], tau, a, sig, s_i, beta, t0)
            ll += _band_ll(b["data"], b["noise"], K)
        return ll if np.isfinite(ll) else -1e100


class _LLEMG:
    """Exponential-PBF + FREE alpha joint logL (the Fig-58 control).
    theta = [tau_1ghz, alpha, sig, t0]; beta fixed at 3.99 (pure exp), s_i irrelevant."""

    def __init__(self, inj):
        self.inj = inj

    def __call__(self, th):
        tau, alpha, sig, t0 = (float(th[i]) for i in range(4))
        ll = 0.0
        for band in ("C", "D"):
            b = self.inj[band]
            K = _kernel(b["freq"], tau, alpha, sig, np.inf, 3.99, t0)  # beta=3.99 -> exp core
            ll += _band_ll(b["data"], b["noise"], K)
        return ll if np.isfinite(ll) else -1e100


# prior bounds
PL_LO = np.array([1e-3, 3.00, -1.0, 1e-3, 0.20 * NT * DT])   # tau,beta,log10 s_i,sig,t0
PL_HI = np.array([2.0, 3.99, 3.0, 5.0, 0.40 * NT * DT])
EMG_LO = np.array([1e-3, 2.0, 1e-3, 0.20 * NT * DT])          # tau,alpha,sig,t0
EMG_HI = np.array([2.0, 6.0, 5.0, 0.40 * NT * DT])


class _PTform:
    """Picklable unit-cube -> box transform (module-level so dynesty.pool can ship it
    to workers under either fork or spawn)."""

    def __init__(self, lo, hi):
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)

    def __call__(self, u):
        return self.lo + u * (self.hi - self.lo)


def fit(loglike, lo, hi, nlive, nproc, seed=0):
    ndim = len(lo)
    pt = _PTform(lo, hi)
    if nproc and nproc > 1:
        import multiprocessing as mp
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            pass
        from dynesty import pool as dypool
        with dypool.Pool(int(nproc), loglike, pt) as pool:
            s = NestedSampler(pool.loglike, pool.prior_transform, ndim, nlive=nlive,
                              sample="rwalk", pool=pool, queue_size=int(nproc), rstate=np.random.default_rng(seed))
            s.run_nested(dlogz=0.5, print_progress=False)
            r = s.results
    else:
        s = NestedSampler(loglike, pt, ndim, nlive=nlive, sample="rwalk", rstate=np.random.default_rng(seed))
        s.run_nested(dlogz=0.5, print_progress=False)
        r = s.results
    from dynesty.utils import resample_equal
    w = np.exp(r.logwt - r.logz[-1]); w /= w.sum()
    eq = resample_equal(r.samples, w)
    return eq, float(r.logz[-1]), float(r.logzerr[-1])


def main():
    mode = sys.argv[1]
    beta = float(sys.argv[2])
    s_i = float(sys.argv[3])
    W = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    nlive = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    nproc = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    tau_1ghz = 0.05
    inj = simulate(tau_1ghz, beta, s_i, W, seed=seed)
    tr = inj["truth"]
    print(f"INJECT tau_1ghz={tau_1ghz} beta={beta} s_i={s_i} alpha_true={tr['alpha']:.3f} "
          f"W/tau={W} sig={tr['sig']:.4f} t0={tr['t0']:.3f}", flush=True)

    def pct(col):
        return np.percentile(col, [5, 50, 95])

    if mode == "recover":
        eq, lnz, lnze = fit(_LLPL(inj), PL_LO, PL_HI, nlive, nproc, seed)
        tauP, betaP, siP, sigP, t0P = (pct(eq[:, i]) for i in range(5))
        aP = np.percentile([alpha_from_beta(b) for b in eq[:, 1]], [5, 50, 95])
        rec = dict(mode="recover", truth=tr, lnZ=lnz,
                   tau_1ghz=tauP.tolist(), beta=betaP.tolist(), log10_s_i=siP.tolist(),
                   alpha=aP.tolist(), sig=sigP.tolist(), t0=t0P.tolist())
        print(f"  PL fit  tau={tauP[1]:.4f}[{tauP[0]:.4f},{tauP[2]:.4f}]  "
              f"beta={betaP[1]:.3f}[{betaP[0]:.3f},{betaP[2]:.3f}]  "
              f"s_i=10^{siP[1]:.2f}[{siP[0]:.2f},{siP[2]:.2f}]  "
              f"alpha={aP[1]:.3f}  sig={sigP[1]:.4f}  lnZ={lnz:.1f}", flush=True)
        ok_b = betaP[0] <= beta <= betaP[2]
        ok_s = (siP[0] <= np.log10(s_i) <= siP[2]) if np.isfinite(s_i) else siP[2] > 2.5
        print(f"  RECOVERY beta_in_90CI={ok_b}  s_i_in_90CI={ok_s}", flush=True)
        json.dump(rec, open(f"{OUTDIR}/plpbf_recover_b{beta}_si{s_i}_W{W}_s{seed}.json", "w"), indent=2)
    elif mode == "embias":
        eq, lnz, lnze = fit(_LLEMG(inj), EMG_LO, EMG_HI, nlive, nproc, seed)
        tauP = pct(eq[:, 0])                                  # tau_apparent (Fig 54 tau-bias)
        aP = pct(eq[:, 1])                                    # alpha_apparent (Fig 58 alpha-bias)
        tau_ratio = tauP[1] / tr["tau_1ghz"]
        print(f"  EMG fit alpha_apparent={aP[1]:.3f}[{aP[0]:.3f},{aP[2]:.3f}]  "
              f"(alpha_true={tr['alpha']:.3f})  bias={aP[1]-tr['alpha']:+.3f}  lnZ={lnz:.1f}", flush=True)
        print(f"  EMG fit tau_apparent={tauP[1]:.4f} (tau_true={tr['tau_1ghz']:.4f})  "
              f"tau_hat/tau={tau_ratio:.3f}", flush=True)
        biased_low = aP[1] < tr["alpha"]
        print(f"  FIG-58 CONTROL alpha_apparent < alpha_true : {biased_low}   "
              f"FIG-54 CONTROL tau_hat/tau > 1 : {tau_ratio > 1.0}", flush=True)
        json.dump(dict(mode="embias", truth=tr, alpha_apparent=aP.tolist(),
                       alpha_true=tr["alpha"], bias=aP[1] - tr["alpha"],
                       tau_apparent=tauP.tolist(), tau_true=tr["tau_1ghz"],
                       tau_ratio=tau_ratio, W_over_tau=tr["W_over_tau"], lnZ=lnz),
                  open(f"{OUTDIR}/plpbf_embias_b{beta}_si{s_i}_W{W}_s{seed}.json", "w"), indent=2)
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
