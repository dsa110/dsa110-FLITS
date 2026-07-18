"""Scint-gain-leakage injection (task #13 successor probe).

Question: can TIME-DECORRELATING scintillation bias the JOINT free-alpha wedge below
the true nu^-4, i.e. is the sub-4 wedge (casey/wilhelm) a scint-gain-marginalization
artifact rather than real chromatic scattering?

Mechanism (from the scint campaign's generator2d.py, generalized here to a
frequency-DEPENDENT scattering timescale): a scatter-broadened burst is the sum over
scattering-DELAY slices of |impulse response|^2; the field at different delays is an
independent complex Gaussian, so each delay slice k carries an INDEPENDENT scintillation
realization G_k(nu) with the same decorrelation bandwidth Delta-nu_d (Lorentzian-ACF
HWHM) and modulation m. A narrow core window sees full scintle contrast; the scattering
tail sums many independent slices, so contrast dilutes ~1/sqrt(N_eff) along the pulse.
A STATIC per-channel spectrum is fully absorbed by the per-channel gain marginalization
(g_f soaks any time-independent modulation); ONLY this time-decorrelating structure --
scintles interacting with the time-frequency covariance -- can leak into the temporal
(tau/alpha) fit. The leakage is worst where the gain prior is weakest: coarse
channelization (few channels), where g_f cannot track the scintle pattern.

Truth: intrinsic Gaussian (x) one-sided exponential scattering with tau(nu)=tau_1ghz *
nu_GHz^-alpha_true (alpha_true=4, the exact thin-screen line). We impose the decorrelating
scintillation in the TARGET band at that band's real channelization (casey C: 64 ch ~6.2
MHz; wilhelm C: 8 ch ~50 MHz), keep the OTHER band clean, and refit with the SAME joint
gain-marginal free-alpha EMG the wedge uses. alpha_recovered < 4 => scint-gain leakage
reproduces (part of) the wedge; the bias magnitude bounds the mechanism's contribution.
scint_decorr=False (static control) should recover alpha~4, proving the leakage is the
DECORRELATION, not the mere presence of spectral structure.

Reuses plpbf_inject (_LLEMG joint free-alpha EMG likelihood, fit, EMG priors, DT/NT/NU0).
Delta-nu_d and m per band come from the 2L window-campaign table (casey/wilhelm) once
sourced; parameterized here so the driver is ready. If casey/wilhelm are 2L
non-detections, inject at the upper-limit Delta-nu_d for a BOUND on the mechanism.

CLI: python scint_leakage_inject.py <burst> <target_band C|D> <nchan_target> \
        <gamma_mhz> <m> <tau_1ghz> [alpha_true=4] [snr=40] [decorr=1] \
        [nlive=300] [nproc=4] [seed=0] [--smoke]
"""
import json
import os
import sys

import numpy as np

REFIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REFIT)

from plpbf_inject import DT, EMG_HI, EMG_LO, NT, NU0, OUTDIR, _LLEMG, fit  # noqa: E402

# fixed CLEAN-band channelizations (non-target band); target band is built per-CLI
FREQ_C_DEFAULT = np.linspace(0.40, 0.80, 64)   # GHz
FREQ_D_DEFAULT = np.linspace(1.31, 1.50, 48)   # GHz
BAND_EDGES = {"C": (0.40, 0.80), "D": (1.31, 1.50)}  # GHz


def _lorentzian_scint_spectrum(n_chan, chan_width_mhz, gamma_mhz, rng):
    """Fully-modulated (m=1) intensity spectrum, Lorentzian-ACF HWHM = gamma_mhz.

    Cite generator.py (scint-injection-harness): complex-Gaussian delay field with a
    one-sided exponential delay power exp(-tau/tau_s), tau_s = 1/(2 pi gamma); |FFT|^2
    is the intensity, unit-mean (Cordes & Rickett 1998 strong-scintillation result).
    """
    B = n_chan * chan_width_mhz
    d_tau = 1.0 / B
    tau = np.arange(n_chan) * d_tau
    tau_s = 1.0 / (2.0 * np.pi * gamma_mhz)
    power = np.exp(-tau / tau_s)
    g = (rng.standard_normal(n_chan) + 1j * rng.standard_normal(n_chan)) * np.sqrt(power / 2.0)
    I = np.abs(np.fft.fft(g)) ** 2
    return I / I.mean()


def make_band(freqs_ghz, tau_1ghz, alpha_true, sigma_t_ms, t_peak_ms, gamma_mhz, m,
              snr, rng, scint_decorr=True):
    """One band's (n_chan, NT) dynamic spectrum: EMG scattered burst with tau(nu)=
    tau_1ghz*nu^-alpha, modulated by scintillation. m=0 -> clean EMG. m>0 with
    scint_decorr True -> independent scint per delay slice (leakage); False -> one
    static scint pattern shared across delays (control, gain-marg should absorb it).
    """
    n_chan = freqs_ghz.size
    chan_width_mhz = abs(freqs_ghz[1] - freqs_ghz[0]) * 1000.0
    t = np.arange(NT) * DT
    tau_scat = tau_1ghz * (freqs_ghz / NU0) ** (-alpha_true)   # ms, per channel
    k = np.arange(NT)
    delay = k * DT                                             # ms, delay grid

    # Within-channel scintle averaging: when Delta-nu_d < channel width, each channel
    # spans N ~ ch_bw/Delta-nu_d independent scintles, so the channel-averaged intensity
    # modulation is suppressed to m_eff = m * sqrt(Delta-nu_d/ch_bw) (variance of an
    # N-fold average of independent unit-modulation scintles); >= ch_bw it saturates at m.
    # This is the physical suppression that happens BEFORE the gain marginalization acts.
    # (Analytic bound; a fine-grid-generate-then-block-average would be the gold standard
    # but is unnecessary for a BOUND that already comes out tiny.) The coarse-grid field
    # from _lorentzian_scint_spectrum is ~white channel-to-channel in this regime, i.e.
    # correctly decorrelated; only its amplitude needs the m_eff rescale.
    m_eff = m * min(1.0, float(np.sqrt(gamma_mhz / chan_width_mhz))) if m > 0 else 0.0

    if m_eff <= 0:
        Gmat = np.ones((n_chan, NT))
    elif scint_decorr:
        Gmat = np.empty((n_chan, NT))
        for kk in range(NT):
            g = _lorentzian_scint_spectrum(n_chan, chan_width_mhz, gamma_mhz, rng)
            Gmat[:, kk] = 1.0 + m_eff * (g - 1.0)
    else:
        g = _lorentzian_scint_spectrum(n_chan, chan_width_mhz, gamma_mhz, rng)
        Gmat = np.repeat((1.0 + m_eff * (g - 1.0))[:, None], NT, axis=1)  # static

    tail = np.exp(-delay[None, :] / tau_scat[:, None])         # [nu, k]
    tt = t[None, :] - (t_peak_ms + delay[:, None])             # [k, t]
    gauss = np.exp(-0.5 * (tt / sigma_t_ms) ** 2)              # [k, t]
    signal = (Gmat * tail) @ gauss                             # [nu, t]
    noise = float(signal.max()) / snr
    data = signal + rng.normal(0.0, noise, signal.shape)
    return dict(freq=freqs_ghz, data=data, noise=noise, m_eff=m_eff,
                chan_width_mhz=chan_width_mhz)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    smoke = "--smoke" in sys.argv[1:]
    burst = args[0]
    tband = args[1].upper()
    nchan_t = int(args[2])
    gamma_mhz = float(args[3])
    m = float(args[4])
    tau_1ghz = float(args[5])
    alpha_true = float(args[6]) if len(args) > 6 else 4.0
    snr = float(args[7]) if len(args) > 7 else 40.0
    decorr = bool(int(args[8])) if len(args) > 8 else True
    nlive = int(args[9]) if len(args) > 9 else 300
    nproc = int(args[10]) if len(args) > 10 else 4
    seed = int(args[11]) if len(args) > 11 else 0
    if tband not in ("C", "D"):
        raise SystemExit("target band must be C or D")
    if smoke:
        nlive, nproc = 40, 1

    rng = np.random.default_rng(seed)
    sigma_t_ms = 0.05
    t_peak_ms = 0.30 * NT * DT   # 6.14 ms, inside the EMG t0 prior [4.10, 8.19]

    lo_g, hi_g = BAND_EDGES[tband]
    freq_t = np.linspace(lo_g, hi_g, nchan_t)
    freq_other = FREQ_D_DEFAULT if tband == "C" else FREQ_C_DEFAULT
    other_band = "D" if tband == "C" else "C"

    inj = {}
    # target band: decorrelating (or static-control) scintillation at its channelization
    inj[tband] = make_band(freq_t, tau_1ghz, alpha_true, sigma_t_ms, t_peak_ms,
                           gamma_mhz, m, snr, rng, scint_decorr=decorr)
    # other band: clean EMG (m=0), so the joint alpha is anchored by a clean band + the
    # contaminated band's remaining shape -- mirrors the real config (dipole in ONE band)
    inj[other_band] = make_band(freq_other, tau_1ghz, alpha_true, sigma_t_ms, t_peak_ms,
                                gamma_mhz, 0.0, snr, rng, scint_decorr=False)
    inj["truth"] = dict(burst=burst, target_band=tband, nchan_target=nchan_t,
                        gamma_mhz=gamma_mhz, m=m, tau_1ghz=tau_1ghz, alpha=alpha_true,
                        snr=snr, decorr=decorr, seed=seed,
                        chan_width_mhz_target=(hi_g - lo_g) * 1000.0 / nchan_t)
    tr = inj["truth"]
    tr["m_eff_target"] = inj[tband]["m_eff"]
    mode = "DECORR" if decorr else "STATIC-control"
    print(f"INJECT-SCINT-LEAK burst={burst} band={tband} nchan={nchan_t} "
          f"({tr['chan_width_mhz_target']:.1f} MHz/ch) gamma={gamma_mhz}MHz m={m} "
          f"-> m_eff={inj[tband]['m_eff']:.4f} (Dnu_d/ch_bw suppression) "
          f"tau={tau_1ghz} alpha_true={alpha_true} [{mode}]", flush=True)

    eq, lnz, lnze = fit(_LLEMG(inj), EMG_LO, EMG_HI, nlive, nproc, seed)
    aP = np.percentile(eq[:, 1], [5, 50, 95])
    tauP = np.percentile(eq[:, 0], [5, 50, 95])
    bias = aP[1] - alpha_true
    verdict = ("LEAKAGE: scint biases joint alpha below nu^-4" if aP[2] < 3.90
               else ("no leakage (alpha recovers ~4)" if aP[0] > 3.90 else "marginal/straddles"))
    print(f"  JOINT free-alpha: alpha_apparent={aP[1]:.3f}[{aP[0]:.3f},{aP[2]:.3f}]  "
          f"bias={bias:+.3f}  tau={tauP[1]:.4g}  lnZ={lnz:.1f}\n  VERDICT: {verdict}", flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    out = f"{OUTDIR}/scintleak_{burst}_{tband}{nchan_t}_g{gamma_mhz}_m{m}_{'dec' if decorr else 'sta'}_s{seed}.json"
    json.dump(dict(mode="scint_leakage", truth=tr, alpha_apparent=aP.tolist(),
                   alpha_true=alpha_true, bias=bias, tau_apparent=tauP.tolist(),
                   lnZ=lnz, verdict=verdict), open(out, "w"), indent=2)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
