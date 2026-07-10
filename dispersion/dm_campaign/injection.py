"""Known-truth DM injection harness (plan-dm-measurement-methods, Phase 0).

Synthetic EMG pulses with a known residual DM are injected into REAL off-pulse
noise drawn from the co-detection products, so every DM estimator is validated
against ground truth before its verdict on the science sample counts.

Sign convention (matches shift_waterfall_residual_dm / chime_dm.measure_dm):
a POSITIVE residual DM means low-frequency channels arrive LATER than the
top-of-band reference (under-dedispersed data); ``disperse_waterfall(+dm)``
therefore delays low channels.

Kernel coupling, stated: the pulse shape reuses ``chime_dm.exgauss`` — the same
EMG family the arrival-regression estimator fits. That coupling is acceptable
for DM-recovery validation (the truth being tested is the dispersive sweep,
not the profile family), and the estimator-independent shape validation
already lives in tests/test_chime_dm.py (numerical-convolution injector).

Units at this API: freq_ghz [GHz], dt_ms [ms] (converted internally to the
MHz/seconds convention of dispersion.chime_dm).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from dispersion.chime_dm import K_DM, _dedisperse, exgauss

K_DM_MS_GHZ2 = K_DM * 1e-3  # 4.148808 ms GHz^2 pc^-1 cm^3 (K_DM is s MHz^2)

# Analysis-product resolution (256 ch, 163.84 us) -- the resolution the battery
# runs on, not raw. CHIME truth range is bounded by the 82 ms singlebeam cutout
# (+-1.5 -> +-29 ms sweep at 400-800 MHz); DSA sweep is unbound at these ranges.
# Search windows are uniform across estimators (uniform-battery constraint).
INSTRUMENTS = {
    "dsa": dict(f_lo_ghz=1.31125, f_hi_ghz=1.49875, nchan=256, dt_ms=0.16384, ntime=1024,
                window=5.0, truth_dm=2.5),
    "chime": dict(f_lo_ghz=0.40019, f_hi_ghz=0.80019, nchan=256, dt_ms=0.16384, ntime=512,
                  window=4.0, truth_dm=1.5),
}


def standard_bright_case(instrument, seed=0, dm_ref=500.0):
    """Adapter contract fixture (Phase 1): S/N 80, 0.5 ms, dDM=+0.7, unscattered.

    Returns (waterfall, freq_ghz, dt_ms, truth) with truth carrying dm_ref,
    dm_true = dm_ref + 0.7, and the instrument's uniform search window.
    """
    geom = INSTRUMENTS[instrument]
    rng = np.random.default_rng(seed)
    freq_ghz = np.linspace(geom["f_lo_ghz"], geom["f_hi_ghz"], geom["nchan"])
    noise = rng.normal(size=(geom["nchan"], geom["ntime"])).astype(np.float32)
    spec = InjectionSpec(dm_offset=0.7, snr=80.0, width_ms=0.5, tau_1ghz_ms=0.0)
    wf, truth = inject_pulse(noise, freq_ghz, geom["dt_ms"], spec, rng)
    truth.update(dm_ref=float(dm_ref), dm_true=float(dm_ref) + spec.dm_offset,
                 window=geom["window"])
    return wf, freq_ghz, geom["dt_ms"], truth


@dataclass
class InjectionSpec:
    """Truth parameters for one injected burst."""

    dm_offset: float  # residual DM [pc/cm^3] applied on top of the noise frame
    snr: float  # band-summed profile peak S/N (MAD-normalized)
    width_ms: float  # intrinsic Gaussian sigma [ms]
    tau_1ghz_ms: float  # scattering time at 1 GHz [ms]; tau(nu) = tau_1ghz * nu^-4
    t0_frac: float = 0.5  # burst center as a fraction of the time window
    gamma: float = 0.0  # power-law spectral index of per-channel amplitude
    components: int = 1  # temporal components (same DM, separated by sep_ms)
    sep_ms: float = 3.0
    amp_ratio: float = 0.5  # amplitude of component k = amp_ratio**k


def disperse_waterfall(wf, freq_ghz, dm, dt_ms, ref="top"):
    """Apply the dispersive delay of ``dm`` to a (n_freq, n_time) waterfall.

    Delay is referenced to the top of the band (zero delay at nu_ref) and
    edges are zero-filled, mirroring chime_dm._dedisperse (which this calls
    with the opposite sign so the two stay inverses by construction).
    """
    freqs_mhz = np.asarray(freq_ghz, float) * 1e3
    nu_ref = freqs_mhz.max() if ref == "top" else float(ref) * 1e3
    return _dedisperse(np.asarray(wf, float), freqs_mhz, dt_ms * 1e-3, -dm, nu_ref)


def inject_pulse(noise, freq_ghz, dt_ms, spec, rng=None):
    """Inject a dispersed EMG burst into ``noise`` (n_freq, n_time).

    Returns ``(waterfall, truth)`` where truth carries the spec fields plus
    the realized scale, t0, and reference frequency. ``rng`` is accepted for
    API stability with matrix runners (noise realizations are drawn upstream
    by make_noise_from_offpulse); the injection itself is deterministic.
    """
    noise = np.asarray(noise, float)
    freq_ghz = np.asarray(freq_ghz, float)
    nf, nt = noise.shape
    t = np.arange(nt) * dt_ms
    t0_ms = spec.t0_frac * nt * dt_ms
    nu_ref = float(freq_ghz.max())
    amp_nu = (freq_ghz / nu_ref) ** spec.gamma

    clean = np.zeros((nf, nt))
    for k in range(spec.components):
        a_k = spec.amp_ratio**k
        t0_k = t0_ms + k * spec.sep_ms
        for j in range(nf):
            tau_nu = max(spec.tau_1ghz_ms * freq_ghz[j] ** -4.0, 1e-6)
            clean[j] += a_k * amp_nu[j] * exgauss(t, t0_k, spec.width_ms, tau_nu, 1.0, 0.0)
    clean = disperse_waterfall(clean, freq_ghz, spec.dm_offset, dt_ms)

    prof_noise = noise.sum(axis=0)
    med = np.median(prof_noise)
    sigma = 1.4826 * np.median(np.abs(prof_noise - med)) + 1e-12
    peak = clean.sum(axis=0).max()
    scale = spec.snr * sigma / (peak + 1e-30)

    truth = {
        **asdict(spec),
        "t0_ms": float(t0_ms),
        "scale": float(scale),
        "nu_ref_ghz": nu_ref,
    }
    return (noise + scale * clean).astype(np.float32), truth


def make_noise_from_offpulse(wf, on_frac, n_time, rng):
    """Bootstrap-resample off-pulse time columns of a real product.

    Whole columns are drawn with replacement from outside the ``on_frac``
    window, preserving per-channel means/variances and cross-channel structure
    (dead channels, bandpass residuals). Temporal correlation is broken —
    acceptable here because the recovered quantity is the injected dispersive
    sweep, not noise timescales.
    """
    wf = np.asarray(wf, float)
    nt = wf.shape[1]
    lo, hi = int(on_frac[0] * nt), int(on_frac[1] * nt)
    cols = np.r_[0:lo, hi:nt]
    return wf[:, rng.choice(cols, size=int(n_time), replace=True)]
