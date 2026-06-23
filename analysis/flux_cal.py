#!/usr/bin/env python
"""Radiometer flux calibration: per-channel S/N -> Jy for FLITS dynamic spectra.

FLITS data is per-channel z-scored S/N (scattering/scat_analysis/pipeline/io.py:131-146:
_bandpass_correct divides each channel by its own off-pulse mean/std and the loader keeps
"units as S/N"). Physical flux density is therefore one radiometer multiply away:

    S_nu(nu,t) [Jy] = (S/N)(nu,t) * sigma_S(nu),
    sigma_S(nu) = SEFD(nu) / (sqrt(n_pol * dnu * dt) * G(theta,phi,nu)),

with SEFD = 2 k_B T_sys / A_eff and G the beam gain (boresight=1). Dividing by the
per-channel off-pulse noise also cancels the bandpass gain, so the S/N spectrum (and hence
this calibration) is free of the band-edge rolloff that rails the fitted gamma_D.

The band fluence integral (Jy*ms*Hz) returned here feeds
analysis/calculate_burst_energies.band_energy_erg with flux_scale=1 -- the per-channel scale
is already folded in. See docs/rse/specs/plan-radiometer-flux-cal.md.
"""

from __future__ import annotations

import numpy as np


def radiometer_sigma_jy(sefd_jy, n_pol, dnu_hz, dt_s, g):
    """Per-sample radiometer noise [Jy]: SEFD / (sqrt(n_pol*dnu*dt) * G).

    sefd_jy: system-equivalent flux density [Jy]; n_pol: summed polarisations;
    dnu_hz: channel bandwidth [Hz]; dt_s: sample time [s]; g: beam gain (boresight=1).
    """
    return sefd_jy / (np.sqrt(n_pol * dnu_hz * dt_s) * g)


def calibrated_band_integral_jy_ms_hz(sn_integrated, sigma_jy, freq_hz, dt_ms):
    """int_band [ sigma_S(nu) * dt_ms * sum_onpulse(S/N)(nu) ] dnu   [Jy*ms*Hz].

    sn_integrated: per-channel sum over the on-pulse window of (data/noise_std) [dimensionless];
    sigma_jy: per-channel sigma_S [Jy]; freq_hz: ascending channel centres [Hz]; dt_ms: sample [ms].
    """
    chan_fluence_jy_ms = sigma_jy * dt_ms * sn_integrated  # [Jy*ms] per channel
    return float(np.trapezoid(chan_fluence_jy_ms, freq_hz))  # [Jy*ms*Hz]


def sn_spectrum_from_npy(inpath, telescope, f_factor=1, t_factor=1, onpulse_thresh=3.0):
    """Per-channel on-pulse S/N spectrum from a burst .npy via the FLITS loader.

    Returns (freq_hz ascending, sn_integrated [per channel, sum over on-pulse of data/noise_std],
    dt_ms, dnu_hz). BurstDataset z-scores each channel (io.py:131-146), so model.data is already
    S/N; dividing by the full-window per-channel noise makes the unit explicit and robust to the
    downsample rescaling.
    """
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scattering"))  # io.py uses package-relative imports (..burstfit)
    from scat_analysis.config_utils import load_telescope_block
    from scat_analysis.pipeline.io import BurstDataset

    tel = load_telescope_block(str(repo / "scattering" / "configs" / "telescopes.yaml"), telescope)
    ds = BurstDataset(
        inpath,
        inpath,
        telescope=tel,
        f_factor=f_factor,
        t_factor=t_factor,
        onpulse_crop=True,
        onpulse_thresh=onpulse_thresh,
    )
    m = ds.model
    noise = np.clip(m.noise_std, 1e-9, None)  # per-channel full-window noise (n_freq,)
    sn = m.data / noise[:, None]  # (n_freq, n_time_onpulse) signal-to-noise
    sn_integrated = np.nansum(sn, axis=1)  # sum over the cropped on-pulse window
    return m.freq * 1e9, sn_integrated, ds.dt_ms, ds.df_MHz * 1e6


def dsa_sigma_jy(freq_hz, dnu_hz, sefd_jy, dt_s, theta_deg, phi_deg, beam_gain_fn):
    """Per-channel radiometer noise sigma_S(nu) [Jy] for DSA (n_pol=2) at a beam offset.

    beam_gain_fn(theta_deg, phi_deg, freq_ghz) -> normalized power-beam gain (boresight=1);
    dnu_hz is the channel bandwidth (downsampled ds.df_MHz*1e6), dt_s the sample time [s].
    """
    g = np.array([beam_gain_fn(theta_deg, phi_deg, f / 1e9) for f in freq_hz])
    return radiometer_sigma_jy(sefd_jy, 2, dnu_hz, dt_s, g)


def dsa_beam_offset(dec_src, dec_pointing):
    """(theta_deg, phi_deg) of the source from the DSA primary-beam boresight.

    DSA-110 is a transit array: at the burst's meridian transit (HA~0) the boresight sits at the
    source RA, so the angular separation is exactly the declination difference; phi=0 (meridian).
    HA refinement (small E-W term) is out of scope -- see the plan.
    """
    return abs(dec_src - dec_pointing), 0.0


def burst_epoch_position(nick):
    """(mjd, ra_deg, dec_deg) for a burst nickname from configs/bursts.yaml."""
    from pathlib import Path

    import yaml

    b = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "bursts.yaml").read_text()
    )
    e = b["bursts"][nick]
    return float(e["mjd"]), float(e["ra_deg"]), float(e["dec_deg"])


def _csv_lookup(filename, key, key_col="burst", val_col=None):
    import csv
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "analysis" / "burst_energies" / filename
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- run the Phase 4 acquisition")
    for row in csv.DictReader(p.open()):
        if row[key_col] == key:
            return row[val_col]
    raise KeyError(f"{key} not in {p}")


def load_dsa_sefd(nick):
    """Measured DSA SEFD [Jy] nearest the burst epoch, from dsa_sefd.csv (Phase 4)."""
    return float(_csv_lookup("dsa_sefd.csv", nick, val_col="sefd_jy"))


def dsa_pointing_dec(nick):
    """DSA primary-beam pointing declination [deg] for a burst, from dsa_pointing.csv (Phase 4)."""
    return float(_csv_lookup("dsa_pointing.csv", nick, val_col="pointing_dec_deg"))


# DSA-110 formed its FRB detection beam by coherently combining 48 antennas in 2022
# (Law et al. 2024, arXiv:2307.03344). The dsa110-rt SEFD store is PER-BASELINE (~per-element;
# estimate_sefd.py:compute_sefd_per_baseline), so the coherent-beam SEFD is SEFD_element / N_ant.
# Cross-check: this beam SEFD (~167 Jy) reproduces the catalog fluences (16.2/26.2/13.2 Jy*ms for
# zach/whitney/oran) to within the ~2x boxcar-vs-full-integral estimator difference; the empirical
# matched-filter anchor is ~234 Jy (=> effective N_eff~34, ~70% beamforming efficiency).
DSA_N_ANT = 48


def load_dsa_sefd_beam(nick):
    """Coherent-beam SEFD [Jy] = per-element dashboard SEFD / N_ant (the detection beam)."""
    return load_dsa_sefd(nick) / DSA_N_ANT


def _dsa_burst_config(nick):
    """(npy_path, f_factor, t_factor) from configs/batch/dsa/<nick>_dsa.yaml (the fit's own binning)."""
    from pathlib import Path

    import yaml

    cfg = Path(__file__).resolve().parents[1] / "configs" / "batch" / "dsa" / f"{nick}_dsa.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"{cfg} missing -- no DSA batch config for {nick}")
    c = yaml.safe_load(cfg.read_text())
    return (cfg.parent / c["path"]).resolve(), int(c.get("f_factor", 1)), int(c.get("t_factor", 1))


def dsa_band_fluence_jy_ms_hz(nick):
    """Calibrated DSA-band fluence integral [Jy*ms*Hz] for a burst (Phases 2-3 composed).

    Loads the burst .npy via its batch config (same f/t binning the joint fit used), turns the
    per-channel on-pulse S/N spectrum into Jy with sigma_S(nu)=SEFD_beam/(sqrt(2*dnu*dt)*G(theta,nu)),
    and integrates over the DSA band. SEFD_beam is the COHERENT-beam SEFD (per-element / N_ant);
    using the per-element SEFD directly over-estimates the fluence by ~N_ant. Raises
    FileNotFoundError if the .npy is not staged locally (data/dsa/ is external -- iacobus:burst_npys).
    """
    from analysis.dsa_beam import beam_gain

    npy, f_factor, t_factor = _dsa_burst_config(nick)
    if not npy.exists():
        raise FileNotFoundError(
            f"{npy} not materialized -- stage the DSA .npy (iacobus:burst_npys) under data/dsa/"
        )
    freq_hz, sn_int, dt_ms, dnu_hz = sn_spectrum_from_npy(npy, "dsa", f_factor, t_factor)
    _mjd, _ra, dec_src = burst_epoch_position(nick)
    theta, phi = dsa_beam_offset(dec_src, dsa_pointing_dec(nick))
    sigma_jy = dsa_sigma_jy(
        freq_hz, dnu_hz, load_dsa_sefd_beam(nick), dt_ms / 1e3, theta, phi, beam_gain
    )
    return calibrated_band_integral_jy_ms_hz(sn_int, sigma_jy, freq_hz, dt_ms)


def _band_burst_config(nick, band):
    """(npy_path, f_factor, t_factor, telescope) from the band's batch config (band 'C'|'D')."""
    from pathlib import Path

    import yaml

    tel = {"C": "chime", "D": "dsa"}[band]
    cfg = Path(__file__).resolve().parents[1] / "configs" / "batch" / tel / f"{nick}_{tel}.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"{cfg} missing -- no {tel} batch config for {nick}")
    c = yaml.safe_load(cfg.read_text())
    return (
        (cfg.parent / c["path"]).resolve(),
        int(c.get("f_factor", 1)),
        int(c.get("t_factor", 1)),
        tel,
    )


def joint_c0_gamma(nick, band):
    """(c0, gamma, c0_frac_err) for a band from the joint fit JSON (band 'C'|'D').

    c0_frac_err is the mean of the asymmetric 16/84 half-widths over the median -- the statistical
    amplitude (hence fluence) uncertainty that propagates into the energy.
    """
    import json
    from pathlib import Path

    p = (
        Path(__file__).resolve().parents[1]
        / "analysis"
        / "scattering-refit-2026-06"
        / "joint_json"
        / f"{nick}_joint_fit.json"
    )
    if not p.exists():
        raise FileNotFoundError(f"{p} missing -- no joint fit for {nick}")
    pct = json.loads(p.read_text())["percentiles"]
    e = pct[f"c0_{band}"]
    c0 = e["median"]
    frac = 0.5 * (e["err_plus"] + e["err_minus"]) / c0 if c0 else float("nan")
    return c0, pct[f"gamma_{band}"]["median"], frac


def _band_noise_grid(npy, telescope, f_factor, t_factor):
    """(freq_hz, noise_std, dt_ms, dnu_hz, valid) loaded exactly as the joint fit did (crop, pad 0.5).

    Matching onpulse_pad_factor=0.5 is essential: c0 is in the joint fit's bandpass-normalized data
    units, so the per-channel noise_std used to convert it to S/N must come from that same window.
    `valid` is the per-channel mask the fit actually used: RFI-masked channels have noise_std ~ 0 and
    must be excluded (else c0/noise_std explodes), so the integral runs over the SAME channels as the fit.
    """
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scattering"))
    from scat_analysis.config_utils import load_telescope_block
    from scat_analysis.pipeline.io import BurstDataset

    tel = load_telescope_block(str(repo / "scattering" / "configs" / "telescopes.yaml"), telescope)
    ds = BurstDataset(
        npy,
        npy,
        telescope=tel,
        f_factor=f_factor,
        t_factor=t_factor,
        onpulse_crop=True,
        onpulse_pad_factor=0.5,
    )
    m = ds.model
    ns = m.noise_std
    if m.valid is not None:
        valid = m.valid if m.valid.ndim == 1 else np.any(m.valid, axis=1)
    else:
        valid = np.isfinite(ns) & (ns > 0)
    valid = valid & np.isfinite(ns) & (ns > 1e-6 * np.nanmedian(ns[valid]))  # drop masked channels
    return m.freq * 1e9, ns, ds.dt_ms, ds.df_MHz * 1e6, valid


def joint_band_fluence_jy_ms_hz(nick, band):
    """Model-based calibrated band fluence [Jy*ms*Hz] from the joint fit (band 'C'|'D').

        F = int sigma_S(nu) * [ c0 (nu/ref)^gamma / noise_std(nu) ] dnu.

    The bracket is int(S/N) dt per channel: int(model)dt = c0 for the area-conserving scattering
    kernel (verified -- tail-complete, no on-pulse window to tune), and dividing by the joint fit's
    per-channel noise_std converts its bandpass-normalized c0 to S/N units (noise_std ~ 0.04, not 1).
    sigma_S is the DSA measured-beam or CHIME documented-cylinder-beam radiometer noise (n_pol=2).
    """
    npy, ff, tf, tel = _band_burst_config(nick, band)
    if not npy.exists():
        raise FileNotFoundError(f"{npy} not materialized -- stage the {tel} .npy under data/{tel}/")
    c0, gamma, _frac = joint_c0_gamma(nick, band)
    freq_hz, noise_std, dt_ms, dnu_hz, valid = _band_noise_grid(npy, tel, ff, tf)
    ref = np.median(freq_hz)  # the fit references c0 to the median of the FULL channel grid
    fz, ns = freq_hz[valid], noise_std[valid]  # integrate over the channels the fit used
    amp_sn = c0 * (fz / ref) ** gamma / ns  # int(S/N) dt per valid channel [S/N*ms]
    if band == "D":
        from analysis.dsa_beam import beam_gain

        _mjd, _ra, dec = burst_epoch_position(nick)
        theta, phi = dsa_beam_offset(dec, dsa_pointing_dec(nick))
        sigma_jy = dsa_sigma_jy(
            fz, dnu_hz, load_dsa_sefd_beam(nick), dt_ms / 1e3, theta, phi, beam_gain
        )
    else:  # CHIME baseband: beam formed at the source -> G ~ 1
        from analysis.chime_beam import chime_sigma_jy, load_chime_sefd

        sigma_jy = chime_sigma_jy(fz, dnu_hz, load_chime_sefd(), dt_ms / 1e3, g=1.0)
    return float(np.trapezoid(sigma_jy * amp_sn, fz))


def _check() -> None:
    # 1. radiometer noise vs analytic: 2000/sqrt(2*1e6*1e-3) == sqrt(2000); G=0.5 doubles it
    s = radiometer_sigma_jy(2000.0, 2, 1e6, 1e-3, 1.0)
    assert abs(s - np.sqrt(2000.0)) < 1e-9, s
    assert abs(radiometer_sigma_jy(2000.0, 2, 1e6, 1e-3, 0.5) - 2.0 * s) < 1e-9
    # 2. flat-band integral oracle: trapz(const) == const*(nu2-nu1)
    freq_hz = np.linspace(1.311e9, 1.499e9, 64)
    i_band = calibrated_band_integral_jy_ms_hz(
        np.full(64, 3.0), np.full(64, 5.0), freq_hz, 0.131072
    )
    oracle = 3.0 * 5.0 * 0.131072 * (freq_hz[-1] - freq_hz[0])
    assert abs(i_band - oracle) / oracle < 1e-9, (i_band, oracle)
    print("self-check OK: radiometer noise + flat-band integral match analytic oracles")


if __name__ == "__main__":
    _check()
