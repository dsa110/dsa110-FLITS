#!/usr/bin/env python3
"""Dump jointmodel NPZ + render the pair figure for a sandbox refit.

Mirrors dump_jointmodel.py (OLS gain recovery) and plot_jointmodel_pair.py
(codetection-style data/model/resid panels), but rebuilds bands through
refit_runner.prepare so per-band onpulse pads stay consistent with the fit.

Usage: python3 dump_plot.py <burst>
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent
REPO = Path(os.environ.get("FABER2026_PIPELINE", Path(__file__).resolve().parents[4]))
RUNS = Path(os.environ.get("FABER2026_REFIT_RUNS", RUN_DIR))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scattering"))
sys.path.insert(0, str(SCRIPT_DIR))

import refit_runner as R  # noqa: E402
from scat_analysis.burstfit import FRBParams  # noqa: E402
from flits.batch.codetection_data import (  # noqa: E402
    chime_toa_shift_ms,
    crop_bands_to_subburst_window,
    toa_offset_ms,
)
from flits.batch.codetection_plots import BandSpectrum, plot_codetection  # noqa: E402


def recover(model, params_list):
    Ks = np.stack([model(replace(p, c0=1.0, gamma=0.0), "M3") for p in params_list])
    d = np.asarray(model.data, float)
    sig = np.clip(np.asarray(model.noise_std, float).reshape(-1), 1e-9, None)
    M = np.einsum("nft,mft->fnm", Ks, Ks)
    b = np.einsum("nft,ft->fn", Ks, d)
    N = len(params_list)
    jit = 1e-9 * max(float(np.einsum("fnn->f", M).mean()), 1e-30)
    g = np.linalg.solve(M + jit * np.eye(N), b[..., None])[..., 0]
    mod = np.einsum("fn,nft->ft", g, Ks)
    valid = model.valid
    v = np.ones(d.shape[0], bool) if valid is None else np.asarray(valid).reshape(-1).astype(bool)
    r = ((d - mod) / sig[:, None])[v]
    r = r[np.isfinite(r)]
    chi2 = float(np.sum(r**2)) / max(int(r.size) - 7, 1)
    # lag-1 autocorr of the whitened residual PROFILE (freq-summed)
    rp = ((d - mod) / sig[:, None])[v].sum(axis=0)
    rp = (rp - rp.mean()) / max(rp.std(), 1e-12)
    lag1 = float(np.corrcoef(rp[:-1], rp[1:])[0, 1])
    return (
        dict(data=d, model=mod, freq=np.asarray(model.freq, float),
             time=np.asarray(model.time, float), noise=sig, valid=v),
        chi2,
        lag1,
        g,
    )


def band_params(p, X, n, tau, beta):
    ddm = float(p.get(f"delta_dm_{X}", 0.0))
    out = []
    for i in range(1, n + 1):
        out.append(FRBParams(c0=1.0, t0=float(p[f"t0_{X}{i}"]), gamma=0.0,
                             zeta=float(p[f"zeta_{X}{i}"]), tau_1ghz=tau,
                             beta=beta, delta_dm=ddm))
    return out


def make_band(D, label):
    return BandSpectrum(
        freq_mhz=D["freq"] * 1e3,
        time_ms=D["time"],
        data=D["data"],
        model=D["model"],
        sigma=D["noise"],
        label=label,
        channel_valid=D["valid"],
    )


def main():
    burst = sys.argv[1]
    s = R.SPEC[burst]
    n_C, n_D = len(s["C"]["comps"]), len(s["D"]["comps"])
    suffix = R.suffix_for(burst)
    out = RUNS / "data/joint"
    d = json.load(open(out / f"{burst}_joint_fit{suffix}.json"))
    p = {k: v["median"] for k, v in d["percentiles"].items()}
    tau, beta = p["tau_1ghz"], p["beta"]

    mC = R.prepare(burst, "chime", s)
    mD = R.prepare(burst, "dsa", s)
    C, chiC, lag1C, gC = recover(mC, band_params(p, "C", n_C, tau, beta))
    D, chiD, lag1D, gD = recover(mD, band_params(p, "D", n_D, tau, beta))

    np.savez_compressed(
        out / f"{burst}_jointmodel{suffix}.npz",
        dataC=C["data"], modelC=C["model"], freqC=C["freq"], timeC=C["time"],
        noiseC=C["noise"], validC=C["valid"],
        dataD=D["data"], modelD=D["model"], freqD=D["freq"], timeD=D["time"],
        noiseD=D["noise"], validD=D["valid"],
        alpha=d["alpha"]["median"], beta=beta, tau_1ghz=tau,
        chi2C=chiC, chi2D=chiD, nC=n_C, nD=n_D, burst=burst,
        gainC=gC, gainD=gD,
    )

    chime, dsa = make_band(C, "CHIME/FRB"), make_band(D, "DSA-110")
    offset = toa_offset_ms(burst)
    if offset is not None:
        shift_c = chime_toa_shift_ms(dsa, chime, offset)
        chime = BandSpectrum(freq_mhz=chime.freq_mhz, time_ms=chime.time_ms + shift_c,
                             data=chime.data, model=chime.model, sigma=chime.sigma,
                             label=chime.label, channel_valid=chime.channel_valid)
    # Per-burst plot pad: zach's trailing DSA sub-bursts (+2.1/+2.6 ms) fall
    # outside the default subburst window and were cut at the right edge.
    PLOT_PAD_MS = {"zach": 6.0, "wilhelm": 6.0}
    pad = PLOT_PAD_MS.get(burst)
    if pad is not None:
        bands = crop_bands_to_subburst_window([chime, dsa], center=True, pad_ms=pad)
    else:
        bands = crop_bands_to_subburst_window([chime, dsa], center=True)
    fig = plot_codetection(
        bands, columns=("data", "model", "resid"), show_model_on_data=False,
        per_band_scale=True, gap_label=False, figsize=(12.4, 4.9),
        band_labels=False, show_column_titles=False, per_band_marginals=True,
        title=None,
    )
    figdir = RUNS / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    stem = figdir / f"{burst}_jointmodel_pair"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")

    diag = dict(burst=burst, suffix=suffix, alpha=d["alpha"]["median"],
                tau_1ghz=tau, beta=beta, chi2C=chiC, chi2D=chiD,
                lag1C=lag1C, lag1D=lag1D,
                lnZ=d["log_evidence"], runtime_s=d.get("runtime_s"))
    (RUNS / f"{burst}_diag.json").write_text(json.dumps(diag, indent=2))
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
