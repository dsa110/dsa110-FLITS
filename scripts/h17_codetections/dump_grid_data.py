"""Dump compact per-burst plotting data for the 12-panel manuscript DM grid.

Re-runs coherent_dedisp + arrival regression (UNIFORM TDS=32/N_SB=6, the authoritative config from
extract_final_parallel.py) and saves, per burst: a freq-binned cropped waterfall (for display) + the
sub-band arrival fits (nu, t0, err, beta). The heavy dedispersion runs once here so the in-repo figure
script can re-render the SVG without docker/baseband. See .agents/audit-chime-side-dm.md P5.
"""

import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, "/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
import numpy as np
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp
from chime_dm import K_DM, _fit_subband_arrival

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
SB = "/data/Faber2026/data/chime-frb"
TDS, N_SB, MIN_SNR, MIN_GOOD, NROW = 32, 6, 4.0, 3, 48
ALL = [
    "zach",
    "casey",
    "freya",
    "hamilton",
    "chromatica",
    "isha",
    "wilhelm",
    "phineas",
    "whitney",
    "oran",
    "johndoeII",
    "mahi",
]  # constrained first (8), then non-detections (4) -- the grid reading order


def _ds(wf, k):
    nt = (wf.shape[1] // k) * k
    return wf[:, :nt].reshape(wf.shape[0], nt // k, k).mean(2)


def _binrows(wf, nrow):
    n = (wf.shape[0] // nrow) * nrow
    if n < nrow:
        return wf
    return wf[:n].reshape(nrow, n // nrow, wf.shape[1]).mean(1)


def _regress(I0, fw, dt):
    order = np.argsort(fw)
    wf, freqs = I0[order], fw[order]
    nu_ref = float(freqs.max())
    prof = (wf - np.median(wf, 1, keepdims=True)).sum(0)
    pk = int(np.argmax(np.convolve(prof, np.ones(max(int(1.5e-3 / dt), 1)), "same")))
    lo, hi = max(pk - int(15e-3 / dt), 0), min(pk + int(45e-3 / dt), wf.shape[1])
    wf = wf[:, lo:hi]
    edges = np.linspace(0, wf.shape[0], N_SB + 1, dtype=int)
    nu, t0, err = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 1:
            continue
        fit = _fit_subband_arrival(np.nansum(np.nan_to_num(wf[a:b]), axis=0), dt, min_snr=MIN_SNR)
        if fit:
            nu.append(float(freqs[a:b].mean()))
            t0.append(fit[0])
            err.append(fit[1])
    n = len(nu)
    out = {
        "n_good": n,
        "nu": nu,
        "t0": t0,
        "err": err,
        "nu_ref": nu_ref,
        "beta": None,
        "dm_offset": None,
        "sigma": None,
    }
    if n >= MIN_GOOD:
        x = K_DM * (1.0 / np.array(nu) ** 2 - 1.0 / nu_ref**2)
        y, w = np.array(t0), 1.0 / np.array(err) ** 2
        X = np.vstack([x, np.ones_like(x)]).T
        cov = np.linalg.inv(X.T @ (X * w[:, None]))
        beta = cov @ ((X * w[:, None]).T @ y)
        chi2 = float(np.sum(w * (y - X @ beta) ** 2) / max(n - 2, 1))
        out["beta"] = beta.tolist()
        out["dm_offset"] = float(beta[0])
        out["sigma"] = float(np.sqrt(abs(cov[0, 0]) * max(chi2, 1.0)))
    return (
        out,
        _binrows(wf, NROW),
        float(wf.shape[1] * dt * 1e3),
        float(freqs.min()),
        float(freqs.max()),
    )


def work(name):
    m = {b["name"]: b for b in json.load(open(ROOT + "/scripts/burst_inputs.json"))["bursts"]}[name]
    dm_dsa = float(m["dm"])
    bb = BBData.from_file(f"{SB}/{name.lower()}/singlebeam_{m['chime_id']}.h5")
    dt0 = float(bb.attrs["delta_time"])
    freq = np.asarray(bb.index_map["freq"]["centre"], float)
    bbdd = coherent_dedisp(bb, dm_dsa)
    inten = np.nan_to_num(np.abs(bbdd[:, 0, :]) ** 2 + np.abs(bbdd[:, 1, :]) ** 2)
    csd = inten.std(1)
    med = np.median(csd[csd > 0]) if np.any(csd > 0) else 0.0
    good = np.isfinite(csd) & (csd > 0.2 * med) & (csd < 8.0 * med)
    r, wf, tmax_ms, fmin, fmax = _regress(_ds(inten[good], TDS), freq[good], dt0 * TDS)
    dm = None if r["dm_offset"] is None else dm_dsa + r["dm_offset"]
    fit = {
        "name": name,
        "dm_dsa": dm_dsa,
        "dm": dm,
        "dm_offset": r["dm_offset"],
        "sigma": r["sigma"],
        "n_good": r["n_good"],
        "constrains": bool(r["sigma"] is not None and r["sigma"] <= 1.0),
        "nu": r["nu"],
        "t0": r["t0"],
        "err": r["err"],
        "nu_ref": r["nu_ref"],
        "beta": r["beta"],
        "extent": [0.0, tmax_ms, fmin, fmax],
    }
    return fit, wf.astype(np.float32)


if __name__ == "__main__":
    with Pool(6) as p:
        out = p.map(work, ALL)
    fits = [f for f, _ in out]
    json.dump(fits, open(ROOT + "/results/chime_dm_grid_fits.json", "w"), indent=2)
    np.savez_compressed(
        ROOT + "/results/chime_dm_grid_waterfalls.npz",
        **{f["name"]: wf for f, wf in zip(fits, [w for _, w in out])},
    )
    nC = sum(f["constrains"] for f in fits)
    print(
        f"dumped {len(fits)} bursts ({nC} constrain) -> chime_dm_grid_fits.json + _waterfalls.npz"
    )
    for f in fits:
        print(
            f"  {f['name']:11s} nsub={f['n_good']} constrains={f['constrains']} "
            f"wf={'ok'} dDM={None if f['dm_offset'] is None else round(f['dm_offset'], 2)}"
        )
