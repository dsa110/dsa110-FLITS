"""Downsampling sweep: dedisperse each burst ONCE (in parallel across cores), then try many
(TDS, N_SB) binnings in-memory. Tests whether coarser downsampling rescues the faint bursts into at
least a wide-sigma DM / exclusion, or whether the scattering bias persists. See audit P5.
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
FAINT = [
    "zach",
    "whitney",
    "oran",
    "isha",
    "wilhelm",
    "phineas",
    "freya",
    "johndoeII",
    "hamilton",
    "mahi",
    "chromatica",
    "casey",
]  # all 12, uniform config below
CONFIGS = [(32, 6)]  # pre-registered: coarsest binning the bright control still resolves cleanly
MIN_SNR, MIN_GOOD = 4.0, 3


def _ds(wf, k):
    nt = (wf.shape[1] // k) * k
    return wf[:, :nt].reshape(wf.shape[0], nt // k, k).mean(2)


def _regress(I0, fw, dt, n_sb):
    order = np.argsort(fw)
    wf, freqs = I0[order], fw[order]
    nu_ref = float(freqs.max())
    prof = (wf - np.median(wf, 1, keepdims=True)).sum(0)
    pk = int(np.argmax(np.convolve(prof, np.ones(max(int(1.5e-3 / dt), 1)), "same")))
    lo, hi = max(pk - int(15e-3 / dt), 0), min(pk + int(45e-3 / dt), wf.shape[1])
    wf = wf[:, lo:hi]
    edges = np.linspace(0, wf.shape[0], n_sb + 1, dtype=int)
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
    if n < MIN_GOOD:
        return {"n": n, "dm_off": None, "sigma": None}
    x = K_DM * (1.0 / np.array(nu) ** 2 - 1.0 / nu_ref**2)
    y, w = np.array(t0), 1.0 / np.array(err) ** 2
    X = np.vstack([x, np.ones_like(x)]).T
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ ((X * w[:, None]).T @ y)
    chi2 = float(np.sum(w * (y - X @ beta) ** 2) / max(n - 2, 1))
    return {
        "n": n,
        "dm_off": float(beta[0]),
        "sigma": float(np.sqrt(abs(cov[0, 0]) * max(chi2, 1.0))),
    }


def work(name):
    bursts = {b["name"]: b for b in json.load(open(ROOT + "/scripts/burst_inputs.json"))["bursts"]}
    m = bursts[name]
    dm = float(m["dm"])
    bb = BBData.from_file(f"{SB}/{name.lower()}/singlebeam_{m['chime_id']}.h5")
    dt0 = float(bb.attrs["delta_time"])
    freq = np.asarray(bb.index_map["freq"]["centre"], float)
    bbdd = coherent_dedisp(bb, dm)  # the ONE expensive call per burst
    inten = np.nan_to_num(np.abs(bbdd[:, 0, :]) ** 2 + np.abs(bbdd[:, 1, :]) ** 2)
    csd = inten.std(1)
    med = np.median(csd[csd > 0]) if np.any(csd > 0) else 0.0
    good = np.isfinite(csd) & (csd > 0.2 * med) & (csd < 8.0 * med)
    base, fw = inten[good], freq[good]
    rows = []
    for tds, n_sb in CONFIGS:
        r = _regress(_ds(base, tds), fw, dt0 * tds, n_sb)
        ddm = None if r["dm_off"] is None else round(r["dm_off"], 2)
        rows.append(
            (name, dm, tds, n_sb, r["n"], ddm, None if r["sigma"] is None else round(r["sigma"], 2))
        )
    return rows


if __name__ == "__main__":
    with Pool(6) as p:  # cap to bound peak memory (each baseband ~1.5 GB)
        allrows = p.map(work, FAINT)
    print(f"\n{'burst':11s} {'DSA':>8} {'TDS':>4} {'NSB':>4} {'ngood':>6} {'dDM':>7} {'sigma':>7}")
    for rows in allrows:
        for name, dm, tds, n_sb, n, ddm, sig in rows:
            print(f"{name:11s} {dm:8.2f} {tds:4d} {n_sb:4d} {n:6d} {ddm!s:>7} {sig!s:>7}")
        print()
