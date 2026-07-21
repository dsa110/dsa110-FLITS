"""Authoritative CHIME-side DM: arrival regression on coherent-once data, UNIFORM config
(TDS=32, N_SB=6 -- the coarsest binning the bright control resolves cleanly; downsampling rescues
the faint bursts that finer binning starved of sub-bands). Parallel dedisp-once per burst, full
precision json + per-burst figure. See .agents/audit-chime-side-dm.md P5.
"""

import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, "/data/research/astrophysics/frbs/chime-dsa-codetections/scripts")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from baseband_analysis.core.bbdata import BBData
from baseband_analysis.core.dedispersion import coherent_dedisp
from chime_dm import K_DM, _fit_subband_arrival

ROOT = "/data/research/astrophysics/frbs/chime-dsa-codetections"
SB = "/data/Faber2026/data/chime-frb"
DIAG = ROOT + "/diagnostics/chime_dm_final"
os.makedirs(DIAG, exist_ok=True)
TDS, N_SB, MIN_SNR, MIN_GOOD = 32, 6, 4.0, 3
ALL = [
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
]


def _ds(wf, k):
    nt = (wf.shape[1] // k) * k
    return wf[:, :nt].reshape(wf.shape[0], nt // k, k).mean(2)


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
    sub = {"nu": nu, "t0": t0, "err": err, "nu_ref": nu_ref, "wf": wf, "freqs": freqs, "dt": dt}
    if n < MIN_GOOD:
        return {
            "n_good": n,
            "dm_offset": None,
            "sigma": None,
            "constrains": False,
            "reason": f"{n} sub-bands < {MIN_GOOD}",
            "sub": sub,
            "beta": None,
        }
    x = K_DM * (1.0 / np.array(nu) ** 2 - 1.0 / nu_ref**2)
    y, w = np.array(t0), 1.0 / np.array(err) ** 2
    X = np.vstack([x, np.ones_like(x)]).T
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ ((X * w[:, None]).T @ y)
    chi2 = float(np.sum(w * (y - X @ beta) ** 2) / max(n - 2, 1))
    sigma = float(np.sqrt(abs(cov[0, 0]) * max(chi2, 1.0)))
    return {
        "n_good": n,
        "dm_offset": float(beta[0]),
        "sigma": sigma,
        "constrains": bool(sigma <= 1.0),
        "chi2_red": chi2,
        "beta": beta.tolist(),
        "sub": sub,
    }


def _figure(name, dm_dsa, r, dm):
    sub = r["sub"]
    wf, fro, dt = sub["wf"], sub["freqs"], sub["dt"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    nf = (wf.shape[0] // 4) * 4
    m = wf[:nf].reshape(nf // 4, 4, wf.shape[1]).mean(1)
    mu, sd = m.mean(1, keepdims=True), m.std(1, keepdims=True) + 1e-9
    ax[0].imshow(
        (m - mu) / sd,
        aspect="auto",
        origin="lower",
        extent=[0, wf.shape[1] * dt * 1e3, fro.min(), fro.max()],
        vmin=-0.5,
        vmax=5,
        cmap="magma",
    )
    ax[0].set(
        title=f"{name}: coherent @ DSA={dm_dsa:g} ({r['n_good']} sub-bands, TDS={TDS} NSB={N_SB})",
        xlabel="t (ms)",
        ylabel="freq MHz",
    )
    if sub["nu"]:
        nu = np.array(sub["nu"])
        x = K_DM * (1.0 / nu**2 - 1.0 / sub["nu_ref"] ** 2)
        ax[1].errorbar(x, np.array(sub["t0"]) * 1e3, yerr=np.array(sub["err"]) * 1e3, fmt="o", ms=5)
        if r["beta"]:
            xs = np.linspace(x.min(), x.max(), 50)
            ax[1].plot(xs, (r["beta"][0] * xs + r["beta"][1]) * 1e3, "r-")
    s = "None" if r["sigma"] is None else f"{r['sigma']:.3f}"
    dms = "None" if dm is None else f"{dm:.2f}"
    ax[1].set(
        title=f"DM={dms} +/- {s} (dDM={'' if dm is None else round(dm - dm_dsa, 2)}) constrains={r['constrains']}",
        xlabel="K_DM (nu^-2 - nu_ref^-2) [s per pc/cm^3]",
        ylabel="t0 (ms)",
    )
    fig.tight_layout()
    fig.savefig(f"{DIAG}/{name}_dm.png", dpi=110)
    plt.close(fig)


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
    r = _regress(_ds(inten[good], TDS), freq[good], dt0 * TDS)
    dm = None if r["dm_offset"] is None else dm_dsa + r["dm_offset"]
    _figure(name, dm_dsa, r, dm)
    return {
        "name": name,
        "chime_id": m["chime_id"],
        "dm_dsa": dm_dsa,
        "dm": dm,
        "dm_err": r["sigma"],
        "dm_offset": r["dm_offset"],
        "n_good_subbands": r["n_good"],
        "constrains_dm": r["constrains"],
        "reason": r["reason"] if "reason" in r else "ok",
        "chi2_red": r.get("chi2_red"),
        "n_chan_ok": int(good.sum()),
        "chime_ra_deg": float(np.asarray(bb["tiedbeam_locations"]["ra"]).ravel()[0]),
        "chime_dec_deg": float(np.asarray(bb["tiedbeam_locations"]["dec"]).ravel()[0]),
    }


if __name__ == "__main__":
    with Pool(6) as p:
        out = p.map(work, ALL)
    json.dump(out, open(ROOT + "/results/chime_dm_final.json", "w"), indent=2)
    man = [
        {
            "path": f"{r['name']}_dm.png",
            "expectation": f"{r['name']}: coherent@DSA={r['dm_dsa']:g}, TDS={TDS}/NSB={N_SB}; "
            f"DM={'None' if r['dm'] is None else round(r['dm'], 2)} ({r['n_good_subbands']} sub-bands) "
            f"constrains={r['constrains_dm']} -- constrained land near DSA, non-det <3 sub-bands",
        }
        for r in out
    ]
    json.dump({"figures": man}, open(f"{DIAG}/figures.manifest.json", "w"), indent=2)
    nC = sum(r["constrains_dm"] for r in out)
    print(f"\nwrote chime_dm_final.json: {nC}/12 constrain")
    for r in sorted(out, key=lambda r: not r["constrains_dm"]):
        dms = "None" if r["dm"] is None else f"{r['dm']:.2f}"
        e = "None" if r["dm_err"] is None else f"{r['dm_err']:.3f}"
        print(
            f"  {r['name']:11s} DSA={r['dm_dsa']:8.2f} DM={dms:>8} +/- {e:>6} "
            f"nsub={r['n_good_subbands']} constrains={r['constrains_dm']}"
        )
