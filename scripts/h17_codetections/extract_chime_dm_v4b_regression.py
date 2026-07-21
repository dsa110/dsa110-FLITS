"""P5 (simplified): independent CHIME DM by sub-band arrival regression on COHERENT-once data.

The expert fix is coherent + full-band (no zero-fill) + no biasing S/N-max seed. Here: one
coherent_dedisp at dm_dsa, tight crop, bin to N_SB sub-bands, fit each sub-band's scatter-deconvolved
arrival (chime_dm EMG), and weighted-linear-regress t0 vs K_DM(nu^-2 - nu_ref^-2). Slope = residual DM
(so DM = dm_dsa + slope, measured AROUND zero -- no +4 S/N-max seed), sigma = chi^2-inflated covariance.
Bright bursts -> tight sigma at DSA; faint -> few sub-bands / large scatter -> wide sigma / not
constrained (no fabricated value). dm enters only as the coherent reference, not as a fitted prior.

chime_dm.py vendored in scripts/. Run:
  DM_TARGETS=zach bin/baseband_analysis_python.sh scripts/extract_chime_dm_v4b_regression.py
"""

import json
import os
import sys
import traceback

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
DIAG = ROOT + "/diagnostics/chime_dm_v4b"
os.makedirs(DIAG, exist_ok=True)
TDS = int(os.environ.get("DM_TDS","16"))
N_SB = int(os.environ.get("DM_NSB", "8"))
MIN_SNR = 4.0
MIN_GOOD = int(os.environ.get("DM_MINGOOD","4"))
TARGETS = os.environ.get("DM_TARGETS", "zach").split(",")


def _downsample_t(wf, k):
    nt = (wf.shape[1] // k) * k
    return wf[:, :nt].reshape(wf.shape[0], nt // k, k).mean(2)


def _regress(wf, freqs, dt):
    """Sub-band arrival regression on coherent-once data. Returns dict with residual DM + sigma."""
    order = np.argsort(freqs)
    wf, freqs = wf[order], freqs[order]
    nu_ref = float(freqs.max())
    edges = np.linspace(0, wf.shape[0], N_SB + 1, dtype=int)
    nu, t0, err, snr = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 1:
            continue
        fit = _fit_subband_arrival(np.nansum(np.nan_to_num(wf[a:b]), axis=0), dt, min_snr=MIN_SNR)
        if fit is None:
            continue
        nu.append(float(freqs[a:b].mean()))
        t0.append(fit[0])
        err.append(fit[1])
        snr.append(fit[2])
    n = len(nu)
    sub = {"nu": nu, "t0": t0, "err": err, "snr": snr, "nu_ref": nu_ref}
    if n < MIN_GOOD:
        return {
            "dm_offset": None,
            "sigma": None,
            "n_good": n,
            "constrains": False,
            "reason": f"{n} sub-bands < {MIN_GOOD}",
            "sub": sub,
        }
    x = K_DM * (1.0 / np.array(nu) ** 2 - 1.0 / nu_ref**2)
    y = np.array(t0)
    w = 1.0 / np.array(err) ** 2
    X = np.vstack([x, np.ones_like(x)]).T
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ ((X * w[:, None]).T @ y)
    resid = y - X @ beta
    chi2_red = float(np.sum(w * resid**2) / max(n - 2, 1))
    sigma = float(np.sqrt(abs(cov[0, 0]) * max(chi2_red, 1.0)))
    return {
        "dm_offset": float(beta[0]),
        "sigma": sigma,
        "n_good": n,
        "constrains": bool(sigma <= 1.0),
        "chi2_red": chi2_red,
        "slope_beta": beta.tolist(),
        "sub": sub,
        "reason": "ok",
    }


def extract_one(path, dm_dsa):
    bb = BBData.from_file(path)
    dt = float(bb.attrs["delta_time"]) * TDS
    freq = np.asarray(bb.index_map["freq"]["centre"], float)
    bbdd = coherent_dedisp(bb, dm_dsa)
    inten = _downsample_t(
        np.nan_to_num(np.abs(bbdd[:, 0, :]) ** 2 + np.abs(bbdd[:, 1, :]) ** 2), TDS
    )
    csd = inten.std(1)
    med = np.median(csd[csd > 0]) if np.any(csd > 0) else 0.0
    good = np.isfinite(csd) & (csd > 0.2 * med) & (csd < 8.0 * med)
    I0, fw = inten[good], freq[good]
    prof = (I0 - np.median(I0, 1, keepdims=True)).sum(0)
    pk = int(np.argmax(np.convolve(prof, np.ones(max(int(1.5e-3 / dt), 1)), "same")))
    lo = max(pk - int(15e-3 / dt), 0)
    hi = min(pk + int(45e-3 / dt), I0.shape[1])
    r = _regress(I0[:, lo:hi], fw, dt)
    dm = None if r["dm_offset"] is None else dm_dsa + r["dm_offset"]
    res = {
        "dm": dm,
        "dm_offset": r["dm_offset"],
        "sigma": r["sigma"],
        "n_good_subbands": r["n_good"],
        "constrains_dm": r["constrains"],
        "reason": r["reason"],
        "chi2_red": r.get("chi2_red"),
        "dm_dsa": dm_dsa,
        "n_chan_ok": int(good.sum()),
        "crop_ms": [round(lo * dt * 1e3, 1), round(hi * dt * 1e3, 1)],
    }
    fig = _figure(I0[:, lo:hi], fw, dt, dm_dsa, r, res)
    return res, fig


def _figure(wf, freq, dt, dm_dsa, r, res):
    order = np.argsort(freq)
    wfo, fro = wf[order], freq[order]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    nf = (wfo.shape[0] // 4) * 4
    m = wfo[:nf].reshape(nf // 4, 4, wfo.shape[1]).mean(1)
    mu, sd = m.mean(1, keepdims=True), m.std(1, keepdims=True) + 1e-9
    ax[0].imshow(
        (m - mu) / sd,
        aspect="auto",
        origin="lower",
        extent=[0, wfo.shape[1] * dt * 1e3, fro.min(), fro.max()],
        vmin=-0.5,
        vmax=5,
        cmap="magma",
    )
    ax[0].set(
        title=f"coherent @ DSA DM={dm_dsa:g} ({res['n_good_subbands']} good sub-bands)",
        xlabel="t (ms)",
        ylabel="freq MHz",
    )
    sub = r["sub"]
    if sub["nu"]:
        nu = np.array(sub["nu"])
        x = K_DM * (1.0 / nu**2 - 1.0 / sub["nu_ref"] ** 2)
        ax[1].errorbar(x, np.array(sub["t0"]) * 1e3, yerr=np.array(sub["err"]) * 1e3, fmt="o", ms=4)
        if r["dm_offset"] is not None:
            xs = np.linspace(x.min(), x.max(), 50)
            b = r["slope_beta"]
            ax[1].plot(xs, (b[0] * xs + b[1]) * 1e3, "r-")
    s = "None" if res["sigma"] is None else f"{res['sigma']:.3f}"
    dms = "None" if res["dm"] is None else f"{res['dm']:.2f}"
    ax[1].set(
        title=f"arrival regression: DM={dms} +/- {s}  constrains={res['constrains_dm']}",
        xlabel="K_DM (nu^-2 - nu_ref^-2)  [s per pc/cm^3]",
        ylabel="t0 (ms)",
    )
    fig.tight_layout()
    return fig


def main():
    bursts = {b["name"]: b for b in json.load(open(ROOT + "/scripts/burst_inputs.json"))["bursts"]}
    out, manifest = [], []
    for name in TARGETS:
        meta = bursts.get(name)
        if meta is None:
            continue
        cid, dm = meta["chime_id"], float(meta["dm"])
        path = f"{SB}/{name.lower()}/singlebeam_{cid}.h5"
        if not os.path.exists(path):
            continue
        try:
            res, fig = extract_one(path, dm)
            png = f"{DIAG}/{name}_dm.png"
            fig.savefig(png, dpi=110)
            plt.close(fig)
            res.update(name=name, chime_id=cid)
            out.append(res)
            dms = "None" if res["dm"] is None else f"{res['dm']:.2f}"
            s = "None" if res["sigma"] is None else f"{res['sigma']:.3f}"
            manifest.append(
                {
                    "path": os.path.basename(png),
                    "expectation": f"{name}: coherent@DSA={dm:g}; arrival regression DM={dms}+/-{s} "
                    f"({res['n_good_subbands']} sub-bands) constrains={res['constrains_dm']}",
                }
            )
            print(
                f"[OK] {name:11s} DSA={dm:8.2f} DM={dms:>8} +/- {s:>6} "
                f"dDM={'' if res['dm'] is None else round(res['dm'] - dm, 2)!s:>6} "
                f"nsub={res['n_good_subbands']} constrains={res['constrains_dm']}",
                flush=True,
            )
        except Exception as exc:
            out.append({"name": name, "chime_id": cid, "status": f"error: {exc}"})
            print(f"[ERR] {name}: {exc}\n{traceback.format_exc()}", flush=True)
        json.dump(out, open(ROOT + "/results/chime_dm_v4b.json", "w"), indent=2)
        json.dump({"figures": manifest}, open(f"{DIAG}/figures.manifest.json", "w"), indent=2)
    print(f"\nwrote results/chime_dm_v4b.json ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()
