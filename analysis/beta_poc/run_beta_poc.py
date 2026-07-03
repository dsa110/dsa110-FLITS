#!/usr/bin/env python
"""Proof-of-concept: physically-consistent beta-coupled joint scattering fit (one burst).

Replaces the free-alpha + fixed-PBF interpretation (alpha sampled independently
inside an exponential PBF whose own beta is held at Kolmogorov) with a single
turbulence index beta that drives BOTH

  1. the PBF tail shape   -> gaussian_powerlaw_convolution(..., beta)   (shape)
  2. the frequency scaling -> alpha = 2*beta/(beta-2)                    (closure)

so PBF shape and alpha are no longer independent knobs. This is FULL PBF(beta)
physics (both bands use the power-law PBF at the SAME sampled beta), not a
beta->alpha-only shortcut: see `coupling` field in the output JSON.

Scope: ONE burst, freya = FRB 20230325A, the C1D1 shared-zeta(nu) setup
(theta = [tau_1ghz, beta, zeta_1ghz, x_zeta, t0_C, ddm_C, t0_D, ddm_D]) -- the
same 8-vector as the canonical joint shared-zeta fit, with beta replacing alpha
at index 1.

Default is the deterministic synthetic freya-like injection-recovery (truth
beta injected, recovered). --real instead builds both bands from the freya
run-configs (issue #99) through the joint driver's own prepare() -- config
load, real .npy, bandpass/trim/downsample/crop, FRBModel + init -- and fits
those: Route A, the independent-likelihood cross-check instrument for
#103/#105. data_source records which path ran. The nlive/maxcall defaults are
smoke-scale; the production real fit (and its runtime budget) is #104's.

  conda run -n flits python analysis/beta_poc/run_beta_poc.py [--nlive N] [--seed S] [--real]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scattering.scat_analysis.burstfit import (  # noqa: E402
    FRBModel,
    FRBParams,
)

# The POC's private alpha_from_beta_thin_screen kernel was superseded when the
# beta co-model landed on main: the same thin-screen closure alpha = 2b/(b-2)
# now lives in scattering.scat_analysis.turbulence as alpha_from_beta.
from scattering.scat_analysis.turbulence import (  # noqa: E402
    alpha_from_beta as alpha_from_beta_thin_screen,
)

# --- freya band geometry (scattering/configs/telescopes.yaml; run_freya_fitting.py) ---
CHIME = dict(f_min=0.40019, f_max=0.80019, df_MHz=0.390625)
DSA = dict(f_min=1.31125, f_max=1.49875, df_MHz=0.03051757812)

# --- synthetic injection truth (freya-like; tau/zeta/x_zeta near the published
#     free-alpha shared-zeta posterior, alpha set BY beta via the closure) ---
BETA_TRUE = 3.70  # -> alpha = 2*3.70/1.70 = 4.353 (just above Kolmogorov 11/3)
# tau_1ghz=0.05 ms: the power-law PBF tail is heavy (~t^-beta/2), so at CHIME 0.4 GHz
# tau(nu)=tau_1ghz*nu^-alpha = 0.05*0.4^-4.353 ~ 2.7 ms; with t0 early in the window
# the 32 ms CHIME grid captures ~10 e-folds of tail (s_c crossover ~3.8) -- enough
# that the tail SHAPE, not its truncation, drives beta. (freya's published exp-PBF
# tau_1ghz~0.12 ms would need a ~150 ms window under a heavy power-law tail.)
TAU_TRUE = 0.05  # tau_1ghz [ms]
ZETA1_TRUE = 0.085  # zeta_1ghz [ms]
X_ZETA_TRUE = -0.79
T0_C_TRUE = 4.0  # ms (early in the window so the heavy tail is captured, not truncated)
T0_D_TRUE = 1.0  # ms
DDM_TRUE = 0.0  # coherently-dedispersed assumption (dm_init=0); isolates scattering

# beta prior: lower bound 3.0 (alpha=6, steep); upper 3.95 sits at the power-law
# PBF validity edge (gaussian_powerlaw_convolution clips beta to <3.99, alpha->4).
# This range CANNOT reach alpha<4 by construction -- the closure's sign caveat.
BETA_LO, BETA_HI = 3.0, 3.95


def _build_band(geom: dict, n_freq: int, t_max: float, n_time: int) -> FRBModel:
    """Empty FRBModel on a freya-like band grid (dm_init=0)."""
    freq = np.linspace(geom["f_min"], geom["f_max"], n_freq)  # ascending (loader convention)
    time = np.linspace(0.0, t_max, n_time)
    return FRBModel(time=time, freq=freq, dm_init=0.0, df_MHz=geom["df_MHz"])


def _inject(m: FRBModel, tau: float, beta: float, z1: float, x: float, t0: float, rng) -> FRBModel:
    """Inject a powerlaw-PBF burst with per-channel scintillation gains + noise."""
    zeta_nu = z1 * m.freq**x
    p = FRBParams(
        c0=1.0, t0=t0, gamma=0.0, zeta=zeta_nu, tau_1ghz=tau, beta=beta, delta_dm=DDM_TRUE
    )
    # forward model derives alpha and dispatches the power-law PBF from p.beta (ADR-0006)
    kernel = m(p, "M3")  # (n_freq, n_time), area-normalized per channel
    # per-channel gain = smooth power-law envelope * lognormal scintillation
    envelope = (m.freq / np.median(m.freq)) ** -1.5
    scint = np.exp(rng.normal(0.0, 0.2, size=m.freq.size))
    gain = 20.0 * envelope * scint
    clean = gain[:, None] * kernel
    sigma = float(np.max(clean)) / 20.0  # peak-channel SNR ~ 20
    data = clean + rng.normal(0.0, sigma, size=clean.shape)
    noise = np.full(m.freq.size, sigma)
    return FRBModel(
        time=m.time, freq=m.freq, data=data, dm_init=0.0, df_MHz=m.df_MHz, noise_std=noise
    )


def _band_ll(m: FRBModel, tau, beta, z1, x, t0, ddm) -> float:
    zeta_nu = z1 * np.asarray(m.freq, float) ** x
    p = FRBParams(c0=1.0, t0=t0, gamma=0.0, zeta=zeta_nu, tau_1ghz=tau, beta=beta, delta_dm=ddm)
    return m.log_likelihood_gain_marginal(p, "M3")


class BetaCoupledLogL:
    """theta = [tau, beta, zeta_1ghz, x_zeta, t0_C, ddm_C, t0_D, ddm_D].

    ONE sampled beta enters both bands' FRBParams; the forward model derives
    the shared alpha via the thin-screen closure and dispatches the power-law
    PBF shape from that same beta (ADR-0006) -- the whole point of the POC.
    """

    def __init__(self, m_C: FRBModel, m_D: FRBModel):
        self.m_C, self.m_D = m_C, m_D

    def __call__(self, theta) -> float:
        tau, beta, z1, x = (float(theta[i]) for i in range(4))
        ll = _band_ll(self.m_C, tau, beta, z1, x, float(theta[4]), float(theta[5])) + _band_ll(
            self.m_D, tau, beta, z1, x, float(theta[6]), float(theta[7])
        )
        return ll if np.isfinite(ll) else -1e100


def _ptform_factory(t0_C, t0_D):
    lo = np.array([np.log(0.01), BETA_LO, np.log(1e-3), -4.0, t0_C - 2.0, -0.5, t0_D - 2.0, -0.5])
    hi = np.array([np.log(5.0), BETA_HI, np.log(1.0), 2.0, t0_C + 2.0, 0.5, t0_D + 2.0, 0.5])
    is_log = np.array([True, False, True, False, False, False, False, False])

    def ptform(u):
        x = lo + u * (hi - lo)
        x[is_log] = np.exp(x[is_log])
        return x

    return ptform


def _matched(m: FRBModel, tau, z1, x, t0, ddm, beta):
    """Data + per-channel matched-filter gain model at fixed shape params
    (shared by goodness-of-fit and the diagnostic figures)."""
    zeta_nu = z1 * m.freq**x
    p = FRBParams(c0=1.0, t0=t0, gamma=0.0, zeta=zeta_nu, tau_1ghz=tau, beta=beta, delta_dm=ddm)
    valid = m.valid
    K = m(replace(p, c0=1.0, gamma=0.0), "M3", freq_subset=valid)
    d = m.data[valid]
    S_dk = np.einsum("ij,ij->i", d, K)
    S_kk = np.einsum("ij,ij->i", K, K)
    gain = np.where(S_kk > 1e-30, S_dk / np.where(S_kk > 1e-30, S_kk, 1.0), 0.0)
    return d, gain[:, None] * K, valid


def _goodness(m: FRBModel, tau, z1, x, t0, ddm, beta) -> dict:
    """Matched-filter reduced chi2 + R^2 + Durbin-Watson at the posterior median."""
    d, model, valid = _matched(m, tau, z1, x, t0, ddm, beta)
    sig = np.clip(m.noise_std[valid], 1e-9, None)
    resid = d - model
    chi2 = float(np.sum((resid / sig[:, None]) ** 2))
    n_chan = int(valid.sum())
    # dof: data points - per-channel gains (n_chan) - 6 shared shape params
    dof = max(resid.size - n_chan - 6, 1)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((d - d.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Durbin-Watson on the band-summed residual profile (time autocorrelation)
    rp = resid.sum(axis=0)
    dw = float(np.sum(np.diff(rp) ** 2) / np.sum(rp**2)) if np.sum(rp**2) > 0 else float("nan")
    return {"red_chi2": chi2 / dof, "r2": r2, "durbin_watson": dw, "n_chan": n_chan, "dof": dof}


def _diagnostic_figure(m: FRBModel, tau, z1, x, t0, ddm, beta, gof: dict, band: str, path):
    """Data-vs-model, residual profile, and Q-Q of noise-normalized residuals at
    the posterior median (the fit-validation contract's diagnostics checklist:
    data-vs-model + residual + Q-Q + Durbin-Watson, the last in gof -- #111)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.special import erfinv

    d, model, valid = _matched(m, tau, z1, x, t0, ddm, beta)
    sig = np.clip(m.noise_std[valid], 1e-9, None)
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 8), height_ratios=[3, 1, 2])
    ax0.plot(m.time, d.sum(axis=0), "k-", lw=0.8, label="data")
    ax0.plot(m.time, model.sum(axis=0), "r-", lw=1.2, label="model (matched gains)")
    ax0.set_ylabel("band-summed flux [S/N units]")
    ax0.legend(frameon=False)
    ax0.set_title(
        f"freya {band}: red_chi2={gof['red_chi2']:.2f}, r2={gof['r2']:.3f}, "
        f"dw={gof['durbin_watson']:.2f}, beta={beta:.3f}"
    )
    ax1.plot(m.time, (d - model).sum(axis=0), "k-", lw=0.8)
    ax1.axhline(0.0, color="r", lw=0.6)
    ax1.set_xlabel("time [ms]")
    ax1.set_ylabel("residual")
    # Q-Q: per-pixel noise-normalized residuals vs standard normal (Hazen positions)
    r = np.sort(((d - model) / sig[:, None]).ravel())
    q = (np.arange(r.size) + 0.5) / r.size
    theo = np.sqrt(2.0) * erfinv(2.0 * q - 1.0)
    ax2.plot(theo, r, "k.", ms=1, rasterized=True)
    lim = [theo[0], theo[-1]]
    ax2.plot(lim, lim, "r-", lw=0.8)
    ax2.set_xlabel("normal quantiles")
    ax2.set_ylabel("residual quantiles [sigma]")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _validate(med: dict, gof_C: dict, gof_D: dict) -> dict:
    """3-level PASS/MARGINAL/FAIL. Level 2 delegates to the kernel's
    classify_fit_quality: chi2_red is the gate; R^2 and Durbin-Watson are
    informational only (low-S/N rationale in its docstring) -- issue #111."""
    from flits.fitting import VALIDATION_THRESHOLDS as T
    from scattering.scat_analysis.burstfit import classify_fit_quality

    fails, marginals, notes = [], [], []
    # Level 1 (hard gates)
    tau, alpha = med["tau_1ghz"], med["alpha"]
    if not (T.WIDTH_MIN / 10 < tau < 100):  # 0.0001 < tau < 100 ms
        fails.append(f"tau_1ghz={tau:.4g} ms outside (1e-4, 100)")
    if not (1.5 < alpha < 6.0):
        fails.append(f"alpha={alpha:.3g} outside (1.5, 6.0)")
    # Level 2 (quality) -- worst band drives the verdict; chi2_red is the gate
    for tag, g in (("C", gof_C), ("D", gof_D)):
        flag, band_notes = classify_fit_quality(g["red_chi2"], g["r2"])
        notes += [f"[{tag}] {n}" for n in band_notes]
        notes.append(f"[{tag}] durbin_watson={g['durbin_watson']:.2f} [informational]")
        if flag == "FAIL":
            fails.append(f"level2[{tag}]: chi2_red={g['red_chi2']:.2f} FAIL")
        elif flag == "MARGINAL":
            marginals.append(f"level2[{tag}]: chi2_red={g['red_chi2']:.2f} MARGINAL")
    verdict = "FAIL" if fails else ("MARGINAL" if marginals else "PASS")
    return {"verdict": verdict, "fails": fails, "marginals": marginals, "notes": notes}


def _write_manifest(outdir: Path, figures: list, data_source: str):
    """figures.manifest.json for the figure-review Stop gate. Merges with any
    existing entries (synthetic and real runs keep separate figure files);
    figures.review.json is written by the reviewer, never here."""
    from datetime import date

    manifest_path = outdir / "figures.manifest.json"
    entries = {}
    if manifest_path.exists():
        for e in json.loads(manifest_path.read_text()).get("figures", []):
            entries[e["file"]] = e
    src = "real freya data" if data_source == "real_freya_npy" else "synthetic injection-recovery"
    for f in figures:
        band = "CHIME" if "chime" in f else "DSA"
        entries[f] = {
            "file": f,
            "expectation": (
                f"beta POC {src}, {band} band: matched-gain model (red) overlays the "
                "band-summed data profile (black) including the scattering tail; "
                "residual panel structureless around zero; Q-Q panel of "
                "noise-normalized residuals hugs the y=x line."
            ),
        }
    manifest_path.write_text(
        json.dumps(
            {
                "directory": "analysis/beta_poc/freya",
                "generated": str(date.today()),
                "campaign": f"beta POC diagnostics ({data_source})",
                "figures": [entries[k] for k in sorted(entries)],
            },
            indent=2,
        )
    )


def _prepare_real_bands():
    """CHIME+DSA FRBModels from the freya run-configs (issue #99) via the joint
    driver's own prepare() -- the completion this file's real-data stub always
    prescribed. Loading pattern as in tests/test_freya_local_runs_smoke.py.

    t0 prior centers come from the RAW data-driven guess, not prepare()'s
    MLE-refined init: the post-#98 refine rails beta to the thin-screen floor
    on freya and drags t0 with it (#101 observed CHIME 17.6 -> 31.4 ms, near
    the window end) -- init-only for nested sampling, but a prior *center*
    must not inherit it.
    """
    import yaml

    local_runs = REPO / "analysis" / "scattering-refit-2026-06" / "local_runs"
    os.environ["FLITS_REPO"] = str(REPO)
    spec = importlib.util.spec_from_file_location("run_joint_fit", local_runs / "run_joint_fit.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    from scattering.scat_analysis.burstfit_init import data_driven_initial_guess

    models, t0s = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for band in ("chime", "dsa"):
            cfg_path = local_runs / "configs" / f"freya_{band}_run.yaml"
            cfg = yaml.safe_load(cfg_path.read_text())
            if not Path(cfg["path"]).exists():
                raise SystemExit(
                    f"real freya .npy not staged: {cfg['path']} "
                    "(pinned manuscript checkout's data symlink)"
                )
            model, _init_mle = driver.prepare(str(cfg_path), f"freya_{band}", tmp)
            raw = data_driven_initial_guess(
                data=model.data,
                freq=model.freq,
                time=model.time,
                dm=float(model.dm_init),
                verbose=False,
            ).params
            models.append(model)
            t0s.append(float(raw.t0))
    return models[0], models[1], t0s[0], t0s[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlive", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20230325)  # FRB 20230325A
    ap.add_argument("--dlogz", type=float, default=0.5)
    ap.add_argument("--maxcall", type=int, default=400_000)
    ap.add_argument(
        "--real",
        action="store_true",
        help="fit the real freya .npy (joint driver's prepare() + the #99 "
        "run-configs) instead of the synthetic injection",
    )
    args = ap.parse_args()

    outdir = REPO / "analysis" / "beta_poc" / "freya"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.real:
        data_source = "real_freya_npy"
        m_C, m_D, t0_C, t0_D = _prepare_real_bands()
        print(
            f"[freya] real prepared bands: CHIME {m_C.data.shape} DSA {m_D.data.shape}; "
            f"raw-init t0_C={t0_C:.2f} ms t0_D={t0_D:.2f} ms; fitting (nlive={args.nlive})...",
            flush=True,
        )
    else:
        data_source = "synthetic_injection"
        # Synthetic freya-like bands (coarse grids -> fast deterministic smoke fit).
        rng = np.random.default_rng(args.seed)
        m_C0 = _build_band(CHIME, n_freq=48, t_max=32.0, n_time=448)
        m_D0 = _build_band(DSA, n_freq=48, t_max=6.0, n_time=320)
        m_C = _inject(m_C0, TAU_TRUE, BETA_TRUE, ZETA1_TRUE, X_ZETA_TRUE, T0_C_TRUE, rng)
        m_D = _inject(m_D0, TAU_TRUE, BETA_TRUE, ZETA1_TRUE, X_ZETA_TRUE, T0_D_TRUE, rng)
        t0_C, t0_D = T0_C_TRUE, T0_D_TRUE
        print(
            f"[freya] injected beta={BETA_TRUE} -> alpha={alpha_from_beta_thin_screen(BETA_TRUE):.3f}, "
            f"tau_1ghz={TAU_TRUE} ms; fitting (nlive={args.nlive})...",
            flush=True,
        )

    from dynesty import NestedSampler

    loglike = BetaCoupledLogL(m_C, m_D)
    ptform = _ptform_factory(t0_C, t0_D)
    names = ["tau_1ghz", "beta", "zeta_1ghz", "x_zeta", "t0_C", "delta_dm_C", "t0_D", "delta_dm_D"]
    sampler = NestedSampler(
        loglike,
        ptform,
        ndim=8,
        nlive=args.nlive,
        sample="rwalk",
        rstate=np.random.default_rng(args.seed + 1),
    )
    sampler.run_nested(dlogz=args.dlogz, maxcall=args.maxcall, print_progress=True)
    res = sampler.results

    w = np.exp(res.logwt - res.logz[-1])
    w /= w.sum()
    med = {}
    for i, n in enumerate(names):
        s = res.samples[:, i]
        lo, m50, hi = _wquantile(s, w, [0.16, 0.5, 0.84])
        med[n] = {"median": float(m50), "err_minus": float(m50 - lo), "err_plus": float(hi - m50)}
    medv = {n: med[n]["median"] for n in names}
    medv["alpha"] = alpha_from_beta_thin_screen(medv["beta"])

    gof_C = _goodness(
        m_C,
        medv["tau_1ghz"],
        medv["zeta_1ghz"],
        medv["x_zeta"],
        medv["t0_C"],
        medv["delta_dm_C"],
        medv["beta"],
    )
    gof_D = _goodness(
        m_D,
        medv["tau_1ghz"],
        medv["zeta_1ghz"],
        medv["x_zeta"],
        medv["t0_D"],
        medv["delta_dm_D"],
        medv["beta"],
    )
    val = _validate(medv, gof_C, gof_D)

    figtag = "_real" if args.real else ""
    figures = []
    for band, m, t0b, ddmb, g in (
        ("chime", m_C, medv["t0_C"], medv["delta_dm_C"], gof_C),
        ("dsa", m_D, medv["t0_D"], medv["delta_dm_D"], gof_D),
    ):
        fname = f"freya_beta_poc_diag_{band}{figtag}.png"
        _diagnostic_figure(
            m,
            medv["tau_1ghz"],
            medv["zeta_1ghz"],
            medv["x_zeta"],
            t0b,
            ddmb,
            medv["beta"],
            g,
            band,
            outdir / fname,
        )
        figures.append(fname)
    _write_manifest(outdir, figures, data_source)

    summary = {
        "burst": "freya",
        "tns": "FRB 20230325A",
        "model": "beta_coupled_thin_screen_powerlaw_pbf",
        "coupling": "full_pbf_beta",  # both PBF shape AND alpha derived from one beta
        "closure": "alpha = 2*beta/(beta-2)  (thin-screen inertial range)",
        "data_source": data_source,
        "setup": "C1D1 shared-zeta(nu); theta=[tau,beta,zeta1,x_zeta,t0_C,ddm_C,t0_D,ddm_D]",
        "pbf": {"C": "powerlaw", "D": "powerlaw", "beta_is_sampled": True},
        "beta_prior": [BETA_LO, BETA_HI],
        "alpha_implied_by_beta_prior": [
            alpha_from_beta_thin_screen(BETA_HI),
            alpha_from_beta_thin_screen(BETA_LO),
        ],
        "median": med,
        "alpha_derived": {
            "median": medv["alpha"],
            "note": "derived from sampled beta via closure; NOT independently fit",
        },
        "goodness_of_fit": {"chime": gof_C, "dsa": gof_D},
        "validation": val,
        "diagnostics": figures,
        "log_evidence": float(res.logz[-1]),
        "log_evidence_err": float(res.logzerr[-1]),
        "ncall": int(np.sum(res.ncall)),
        "limitation": (
            "Full self-consistency restricts the power-law PBF to beta<4, so this "
            "closure yields alpha>=4 only (alpha=4 at beta=4). A burst with "
            "empirical alpha<4 (e.g. casey ~3.9) cannot be a thin-screen power-law "
            "screen under this mapping -- it is NOT a sub-Kolmogorov (smaller) beta. "
            "The deprecated exp-era fit *suggested* alpha~4.35 for freya (maps to "
            "beta~3.70, inside the valid range); whether that survives under the "
            "beta co-model is the hypothesis the real-data fit tests."
        ),
    }
    if data_source == "synthetic_injection":
        # Injection-recovery bookkeeping is only meaningful when a truth exists.
        truth = {
            "beta": BETA_TRUE,
            "alpha": alpha_from_beta_thin_screen(BETA_TRUE),
            "tau_1ghz": TAU_TRUE,
            "zeta_1ghz": ZETA1_TRUE,
            "x_zeta": X_ZETA_TRUE,
        }
        summary["recovery"] = {
            k: {
                "true": truth[k],
                "fit": medv[k],
                "rel_err": abs(medv[k] - truth[k]) / abs(truth[k]) if truth[k] else None,
            }
            for k in ("beta", "alpha", "tau_1ghz", "zeta_1ghz", "x_zeta")
        }

    out = outdir / ("freya_beta_poc_fit_real.json" if args.real else "freya_beta_poc_fit.json")
    out.write_text(json.dumps(summary, indent=2))
    truth_tag = (
        f"(true beta={BETA_TRUE}, alpha={alpha_from_beta_thin_screen(BETA_TRUE):.3f})  "
        if data_source == "synthetic_injection"
        else "(real data; no truth)  "
    )
    print(
        f"\n[freya] beta={medv['beta']:.3f} -> alpha={medv['alpha']:.3f} "
        f"{truth_tag}"
        f"tau_1ghz={medv['tau_1ghz']:.4g} ms  lnZ={res.logz[-1]:.1f}",
        flush=True,
    )
    print(
        f"[freya] validation: {val['verdict']}  fails={val['fails']}  marginals={val['marginals']}"
    )
    print(f"[freya] wrote {out}")
    return 0


def _wquantile(x, w, q):
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return np.interp(q, cdf, x)


if __name__ == "__main__":
    raise SystemExit(main())
