#!/usr/bin/env python3
"""Multi-component joint CHIME+DSA refits for the bad-fit bursts
(hamilton, casey, zach, wilhelm), whitney-C2D2-cwin precedent.

Per-component t0/zeta windows from profile inspection (2026-07-07, sandbox).
Disjoint windows -> plain _JointPriorTransform (whitney cwin precedent);
overlapping windows (casey CHIME) -> _JointPriorTransformOrdered + dt_min.

Usage: python3 refit_runner.py <burst> [--nlive 160] [--nproc 4] [--dlogz 0.5]
Writes RUNS/data/joint/<burst>_joint_fit<suffix>.json + samples npz + configs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time as _time
from pathlib import Path

import numpy as np
import yaml

RUN_DIR = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("FABER2026_PIPELINE", Path(__file__).resolve().parents[4]))
DATA = Path(
    os.environ.get("FABER2026_BURST_DATA", "/Users/jakobfaber/Data/Faber2026/dsa110/DSA_bursts")
)
RUNS = Path(os.environ.get("FABER2026_REFIT_RUNS", RUN_DIR))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scattering"))

from dynesty import NestedSampler  # noqa: E402
from scat_analysis.burstfit_joint import (  # noqa: E402
    _JointLogLikelihoodGainMulti,
    _JointPriorTransform,
    _JointPriorTransformOrdered,
    _append_derived_alpha_percentiles,
    _weighted_percentiles,
    alpha_from_beta,
)
from scat_analysis.config_utils import load_telescope_block  # noqa: E402
from scat_analysis.pipeline.io import BurstDataset  # noqa: E402

# ---------------------------------------------------------------------------
# Per-burst spec. t0 windows are on the BurstDataset-cropped time axis with the
# stated onpulse_pad_factor -- keep pads in sync with inspect_profiles output.
# comps: list of ((t0_lo, t0_hi), (zeta_lo, zeta_hi)) per component.
# ordered=True -> group window = comp-1 window, dt_min separation (overlap OK).
SPEC = {
    "hamilton": dict(
        dm=518.799,
        tau=(0.002, 0.15),
        pad=dict(chime=0.5, dsa=0.5),
        C=dict(
            # round 2: + trailing component around the 12.90 ms peak (10.7 sig)
            comps=[
                ((11.45, 11.78), (0.02, 0.12)),
                ((11.80, 12.10), (0.02, 0.12)),
                ((12.60, 13.20), (0.02, 0.20)),
            ],
            ordered=False,
            ddm=(-0.08, 0.08),
        ),
        D=dict(
            comps=[((14.90, 15.40), (0.02, 0.15))],
            ordered=False,
            ddm=(-0.08, 0.08),
        ),
        note="C2D1: leading CHIME component at ~-0.31 ms (69 sig prominence); "
        "third trailing CHIME peak (+0.98 ms, 10.7 sig) deliberately left out "
        "of this first pass.",
    ),
    "zach": dict(
        dm=262.368,
        tau=(0.05, 1.2),
        pad=dict(chime=0.5, dsa=0.5),
        C=dict(
            # t0 = pulse RISE, not peak: tau(0.6 GHz) ~ 2 ms lags the peak, so
            # the window opens ~2.5 tau before the 13.58 ms profile peak
            # (first pass railed at a 13.30 floor).
            # round 2: + trailing CHIME component (the +2.5-4 ms residual patch
            # coincident with the trailing DSA complex; lag1_C=0.86 first pass)
            # The first C2D4 pass railed this component at t0=15.00 ms and
            # zeta=1.00, so open the rise-time window before rerunning.
            comps=[((12.40, 13.90), (0.03, 0.40)), ((14.00, 17.50), (0.05, 2.00))],
            ordered=False,
            ddm=(-0.5, 0.5),  # CHIME flagged under-dedispersed
        ),
        D=dict(
            comps=[
                ((9.30, 9.80), (0.02, 0.20)),
                # Owner correction: DSA has one initial pulse, then a trailing
                # three-component cluster. The first two cluster peaks are
                # obvious in the profile (11.665, 12.124 ms); the last window
                # covers the weaker trailing shoulder near 12.58 ms.
                ((11.35, 11.85), (0.02, 0.30)),
                ((11.85, 12.35), (0.02, 0.40)),
                ((12.35, 12.85), (0.02, 0.40)),
            ],
            ordered=False,
            ddm=(-0.08, 0.08),
        ),
        note="C2D4: CHIME trailing component plus DSA morphology corrected to "
        "one initial pulse and a trailing three-component cluster; wide "
        "delta_dm_C for the known CHIME dedispersion residual.",
    ),
    "wilhelm": dict(
        dm=602.346,
        tau=(0.03, 1.0),
        pad=dict(chime=0.5, dsa=2.0),  # dsa pad 0.5 crops to 1.5 ms -- too tight
        # Round 2: native 32.8 us tests whether the coherent bright-pulse
        # residual is a resolution artifact. This is not an EMG-family rejection:
        # beta-coherent fits drive wilhelm to the beta~4 exponential limit.
        t_factor=dict(dsa=1),
        suffix_extra="_tf1",
        C=dict(
            # rise-not-peak windows: tau(0.6 GHz) up to ~1-3 ms for
            # tau_1ghz in [0.03,1.0] -> open well before the 2.70/6.51 peaks.
            comps=[((1.40, 2.95), (0.03, 0.30)), ((5.60, 6.85), (0.03, 0.40))],
            ordered=False,
            ddm=(-0.08, 0.08),
        ),
        D=dict(
            comps=[((0.30, 0.90), (0.02, 0.15)), ((1.45, 2.10), (0.02, 0.15))],
            ordered=False,
            ddm=(-0.08, 0.08),
        ),
        note="C2D2: leading DSA component at ~-1.2 ms (5.9 sig) is the "
        "plan-doc hidden pulse; trailing CHIME 6.4 sig at +3.8 ms.",
    ),
    "casey": dict(
        dm=491.207,
        tau=(0.002, 0.15),
        pad=dict(chime=0.5, dsa=0.5),
        C=dict(
            comps=[((10.40, 11.35), (0.015, 0.10)), ((10.40, 11.35), (0.015, 0.10))],
            ordered=True,
            dt_min=0.15,  # ~2.5 CHIME time samples at t_factor=24
            ddm=(-0.08, 0.08),
        ),
        D=dict(
            comps=[((28.40, 28.90), (0.02, 0.12))],
            ordered=False,
            ddm=(-0.08, 0.08),
        ),
        note="C2D1: band-integrated CHIME profile shows ONE narrow peak even at "
        "15 us, but the pair figure shows two-streak structure; overlapping "
        "windows + dt_min let the evidence/residuals decide.",
    ),
}
BAND = {
    "chime": dict(f_factor=16, t_factor=24, dm_init=0.0),
    "dsa": dict(f_factor=384, t_factor=2, dm_init="catalog"),
}


def t_factor_for(spec, band: str) -> int:
    return int(spec.get("t_factor", {}).get(band, BAND[band]["t_factor"]))


def suffix_for(burst: str) -> str:
    s = SPEC[burst]
    n_C, n_D = len(s["C"]["comps"]), len(s["D"]["comps"])
    return f"_C{n_C}D{n_D}_cwin" + s.get("suffix_extra", "")


def data_path(burst: str, band: str) -> Path:
    hits = sorted(DATA.glob(f"{burst}_{band}_I_*.npy"))
    assert len(hits) == 1, (burst, band, hits)
    return hits[0]


def write_config(burst: str, band: str, spec) -> Path:
    """Freya local-runs pattern config so dump_jointmodel.py can rebuild bands."""
    bc = BAND[band]
    cfg = dict(
        chunk_size=2000,
        diagnostics=True,
        dlogz=0.5,
        dm_init=spec["dm"] if bc["dm_init"] == "catalog" else bc["dm_init"],
        extend_chain=True,
        f_factor=bc["f_factor"],
        fitting_method="nested",
        max_chunks=5,
        model_scan=True,
        nlive=400,
        nlive_walks=15,
        nproc=8,
        outer_trim=0.15,
        onpulse_pad_factor=spec["pad"][band],
        path=str(data_path(burst, band)),
        plot=True,
        sampcfg_path=str(REPO / "scattering/configs/sampler.yaml"),
        steps=10000,
        t_factor=t_factor_for(spec, band),
        telcfg_path=str(REPO / "scattering/configs/telescopes.yaml"),
        telescope=band,
    )
    out = RUNS / "configs"
    out.mkdir(parents=True, exist_ok=True)
    fp = out / f"{burst}_{band}_run.yaml"
    fp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return fp


def prepare(burst: str, band: str, spec):
    tel = load_telescope_block(str(REPO / "scattering/configs/telescopes.yaml"), band)
    ds = BurstDataset(
        str(data_path(burst, band)),
        str(RUNS / "prep"),
        name=f"{burst}_{band}",
        telescope=tel,
        f_factor=BAND[band]["f_factor"],
        t_factor=t_factor_for(spec, band),
        outer_trim=0.15,
        onpulse_crop=True,
        onpulse_pad_factor=spec["pad"][band],
    )
    m = ds.model
    m.dm_init = spec["dm"] if BAND[band]["dm_init"] == "catalog" else BAND[band]["dm_init"]
    return m


def build_spec(burst: str):
    s = SPEC[burst]
    names: list[str] = ["tau_1ghz", "beta"]
    rows: list[tuple[str, tuple[float, float], bool]] = [
        ("tau_1ghz", s["tau"], True),
        ("beta", (3.0, 4.0), False),
    ]
    t0_groups: list[list[int]] = []
    dt_mins: list[float] = []
    any_ordered = False
    for tag in ("C", "D"):
        b = s[tag]
        grp = []
        for i, (t0w, zw) in enumerate(b["comps"], start=1):
            names += [f"t0_{tag}{i}", f"zeta_{tag}{i}"]
            grp.append(len(rows))
            rows.append((f"t0_{tag}{i}", tuple(t0w), False))
            rows.append((f"zeta_{tag}{i}", tuple(zw), True))
        names.append(f"delta_dm_{tag}")
        rows.append((f"delta_dm_{tag}", tuple(b["ddm"]), False))
        t0_groups.append(grp)
        if b.get("ordered") and len(grp) > 1:
            any_ordered = True
            dt_mins.append(float(b["dt_min"]))
        else:
            dt_mins.append(0.0)  # plain windows already disjoint
    if any_ordered:
        ptform = _JointPriorTransformOrdered(rows, t0_groups, dt_mins)
    else:
        ptform = _JointPriorTransform(rows)
    return tuple(names), rows, ptform


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("burst", choices=sorted(SPEC))
    ap.add_argument("--nlive", type=int, default=160)
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--dlogz", type=float, default=0.5)
    args = ap.parse_args()

    burst = args.burst
    s = SPEC[burst]
    n_C, n_D = len(s["C"]["comps"]), len(s["D"]["comps"])
    suffix = suffix_for(burst)

    for band in ("chime", "dsa"):
        write_config(burst, band, s)
    out = RUNS / "data" / "joint"
    out.mkdir(parents=True, exist_ok=True)

    model_C = prepare(burst, "chime", s)
    model_D = prepare(burst, "dsa", s)
    names, rows, ptform = build_spec(burst)
    loglike = _JointLogLikelihoodGainMulti(model_C, model_D, n_C=n_C, n_D=n_D, s2=None)
    ndim = len(names)
    print(f"[{burst}] ndim={ndim} n_C={n_C} n_D={n_D} nlive={args.nlive} suffix={suffix}", flush=True)
    t0 = _time.time()

    with mp.Pool(args.nproc) as pool:
        sampler = NestedSampler(
            loglike,
            ptform,
            ndim=ndim,
            nlive=args.nlive,
            sample="rwalk",
            queue_size=args.nproc,
            pool=pool,
        )
        sampler.run_nested(dlogz=args.dlogz, print_progress=True)
        res = sampler.results

    samples = np.asarray(res.samples)
    weights = np.exp(np.asarray(res.logwt) - float(res.logz[-1]))
    pct = _weighted_percentiles(samples, weights, names)
    pct = _append_derived_alpha_percentiles(pct, samples, weights, names)

    fit = {
        "burst": burst,
        "fit_note": s["note"] + " Sandbox refit 2026-07-07 (bad-fit remediation).",
        "marginalize_gain": False,
        "marginalize_gain_gp": False,
        "shared_zeta": False,
        "beta": pct["beta"],
        "beta_bounds": [3.0, 4.0],
        "alpha": pct["alpha"],
        "tau_1ghz": pct["tau_1ghz"],
        "log_evidence": float(res.logz[-1]),
        "log_evidence_err": float(res.logzerr[-1]),
        "alpha_bounds": [alpha_from_beta(4.0), alpha_from_beta(3.0)],
        "components_C": n_C,
        "components_D": n_D,
        "component_windows": {name: list(bounds) for name, bounds, _ in rows},
        "gain_s2": None,
        "percentiles": {name: pct[name] for name in names + ("alpha",)},
        "ncall": int(np.sum(res.ncall)),
        "runtime_s": round(_time.time() - t0, 1),
    }
    fit_path = out / f"{burst}_joint_fit{suffix}.json"
    fit_path.write_text(json.dumps(fit, indent=2))
    np.savez_compressed(
        out / f"{burst}_joint_samples{suffix}.npz",
        samples=samples,
        weights=weights,
        param_names=np.asarray(names, dtype=object),
        alpha_bounds=np.asarray(fit["alpha_bounds"]),
        freq_C=np.asarray(model_C.freq),
        freq_D=np.asarray(model_D.freq),
    )
    print(f"wrote {fit_path}  (runtime {fit['runtime_s']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
