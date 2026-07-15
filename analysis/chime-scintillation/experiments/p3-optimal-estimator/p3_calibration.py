#!/usr/bin/env python3
"""Blinded G1″/G2″ calibration of the P3′ matched Δν_d estimator.

Implements experiment P3′ from the predeclared record
``docs/rse/specs/experiment-chime-scint-p3-optimal-estimator.md`` (Faber2026,
§P3′ amendment): the delay-domain matched (optimal quadratic) estimator on
P2's S2 split-ratio spectra, calibrated behind a blinded gate before any
on-pulse statistic exists.

Subcommands enforce the blinding boundary structurally (the P2 pattern):

  freeze     load data, build the scan template bank + calibration-null
             weights, write frozen_config.json (hashed) + scan_assets.npz
  tbattery   T1/T2/T3 at scale on synthetic frames (T4 is baked into the MC
             templates; T5 is resolved by the amendment; T6 is a unit test)
  g1         injection-recovery grid on off-pulse frames relabeled pseudo-on
  g2         evaluation-null campaign (trials-corrected max-z threshold)
  select     aggregate tbattery+G1+G2 -> verdict
  unblind    REFUSES unless the selection is 'calibrated' AND the explicit
             --unblind-i-know-what-i-am-doing flag is passed; orchestrator
             one-shot only.

Frozen seed spaces: G1 ``1000*cell + realization`` (P2 convention);
calibration nulls ``900000+i, i=0..99``; evaluation nulls ``900100..900199``;
scan templates ``750000 + 1000*grid_index + j``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
P2_DIR = HERE.parent / "p2-routeb-voltage"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(P2_DIR))

import routeb_calibration as p2  # noqa: E402  (Products, grid, gate constants)
from scintillation.scint_analysis import optimal_dnu as od  # noqa: E402
from scintillation.scint_analysis import routeb_voltage as rb  # noqa: E402

EXPERIMENT_ID = "p3-optimal-estimator"
RECORD = "docs/rse/specs/experiment-chime-scint-p3-optimal-estimator.md"

# ---- frozen null split (record §P3' amendment item 3) ------------------------
N_NULL_CAL = 100  # seeds 900000 + (0..99): weights + per-dnu mean/sigma
N_NULL_EVAL = 100  # seeds 900100 + (0..99): trials-corrected max-z threshold
EVAL_SEED_BASE = p2.G2_SEED_BASE + N_NULL_CAL
G2_CONSISTENCY_TOL = 0.15  # |p95_cal - p95_eval| <= tol * p95_eval
G2_FWER = 0.05  # threshold percentile on the eval max-z distribution
T3_SIGMA_BAND = (0.8, 1.2)

# ---- G1'' certification (record: >=213 kHz cells must certify) ----------------
DETECTABLE_DNU_KHZ = 213.0

FROZEN = HERE / "frozen_config.json"
ASSETS = HERE / "scan_assets.npz"
UNBLIND_FLAG = "--unblind-i-know-what-i-am-doing"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_products():
    class Args:
        pol0 = p2.DEFAULT_POL0
        pol1 = p2.DEFAULT_POL1
        frequencies = p2.DEFAULT_FREQUENCIES
        time0_metadata = p2.DEFAULT_METADATA

    return p2.Products(Args())


def _null_fields(products, seed: int, *, on_gain=None):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(products.off_pool)
    on = np.sort(perm[: p2.N_ON])
    off = np.sort(perm[p2.N_ON :])
    return od.split_ratio_fields(products.dynamic, on, off, on_gain=on_gain), rng


def _null_power(products, seed: int, *, on_gain=None):
    (f1, f2), rng = _null_fields(products, seed, on_gain=on_gain)
    return od.cross_power(f1, f2), rng


def _build_scan(products) -> od.MatchedScan:
    """Rebuild the frozen MatchedScan from scan_assets.npz."""
    assets = np.load(ASSETS)
    scan = od.MatchedScan(
        assets["dnu_khz"], assets["templates"], assets["variance_smoothed"], kmin=od.KMIN
    )
    scan.null_mean = assets["null_mean"]
    scan.null_sigma = assets["null_sigma"]
    return scan


def _frozen_or_die() -> dict:
    if not FROZEN.exists():
        raise SystemExit("frozen_config.json missing -- run the freeze subcommand first")
    return json.loads(FROZEN.read_text())


def _frozen_sha() -> str:
    return _sha256(FROZEN)


# --------------------------------------------------------------------------- #
# freeze
# --------------------------------------------------------------------------- #
def cmd_freeze(args: argparse.Namespace) -> int:
    products = _load_products()
    channel_width_khz = products.channel_width_mhz * 1e3

    templates = np.vstack(
        [
            od.lorentzian_template(
                dnu,
                gi,
                n_channels=products.n_band_channels,
                channel_width_khz=channel_width_khz,
                good_mask=products.good_channels,
            )
            for gi, dnu in enumerate(od.DNU_SCAN_KHZ)
        ]
    )
    cal_powers = np.array(
        [_null_power(products, p2.G2_SEED_BASE + i)[0] for i in range(N_NULL_CAL)]
    )
    variance = od.smooth_variance(cal_powers.var(axis=0, ddof=1))
    scan = od.MatchedScan(od.DNU_SCAN_KHZ, templates, variance, kmin=od.KMIN)
    scan.calibrate(cal_powers)
    np.savez_compressed(
        ASSETS,
        dnu_khz=od.DNU_SCAN_KHZ,
        templates=templates,
        variance_smoothed=variance,
        null_mean=scan.null_mean,
        null_sigma=scan.null_sigma,
    )

    config = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "record": RECORD,
        "amendment": "P3' (2026-07-15): no block demeaning; kmin=11; null-mean-subtracted z",
        "band_mhz": list(p2.BAND_MHZ),
        "lte_exclusion_mhz": list(p2.LTE_EXCLUSION_MHZ),
        "on_pulse_window_blinded": list(p2.ON_PULSE_WINDOW),
        "off_pool_windows": [list(w) for w in p2.OFF_POOL_WINDOWS],
        "n_band_channels": products.n_band_channels,
        "n_good_channels": int(products.good_channels.sum()),
        "channel_width_khz": channel_width_khz,
        "estimator": {
            "construction": "S2 split-ratio fields (P2), pol-mean halves",
            "transform": "global demean + full-band rfft, DC dropped (P3' amendment)",
            "kmin": od.KMIN,
            "dnu_scan_khz": [float(d) for d in od.DNU_SCAN_KHZ],
            "n_template": od.N_TEMPLATE,
            "template_seed_base": od.TEMPLATE_SEED_BASE,
            "variance_smoothing_bands": od.N_VAR_BANDS,
            "burst_flux_fraction": p2.BURST_FLUX_FRACTION,
            "n_on_samples": p2.N_ON,
        },
        "null_split": {
            "calibration": {"seed_base": p2.G2_SEED_BASE, "n": N_NULL_CAL},
            "evaluation": {"seed_base": EVAL_SEED_BASE, "n": N_NULL_EVAL},
        },
        "injection_grid": {
            "modulations": list(p2.MODULATIONS),
            "dnu_khz": list(p2.DNU_KHZ),
            "control_dnu_khz": p2.CONTROL_DNU_KHZ,
            "detectable_dnu_khz": DETECTABLE_DNU_KHZ,
            "n_realizations": p2.N_REALIZATIONS,
            "seed_formula": "1000*cell_index + realization ; cell_index = m_index*5 + dnu_index",
        },
        "gates": {
            "g1": {
                "detectable_cells_must_certify": ">= 213 kHz (both m)",
                "control_must_not_certify": True,
                "dnu_fractional_tolerance": p2.G1_DNU_TOL,
                "amplitude_pull_max": p2.G1_AMP_PULL_MAX,
                "convergence_min": p2.G1_CONVERGENCE_MIN,
                "recovered_dnu": "argmax z over the 25-point scan grid",
                "amplitude_pull": "(a_hat - a_true)/sigma_null at grid point nearest injection",
            },
            "g2": {
                "threshold": f"p{int((1-G2_FWER)*100)} of eval-null max-z",
                "consistency": f"|p95_cal - p95_eval| <= {G2_CONSISTENCY_TOL} * p95_eval",
                "t3_sigma_band": list(T3_SIGMA_BAND),
            },
            "g3_admissibility_after_unblind": {
                "detection_requires_dnu_in_gate0_window": True,
                "gate0_window_khz": {"m0.17": 77.0, "m0.15": 127.0},
                "significance_min": 5.0,
            },
        },
        "inputs": products.inputs,
        "inputs_sha256": products.inputs_sha256,
        "scan_assets_sha256": _sha256(ASSETS),
        "blinding": {
            "on_pulse_guard": list(rb.ON_PULSE_GUARD),
            "unblind_flag": UNBLIND_FLAG,
            "policy": "no statistic on samples 250-350 without the explicit flag",
        },
    }
    FROZEN.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "frozen_config_sha256": _frozen_sha(),
                "n_good_channels": config["n_good_channels"],
                "sigma_analytic_at_213": float(
                    scan.sigma_analytic[scan.nearest_grid_index(213.0)]
                ),
                "null_sigma_at_213": float(scan.null_sigma[scan.nearest_grid_index(213.0)]),
            },
            sort_keys=True,
        )
    )
    return 0


# --------------------------------------------------------------------------- #
# T battery at scale (synthetic; T4 is inside the MC templates; T5 resolved
# by the amendment; T6 is unit-tested in test_optimal_dnu.py)
# --------------------------------------------------------------------------- #
def cmd_tbattery(args: argparse.Namespace) -> int:
    _frozen_or_die()
    products = _load_products()
    scan = _build_scan(products)
    n = products.n_band_channels
    rng = np.random.default_rng(4242)
    n_times = 437
    # synthetic frames still respect the structural blinding guard, so the
    # windows mirror the real off-pool geometry (no sample in [250, 350))
    on = np.arange(10, 110)
    off = np.concatenate([np.arange(120, 200), np.arange(360, n_times)])

    # T1: bandpass invariance at full scale with the measured-shape gain
    dynamic = [1.0 + 0.05 * rng.standard_normal((n, n_times)) for _ in range(2)]
    gain = 1.0 + 0.586 * rb.lorentzian_gain_field(rng, n_channels=n, width_channels=5.8)
    gain = np.abs(gain) + 0.05
    base = od.split_ratio_fields(dynamic, on, off)
    scaled = od.split_ratio_fields([d * gain[:, None] for d in dynamic], on, off)
    t1_dev = float(max(np.nanmax(np.abs(a - b)) for a, b in zip(base, scaled)))

    # T2/T3: unbiasedness and sigma calibration on pure synthetic at the real
    # per-sample noise scale (matched to the real off-pulse fractional RMS)
    frac_rms = float(
        np.nanmedian(
            [np.nanmedian(p2._row_nanstd(d[:, off]) / np.nanmedian(d[:, off])) for d in products.dynamic]
        )
    )
    m, dnu = 0.17, 213.0
    grid = scan.nearest_grid_index(dnu)
    a_true = (p2.BURST_FLUX_FRACTION * m) ** 2
    width_channels = dnu / (products.channel_width_mhz * 1e3)
    null_powers, sig_powers = [], []
    for j in range(100):
        srng = np.random.default_rng(5_000_000 + j)
        dyn = [1.0 + frac_rms * srng.standard_normal((n, n_times)) for _ in range(2)]
        null_powers.append(od.cross_power(*od.split_ratio_fields(dyn, on, off)))
        delta = rb.lorentzian_gain_field(srng, n_channels=n, width_channels=width_channels)
        inj = 1.0 + p2.BURST_FLUX_FRACTION * (1.0 + m * delta)
        sig_powers.append(
            od.cross_power(*od.split_ratio_fields(dyn, on, off, on_gain=inj))
        )
    null_powers = np.asarray(null_powers)
    # weights and null mean/sigma from the first synthetic-null half; the
    # second half is the independent evaluation set (mirrors the real-data
    # calibration/evaluation split of G2'')
    syn_scan = od.MatchedScan(
        scan.dnu_khz,
        scan.templates,
        od.smooth_variance(null_powers[:50].var(axis=0, ddof=1)),
        kmin=od.KMIN,
    )
    syn_scan.calibrate(null_powers[:50])
    recovered = np.array([syn_scan.amplitudes(p)[grid] for p in sig_powers])
    recovered -= syn_scan.null_mean[grid]
    stderr = recovered.std(ddof=1) / np.sqrt(recovered.size)
    t2_bias_se = float((recovered.mean() - a_true) / stderr)
    # T3: predicted null error (calibration half) vs the empirical scatter on
    # the independent evaluation half.  Signal-present scatter is recorded as
    # informative only — it legitimately exceeds the null error by the
    # scintle self-noise.
    a_eval = np.array([syn_scan.amplitudes(p)[grid] for p in null_powers[50:]])
    t3_ratio = float(a_eval.std(ddof=1) / syn_scan.null_sigma[grid])
    signal_scatter_ratio = float(recovered.std(ddof=1) / syn_scan.null_sigma[grid])

    result = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": _frozen_sha(),
        "t1_max_deviation": t1_dev,
        "t1_pass": bool(t1_dev < 1e-10),
        "t2_bias_over_stderr": t2_bias_se,
        "t2_pass": bool(abs(t2_bias_se) <= 4.0),
        "t3_sigma_ratio": t3_ratio,
        "t3_pass": bool(T3_SIGMA_BAND[0] <= t3_ratio <= T3_SIGMA_BAND[1]),
        "signal_scatter_over_null_sigma": signal_scatter_ratio,
        "t4_note": "demeaning/mask transfer baked into MC templates by construction",
        "t5_note": "resolved by the P3' amendment (full-band, no demeaning)",
        "t6_note": "unit-tested (test_optimal_dnu.py::test_t6_blinding_guard)",
        "synthetic_fractional_rms": frac_rms,
    }
    result["tbattery_pass"] = bool(result["t1_pass"] and result["t2_pass"] and result["t3_pass"])
    (HERE / "tbattery.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("t1_pass", "t2_pass", "t3_pass", "tbattery_pass", "t2_bias_over_stderr", "t3_sigma_ratio")}, sort_keys=True))
    return 0 if result["tbattery_pass"] else 2


# --------------------------------------------------------------------------- #
# G1'' injection recovery
# --------------------------------------------------------------------------- #
def cmd_g1(args: argparse.Namespace) -> int:
    frozen = _frozen_or_die()
    products = _load_products()
    scan = _build_scan(products)
    channel_width_khz = products.channel_width_mhz * 1e3
    cells = []
    for m_index, m in enumerate(p2.MODULATIONS):
        for dnu_index, dnu_khz in enumerate(p2.DNU_KHZ):
            cell_index = m_index * len(p2.DNU_KHZ) + dnu_index
            grid = scan.nearest_grid_index(dnu_khz)
            a_true = (p2.BURST_FLUX_FRACTION * m) ** 2
            recovered_dnu, pulls, zmax = [], [], []
            for realization in range(p2.N_REALIZATIONS):
                seed = p2.SEED_MULTIPLIER * cell_index + realization
                rng = np.random.default_rng(seed)
                perm = rng.permutation(products.off_pool)
                on = np.sort(perm[: p2.N_ON])
                off = np.sort(perm[p2.N_ON :])
                delta = rb.lorentzian_gain_field(
                    rng,
                    n_channels=products.n_band_channels,
                    width_channels=dnu_khz / channel_width_khz,
                )
                gain = 1.0 + p2.BURST_FLUX_FRACTION * (1.0 + m * delta)
                f1, f2 = od.split_ratio_fields(products.dynamic, on, off, on_gain=gain)
                result = scan.zscan(od.cross_power(f1, f2))
                recovered_dnu.append(result["dnu_khz_argmax"])
                pulls.append(
                    (result["a_hat"][grid] - scan.null_mean[grid] - a_true)
                    / scan.null_sigma[grid]
                )
                zmax.append(result["z_max"])
            recovered_dnu = np.asarray(recovered_dnu)
            pulls = np.asarray(pulls)
            finite = np.isfinite(recovered_dnu) & np.isfinite(pulls)
            convergence = float(finite.mean())
            median_dnu = float(np.median(recovered_dnu[finite])) if finite.any() else np.nan
            bias = median_dnu / dnu_khz - 1.0 if np.isfinite(median_dnu) else np.nan
            median_pull = float(np.median(pulls[finite])) if finite.any() else np.nan
            certify = bool(
                convergence >= p2.G1_CONVERGENCE_MIN
                and np.isfinite(bias)
                and abs(bias) <= p2.G1_DNU_TOL
                and np.isfinite(median_pull)
                and abs(median_pull) <= p2.G1_AMP_PULL_MAX
            )
            summary = {
                "cell_index": cell_index,
                "modulation": m,
                "dnu_khz": dnu_khz,
                "amplitude_true": a_true,
                "convergence": convergence,
                "median_recovered_dnu_khz": median_dnu,
                "dnu_fractional_bias": bias,
                "median_amplitude_pull": median_pull,
                "median_z_max": float(np.median(np.asarray(zmax)[finite])) if finite.any() else np.nan,
                "recovered_dnu_khz": recovered_dnu.tolist(),
                "pulls": pulls.tolist(),
                "is_control": dnu_khz == p2.CONTROL_DNU_KHZ,
                "is_detectable_gate": dnu_khz >= DETECTABLE_DNU_KHZ,
                "certify": certify,
            }
            cells.append(summary)
            print(
                json.dumps(
                    {
                        "m": m,
                        "dnu_khz": dnu_khz,
                        "certify": certify,
                        "median_dnu_khz": None if not np.isfinite(median_dnu) else round(median_dnu, 1),
                        "bias": None if not np.isfinite(bias) else round(bias, 3),
                        "pull": None if not np.isfinite(median_pull) else round(median_pull, 2),
                        "median_z_max": None if not finite.any() else round(summary["median_z_max"], 2),
                    }
                ),
                flush=True,
            )
    detectable = [c for c in cells if c["is_detectable_gate"]]
    control = [c for c in cells if c["is_control"]]
    g1_pass = bool(all(c["certify"] for c in detectable) and not any(c["certify"] for c in control))
    result = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": _frozen_sha(),
        "g1_pass": g1_pass,
        "control_certifies": bool(any(c["certify"] for c in control)),
        "detectable_all_certify": bool(all(c["certify"] for c in detectable)),
        "cells": cells,
        "gates": frozen["gates"]["g1"],
    }
    (HERE / "g1_matched.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"g1_pass": g1_pass, "control_certifies": result["control_certifies"]}, sort_keys=True))
    return 0 if g1_pass else 2


# --------------------------------------------------------------------------- #
# G2'' evaluation-null campaign
# --------------------------------------------------------------------------- #
def cmd_g2(args: argparse.Namespace) -> int:
    frozen = _frozen_or_die()
    products = _load_products()
    scan = _build_scan(products)
    grid213 = scan.nearest_grid_index(213.0)

    def max_z_set(seed_base: int, n: int):
        maxima, a213 = [], []
        for i in range(n):
            power, _ = _null_power(products, seed_base + i)
            result = scan.zscan(power)
            maxima.append(result["z_max"])
            a213.append(result["a_hat"][grid213])
        return np.asarray(maxima), np.asarray(a213)

    cal_max, _ = max_z_set(p2.G2_SEED_BASE, N_NULL_CAL)
    eval_max, eval_a213 = max_z_set(EVAL_SEED_BASE, N_NULL_EVAL)
    p95_cal = float(np.percentile(cal_max, 100 * (1 - G2_FWER)))
    p95_eval = float(np.percentile(eval_max, 100 * (1 - G2_FWER)))
    consistency = abs(p95_cal - p95_eval) <= G2_CONSISTENCY_TOL * p95_eval
    t3_ratio = float(eval_a213.std(ddof=1) / scan.null_sigma[grid213])
    t3_ok = T3_SIGMA_BAND[0] <= t3_ratio <= T3_SIGMA_BAND[1]
    g2_pass = bool(consistency and t3_ok)
    result = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": _frozen_sha(),
        "n_cal": N_NULL_CAL,
        "n_eval": N_NULL_EVAL,
        "fwer": G2_FWER,
        "z_trials_threshold": p95_eval,
        "p95_cal": p95_cal,
        "p95_eval": p95_eval,
        "consistency_pass": bool(consistency),
        "eval_sigma_ratio_at_213": t3_ratio,
        "t3_pass": bool(t3_ok),
        "g2_pass": g2_pass,
        "cal_max_z": cal_max.tolist(),
        "eval_max_z": eval_max.tolist(),
        "gates": frozen["gates"]["g2"],
    }
    (HERE / "g2_matched.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "g2_pass": g2_pass,
                "z_trials_threshold": round(p95_eval, 3),
                "p95_cal": round(p95_cal, 3),
                "eval_sigma_ratio_at_213": round(t3_ratio, 3),
            },
            sort_keys=True,
        )
    )
    return 0 if g2_pass else 2


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #
def cmd_select(args: argparse.Namespace) -> int:
    _frozen_or_die()
    current = _frozen_sha()
    parts = {}
    for name in ("tbattery", "g1_matched", "g2_matched"):
        path = HERE / f"{name}.json"
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("frozen_config_sha256") == current:
                parts[name] = data
    ok = (
        parts.get("tbattery", {}).get("tbattery_pass") is True
        and parts.get("g1_matched", {}).get("g1_pass") is True
        and parts.get("g2_matched", {}).get("g2_pass") is True
    )
    verdict = "calibrated" if ok else "DOCUMENTED-FAIL"
    result = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": current,
        "verdict": verdict,
        "tbattery_pass": parts.get("tbattery", {}).get("tbattery_pass"),
        "g1_pass": parts.get("g1_matched", {}).get("g1_pass"),
        "g2_pass": parts.get("g2_matched", {}).get("g2_pass"),
        "note": (
            "verdict 'calibrated' means the estimator may be unblinded once, by the "
            "orchestrator, per the record's unblinding rule; 'DOCUMENTED-FAIL' is a "
            "terminal, reportable outcome with no unblinding."
        ),
    }
    (HERE / "selection.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict}, sort_keys=True))
    return 0 if ok else 2


# --------------------------------------------------------------------------- #
# unblind (orchestrator-only)
# --------------------------------------------------------------------------- #
def cmd_unblind(args: argparse.Namespace) -> int:
    if not args.unblind_i_know_what_i_am_doing:
        raise SystemExit(
            f"REFUSED: unblinding computes the on-pulse (250-350) statistic. "
            f"Pass {UNBLIND_FLAG} only for the sanctioned one-shot orchestrator step."
        )
    _frozen_or_die()
    selection_path = HERE / "selection.json"
    if not selection_path.exists():
        raise SystemExit("selection.json missing -- run select first; staying blind")
    selection = json.loads(selection_path.read_text())
    if selection["frozen_config_sha256"] != _frozen_sha():
        raise SystemExit("frozen_config.json changed after selection -- new experiment required")
    if selection.get("verdict") != "calibrated":
        raise SystemExit("selection verdict is DOCUMENTED-FAIL -- the on-pulse fit stays blind")
    g2 = json.loads((HERE / "g2_matched.json").read_text())
    products = _load_products()
    scan = _build_scan(products)
    on = rb.samples_from_window(p2.ON_PULSE_WINDOW)
    f1, f2 = od.split_ratio_fields(
        products.dynamic, on, products.off_pool, allow_unblind=True
    )
    result = scan.zscan(od.cross_power(f1, f2))
    payload = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": _frozen_sha(),
        "z_trials_threshold": g2["z_trials_threshold"],
        "onpulse": {
            "z_max": result["z_max"],
            "dnu_khz_argmax": result["dnu_khz_argmax"],
            "z_by_dnu": {str(float(d)): float(z) for d, z in zip(scan.dnu_khz, result["z"])},
            "a_hat_by_dnu": {
                str(float(d)): float(a) for d, a in zip(scan.dnu_khz, result["a_hat"])
            },
        },
        "g3": {
            "gate0_window_khz": {"m0.17": 77.0, "m0.15": 127.0},
            "significance_min": 5.0,
        },
        "warning": "on-pulse window read under explicit unblind flag",
    }
    (HERE / "unblind_onpulse.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"z_max": result["z_max"], "dnu_khz_argmax": result["dnu_khz_argmax"], "z_trials_threshold": g2["z_trials_threshold"]}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "tbattery", "g1", "g2", "select"):
        sub.add_parser(name)
    up = sub.add_parser("unblind")
    up.add_argument(UNBLIND_FLAG, dest="unblind_i_know_what_i_am_doing", action="store_true")
    args = parser.parse_args()
    return {
        "freeze": cmd_freeze,
        "tbattery": cmd_tbattery,
        "g1": cmd_g1,
        "g2": cmd_g2,
        "select": cmd_select,
        "unblind": cmd_unblind,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
