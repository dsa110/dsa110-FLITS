#!/usr/bin/env python3
"""Blinded G1/G2 calibration of the Route-B common-mode-immune Δν_d statistics.

Implements experiment P2 (Route B) from the predeclared record
``docs/rse/specs/experiment-chime-scint-routeb-voltage.md`` (Faber2026): the
S1/S2/S3 on/off-ratio cross-ACF statistics whose ratio construction cancels the
instrumental common mode ``g(ν)`` algebraically, calibrated behind a blinded
gate before any on-pulse statistic exists.

Subcommands enforce the blinding boundary structurally:

  freeze            prove data load + windows, write frozen_config.json (hashed)
  g1 --statistic S  injection-recovery grid on off-pulse frames relabeled
                    pseudo-on (never reads the on-pulse window)
  g2 --statistic S  >=24 off-pulse-only pseudo-on null realizations at the
                    operating point (pseudo-on window disjoint from reference)
  select            aggregate G1/G2 -> pick the statistic by calibration score
  unblind           REFUSES unless G1+G2 pass AND the explicit
                    --unblind-i-know-what-i-am-doing flag is passed; only then
                    does the on-pulse (250-350) statistic get computed.  This is
                    the orchestrator's one-shot step; the builder never runs it.

No code path outside `unblind` reads a sample in the on-pulse window, and the
Route-B statistics refuse such a window unless allow_unblind is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from scipy.stats import norm as _norm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from scintillation.scint_analysis import routeb_voltage as rb  # noqa: E402

DATA = Path.home() / "Data/Faber2026/dsa110/upchan_codetections/crossacf-2026-07-14"
DEFAULT_POL0 = DATA / "freya_chime_pol0_upchan.npy"
DEFAULT_POL1 = DATA / "freya_chime_pol1_upchan.npy"
DEFAULT_FREQUENCIES = DATA / "freya_chime_freq.npy"
DEFAULT_METADATA = DATA / "freya_crossacf_metadata.json"

EXPERIMENT_ID = "p2-routeb-voltage"

# ---- frozen product geometry (matches the common-mode research record) -------
BAND_MHZ = (627.0, 800.0)
LTE_EXCLUSION_MHZ = (730.0, 760.0)
ON_PULSE_WINDOW = (250, 350)  # blinded numerator window; never read while blind
# Off-pulse sample pool: pre-burst 10-200 and post-burst remainder (10-sample
# guard after the 350 on-window edge to avoid burst-tail bleed).
OFF_POOL_WINDOWS = ((10, 200), (360, 437))
CHANNELS_PER_COARSE = 64

# ---- frozen estimator hyperparameters ---------------------------------------
N_ON = 100  # pseudo-on samples per G1 realization (matches the 250-350 width)
MAX_LAG_BINS = 48  # <64 so every lag stays inside a coarse block (defined lag)
FIRST_LAG_BIN = 2  # lag-0 excluded structurally; drop lag-1 too (noise spike)
FIT_MAX_MHZ = 0.50  # Lorentzian HWHM upper bound (brackets the 352 kHz cell)
BURST_FLUX_FRACTION = 0.05  # f_b: measured on/off mean contrast (1.05)

# ---- frozen injection grid (verbatim from the record) -----------------------
MODULATIONS = (0.15, 0.17)
DNU_KHZ = (35.0, 77.0, 127.0, 213.0, 352.0)  # 35 = must-fail control
CONTROL_DNU_KHZ = 35.0
DETECTABLE_DNU_KHZ = 127.0  # cells with Δν_d >= this must certify for G1 PASS
N_REALIZATIONS = 50
# seed = 1000 * cell_index + realization ; cell_index = m_index*len(DNU)+dnu_index
SEED_MULTIPLIER = 1000

# ---- frozen gates (verbatim from the record) --------------------------------
G1_DNU_TOL = 0.30  # median recovered Δν_d within +-30% of injected
G1_AMP_PULL_MAX = 2.0  # |median amplitude pull| <= 2
G1_CONVERGENCE_MIN = 0.90  # >= 90% of realizations converged
G2_N_WINDOWS = 24  # >= 24 off-pulse-only pseudo-on null realizations
G2_FWER = 0.05  # Šidák family-wise error rate
G2_SEED_BASE = 900_000  # null-realization seeds, disjoint from the G1 seed space
# Amplitude below this is the fit railing to zero -> the strongest null, not a
# detection (its modulation error is undefined only because m -> 0).
AMPLITUDE_NULL_FLOOR = 1e-10

CHANNEL_WIDTH_KHZ = 6.103608758678547  # median fine-channel spacing (record)

FROZEN = HERE / "frozen_config.json"
UNBLIND_FLAG = "--unblind-i-know-what-i-am-doing"


# --------------------------------------------------------------------------- #
# small self-contained helpers (kept local so the harness has no analysis-dir
# import coupling; the channel mask matches validate_freya_highband_crossacf)
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating | float):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return np.nan
    return 1.4826 * float(np.median(np.abs(finite - np.median(finite))))


def _row_nanstd(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = rb.row_nanmean(values)
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    squared = np.where(finite, (values - mean[:, None]) ** 2, 0.0).sum(axis=1)
    return np.sqrt(np.divide(squared, count, out=np.full(values.shape[0], np.nan), where=count > 0))


def _channel_mask(power_by_pol: list[np.ndarray], frequencies: np.ndarray) -> np.ndarray:
    """Good-channel mask: finite positive gain, sane fractional RMS, no LTE.

    Matches ``validate_freya_highband_crossacf._channel_mask`` so Route B keeps
    the same channel population as the earlier routes.
    """
    good = np.ones(frequencies.size, dtype=bool)
    lo, hi = OFF_POOL_WINDOWS[0]
    for power in power_by_pol:
        off = np.asarray(power[:, lo:hi], dtype=float)
        gain = np.nanmedian(off, axis=1)
        fractional_rms = _row_nanstd(off) / gain
        center = float(np.nanmedian(fractional_rms))
        scale = _mad(fractional_rms)
        good &= np.isfinite(gain) & (gain > 0) & np.isfinite(fractional_rms)
        if np.isfinite(scale) and scale > 0:
            good &= np.abs(fractional_rms - center) <= 5.0 * scale
    good &= ~((frequencies >= LTE_EXCLUSION_MHZ[0]) & (frequencies <= LTE_EXCLUSION_MHZ[1]))
    return good


def _off_pool_indices() -> np.ndarray:
    return np.concatenate([np.arange(lo, hi, dtype=int) for lo, hi in OFF_POOL_WINDOWS])


def _cell_index(m_index: int, dnu_index: int) -> int:
    return m_index * len(DNU_KHZ) + dnu_index


def _width_channels(dnu_khz: float) -> float:
    return float(dnu_khz) / CHANNEL_WIDTH_KHZ


def _sidak_z(n_comparisons: int, fwer: float) -> float:
    per_comparison = 1.0 - (1.0 - fwer) ** (1.0 / n_comparisons)
    return float(_norm.isf(per_comparison / 2.0))  # two-sided


# --------------------------------------------------------------------------- #
# product load (raw detected intensity; NO common-mode correction: the ratio
# statistic removes g(ν) algebraically, so pre-correcting would corrupt it)
# --------------------------------------------------------------------------- #
class Products:
    def __init__(self, args: argparse.Namespace) -> None:
        frequencies_full = np.load(args.frequencies)
        select = (frequencies_full >= BAND_MHZ[0]) & (frequencies_full <= BAND_MHZ[1])
        self.frequencies = np.asarray(frequencies_full[select], dtype=float)
        power = [
            np.asarray(np.load(path, mmap_mode="r")[select], dtype=float)
            for path in (args.pol0, args.pol1)
        ]
        self.n_band_channels = int(self.frequencies.size)
        self.good_channels = _channel_mask(power, self.frequencies)
        # RFI/LTE/bad channels -> NaN so the cross-ACF skips them.
        self.dynamic = []
        for item in power:
            masked = np.array(item, dtype=float, copy=True)
            masked[~self.good_channels] = np.nan
            self.dynamic.append(masked)
        metadata = json.loads(Path(args.time0_metadata).read_text())
        coarse = np.asarray(metadata["freq_mhz"], dtype=float)
        # parent coarse channel per fine channel -> block ids for demeaning and
        # for keeping every ACF lag inside one contiguous 6.1 kHz-spaced block.
        self.parent = np.argmin(np.abs(self.frequencies[:, None] - coarse[None, :]), axis=1)
        self.channel_width_mhz = float(np.nanmedian(np.diff(self.frequencies)))
        self.off_pool = _off_pool_indices()
        self.metadata = metadata
        self.inputs = {
            "pol0": str(args.pol0),
            "pol1": str(args.pol1),
            "frequencies": str(args.frequencies),
            "metadata": str(args.time0_metadata),
        }
        self.inputs_sha256 = {
            "pol0": _sha256(args.pol0),
            "pol1": _sha256(args.pol1),
            "frequencies": _sha256(args.frequencies),
            "metadata": _sha256(args.time0_metadata),
        }

    def statistic(self, name: str):
        return rb.STATISTICS[name]

    def run_statistic(self, name, on_samples, off_samples, *, on_gain=None, fit_max_mhz=FIT_MAX_MHZ):
        return self.statistic(name)(
            self.dynamic,
            on_samples,
            off_samples,
            self.parent,
            channel_width_mhz=self.channel_width_mhz,
            on_gain=on_gain,
            max_lag_bins=MAX_LAG_BINS,
            first_lag_bin=FIRST_LAG_BIN,
            fit_max_mhz=fit_max_mhz,
            block_length=CHANNELS_PER_COARSE,
        )


# --------------------------------------------------------------------------- #
# G1 injection recovery
# --------------------------------------------------------------------------- #
def _inject_realization(products: Products, statistic: str, m: float, dnu_khz: float, seed: int):
    """One G1 realization: split the off pool into pseudo-on/reference subsets,
    inject a multiplicative scintillated-burst gain, run the statistic + fit."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(products.off_pool)
    on_samples = np.sort(perm[:N_ON])
    off_samples = np.sort(perm[N_ON:])
    delta = rb.lorentzian_gain_field(
        rng, n_channels=products.n_band_channels, width_channels=_width_channels(dnu_khz)
    )
    # A real scintillated burst rides on the same common mode as the background,
    # so multiply the g-bearing off-pulse pseudo-on frame by the burst gain; the
    # ratio then still cancels g and leaves f_b*m*delta as the recoverable signal.
    gain = 1.0 + BURST_FLUX_FRACTION * (1.0 + m * delta)
    result = products.run_statistic(statistic, on_samples, off_samples, on_gain=gain)
    return result.fit


def _amplitude_error(fit: dict) -> float:
    # amp = m^2 -> sigma_amp = 2*m*sigma_m (fit reports m and m_err, not amp_err)
    m = fit.get("m", np.nan)
    m_err = fit.get("m_err", np.nan)
    if not (np.isfinite(m) and np.isfinite(m_err)):
        return np.nan
    return 2.0 * m * m_err


def _cell_summary(records, m, dnu_khz):
    truth_mhz = dnu_khz / 1000.0
    amp_true = BURST_FLUX_FRACTION * m  # this is m_true on the ratio (= sqrt(A_true))
    amp_true = amp_true**2
    converged = [
        r
        for r in records
        if r["fit"] is not None
        and np.isfinite(r["fit"].get("dnu_err_mhz", np.nan))
        and _amplitude_error(r["fit"]) > 0
        and np.isfinite(_amplitude_error(r["fit"]))
    ]
    convergence = len(converged) / len(records)
    if converged:
        recovered = np.array([r["fit"]["dnu_mhz"] for r in converged])
        median_dnu = float(np.median(recovered))
        dnu_ratio_bias = median_dnu / truth_mhz - 1.0
        pulls = np.array(
            [(r["fit"]["amplitude"] - amp_true) / _amplitude_error(r["fit"]) for r in converged]
        )
        median_pull = float(np.median(pulls))
    else:
        median_dnu = np.nan
        dnu_ratio_bias = np.nan
        median_pull = np.nan
        pulls = np.array([])
    certify = bool(
        convergence >= G1_CONVERGENCE_MIN
        and np.isfinite(dnu_ratio_bias)
        and abs(dnu_ratio_bias) <= G1_DNU_TOL
        and np.isfinite(median_pull)
        and abs(median_pull) <= G1_AMP_PULL_MAX
    )
    return {
        "modulation": m,
        "dnu_khz": dnu_khz,
        "injected_dnu_mhz": truth_mhz,
        "amplitude_true": amp_true,
        "n_realizations": len(records),
        "n_converged": len(converged),
        "convergence": convergence,
        "median_recovered_dnu_mhz": median_dnu,
        "dnu_fractional_bias": dnu_ratio_bias,
        "median_amplitude_pull": median_pull,
        "pulls": pulls.tolist(),
        "recovered_dnu_mhz": [r["fit"]["dnu_mhz"] for r in converged],
        "is_control": dnu_khz == CONTROL_DNU_KHZ,
        "is_detectable_gate": dnu_khz >= DETECTABLE_DNU_KHZ,
        "certify": certify,
    }


def cmd_g1(args: argparse.Namespace) -> int:
    frozen = _frozen_or_die()
    products = Products(args)
    statistic = args.statistic
    cells = []
    for m_index, m in enumerate(MODULATIONS):
        for dnu_index, dnu_khz in enumerate(DNU_KHZ):
            cell_index = _cell_index(m_index, dnu_index)
            records = []
            for realization in range(N_REALIZATIONS):
                seed = SEED_MULTIPLIER * cell_index + realization
                fit = _inject_realization(products, statistic, m, dnu_khz, seed)
                records.append({"realization": realization, "seed": seed, "fit": fit})
            summary = _cell_summary(records, m, dnu_khz)
            summary["cell_index"] = cell_index
            cells.append(summary)
            print(
                json.dumps(
                    {
                        "statistic": statistic,
                        "m": m,
                        "dnu_khz": dnu_khz,
                        "certify": summary["certify"],
                        "convergence": round(summary["convergence"], 3),
                        "median_dnu_khz": None
                        if not np.isfinite(summary["median_recovered_dnu_mhz"])
                        else round(summary["median_recovered_dnu_mhz"] * 1e3, 1),
                        "dnu_bias": None
                        if not np.isfinite(summary["dnu_fractional_bias"])
                        else round(summary["dnu_fractional_bias"], 3),
                        "amp_pull": None
                        if not np.isfinite(summary["median_amplitude_pull"])
                        else round(summary["median_amplitude_pull"], 2),
                    }
                ),
                flush=True,
            )

    detectable = [c for c in cells if c["is_detectable_gate"]]
    control = [c for c in cells if c["is_control"]]
    g1_pass = bool(
        all(c["certify"] for c in detectable) and not any(c["certify"] for c in control)
    )
    # calibration score: mean recovery accuracy over the detectable cells,
    # only meaningful when the statistic passes G1 (else null).
    if g1_pass:
        score = float(
            np.mean([1.0 - min(1.0, abs(c["dnu_fractional_bias"])) for c in detectable])
        )
    else:
        score = None
    result = {
        "experiment": EXPERIMENT_ID,
        "statistic": statistic,
        "frozen_config_sha256": _frozen_sha(),
        "burst_flux_fraction": BURST_FLUX_FRACTION,
        "g1_pass": g1_pass,
        "calibration_score": score,
        "control_certifies": bool(any(c["certify"] for c in control)),
        "detectable_all_certify": bool(all(c["certify"] for c in detectable)),
        "cells": cells,
        "gates": frozen["gates"]["g1"],
    }
    out = HERE / f"g1_{statistic}.json"
    out.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "statistic": statistic,
                "g1_pass": g1_pass,
                "calibration_score": score,
                "control_certifies": result["control_certifies"],
            },
            sort_keys=True,
        )
    )
    return 0 if g1_pass else 2


# --------------------------------------------------------------------------- #
# G2 null campaign
# --------------------------------------------------------------------------- #
def _null_detection(fit: dict | None) -> tuple[float | None, bool]:
    """(amplitude z, is-detection) for one off-pulse null window.

    An amplitude railed to ~0 is the strongest null (its modulation error is
    undefined only because m -> 0), so it is not a detection.  A bound-clear
    non-zero amplitude with an unusable error IS fail-closed to a detection.
    """
    if fit is None:
        return None, True  # cannot evaluate -> fail closed
    amp = float(fit["amplitude"])
    if amp <= AMPLITUDE_NULL_FLOOR:
        return 0.0, False  # amplitude at the floor: cleanest possible null
    amp_err = _amplitude_error(fit)
    if not (np.isfinite(amp_err) and amp_err > 0):
        return None, True  # positive amplitude, unusable error -> fail closed
    return float(amp / amp_err), None  # sentinel None -> decided by threshold below


def cmd_g2(args: argparse.Namespace) -> int:
    frozen = _frozen_or_die()
    products = Products(args)
    statistic = args.statistic
    pool = products.off_pool
    # >=24 off-pulse-only null realizations at the G1 operating point: each is a
    # seeded permutation split of the off pool into an n_on pseudo-on window and
    # a disjoint reference (pseudo-on ⟂ reference), with NO injection.  This
    # matches the G1 injection S/N; a strict-disjoint partition of the 267-sample
    # pool would force ~11-sample windows whose low-count Lorentzian fits invent
    # spurious features (an artifact, not a real detection).
    n = G2_N_WINDOWS
    z_crit = _sidak_z(n, G2_FWER)
    records = []
    for index in range(n):
        rng = np.random.default_rng(G2_SEED_BASE + index)
        perm = rng.permutation(pool)
        on_samples = np.sort(perm[:N_ON])
        off_samples = np.sort(perm[N_ON:])
        result = products.run_statistic(statistic, on_samples, off_samples, on_gain=None)
        fit = result.fit
        z, decided = _null_detection(fit)
        if decided is None:  # threshold decides
            detection = bool(abs(z) > z_crit)
        else:
            detection = decided
        records.append(
            {
                "seed": G2_SEED_BASE + index,
                "n_on": N_ON,
                "n_off": int(off_samples.size),
                "amplitude": None if fit is None else fit["amplitude"],
                "amplitude_z": z,
                "dnu_mhz": None if fit is None else fit["dnu_mhz"],
                "detection": detection,
            }
        )
    z_values = [r["amplitude_z"] for r in records if r["amplitude_z"] is not None]
    detections = [r for r in records if r["detection"]]
    g2_pass = bool(len(z_values) == len(records) and not detections)
    result = {
        "experiment": EXPERIMENT_ID,
        "statistic": statistic,
        "frozen_config_sha256": _frozen_sha(),
        "n_windows": n,
        "family_wise_error_rate": G2_FWER,
        "sidak_z_threshold": z_crit,
        "max_abs_z": max((abs(z) for z in z_values), default=None),
        "n_detections": len(detections),
        "g2_pass": g2_pass,
        "records": records,
        "gates": frozen["gates"]["g2"],
    }
    out = HERE / f"g2_{statistic}.json"
    out.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "statistic": statistic,
                "g2_pass": g2_pass,
                "max_abs_z": result["max_abs_z"],
                "sidak_z_threshold": round(z_crit, 3),
                "n_detections": len(detections),
            },
            sort_keys=True,
        )
    )
    return 0 if g2_pass else 2


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
TIE_BREAK = ("S2", "S1", "S3")  # bias immunity preferred over raw sensitivity


def cmd_select(args: argparse.Namespace) -> int:
    _frozen_or_die()
    current = _frozen_sha()
    g1 = {}
    g2 = {}
    for statistic in ("S1", "S2", "S3"):
        g1_path = HERE / f"g1_{statistic}.json"
        g2_path = HERE / f"g2_{statistic}.json"
        if g1_path.exists():
            data = json.loads(g1_path.read_text())
            if data.get("frozen_config_sha256") == current:
                g1[statistic] = data
        if g2_path.exists():
            data = json.loads(g2_path.read_text())
            if data.get("frozen_config_sha256") == current:
                g2[statistic] = data

    candidates = []
    for statistic, data in g1.items():
        g2data = g2.get(statistic)
        passes = bool(data["g1_pass"] and g2data is not None and g2data["g2_pass"])
        candidates.append(
            {
                "statistic": statistic,
                "g1_pass": data["g1_pass"],
                "g2_pass": None if g2data is None else g2data["g2_pass"],
                "calibration_score": data["calibration_score"],
                "qualifies": passes,
            }
        )

    qualifying = [c for c in candidates if c["qualifies"]]

    def sort_key(candidate):
        score = candidate["calibration_score"] or 0.0
        return (-score, TIE_BREAK.index(candidate["statistic"]))

    ranked = sorted(qualifying, key=sort_key)
    selected = ranked[0]["statistic"] if ranked else None
    verdict = "calibrated" if selected else "DOCUMENTED-FAIL"
    result = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": current,
        "selected_statistic": selected,
        "selection_rule": "best G1 calibration score; ties break S2 -> S1 -> S3",
        "verdict": verdict,
        "candidates": candidates,
        "s3_status": (
            "run"
            if "S3" in g1
            else (
                "not_run: h17 reachable but the complex fine-channel voltage products "
                "S3 requires are not staged (only detected per-pol power exists); "
                "regenerating them via the P1 voltage worker is out of P2 scope. S1 "
                "already operates on the detected per-pol |V_p|^2; S3 adds only the "
                "P1 grouped-bin noise normalization on complex voltages."
            )
        ),
        "note": (
            "verdict 'calibrated' means G1+G2 passed for the selected statistic; "
            "the on-pulse fit remains blinded and is the orchestrator's one-shot "
            "unblind step. 'DOCUMENTED-FAIL' is a terminal, reportable outcome."
        ),
    }
    (HERE / "selection.json").write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"selected_statistic": selected, "verdict": verdict}, sort_keys=True))
    return 0 if selected else 2


# --------------------------------------------------------------------------- #
# freeze
# --------------------------------------------------------------------------- #
def _frozen_config(products: Products) -> dict:
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "record": "docs/rse/specs/experiment-chime-scint-routeb-voltage.md",
        "band_mhz": list(BAND_MHZ),
        "lte_exclusion_mhz": list(LTE_EXCLUSION_MHZ),
        "on_pulse_window_blinded": list(ON_PULSE_WINDOW),
        "off_pool_windows": [list(w) for w in OFF_POOL_WINDOWS],
        "channels_per_coarse": CHANNELS_PER_COARSE,
        "channel_width_mhz": products.channel_width_mhz,
        "n_band_channels": products.n_band_channels,
        "n_good_channels": int(products.good_channels.sum()),
        "estimator": {
            "n_on_samples": N_ON,
            "max_lag_bins": MAX_LAG_BINS,
            "first_lag_bin": FIRST_LAG_BIN,
            "fit_max_mhz": FIT_MAX_MHZ,
            "block_length": CHANNELS_PER_COARSE,
            "burst_flux_fraction": BURST_FLUX_FRACTION,
            "ratio_cancels_common_mode": "g(nu) divides out of on/off ratio (algebraic)",
        },
        "injection_grid": {
            "modulations": list(MODULATIONS),
            "dnu_khz": list(DNU_KHZ),
            "control_dnu_khz": CONTROL_DNU_KHZ,
            "detectable_dnu_khz": DETECTABLE_DNU_KHZ,
            "n_realizations": N_REALIZATIONS,
            "seed_formula": "1000*cell_index + realization ; cell_index = m_index*5 + dnu_index",
        },
        "gates": {
            "g1": {
                "detectable_cells_must_certify": True,
                "control_must_not_certify": True,
                "dnu_fractional_tolerance": G1_DNU_TOL,
                "amplitude_pull_max": G1_AMP_PULL_MAX,
                "convergence_min": G1_CONVERGENCE_MIN,
                "certify": "median|dnu_bias|<=0.30 AND |median amp pull|<=2 AND convergence>=0.90",
            },
            "g2": {
                "n_windows_min": G2_N_WINDOWS,
                "family_wise_error_rate": G2_FWER,
                "correction": "Sidak two-sided",
                "null_construction": (
                    "off-pulse-only pseudo-on null realizations at the G1 "
                    "operating point (n_on=100 seeded permutation split, "
                    "pseudo-on perpendicular reference, no injection); "
                    "seed = 900000 + index"
                ),
                "pass": "no fitted-amplitude z exceeds the Sidak threshold; all windows fit",
            },
            "g3_admissibility_after_unblind": {
                "detection_requires_dnu_in_gate0_window": True,
                "gate0_window_khz": {"m0.17": 77.0, "m0.15": 127.0},
                "significance_min": 5.0,
            },
        },
        "selection_rule": "best G1 calibration score; ties break S2 -> S1 -> S3",
        "statistics": ["S1", "S2", "S3"],
        "inputs": products.inputs,
        "inputs_sha256": products.inputs_sha256,
        "blinding": {
            "on_pulse_guard": list(rb.ON_PULSE_GUARD),
            "unblind_flag": UNBLIND_FLAG,
            "policy": "no statistic on samples 250-350 without the explicit flag",
        },
    }


def cmd_freeze(args: argparse.Namespace) -> int:
    products = Products(args)
    # prerequisite sanity (NOT a gate): a single real off-pulse ratio null must
    # load and fit without error, proving the pipeline runs end to end.
    pool = products.off_pool
    half = pool.size // 2
    probe = products.run_statistic("S1", pool[:half], pool[half:], on_gain=None)
    config = _frozen_config(products)
    config["prerequisites"] = {
        "data_loaded": True,
        "n_good_channels": int(products.good_channels.sum()),
        "offpulse_probe_fit_finite": bool(probe.fit is not None),
        "offpulse_probe_dnu_mhz": None if probe.fit is None else probe.fit["dnu_mhz"],
    }
    FROZEN.write_text(json.dumps(_jsonable(config), indent=2, sort_keys=True) + "\n")
    digest = _sha256(FROZEN)
    print(
        json.dumps(
            {
                "frozen_config_sha256": digest,
                "n_good_channels": config["n_good_channels"],
                "n_band_channels": config["n_band_channels"],
                "channel_width_khz": round(products.channel_width_mhz * 1e3, 4),
            },
            sort_keys=True,
        )
    )
    return 0


def _frozen_or_die() -> dict:
    if not FROZEN.exists():
        raise SystemExit("frozen_config.json missing -- run the freeze subcommand first")
    return json.loads(FROZEN.read_text())


def _frozen_sha() -> str:
    return _sha256(FROZEN)


# --------------------------------------------------------------------------- #
# unblind (orchestrator-only; refuses without gates + explicit flag)
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
    if not selection.get("selected_statistic"):
        raise SystemExit("selection verdict is DOCUMENTED-FAIL -- the on-pulse fit stays blind")
    statistic = selection["selected_statistic"]
    products = Products(args)
    on = rb.samples_from_window(ON_PULSE_WINDOW)
    off = products.off_pool
    result = products.statistic(statistic)(
        products.dynamic,
        on,
        off,
        products.parent,
        channel_width_mhz=products.channel_width_mhz,
        on_gain=None,
        max_lag_bins=MAX_LAG_BINS,
        first_lag_bin=FIRST_LAG_BIN,
        fit_max_mhz=FIT_MAX_MHZ,
        block_length=CHANNELS_PER_COARSE,
        allow_unblind=True,
    )
    payload = {
        "experiment": EXPERIMENT_ID,
        "selected_statistic": statistic,
        "frozen_config_sha256": _frozen_sha(),
        "onpulse_fit": result.fit,
        "warning": "on-pulse window read under explicit unblind flag",
    }
    (HERE / "unblind_onpulse.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    print(json.dumps({"selected_statistic": statistic, "onpulse_fit": _jsonable(result.fit)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pol0", type=Path, default=DEFAULT_POL0)
    parser.add_argument("--pol1", type=Path, default=DEFAULT_POL1)
    parser.add_argument("--frequencies", type=Path, default=DEFAULT_FREQUENCIES)
    parser.add_argument("--time0-metadata", type=Path, default=DEFAULT_METADATA)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    for name in ("g1", "g2"):
        p = sub.add_parser(name)
        p.add_argument("--statistic", choices=("S1", "S2", "S3"), required=True)
    sub.add_parser("select")
    up = sub.add_parser("unblind")
    up.add_argument(UNBLIND_FLAG, dest="unblind_i_know_what_i_am_doing", action="store_true")
    args = parser.parse_args()
    return {
        "freeze": cmd_freeze,
        "g1": cmd_g1,
        "g2": cmd_g2,
        "select": cmd_select,
        "unblind": cmd_unblind,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
