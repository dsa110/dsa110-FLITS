#!/usr/bin/env python3
"""Blinded C1 all-pairs cross-ACF qualification for Freya CHIME.

Implements the c1-allpairs-crossgp route from the owner decision
(Faber2026 docs/rse/specs/decision-2026-07-14-figure1-and-chime-c1.md):
all admissible distinct-time polarization/time-fold cross-products on the
retained product, nuisance templates trained only on held-out off-pulse
blocks with leave-one-out rotation, and a blinded real-background
calibration matrix that must pass before the on-pulse fit exists.

Subcommands enforce the blinding boundary structurally:

  freeze      prove window/mask/alignment, write frozen_config.json
  calibrate   run one (modulation, width) injection cell -> checkpoint
  nulls       held-out off-pulse + pairing-scramble null campaign
  aggregate   combine checkpoints + nulls -> calibration_verdict.json
  unblind     REFUSES unless the verdict passes; then the on-pulse fit
              plus post-unblind stability/scramble gates -> validation.json

No code path outside `unblind` fits the on-pulse window.
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
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import norm as _norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
B4_ROOT = ROOT / "analysis" / "chime-recovery-2026-07-12"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(B4_ROOT))

from scintillation.scint_analysis.cross_acf import (  # noqa: E402
    all_pairs_cross_acf,
    fit_cross_lorentzian,
)
import validate_freya_highband_crossacf as b4  # noqa: E402

HERE = Path(__file__).parent
FROZEN = HERE / "frozen_config.json"
CALIBRATION_DIR = HERE / "calibration"
VERDICT = HERE / "calibration_verdict.json"
VALIDATION_DIR = HERE / "validation"

EXPERIMENT_ID = "c1-allpairs-crossgp"
MODULATION_INDICES = (0.10, 0.15, 0.17, 0.20, 0.30, 1.00)
GATED_MODULATIONS = (0.15, 0.17)
WIDTH_CHANNELS = (3.0, 6.0, 10.0, 16.0)
N_TRIALS = 128
SEED_BASE = 20260714
# Detection criterion for fit-level nulls (predeclared): a bound-clear fit
# whose modulation index is >= 3 sigma from zero counts as a false positive.
DETECTION_M_SIGMA = 3.0


def _seed(width_index: int, modulation_index: int, trial: int) -> int:
    return SEED_BASE + 1_000_000 * width_index + 10_000 * modulation_index + trial


def _off_starts() -> list[int]:
    width = b4.BURST_WINDOW[1] - b4.BURST_WINDOW[0]
    return list(range(b4.OFF_PULSE[0], b4.OFF_PULSE[1] - width + 1, width))[:12]


def _family_wise_z(n_comparisons: int, alpha: float = 0.01) -> float:
    return float(_norm.isf(alpha / (2.0 * n_comparisons)))


class Products:
    """Retained-product load shared by every subcommand (no on-pulse fits)."""

    def __init__(self, args: argparse.Namespace) -> None:
        frequencies_full = np.load(args.frequencies)
        select = (frequencies_full >= b4.BAND_MHZ[0]) & (frequencies_full <= b4.BAND_MHZ[1])
        self.frequencies = np.asarray(frequencies_full[select], dtype=float)
        power = [
            np.asarray(np.load(path, mmap_mode="r")[select], dtype=float)
            for path in (args.pol0, args.pol1)
        ]
        stokes = np.asarray(np.load(args.stokes, mmap_mode="r")[select], dtype=float)
        parity_max = float(np.nanmax(np.abs((power[0] + power[1]) - stokes)))
        parity_scale = float(np.nanmax(np.abs(stokes)))
        self.producer_parity = bool(parity_max <= max(1e-5 * parity_scale, 1e-6))
        self.metadata = json.loads(args.time0_metadata.read_text())
        self.provenance = b4._metadata_gate(
            self.metadata,
            {
                "pol0": args.pol0,
                "pol1": args.pol1,
                "stokes": args.stokes,
                "frequencies": args.frequencies,
            },
        )
        coarse = np.asarray(self.metadata["freq_mhz"], dtype=float)
        target = b4.load_chime_target("freya")
        dt_s = 2.56e-6 * 2 * int(target["upchannel_factor"])
        offsets = b4.coarse_alignment_offsets(
            coarse,
            np.asarray(self.metadata["fpga_count"]),
            delta_time_s=float(self.metadata["delta_time"]),
            dm=float(target["dm"]),
            dt_s=dt_s,
        )
        self.good_channels = b4._channel_mask(power, self.frequencies)
        self.dynamic = [
            b4._build_polarization_product(
                item, self.frequencies, coarse, offsets, self.good_channels
            )[0]
            for item in power
        ]
        self.baselines = [
            b4._row_nanmean(item[:, b4.OFF_PULSE[0] : b4.OFF_PULSE[1]]) for item in self.dynamic
        ]
        self.residuals = [
            item - baseline[:, None]
            for item, baseline in zip(self.dynamic, self.baselines, strict=True)
        ]
        self.parent = np.argmin(
            np.abs(self.frequencies[:, None] - coarse[None, :]), axis=1
        )
        self.channel_width = float(np.nanmedian(np.diff(self.frequencies)))

        window = b4.BURST_WINDOW
        profile = np.nansum(
            [item[:, window[0] : window[1]] for item in self.residuals], axis=0
        )
        envelope = np.nanmean(profile, axis=0)
        if not np.isfinite(envelope).all() or float(np.mean(envelope)) <= 0:
            raise SystemExit("burst envelope is not positive; check alignment")
        self.envelope = envelope / float(np.mean(envelope))
        # Per-fold on-pulse envelope normalization (time profile only — no
        # spectral fit is formed here); the same norms scale off-pulse nulls
        # so on and off share one amplitude convention.
        self.on_norms = [
            np.nanmean(item[:, window[0] : window[1]], axis=0) for item in self.residuals
        ]
        self.off_sigmas = [
            b4._row_nanstd(item[:, b4.OFF_PULSE[0] : b4.OFF_PULSE[1]]) for item in self.dynamic
        ]
        self.pol_norms = tuple(float(np.nanmean(norm)) for norm in self.on_norms)

    def alignment_gate(self) -> dict:
        window = b4.BURST_WINDOW
        aligned_profile = np.nansum(np.stack(self.residuals), axis=(0, 1))
        peak_bin = int(np.nanargmax(aligned_profile))
        burst_finite = float(
            np.mean(np.isfinite(np.stack([r[:, window[0] : window[1]] for r in self.residuals])))
        )
        off_finite = float(
            np.mean(
                np.isfinite(
                    np.stack([r[:, b4.OFF_PULSE[0] : b4.OFF_PULSE[1]] for r in self.residuals])
                )
            )
        )
        return {
            "pass": bool(
                window[0] <= peak_bin < window[1] and burst_finite >= 0.75 and off_finite >= 0.75
            ),
            "peak_bin": peak_bin,
            "required_peak_window": list(window),
            "burst_finite_fraction": burst_finite,
            "offpulse_finite_fraction": off_finite,
            "minimum_finite_fraction": 0.75,
        }

    def window_folds(self, window: tuple[int, int]) -> list[np.ndarray]:
        return [item[:, window[0] : window[1]] for item in self.residuals]

    def c1_cross(self, folds: list[np.ndarray]):
        return all_pairs_cross_acf(
            folds,
            self.parent,
            max_lag_bins=b4.MAX_LAG_BINS,
            exclude_same_time=True,
            normalizations=self.on_norms,
        )

    def off_crosses(self) -> list:
        crosses = []
        width = b4.BURST_WINDOW[1] - b4.BURST_WINDOW[0]
        for start in _off_starts():
            crosses.append(self.c1_cross(self.window_folds((start, start + width))))
        return crosses


def _pairing_scramble(folds: list[np.ndarray], parent: np.ndarray, seed: int) -> list[np.ndarray]:
    """Within-coarse-block circular shift of every pol-1 fold (seeded).

    Destroys cross-polarization common spectral structure at sub-block lags
    while preserving each fold's own statistics — the predeclared
    pairing-scramble null of the decision doc.
    """
    rng = np.random.default_rng(seed)
    scrambled = [folds[0]]
    for fold_set in folds[1:]:
        out = np.array(fold_set, dtype=float, copy=True)
        for block in np.unique(parent):
            rows = np.flatnonzero(parent == block)
            shift = int(rng.integers(rows.size // 4, 3 * rows.size // 4 + 1))
            out[rows] = np.roll(out[rows], shift, axis=0)
        scrambled.append(out)
    return scrambled


def _fit(cross, channel_width: float, fit_max: float | None = None):
    return fit_cross_lorentzian(
        cross,
        channel_width_mhz=channel_width,
        first_lag_bin=b4.FIRST_LAG_BIN,
        fit_max_mhz=b4.FIT_MAXIMA_MHZ[-1] if fit_max is None else fit_max,
        block_length=b4.CHANNELS_PER_COARSE,
    )


def _is_detection(fit: dict | None, channel_width: float, fit_max: float) -> bool:
    if fit is None:
        return False
    bound_clear = 0.55 * channel_width < fit["dnu_mhz"] < 0.95 * fit_max
    if not bound_clear:
        return False
    # fail closed: a bound-clear control fit with an invalid uncertainty
    # estimate cannot be certified insignificant
    m_err = fit.get("m_err", np.nan)
    if not (np.isfinite(m_err) and m_err > 0):
        return True
    return bool(fit["m"] / m_err >= DETECTION_M_SIGMA)


def _frozen_or_die() -> dict:
    if not FROZEN.exists():
        raise SystemExit("frozen_config.json missing — run the freeze subcommand first")
    return json.loads(FROZEN.read_text())


def _frozen_sha() -> str:
    return hashlib.sha256(FROZEN.read_bytes()).hexdigest()


def cmd_freeze(args: argparse.Namespace) -> int:
    products = Products(args)
    alignment = products.alignment_gate()
    config = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "band_mhz": list(b4.BAND_MHZ),
        "lte_exclusion_mhz": list(b4.LTE_EXCLUSION_MHZ),
        "off_pulse": list(b4.OFF_PULSE),
        "burst_window": list(b4.BURST_WINDOW),
        "off_window_starts": _off_starts(),
        "max_lag_bins": b4.MAX_LAG_BINS,
        "first_lag_bin": b4.FIRST_LAG_BIN,
        "channels_per_coarse": b4.CHANNELS_PER_COARSE,
        "fit_maxima_mhz": list(b4.FIT_MAXIMA_MHZ),
        "alpha_scaling": b4.ALPHA_SCALING,
        "estimator": {
            "name": "all_pairs_cross_acf",
            "exclude_same_time": True,
            "normalizations": "per-fold on-pulse envelope means (time profile)",
        },
        "calibration": {
            "modulation_indices": list(MODULATION_INDICES),
            "gated_modulations": list(GATED_MODULATIONS),
            "width_channels": list(WIDTH_CHANNELS),
            "n_trials": N_TRIALS,
            "seed_formula": "20260714 + 1e6*width_index + 1e4*modulation_index + trial",
        },
        "gates": {
            "width_bias_limit": "max(0.10*truth, 0.25*channel_width)",
            "coverage_68": [0.53, 0.83],
            "modulation_bias_limit": "max(0.10*truth, 0.05)",
            "null_family_wise_alpha": 0.01,
            "null_n_comparisons": 24 * b4.MAX_LAG_BINS,
            "null_max_abs_z": _family_wise_z(24 * b4.MAX_LAG_BINS),
            "detection_m_sigma": DETECTION_M_SIGMA,
            "fit_window_max_movement": 0.20,
        },
        "scrambles": {
            "pairing_scramble": "seeded within-coarse-block circular channel shift of pol-1 folds",
            "post_unblind": ["pol_scramble", "time_scramble_onxoff", "coarse_phase_shift"],
        },
        "channel_width_mhz": products.channel_width,
        "n_good_channels": int(products.good_channels.sum()),
        "inputs_sha256": {
            "pol0": b4._sha256(args.pol0),
            "pol1": b4._sha256(args.pol1),
            "stokes": b4._sha256(args.stokes),
            "frequencies": b4._sha256(args.frequencies),
            "metadata": b4._sha256(args.time0_metadata),
        },
        "prerequisites": {
            "producer_parity": products.producer_parity,
            "provenance": products.provenance["pass"],
            "alignment": alignment,
        },
        "provenance_note": (
            "Hyperparameters inherit the B4 predeclarations; the 2026-07-14 "
            "single-template diagnostic was pre-freeze exploratory and its "
            "unconstrained on-pulse numbers informed nothing here."
        ),
    }
    FROZEN.write_text(json.dumps(b4._jsonable(config), indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    print(json.dumps({"frozen_config_sha256": digest, "prerequisites": config["prerequisites"]}))
    ok = products.producer_parity and products.provenance["pass"] and alignment["pass"]
    return 0 if ok else 2


def cmd_calibrate(args: argparse.Namespace) -> int:
    frozen = _frozen_or_die()
    products = Products(args)
    width_index = WIDTH_CHANNELS.index(args.width)
    modulation_index = MODULATION_INDICES.index(args.modulation)
    CALIBRATION_DIR.mkdir(exist_ok=True)
    checkpoint = CALIBRATION_DIR / f"cell_m{args.modulation:.2f}_w{args.width:g}.json"
    if checkpoint.exists() and not args.force:
        existing = json.loads(checkpoint.read_text())
        if existing.get("frozen_config_sha256") == _frozen_sha():
            print(json.dumps({"cell": checkpoint.name, "status": "already complete"}))
            return 0
        # checkpoint predates the current freeze: rerun rather than reuse

    starts = _off_starts()
    window_width = b4.BURST_WINDOW[1] - b4.BURST_WINDOW[0]
    off_crosses = products.off_crosses()
    truth = args.width * products.channel_width
    fit_max = b4.FIT_MAXIMA_MHZ[-1]
    records = []
    for trial in range(args.trials):
        seed = _seed(width_index, modulation_index, trial)
        rng = np.random.default_rng(seed)
        common = b4._stationary_lorentzian(
            rng, n_channels=products.parent.size, width_bins=args.width
        )
        host = trial % len(starts)
        start = starts[host]
        shared_noise = rng.normal(size=(products.parent.size, window_width))
        injected_folds = []
        for residual, sigma, pol_norm in zip(
            products.residuals, products.off_sigmas, products.pol_norms, strict=True
        ):
            signal = pol_norm * products.envelope[None, :] * (
                1.0 + args.modulation * common[:, None]
            )
            excess = np.sqrt(np.maximum((1.0 + signal) ** 2 - 1.0, 0.0))
            injected_folds.append(
                residual[:, start : start + window_width]
                + signal
                + sigma[:, None] * excess * shared_noise
            )
        cross = products.c1_cross(injected_folds)
        controls = off_crosses[:host] + off_crosses[host + 1 :]
        cross = b4._remove_instrument_template(cross, controls)
        fit = _fit(cross, products.channel_width)
        records.append(
            {
                "trial": trial,
                "seed": seed,
                "host_window": [start, start + window_width],
                "fit": fit,
            }
        )

    finite = [record for record in records if record["fit"] is not None]
    recovered = np.asarray([record["fit"]["dnu_mhz"] for record in finite])
    errors = np.asarray([record["fit"]["dnu_err_mhz"] for record in finite])
    recovered_m = np.asarray([record["fit"]["m"] for record in finite])
    width_bias = float(np.median(np.abs(recovered - truth))) if recovered.size else np.inf
    coverage = float(np.mean(np.abs(recovered - truth) <= errors)) if recovered.size else 0.0
    m_bias = (
        float(np.median(np.abs(recovered_m - args.modulation))) if recovered_m.size else np.inf
    )
    width_limit = max(0.10 * truth, 0.25 * products.channel_width)
    m_limit = max(0.10 * args.modulation, 0.05)
    passed = bool(
        len(finite) == args.trials
        and width_bias < width_limit
        and 0.53 <= coverage <= 0.83
        and m_bias < m_limit
    )
    cell = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
        "modulation_index": args.modulation,
        "width_channels": args.width,
        "truth_width_mhz": truth,
        "n_trials": args.trials,
        "n_finite": len(finite),
        "median_absolute_width_bias_mhz": width_bias,
        "width_bias_limit_mhz": width_limit,
        "coverage_68": coverage,
        "coverage_limits": [0.53, 0.83],
        "median_absolute_modulation_bias": m_bias,
        "modulation_bias_limit": m_limit,
        "gating_cell": args.modulation in GATED_MODULATIONS,
        "stress_cell": args.modulation == 0.10,
        "pass": passed,
        "records": records,
        "fit_max_mhz": fit_max,
        "frozen_gates": frozen["gates"],
    }
    tmp = checkpoint.with_suffix(".tmp")
    tmp.write_text(json.dumps(b4._jsonable(cell), indent=2, sort_keys=True) + "\n")
    tmp.rename(checkpoint)
    print(
        json.dumps(
            {
                "cell": checkpoint.name,
                "pass": passed,
                "n_finite": len(finite),
                "width_bias_mhz": width_bias,
                "coverage_68": coverage,
                "m_bias": m_bias,
            }
        )
    )
    return 0


def cmd_nulls(args: argparse.Namespace) -> int:
    _frozen_or_die()
    products = Products(args)
    starts = _off_starts()
    window_width = b4.BURST_WINDOW[1] - b4.BURST_WINDOW[0]
    off_crosses = products.off_crosses()
    threshold = _family_wise_z(2 * len(starts) * b4.MAX_LAG_BINS)
    fit_max = b4.FIT_MAXIMA_MHZ[-1]

    records = []
    for index, start in enumerate(starts):
        controls = off_crosses[:index] + off_crosses[index + 1 :]
        held_out = b4._remove_instrument_template(off_crosses[index], controls)
        fit = _fit(held_out, products.channel_width)
        finite = np.isfinite(held_out.acf) & np.isfinite(held_out.error) & (held_out.error > 0)
        z = held_out.acf[finite] / held_out.error[finite]
        records.append(
            {
                "kind": "held_out_offpulse",
                "window": [start, start + window_width],
                "max_abs_z": float(np.max(np.abs(z))) if z.size else None,
                "detection": _is_detection(fit, products.channel_width, fit_max),
                "fit": fit,
                "acf": held_out.acf.tolist(),
                "error": held_out.error.tolist(),
            }
        )
        folds = products.window_folds((start, start + window_width))
        scrambled = products.c1_cross(
            _pairing_scramble(folds, products.parent, seed=SEED_BASE + 555 + index)
        )
        scrambled = b4._remove_instrument_template(scrambled, controls)
        s_fit = _fit(scrambled, products.channel_width)
        s_finite = np.isfinite(scrambled.acf) & np.isfinite(scrambled.error) & (scrambled.error > 0)
        s_z = scrambled.acf[s_finite] / scrambled.error[s_finite]
        records.append(
            {
                "kind": "pairing_scramble",
                "window": [start, start + window_width],
                "max_abs_z": float(np.max(np.abs(s_z))) if s_z.size else None,
                "detection": _is_detection(s_fit, products.channel_width, fit_max),
                "fit": s_fit,
                "acf": scrambled.acf.tolist(),
                "error": scrambled.error.tolist(),
            }
        )

    z_values = [record["max_abs_z"] for record in records if record["max_abs_z"] is not None]
    z_pass = bool(len(z_values) == len(records) and max(z_values) <= threshold)
    detections = [record for record in records if record["detection"]]
    result = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": _frozen_sha(),
        "pass": bool(z_pass and not detections),
        "family_wise_threshold": threshold,
        "family_wise_alpha": 0.01,
        "n_comparisons": 2 * len(starts) * b4.MAX_LAG_BINS,
        "max_abs_z": max(z_values) if z_values else None,
        "n_detections": len(detections),
        "records": records,
    }
    CALIBRATION_DIR.mkdir(exist_ok=True)
    (CALIBRATION_DIR / "nulls.json").write_text(
        json.dumps(b4._jsonable(result), indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "pass": result["pass"],
                "max_abs_z": result["max_abs_z"],
                "threshold": threshold,
                "n_detections": len(detections),
            }
        )
    )
    return 0 if result["pass"] else 2


def cmd_aggregate(args: argparse.Namespace) -> int:
    frozen = _frozen_or_die()
    nulls_path = CALIBRATION_DIR / "nulls.json"
    if not nulls_path.exists():
        raise SystemExit("nulls.json missing — run the nulls subcommand first")
    nulls = json.loads(nulls_path.read_text())
    current_sha = _frozen_sha()
    cells = []
    missing = []
    if nulls.get("frozen_config_sha256") != current_sha:
        missing.append("nulls.json (stale or missing frozen_config_sha256)")
    for modulation in MODULATION_INDICES:
        for width in WIDTH_CHANNELS:
            path = CALIBRATION_DIR / f"cell_m{modulation:.2f}_w{width:g}.json"
            if not path.exists():
                missing.append(path.name)
                continue
            cell = json.loads(path.read_text())
            if cell.get("frozen_config_sha256") != current_sha:
                missing.append(f"{path.name} (stale frozen config)")
                continue
            if cell["n_trials"] < N_TRIALS:
                missing.append(f"{path.name} (undersized: {cell['n_trials']} < {N_TRIALS})")
                continue
            cells.append(cell)
    gated = [cell for cell in cells if cell["gating_cell"]]
    n_gated_expected = len(GATED_MODULATIONS) * len(WIDTH_CHANNELS)
    prerequisites = frozen["prerequisites"]
    go = bool(
        not missing
        and prerequisites["producer_parity"]
        and prerequisites["provenance"]
        and prerequisites["alignment"]["pass"]
        and nulls["pass"]
        and len(gated) == n_gated_expected
        and all(cell["pass"] for cell in gated)
    )
    verdict = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": current_sha,
        "go": go,
        "missing_cells": missing,
        "nulls_pass": nulls["pass"],
        "gated_cells": [
            {
                "modulation_index": cell["modulation_index"],
                "width_channels": cell["width_channels"],
                "pass": cell["pass"],
            }
            for cell in gated
        ],
        "all_cells": [
            {
                "modulation_index": cell["modulation_index"],
                "width_channels": cell["width_channels"],
                "pass": cell["pass"],
                "stress_cell": cell["stress_cell"],
                "coverage_68": cell["coverage_68"],
                "median_absolute_width_bias_mhz": cell["median_absolute_width_bias_mhz"],
                "median_absolute_modulation_bias": cell["median_absolute_modulation_bias"],
            }
            for cell in cells
        ],
    }
    VERDICT.write_text(json.dumps(b4._jsonable(verdict), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"go": go, "missing_cells": missing, "n_cells": len(cells)}))
    return 0 if go else 2


def cmd_unblind(args: argparse.Namespace) -> int:
    frozen = _frozen_or_die()
    if not VERDICT.exists():
        raise SystemExit("calibration_verdict.json missing — aggregate first; staying blind")
    verdict = json.loads(VERDICT.read_text())
    if not verdict.get("go"):
        raise SystemExit("calibration verdict is NO-GO — the on-pulse fit stays blind")
    if verdict["frozen_config_sha256"] != _frozen_sha():
        raise SystemExit("frozen_config.json changed after the verdict — new experiment required")

    products = Products(args)
    off_crosses = products.off_crosses()
    window = b4.BURST_WINDOW
    on_folds = products.window_folds(window)
    on_cross = b4._remove_instrument_template(products.c1_cross(on_folds), off_crosses)

    fits = {
        f"{fit_max:.2f}": _fit(on_cross, products.channel_width, fit_max)
        for fit_max in b4.FIT_MAXIMA_MHZ
    }
    on_fit = fits[f"{b4.FIT_MAXIMA_MHZ[-1]:.2f}"]
    finite_fits = [fit for fit in fits.values() if fit is not None]
    widths = np.asarray([fit["dnu_mhz"] for fit in finite_fits])
    bound_clear = all(
        fit is not None
        and fit["dnu_mhz"] > 0.55 * products.channel_width
        and fit["dnu_mhz"] < 0.95 * float(key)
        for key, fit in fits.items()
    )
    movement = (
        float((widths.max() - widths.min()) / np.median(widths))
        if widths.size == len(b4.FIT_MAXIMA_MHZ)
        else np.inf
    )
    fit_window_gate = {
        "pass": bool(len(finite_fits) == len(b4.FIT_MAXIMA_MHZ) and bound_clear and movement < 0.20),
        "max_fractional_movement": movement,
        "fits": fits,
    }

    fit_max = b4.FIT_MAXIMA_MHZ[-1]
    scrambles = {}
    scrambled_folds = _pairing_scramble(on_folds, products.parent, seed=SEED_BASE + 777)
    pol_cross = b4._remove_instrument_template(products.c1_cross(scrambled_folds), off_crosses)
    scrambles["pol_scramble"] = {
        "fit": _fit(pol_cross, products.channel_width),
        "detection": _is_detection(
            _fit(pol_cross, products.channel_width), products.channel_width, fit_max
        ),
    }
    width_bins = window[1] - window[0]
    off_host = _off_starts()[5]
    mixed = [
        products.residuals[0][:, window[0] : window[1]],
        products.residuals[1][:, off_host : off_host + width_bins],
    ]
    mixed_cross = b4._remove_instrument_template(products.c1_cross(mixed), off_crosses)
    scrambles["time_scramble_onxoff"] = {
        "fit": _fit(mixed_cross, products.channel_width),
        "detection": _is_detection(
            _fit(mixed_cross, products.channel_width), products.channel_width, fit_max
        ),
    }
    rng = np.random.default_rng(SEED_BASE + 999)
    shifted = [np.array(on_folds[0], copy=True), np.array(on_folds[1], copy=True)]
    for block in np.unique(products.parent):
        rows = np.flatnonzero(products.parent == block)
        shifted[1][rows] = np.roll(
            shifted[1][rows], int(rng.integers(rows.size // 4, 3 * rows.size // 4 + 1)), axis=0
        )
    phase_cross = b4._remove_instrument_template(products.c1_cross(shifted), off_crosses)
    scrambles["coarse_phase_shift"] = {
        "fit": _fit(phase_cross, products.channel_width),
        "detection": _is_detection(
            _fit(phase_cross, products.channel_width), products.channel_width, fit_max
        ),
    }
    scramble_gate = {
        "pass": bool(not any(item["detection"] for item in scrambles.values())),
        "controls": scrambles,
    }

    nu_reference = float(np.nanmean(products.frequencies[products.good_channels]))
    selections = [
        ("early", np.ones(products.parent.size, dtype=bool), (window[0], 260)),
        ("late", np.ones(products.parent.size, dtype=bool), (260, window[1])),
        ("low_highband", products.frequencies < 713.5, window),
        ("upper_highband", products.frequencies >= 713.5, window),
    ]
    compatibility_records = []
    for name, channel_select, sub_window in selections:
        folds = [item[channel_select] for item in products.window_folds(sub_window)]
        try:
            local = all_pairs_cross_acf(
                folds,
                products.parent[channel_select],
                max_lag_bins=b4.MAX_LAG_BINS,
                exclude_same_time=True,
                normalizations=[
                    np.nanmean(item, axis=0) for item in folds
                ],
            )
        except ValueError:
            local = None
        fit = _fit(local, products.channel_width) if local is not None else None
        selected = channel_select & products.good_channels
        nu_selection = (
            float(np.nanmean(products.frequencies[selected])) if selected.any() else np.nan
        )
        scale = (
            (nu_reference / nu_selection) ** b4.ALPHA_SCALING
            if name in ("low_highband", "upper_highband") and np.isfinite(nu_selection)
            else 1.0
        )
        compatibility_records.append(
            {
                "name": name,
                "fit": fit,
                "mean_frequency_mhz": nu_selection,
                "width_scale_to_reference": scale,
            }
        )
    comparison_fits = [record["fit"] for record in compatibility_records]
    compatible = on_fit is not None and all(fit is not None for fit in comparison_fits)
    if compatible:
        for record in compatibility_records:
            fit = record["fit"]
            scale = record["width_scale_to_reference"]
            difference = abs(scale * fit["dnu_mhz"] - on_fit["dnu_mhz"])
            sigma = np.hypot(scale * fit["dnu_err_mhz"], on_fit["dnu_err_mhz"])
            compatible &= difference <= max(0.25 * on_fit["dnu_mhz"], 2.0 * sigma)
    compatibility = {
        "pass": bool(compatible),
        "alpha_scaling": b4.ALPHA_SCALING,
        "reference_frequency_mhz": nu_reference,
        "records": compatibility_records,
    }

    validated_low = WIDTH_CHANNELS[0] * products.channel_width
    validated_high = WIDTH_CHANNELS[-1] * products.channel_width
    envelope_gate = {
        "pass": bool(on_fit is not None and validated_low <= on_fit["dnu_mhz"] <= validated_high),
        "validated_envelope_mhz": [validated_low, validated_high],
        "onpulse_width_mhz": None if on_fit is None else on_fit["dnu_mhz"],
    }

    machine_pass = bool(
        fit_window_gate["pass"]
        and scramble_gate["pass"]
        and compatibility["pass"]
        and envelope_gate["pass"]
    )
    # a numeric pass is not a pass until the diagnostic figures are reviewed
    # (repo validation contract): the review file must exist beside the
    # validation record and every figure verdict must be "match"
    review_path = VALIDATION_DIR / "figures.review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else None
    reviewed_figures = review.get("figures", []) if isinstance(review, dict) else []
    figure_review_pass = bool(reviewed_figures) and all(
        item.get("verdict") == "match" for item in reviewed_figures
    )
    if machine_pass and figure_review_pass:
        machine_status = "pass"
    elif machine_pass:
        machine_status = "pass_pending_figure_review"
    else:
        machine_status = "documented_fail"
    result = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": verdict["frozen_config_sha256"],
        "calibration_go": True,
        "channel_width_mhz": products.channel_width,
        "onpulse_fit": on_fit,
        "gates": {
            "fit_window_stability": fit_window_gate,
            "post_unblind_scrambles": scramble_gate,
            "split_compatibility": compatibility,
            "onpulse_width_within_validated_envelope": envelope_gate,
            "manual_figure_review": {
                "pass": figure_review_pass,
                "status": "reviewed" if figure_review_pass else "pending",
                "review_file": str(review_path),
            },
        },
        "machine_status": machine_status,
        "science_status": "diagnostic_only",
    }
    VALIDATION_DIR.mkdir(exist_ok=True)
    figure_dir = VALIDATION_DIR / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    lags = on_cross.lag_bins * products.channel_width * 1e3
    ax.errorbar(lags, on_cross.acf, yerr=on_cross.error, fmt=".", ms=4, alpha=0.8, label="C1 ACF")
    if on_fit is not None:
        ax.plot(
            np.asarray(on_fit["fit_lags_mhz"]) * 1e3, on_fit["model_acf"], lw=2, label="fit"
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Frequency lag (kHz)", ylabel="Cross covariance", title="Freya C1 on-pulse ACF")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    figure_path = figure_dir / "freya_c1_onpulse_acf.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    result["figures"] = [str(figure_path)]
    (VALIDATION_DIR / "validation.json").write_text(
        json.dumps(b4._jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "machine_status": result["machine_status"],
                "gates": {name: gate["pass"] for name, gate in result["gates"].items()},
            },
            sort_keys=True,
        )
    )
    return 0 if machine_pass and figure_review_pass else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pol0", type=Path, default=b4.DEFAULT_POL0)
    parser.add_argument("--pol1", type=Path, default=b4.DEFAULT_POL1)
    parser.add_argument("--stokes", type=Path, default=b4.DEFAULT_STOKES)
    parser.add_argument("--frequencies", type=Path, default=b4.DEFAULT_FREQUENCIES)
    parser.add_argument("--time0-metadata", type=Path, default=b4.DEFAULT_METADATA)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--modulation", type=float, required=True)
    calibrate.add_argument("--width", type=float, required=True)
    calibrate.add_argument("--trials", type=int, default=N_TRIALS)
    calibrate.add_argument("--force", action="store_true")
    subparsers.add_parser("nulls")
    subparsers.add_parser("aggregate")
    subparsers.add_parser("unblind")
    args = parser.parse_args()
    if args.command == "calibrate":
        if args.modulation not in MODULATION_INDICES or args.width not in WIDTH_CHANNELS:
            raise SystemExit("modulation/width must come from the frozen calibration grid")
    return {
        "freeze": cmd_freeze,
        "calibrate": cmd_calibrate,
        "nulls": cmd_nulls,
        "aggregate": cmd_aggregate,
        "unblind": cmd_unblind,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
