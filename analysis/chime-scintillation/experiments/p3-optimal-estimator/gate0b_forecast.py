#!/usr/bin/env python3
"""Gate 0b: measured-noise detectability forecast for the P3 optimal estimator.

Predeclared in ``docs/rse/specs/experiment-chime-scint-p3-optimal-estimator.md``
(Faber2026): before any P3 implementation, recompute the Gate-0 detectability
ceiling replacing the idealized radiometer algebra with the **measured** null
variance of the delay-domain cross power on real off-pulse data.

Frozen floor: at m = 0.17, Δν_d = 213 kHz the forecast must reach SNR >= 3 for
the build to proceed; below 3 the experiment terminates as
DOCUMENTED-FAIL-BY-FORECAST with nothing built.

Construction (inherits P2's machinery verbatim via import):

* Null realizations: seeded permutation splits of the off-pulse pool into a
  100-sample pseudo-on and a disjoint reference (P2 G2 convention,
  seed = 900000 + i), through the S2 split-ratio construction -> two ratio
  fields with independent noise.
* Transform: 64-channel block demeaning (identical ``_demean_by_block``), then
  either a per-coarse-block 64-point FFT (variant "perblock") or a full-band
  FFT over all 23 064 fine channels (variant "fullband").  Both variants are
  forecast; the frozen floor passes if EITHER reaches 3 sigma at the floor
  cell — T5 later freezes the winner before G1''.
* Cross power: P(k) = Re[F1(k) conj F2(k)] — noise-bias-free because the two
  splits carry independent noise.
* Template: T(k) = <|D(k)|^2> over synthetic unit-variance Lorentzian-ACF gain
  fields pushed through the SAME mask + demeaning + transform (this bakes the
  block-demeaning transfer function into the template — the T4 correction —
  because demeaning removes most of a scintle wider than the 390.6 kHz block).
  The expected signal cross power is a * T(k) with a = (f_b * m)^2: the ratio
  field carries f_b*m*delta(nu) in both splits (the constant f_b offset dies
  in demeaning).
* Forecast: SNR(m, dnu) = (f_b*m)^2 * sqrt(sum_k T(k)^2 / Var_null(k)) — the
  matched-filter significance with empirical weights (w = 1/Var).
* Empirical cross-check at the floor cell: 50 multiplicative injections
  (gain = 1 + f_b(1 + m*delta), the G1 convention, seeds 8000+r = G1 cell 8)
  through the matched estimator; median(a_hat)/sigma_null must agree with the
  formula to within ~30% for the formula verdict to stand.

Seed spaces: nulls 900000+i (P2 G2), injections 1000*cell+r (P2 G1, cell 8),
templates 700000 + 1000*dnu_index + j (new, disjoint from both).

Blinding: only off-pulse samples are ever read (the P2 harness' structural
guard applies); no on-pulse data enters this forecast.

Run: conda py312; ``python gate0b_forecast.py`` writes gate0b_forecast.json
and gate0b_forecast.png next to this file and prints the verdict JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
P2_DIR = HERE.parent / "p2-routeb-voltage"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(P2_DIR))

import routeb_calibration as p2  # noqa: E402  (P2 harness: Products, constants)
from scintillation.scint_analysis import routeb_voltage as rb  # noqa: E402
from scintillation.scint_analysis.cross_acf import _demean_by_block  # noqa: E402

# ---- frozen Gate-0b parameters (record: experiment-chime-scint-p3-...) -------
N_NULL = 100
NULL_SEED_BASE = p2.G2_SEED_BASE  # 900000 + i
N_TEMPLATE = 200
TEMPLATE_SEED_BASE = 700_000  # + 1000*dnu_index + j ; disjoint from G1/G2 spaces
N_INJECTION_CHECK = 50
FLOOR_M = 0.17
FLOOR_DNU_KHZ = 213.0
FLOOR_SNR = 3.0
INJECTION_CELL_SEED = 1000 * 8  # G1 cell 8 = (m=0.17, dnu=213 kHz), + realization
MODULATIONS = p2.MODULATIONS  # (0.15, 0.17)
DNU_KHZ = p2.DNU_KHZ  # (35, 77, 127, 213, 352)
F_B = p2.BURST_FLUX_FRACTION  # 0.05
N_VAR_BANDS = 48  # log-spaced delay bands for smoothing the full-band Var(k)
MIN_BLOCK_FINITE = 48  # perblock variant: skip blocks with < 48/64 good channels


def split_ratio_fields(products, on_samples, off_samples, *, on_gain=None):
    """The S2 split-ratio fields, verbatim P2 construction (pol-mean halves).

    Returns (field1, field2): independent-noise ratio spectra whose shared
    content is f_b*m*delta(nu) (plus the demeaning-removed f_b offset).
    """
    on = rb.assert_offpulse_samples(on_samples, name="on_samples")
    off = rb.assert_offpulse_samples(off_samples, name="off_samples")
    on_a, on_b = rb._split_halves(on)
    off_a, off_b = rb._split_halves(off)

    def half(on_h, off_h):
        per_pol = [
            rb._ratio(rb._mean_frame(dyn, on_h, on_gain), rb._mean_frame(dyn, off_h, None))
            for dyn in products.dynamic
        ]
        return 0.5 * (per_pol[0] + per_pol[1])

    return half(on_a, off_a), half(on_b, off_b)


class DelayTransforms:
    """Block-demeaned delay-domain transforms shared by data and templates."""

    def __init__(self, block_ids: np.ndarray) -> None:
        _, self.codes = np.unique(np.asarray(block_ids), return_inverse=True)
        self.n_blocks = int(self.codes.max() + 1)
        # perblock variant: only full 64-channel blocks participate, so every
        # block shares one delay grid (tau_k = k / 390.625 kHz).
        self.full_blocks = [
            np.flatnonzero(self.codes == b)
            for b in range(self.n_blocks)
            if (self.codes == b).sum() == p2.CHANNELS_PER_COARSE
        ]

    def demean(self, field: np.ndarray) -> np.ndarray:
        return _demean_by_block(np.asarray(field, dtype=float), self.codes, self.n_blocks)

    def fullband(self, field: np.ndarray) -> np.ndarray:
        """rfft of the demeaned field, NaN->0 (mask acts as a window; the
        template sees the identical window, so the transfer cancels in a/T)."""
        x = self.demean(field)
        x = np.where(np.isfinite(x), x, 0.0)
        return np.fft.rfft(x)

    def perblock(self, field: np.ndarray) -> np.ndarray:
        """Stack of per-block rffts (blocks with >= MIN_BLOCK_FINITE good
        channels), NaN->0 after demeaning; DC bin dropped by the caller."""
        x = self.demean(field)
        rows = []
        for idx in self.full_blocks:
            block = x[idx]
            finite = np.isfinite(block)
            if finite.sum() < MIN_BLOCK_FINITE:
                continue
            rows.append(np.fft.rfft(np.where(finite, block, 0.0)))
        return np.asarray(rows)

    def cross_power(self, variant: str, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        t1, t2 = getattr(self, variant)(f1), getattr(self, variant)(f2)
        if variant == "perblock":
            n = min(len(t1), len(t2))
            power = np.real(t1[:n] * np.conj(t2[:n])).mean(axis=0)
            return power[1:]  # drop DC (demeaned anyway)
        return np.real(t1 * np.conj(t2))[1:]


def smooth_variance(var: np.ndarray) -> np.ndarray:
    """Log-spaced band-average of Var(k): 100 null samples per raw bin is too
    noisy for stable 1/Var weights over ~11k full-band bins; band-averaging is
    slightly conservative (it can only understate the achievable SNR)."""
    n = var.size
    if n <= N_VAR_BANDS:
        return var
    edges = np.unique(np.geomspace(1, n, N_VAR_BANDS + 1).astype(int))
    smoothed = np.empty_like(var)
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = max(hi, lo + 1)
        smoothed[lo - 1 : hi] = var[lo - 1 : hi].mean()
    return smoothed


def template_bank(products, transforms, variant: str) -> dict[float, np.ndarray]:
    """T(k) per Δν_d: mean delay power of unit-variance Lorentzian-ACF fields
    through the identical mask + demeaning + transform (T4 baked in)."""
    bad = ~products.good_channels
    bank = {}
    for dnu_index, dnu_khz in enumerate(DNU_KHZ):
        acc = None
        for j in range(N_TEMPLATE):
            rng = np.random.default_rng(TEMPLATE_SEED_BASE + 1000 * dnu_index + j)
            delta = rb.lorentzian_gain_field(
                rng,
                n_channels=products.n_band_channels,
                width_channels=p2._width_channels(dnu_khz),
            )
            delta = delta.astype(float)
            delta[bad] = np.nan
            power = transforms.cross_power(variant, delta, delta)
            acc = power if acc is None else acc + power
        bank[dnu_khz] = acc / N_TEMPLATE
    return bank


def matched_estimate(power: np.ndarray, template: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(np.sum(weights * template**2))
    return float(np.sum(weights * power * template) / denominator)


def main() -> int:
    class Args:
        pol0 = p2.DEFAULT_POL0
        pol1 = p2.DEFAULT_POL1
        frequencies = p2.DEFAULT_FREQUENCIES
        time0_metadata = p2.DEFAULT_METADATA

    products = p2.Products(Args())
    transforms = DelayTransforms(products.parent)
    pool = products.off_pool

    # -- null campaign: measured delay-domain cross-power variance -------------
    null_powers = {"perblock": [], "fullband": []}
    for i in range(N_NULL):
        rng = np.random.default_rng(NULL_SEED_BASE + i)
        perm = rng.permutation(pool)
        on = np.sort(perm[: p2.N_ON])
        off = np.sort(perm[p2.N_ON :])
        f1, f2 = split_ratio_fields(products, on, off)
        for variant in null_powers:
            null_powers[variant].append(transforms.cross_power(variant, f1, f2))
    results = {}
    for variant, stack_list in null_powers.items():
        stack = np.asarray(stack_list)
        variance = stack.var(axis=0, ddof=1)
        if variant == "fullband":
            variance = smooth_variance(variance)
        weights = 1.0 / variance
        bank = template_bank(products, transforms, variant)
        forecast = {}
        null_ahat = {}
        for dnu_khz, template in bank.items():
            sigma = float(np.sum(template**2 / variance) ** -0.5)
            forecast[dnu_khz] = {
                m: (F_B * m) ** 2 / sigma for m in MODULATIONS
            }
            a_null = [matched_estimate(p, template, weights) for p in stack]
            null_ahat[dnu_khz] = {
                "sigma_analytic": sigma,
                "sigma_empirical": float(np.std(a_null, ddof=1)),
                "mean": float(np.mean(a_null)),
            }
        results[variant] = {
            "forecast_snr": forecast,
            "null_ahat": null_ahat,
            "variance": variance,
            "weights": weights,
            "bank": bank,
            "stack": stack,
        }

    # -- empirical injection cross-check at the floor cell ---------------------
    variant_at_floor = max(
        results, key=lambda v: results[v]["forecast_snr"][FLOOR_DNU_KHZ][FLOOR_M]
    )
    bank = results[variant_at_floor]["bank"]
    weights = results[variant_at_floor]["weights"]
    template = bank[FLOOR_DNU_KHZ]
    injected = []
    for r in range(N_INJECTION_CHECK):
        rng = np.random.default_rng(INJECTION_CELL_SEED + r)
        perm = rng.permutation(pool)
        on = np.sort(perm[: p2.N_ON])
        off = np.sort(perm[p2.N_ON :])
        delta = rb.lorentzian_gain_field(
            rng,
            n_channels=products.n_band_channels,
            width_channels=p2._width_channels(FLOOR_DNU_KHZ),
        )
        gain = 1.0 + F_B * (1.0 + FLOOR_M * delta)
        f1, f2 = split_ratio_fields(products, on, off, on_gain=gain)
        power = transforms.cross_power(variant_at_floor, f1, f2)
        injected.append(matched_estimate(power, template, weights))
    sigma_null = results[variant_at_floor]["null_ahat"][FLOOR_DNU_KHZ]["sigma_empirical"]
    a_true = (F_B * FLOOR_M) ** 2
    empirical_snr = float(np.median(injected) / sigma_null)
    formula_snr = results[variant_at_floor]["forecast_snr"][FLOOR_DNU_KHZ][FLOOR_M]

    floor_snr = max(
        results[v]["forecast_snr"][FLOOR_DNU_KHZ][FLOOR_M] for v in results
    )
    verdict = "PROCEED" if floor_snr >= FLOOR_SNR else "DOCUMENTED-FAIL-BY-FORECAST"

    # -- figure (visual vetting) ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for variant, marker in (("perblock", "o"), ("fullband", "s")):
        for m, style in zip(MODULATIONS, ("--", "-")):
            snr = [results[variant]["forecast_snr"][d][m] for d in DNU_KHZ]
            axes[0].plot(DNU_KHZ, snr, style, marker=marker, label=f"{variant} m={m}")
    axes[0].axhline(FLOOR_SNR, color="r", lw=1, label="floor 3σ")
    axes[0].axvline(FLOOR_DNU_KHZ, color="r", lw=0.5, ls=":")
    axes[0].set(
        xscale="log", yscale="log", xlabel="Δν_d [kHz]",
        ylabel="forecast SNR (measured noise)", title="Gate 0b forecast",
    )
    axes[0].legend(fontsize=7)
    for variant in results:
        var = results[variant]["variance"]
        axes[1].plot(np.arange(1, var.size + 1), var, lw=0.7, label=variant)
    axes[1].set(
        xscale="log", yscale="log", xlabel="delay bin k",
        ylabel="Var_null[P(k)]", title=f"measured null variance (N={N_NULL})",
    )
    axes[1].legend(fontsize=8)
    axes[2].hist(injected, bins=15, alpha=0.7, label="injected â (floor cell)")
    axes[2].axvline(a_true, color="g", label=f"a_true={a_true:.2e}")
    axes[2].axvline(0, color="k", lw=0.5)
    axes[2].axvline(sigma_null * FLOOR_SNR, color="r", ls=":", label="3σ_null")
    axes[2].set(
        xlabel="â", ylabel="count",
        title=f"injection check {variant_at_floor}: emp {empirical_snr:.2f} vs formula {formula_snr:.2f}",
    )
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(HERE / "gate0b_forecast.png", dpi=150)

    payload = {
        "experiment": "p3-optimal-estimator",
        "gate": "0b",
        "record": "docs/rse/specs/experiment-chime-scint-p3-optimal-estimator.md",
        "inputs_sha256": products.inputs_sha256,
        "n_null": N_NULL,
        "n_template": N_TEMPLATE,
        "n_injection_check": N_INJECTION_CHECK,
        "seed_bases": {
            "null": NULL_SEED_BASE,
            "template": TEMPLATE_SEED_BASE,
            "injection_check": INJECTION_CELL_SEED,
        },
        "burst_flux_fraction": F_B,
        "floor": {"m": FLOOR_M, "dnu_khz": FLOOR_DNU_KHZ, "snr_min": FLOOR_SNR},
        "forecast_snr": {
            variant: {
                str(d): {str(m): results[variant]["forecast_snr"][d][m] for m in MODULATIONS}
                for d in DNU_KHZ
            }
            for variant in results
        },
        "sigma_calibration": {
            variant: {
                str(d): {
                    "analytic": results[variant]["null_ahat"][d]["sigma_analytic"],
                    "empirical": results[variant]["null_ahat"][d]["sigma_empirical"],
                    "null_mean": results[variant]["null_ahat"][d]["mean"],
                }
                for d in DNU_KHZ
            }
            for variant in results
        },
        "injection_check": {
            "variant": variant_at_floor,
            "a_true": a_true,
            "a_hat_median": float(np.median(injected)),
            "a_hat_values": [float(v) for v in injected],
            "empirical_snr": empirical_snr,
            "formula_snr": formula_snr,
        },
        "floor_snr": floor_snr,
        "floor_variant": variant_at_floor,
        "verdict": verdict,
    }
    (HERE / "gate0b_forecast.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "floor_snr": round(floor_snr, 3),
                "floor_variant": variant_at_floor,
                "empirical_check_snr": round(empirical_snr, 3),
                "formula_snr_at_floor": round(formula_snr, 3),
                "perblock_213_m017": round(results["perblock"]["forecast_snr"][213.0][0.17], 3),
                "fullband_213_m017": round(results["fullband"]["forecast_snr"][213.0][0.17], 3),
            },
            sort_keys=True,
        )
    )
    return 0 if verdict == "PROCEED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
