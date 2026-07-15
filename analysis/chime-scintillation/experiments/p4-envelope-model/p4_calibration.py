#!/usr/bin/env python3
"""P4 exploratory intrinsic-envelope modeling + residual scintillation search.

Implements experiment P4 from the predeclared record
``docs/rse/specs/experiment-chime-scint-p4-envelope-model.md`` (Faber2026):
freya's CHIME on-pulse ratio spectrum is dominated by broad intrinsic
structure (P3' unblinding); P4 fits three frozen envelope-model families,
divides them out, and searches the residual with the P3' matched scan whose
templates and null calibration are REBUILT through the identical
model+subtract chain per operating point (the Gate-0b transfer lesson).

Everything on-pulse here is post-unblinding, owner-sanctioned 2026-07-15;
every ``allow_unblind=True`` call site carries the sanction comment. The
record's look accounting still binds: the real on-pulse residual is scanned
only in ``e3``, once per surviving family at its E2-frozen operating point.

Subcommands:

  freeze    load data, compute the reference envelope + injection weight +
            per-channel noise level, write p4_frozen_config.json (hashed)
  e0        descriptive envelope characterization + profile/component rule
  e2        injection calibration: recovery arm, noise/null arm, surrogate
            model-mismatch control; freeze operating points (fail branch
            terminates P4 here)
  e3        the (up to) three real-residual looks + discriminants E3a/E3b/E3c
  verdict   aggregate -> frozen verdict taxonomy

Frozen seed spaces (disjoint from every prior campaign space):
injections ``600000 + 1000*cell + r`` (cell = m_index*4 + dnu_index);
noise nulls ``650000 + i``; surrogates ``680000 + 1000*a_index + r``;
templates ``760000 + 1000*grid_index + j``; E3a sub-band templates
``770000 + 100000*band + 1000*grid_index + j``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
P2_DIR = HERE.parent / "p2-routeb-voltage"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(P2_DIR))

import routeb_calibration as p2  # noqa: E402
from scintillation.scint_analysis import envelope_model as em  # noqa: E402
from scintillation.scint_analysis import optimal_dnu as od  # noqa: E402
from scintillation.scint_analysis import routeb_voltage as rb  # noqa: E402

EXPERIMENT_ID = "p4-envelope-model"
RECORD = "docs/rse/specs/experiment-chime-scint-p4-envelope-model.md"
SANCTION = "P4 exploratory: post-unblinding, owner-sanctioned 2026-07-15"

# ---- frozen seed spaces (record §E2) -----------------------------------------
INJ_SEED_BASE = 600_000  # + 1000*cell + r ; cell = m_index*4 + dnu_index
NULL_SEED_BASE = 650_000  # + i
SURR_SEED_BASE = 680_000  # + 1000*a_index + r ; a_index = source-family index
TEMPLATE_SEED_BASE = 760_000  # + 1000*grid_index + j
SUBBAND_TEMPLATE_BASE = 770_000  # + 100000*band + 1000*grid_index + j

# ---- frozen grids and gates (record §E1/E2/E3) --------------------------------
P4_DNU_KHZ = (77.0, 127.0, 213.0, 352.0)
N_REALIZATIONS = 50
N_NULL = 100
N_SURROGATE = 50
CERT_DNU_TOL = 0.30
CERT_PULL_MAX = 2.0
CERT_CONVERGENCE_MIN = 0.90
CONTROL_CLEAN_FACTOR = 1.5  # surrogate p95 max-z <= noise p95 max-z * factor
DROP_MIN_DNU_KHZ = 127.0  # family drops with zero certified cells >= this
CANDIDATE_PERCENTILE = 95.0  # trials threshold on the null max-over-families
ADMISSIBLE_M = (0.05, 0.30)  # implied modulation range at the argmax
E3A_MIN_Z = 5.0  # sub-band scaling runs only above this (power)
E3A_ALPHA_BAND = (3.0, 5.0)
E3A_N_SUBBANDS = 4
E3B_SMOOTH_SIGMA = 2.0  # samples; frozen component rule
E3B_PEAK_SNR_MIN = 5.0
E3B_PEAK_SEP_MIN = 3
E3C_FACTOR = 3.0
E3C_SCALING_INDEX = 4.4

# ---- frozen chain constants ---------------------------------------------------
M_TEMPLATE = 0.16  # template linearization amplitude (mid-prior modulation)
REFERENCE_ENVELOPE_KENV = od.KMIN  # the scan's own envelope definition (k < 11)
SURROGATE_SOURCE_SCALE = {"M1_spline": 0.5, "M2_gp": 0.5, "M3_delaycut": 200}
FAMILIES = tuple(em.FAMILY_SCALES)

FROZEN = HERE / "p4_frozen_config.json"
ASSETS = HERE / "p4_assets.npz"
FIGDIR = HERE / "figures"


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


def _frozen_or_die() -> dict:
    if not FROZEN.exists():
        raise SystemExit("p4_frozen_config.json missing -- run the freeze subcommand first")
    return json.loads(FROZEN.read_text())


def _frozen_sha() -> str:
    return _sha256(FROZEN)


def _assets() -> dict:
    return dict(np.load(ASSETS))


def _delay_transform_batch(fields: np.ndarray) -> np.ndarray:
    """Batched twin of optimal_dnu.delay_transform (row-demeaned, NaN->0,
    rfft, DC dropped) -- must stay bit-identical in semantics."""
    x = np.asarray(fields, dtype=float)
    x = x - np.nanmean(x, axis=-1, keepdims=True)
    x = np.where(np.isfinite(x), x, 0.0)
    return np.fft.rfft(x, axis=-1)[..., 1:]


class OpChain:
    """One (family, scale) model+subtract chain over a channel selection."""

    def __init__(self, family: str, scale, nu_mhz, good_mask, noise_variance: float):
        self.family = family
        self.scale = scale
        self.good = np.asarray(good_mask, dtype=bool)
        self.chain = em.EnvelopeChain(family, scale, nu_mhz, good_mask, noise_variance)

    def residuals(self, field_rows: np.ndarray) -> np.ndarray:
        """Split-ratio fields (ratio - 1) -> multiplicative residuals."""
        spectra = 1.0 + np.atleast_2d(np.asarray(field_rows, dtype=float))
        return em.residual(spectra, self.chain.envelope(spectra))

    def cross_powers(self, fields1: np.ndarray, fields2: np.ndarray) -> np.ndarray:
        """Per-half envelope fits keep the halves' noise independent, so the
        cross power of the residuals stays noise-bias-free (P3 construction)."""
        d1 = _delay_transform_batch(self.residuals(fields1))
        d2 = _delay_transform_batch(self.residuals(fields2))
        return np.real(d1 * np.conj(d2))

    def build_scan(self, e_ref: np.ndarray, channel_width_khz: float,
                   null_pairs: np.ndarray, *, dnu_grid=None,
                   template_seed_base: int = TEMPLATE_SEED_BASE) -> od.MatchedScan:
        """MatchedScan with templates + null calibration rebuilt through THIS
        chain (record §E1: the transfer lives inside the Monte-Carlo bank).

        Template construction mirrors the physical signal structure
        R = E_true + (E_true - 1) * m * delta: noiseless synthetic spectra
        R_syn = E_ref + M_TEMPLATE * (E_ref - 1) * delta through the chain,
        normalized by (M_TEMPLATE * f_b)^2 so a_hat stays in the P3' units
        a = (f_b * m)^2 and the record's m = sqrt(a_res)/f_b applies.
        """
        grid = od.DNU_SCAN_KHZ if dnu_grid is None else np.asarray(dnu_grid, dtype=float)
        n = e_ref.size
        templates = np.empty((grid.size, n // 2), dtype=float)
        carrier = M_TEMPLATE * (e_ref - 1.0)
        for gi, dnu in enumerate(grid):
            width_channels = float(dnu) / channel_width_khz
            fields = np.empty((od.N_TEMPLATE, n), dtype=float)
            for j in range(od.N_TEMPLATE):
                rng = np.random.default_rng(template_seed_base + 1000 * gi + j)
                fields[j] = rb.lorentzian_gain_field(
                    rng, n_channels=n, width_channels=width_channels
                )
            syn = e_ref[None, :] + carrier[None, :] * fields
            syn[:, ~self.good] = np.nan  # identical mask window as the data
            res = em.residual(syn, self.chain.envelope(syn))
            spectra = _delay_transform_batch(res)
            templates[gi] = (np.abs(spectra) ** 2).mean(axis=0) / (
                M_TEMPLATE * p2.BURST_FLUX_FRACTION
            ) ** 2
        null_powers = self.cross_powers(null_pairs[:, 0], null_pairs[:, 1])
        variance = od.smooth_variance(null_powers.var(axis=0, ddof=1))
        scan = od.MatchedScan(grid, templates, variance, kmin=od.KMIN)
        scan.calibrate(null_powers)
        self.null_powers = null_powers
        return scan


def _null_field_pairs(products, n: int = N_NULL) -> np.ndarray:
    """Noise/null arm: off-pulse permutation pseudo-on nulls (seeds 650000+i)."""
    pairs = np.empty((n, 2, products.n_band_channels), dtype=float)
    for i in range(n):
        rng = np.random.default_rng(NULL_SEED_BASE + i)
        perm = rng.permutation(products.off_pool)
        on = np.sort(perm[: p2.N_ON])
        off = np.sort(perm[p2.N_ON :])
        pairs[i] = od.split_ratio_fields(products.dynamic, on, off)
    return pairs


def _onpulse_fields(products) -> tuple[np.ndarray, np.ndarray]:
    on = rb.samples_from_window(p2.ON_PULSE_WINDOW)
    return od.split_ratio_fields(
        products.dynamic, on, products.off_pool,
        allow_unblind=True,  # P4 exploratory: post-unblinding, owner-sanctioned 2026-07-15
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- #
# freeze
# --------------------------------------------------------------------------- #
def cmd_freeze(args: argparse.Namespace) -> int:
    products = _load_products()
    f1, f2 = _onpulse_fields(products)
    field_mean = 0.5 * (f1 + f2)
    r_mean = 1.0 + field_mean
    # reference envelope: delay low-pass at the scan's own envelope boundary
    # (k < KMIN = structure smoother than ~12.8 MHz) -- no new free scale
    e_ref = em.fit_delay_lowpass(r_mean, REFERENCE_ENVELOPE_KENV)
    # injection weight: a real scintle modulates only the burst component, so
    # the on-frame gain is 1 + m*delta*(E-1)/E (signal f_b*s*m*delta)
    inj_weight = (e_ref - 1.0) / e_ref

    null_pairs = _null_field_pairs(products)
    flat = null_pairs.reshape(-1, products.n_band_channels)
    sigma_ch = np.nanstd(flat, axis=0, ddof=1)
    noise_variance = float(np.nanmedian(sigma_ch[products.good_channels] ** 2))

    np.savez_compressed(
        ASSETS,
        field_half1=f1,
        field_half2=f2,
        e_ref=e_ref,
        inj_weight=inj_weight,
        sigma_ch=sigma_ch,
        noise_variance=noise_variance,
    )
    config = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "record": RECORD,
        "evidential_class": "exploratory (post-unblinding, owner-sanctioned 2026-07-15)",
        "band_mhz": list(p2.BAND_MHZ),
        "lte_exclusion_mhz": list(p2.LTE_EXCLUSION_MHZ),
        "on_pulse_window": list(p2.ON_PULSE_WINDOW),
        "off_pool_windows": [list(w) for w in p2.OFF_POOL_WINDOWS],
        "n_band_channels": products.n_band_channels,
        "n_good_channels": int(products.good_channels.sum()),
        "channel_width_khz": products.channel_width_mhz * 1e3,
        "families": {k: list(v) for k, v in em.FAMILY_SCALES.items()},
        "chain": {
            "residual": "r = R/E - 1 per split half; per-half envelope fits keep the cross power noise-bias-free",
            "clip_percentile": em.ENVELOPE_CLIP_PERCENTILE,
            "gp_decimation": em.GP_DECIMATION,
            "reference_envelope": f"delay low-pass k<{REFERENCE_ENVELOPE_KENV} of the on-pulse mean ratio spectrum",
            "injection_weight": "(E_ref-1)/E_ref (scintle rides the burst component only)",
            "template_construction": "R_syn = E_ref + M_TEMPLATE*(E_ref-1)*delta through the chain; T normalized by (M_TEMPLATE*f_b)^2 so a == (f_b*m)^2",
            "m_template": M_TEMPLATE,
            "burst_flux_fraction": p2.BURST_FLUX_FRACTION,
            "matched_scan": {
                "kmin": od.KMIN,
                "dnu_scan_khz": [float(d) for d in od.DNU_SCAN_KHZ],
                "n_template": od.N_TEMPLATE,
                "variance_smoothing_bands": od.N_VAR_BANDS,
            },
        },
        "seeds": {
            "injections": f"{INJ_SEED_BASE} + 1000*cell + r ; cell = m_index*4 + dnu_index",
            "nulls": f"{NULL_SEED_BASE} + i (n={N_NULL})",
            "surrogates": f"{SURR_SEED_BASE} + 1000*a_index + r (n={N_SURROGATE} per source family)",
            "templates": f"{TEMPLATE_SEED_BASE} + 1000*grid_index + j",
            "subband_templates": f"{SUBBAND_TEMPLATE_BASE} + 100000*band + 1000*grid_index + j",
        },
        "e2": {
            "modulations": list(p2.MODULATIONS),
            "dnu_khz": list(P4_DNU_KHZ),
            "n_realizations": N_REALIZATIONS,
            "certification": {
                "dnu_fractional_tolerance": CERT_DNU_TOL,
                "amplitude_pull_max": CERT_PULL_MAX,
                "convergence_min": CERT_CONVERGENCE_MIN,
                "a_true": "(f_b*m)^2",
            },
            "surrogate_source_scale": SURROGATE_SOURCE_SCALE,
            "control_clean": f"surrogate p95 max-z <= noise p95 max-z * {CONTROL_CLEAN_FACTOR}",
            "operating_point_rule": "among control-clean scales: max certified cells; ties -> 213 kHz certification, then smaller lambda/ell (larger k_env)",
            "drop_rule": f"zero certified cells at dnu >= {DROP_MIN_DNU_KHZ} kHz, or no control-clean scale",
        },
        "e3": {
            "looks": "one real on-pulse residual scan per surviving family at its operating point",
            "trials": f"max over surviving families; threshold = p{CANDIDATE_PERCENTILE:.0f} of the same statistic on the {N_NULL} noise-arm nulls",
            "amplitude_admissibility_m": list(ADMISSIBLE_M),
            "e3a": {"min_z": E3A_MIN_Z, "alpha_band": list(E3A_ALPHA_BAND), "n_subbands": E3A_N_SUBBANDS},
            "e3b": {"smooth_sigma": E3B_SMOOTH_SIGMA, "peak_snr_min": E3B_PEAK_SNR_MIN, "peak_sep_min": E3B_PEAK_SEP_MIN},
            "e3c": {"factor": E3C_FACTOR, "scaling_index": E3C_SCALING_INDEX},
        },
        "noise_variance": noise_variance,
        "inputs": products.inputs,
        "inputs_sha256": products.inputs_sha256,
        "assets_sha256": _sha256(ASSETS),
    }
    _write_json(FROZEN, config)
    print(json.dumps({
        "frozen_config_sha256": _frozen_sha(),
        "n_good_channels": config["n_good_channels"],
        "noise_variance": noise_variance,
        "envelope_mean_contrast": float(np.nanmean(e_ref - 1.0)),
        "envelope_p95_contrast": float(np.nanpercentile(e_ref - 1.0, 95)),
    }, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# e0 -- descriptive envelope characterization + frozen component rule
# --------------------------------------------------------------------------- #
def cmd_e0(args: argparse.Namespace) -> int:
    _frozen_or_die()
    products = _load_products()
    assets = _assets()
    f1, f2 = assets["field_half1"], assets["field_half2"]
    e_ref = assets["e_ref"]
    field_mean = 0.5 * (f1 + f2)
    good = products.good_channels
    nu = products.frequencies

    finite = good & np.isfinite(f1) & np.isfinite(f2)
    half_corr = float(np.corrcoef(f1[finite], f2[finite])[0, 1])

    cross_full = np.real(
        _delay_transform_batch(f1[None]) * np.conj(_delay_transform_batch(f2[None]))
    )[0]
    power_mean = np.abs(_delay_transform_batch(field_mean[None])[0]) ** 2

    acf = np.fft.irfft(np.concatenate([[0.0], power_mean]))
    acf = acf[: acf.size // 2] / acf[0]
    below = np.flatnonzero(acf < 0.5)
    hwhm_mhz = float(below[0] * products.channel_width_mhz) if below.size else np.nan

    amp = field_mean[finite]
    amplitude_stats = {
        "mean": float(np.mean(amp)),
        "median": float(np.median(amp)),
        "p5": float(np.percentile(amp, 5)),
        "p95": float(np.percentile(amp, 95)),
        "min": float(np.min(amp)),
        "max": float(np.max(amp)),
        "fraction_negative": float(np.mean(amp < 0)),
    }

    # profile + frozen component rule (E3b availability). On-pulse samples are
    # read here: P4 exploratory: post-unblinding, owner-sanctioned 2026-07-15
    pol_mean = 0.5 * (products.dynamic[0] + products.dynamic[1])
    profile = np.nanmean(pol_mean[good], axis=0)
    off_idx = products.off_pool
    baseline = float(np.median(profile[off_idx]))
    noise = float(np.std(profile[off_idx], ddof=1))
    snr = (profile - baseline) / noise
    snr_smooth = gaussian_filter1d(snr, E3B_SMOOTH_SIGMA)
    lo, hi = p2.ON_PULSE_WINDOW
    components = []
    for t in range(lo + 1, hi - 1):
        if snr_smooth[t] >= E3B_PEAK_SNR_MIN and snr_smooth[t] > snr_smooth[t - 1] and snr_smooth[t] >= snr_smooth[t + 1]:
            if components and t - components[-1]["sample"] < E3B_PEAK_SEP_MIN:
                if snr_smooth[t] > components[-1]["snr"]:
                    components[-1] = {"sample": t, "snr": float(snr_smooth[t])}
                continue
            components.append({"sample": t, "snr": float(snr_smooth[t])})
    # component windows: split at the smoothed-profile minima between peaks
    windows = []
    if len(components) >= 2:
        peaks = [c["sample"] for c in components]
        edges = [lo]
        for a, b in zip(peaks[:-1], peaks[1:]):
            edges.append(a + int(np.argmin(snr_smooth[a:b])))
        edges.append(hi)
        windows = [[int(a), int(b)] for a, b in zip(edges[:-1], edges[1:])]

    payload = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": _frozen_sha(),
        "half_to_half_correlation": half_corr,
        "acf_hwhm_mhz": hwhm_mhz,
        "amplitude_stats": amplitude_stats,
        "profile": {
            "baseline": baseline,
            "noise": noise,
            "peak_snr_smoothed": float(np.max(snr_smooth[lo:hi])),
            "components": components,
            "component_windows": windows,
            "e3b_available": len(components) >= 2,
        },
        "delay_power_mean_k1_10": float(power_mean[: od.KMIN - 1].mean()),
        "delay_power_mean_k11_100": float(power_mean[od.KMIN - 1 : 100].mean()),
    }
    _write_json(HERE / "e0_envelope.json", payload)

    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(nu[good], field_mean[good], lw=0.2, color="0.6", label="on-pulse ratio - 1 (half mean)")
    ax.plot(nu, e_ref - 1.0, lw=1.5, color="crimson", label=f"reference envelope (k<{REFERENCE_ENVELOPE_KENV})")
    ax.set_xlabel("frequency [MHz]"); ax.set_ylabel("R - 1")
    ax.legend(fontsize=8); ax.set_title("freya CHIME on-pulse ratio spectrum")
    ax = axes[0, 1]
    k = np.arange(1, power_mean.size + 1)
    ax.loglog(k, power_mean, lw=0.6, label="|D(mean)|^2")
    ax.loglog(k, np.abs(cross_full), lw=0.4, alpha=0.6, label="|cross P(k)|")
    ax.axvline(od.KMIN, color="crimson", ls="--", lw=1, label=f"KMIN={od.KMIN}")
    ax.set_xlabel("delay bin k"); ax.set_ylabel("delay power")
    ax.legend(fontsize=8); ax.set_title("delay power (full k)")
    ax = axes[1, 0]
    ax.plot(f1[finite][::7], f2[finite][::7], ".", ms=1, alpha=0.3)
    ax.set_xlabel("half 1 field"); ax.set_ylabel("half 2 field")
    ax.set_title(f"half-to-half correlation r = {half_corr:.3f}")
    ax = axes[1, 1]
    ax.hist(amp, bins=200, histtype="step", color="k")
    ax.set_xlabel("R - 1"); ax.set_ylabel("channels")
    ax.set_title("amplitude distribution (good channels)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "e0_envelope.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    t = np.arange(profile.size)
    ax.plot(t, snr, lw=0.5, color="0.6", label="band-mean S/N")
    ax.plot(t, snr_smooth, lw=1.5, color="k", label=f"Gaussian sigma={E3B_SMOOTH_SIGMA}")
    ax.axvspan(lo, hi, color="crimson", alpha=0.08, label="on-pulse window")
    ax.axhline(E3B_PEAK_SNR_MIN, color="crimson", ls=":", lw=1, label="component S/N >= 5")
    for c in components:
        ax.axvline(c["sample"], color="royalblue", ls="--", lw=1)
    ax.set_xlabel("time sample"); ax.set_ylabel("S/N")
    ax.legend(fontsize=8)
    ax.set_title(f"freya CHIME profile -- {len(components)} qualifying component(s); E3b {'available' if len(components) >= 2 else 'unavailable'}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "e0_profile.png", dpi=150)
    plt.close(fig)

    print(json.dumps({
        "half_to_half_correlation": round(half_corr, 4),
        "acf_hwhm_mhz": None if not np.isfinite(hwhm_mhz) else round(hwhm_mhz, 3),
        "n_components": len(components),
        "e3b_available": len(components) >= 2,
        "peak_snr": round(payload["profile"]["peak_snr_smoothed"], 1),
    }, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# e2 -- injection calibration, noise/null arm, surrogate control, op freeze
# --------------------------------------------------------------------------- #
def _injection_field_pairs(products, inj_weight: np.ndarray) -> dict:
    """Recovery arm: real on-pulse frames x (1 + m*delta*w) per record cell."""
    on = rb.samples_from_window(p2.ON_PULSE_WINDOW)
    channel_width_khz = products.channel_width_mhz * 1e3
    out = {}
    for m_index, m in enumerate(p2.MODULATIONS):
        for dnu_index, dnu_khz in enumerate(P4_DNU_KHZ):
            cell = m_index * len(P4_DNU_KHZ) + dnu_index
            pairs = np.empty((N_REALIZATIONS, 2, products.n_band_channels), dtype=float)
            for r in range(N_REALIZATIONS):
                rng = np.random.default_rng(INJ_SEED_BASE + 1000 * cell + r)
                delta = rb.lorentzian_gain_field(
                    rng,
                    n_channels=products.n_band_channels,
                    width_channels=dnu_khz / channel_width_khz,
                )
                gain = 1.0 + m * delta * inj_weight
                pairs[r] = od.split_ratio_fields(
                    products.dynamic, on, products.off_pool, on_gain=gain,
                    allow_unblind=True,  # P4 exploratory: post-unblinding, owner-sanctioned 2026-07-15
                )
            out[(m, dnu_khz, cell)] = pairs
    return out


def _surrogate_field_pairs(products, assets: dict, noise_variance: float) -> dict:
    """Model-mismatch control: E_A*(1 + noise) halves per source family A."""
    r_mean = 1.0 + 0.5 * (assets["field_half1"] + assets["field_half2"])
    sigma = assets["sigma_ch"]
    good = products.good_channels
    out = {}
    for a_index, family in enumerate(FAMILIES):
        chain = em.EnvelopeChain(
            family, SURROGATE_SOURCE_SCALE[family],
            products.frequencies, good, noise_variance,
        )
        e_a = chain.envelope(r_mean)
        pairs = np.empty((N_SURROGATE, 2, products.n_band_channels), dtype=float)
        for r in range(N_SURROGATE):
            rng = np.random.default_rng(SURR_SEED_BASE + 1000 * a_index + r)
            for h in range(2):
                spec = e_a * (1.0 + sigma * rng.standard_normal(e_a.size))
                spec[~good] = np.nan
                pairs[r, h] = spec - 1.0  # chain API takes ratio-1 fields
        out[family] = pairs
    return out


def cmd_e2(args: argparse.Namespace) -> int:
    frozen = _frozen_or_die()
    products = _load_products()
    assets = _assets()
    e_ref = assets["e_ref"]
    noise_variance = float(assets["noise_variance"])
    channel_width_khz = products.channel_width_mhz * 1e3

    print(json.dumps({"stage": "fields"}), flush=True)
    null_pairs = _null_field_pairs(products)
    injections = _injection_field_pairs(products, assets["inj_weight"])
    surrogates = _surrogate_field_pairs(products, assets, noise_variance)

    ops = []
    for family in FAMILIES:
        for scale in em.FAMILY_SCALES[family]:
            print(json.dumps({"stage": "op", "family": family, "scale": scale}), flush=True)
            op = OpChain(family, scale, products.frequencies, products.good_channels, noise_variance)
            scan = op.build_scan(e_ref, channel_width_khz, null_pairs)
            null_results = [scan.zscan(p) for p in op.null_powers]
            null_maxz = np.array([r["z_max"] for r in null_results])
            noise_p95 = float(np.percentile(null_maxz, 95))

            cells = []
            for (m, dnu_khz, cell), pairs in injections.items():
                grid = scan.nearest_grid_index(dnu_khz)
                a_true = (p2.BURST_FLUX_FRACTION * m) ** 2
                powers = op.cross_powers(pairs[:, 0], pairs[:, 1])
                results = [scan.zscan(p) for p in powers]
                rec_dnu = np.array([r["dnu_khz_argmax"] for r in results])
                pulls = np.array([
                    (r["a_hat"][grid] - scan.null_mean[grid] - a_true) / scan.null_sigma[grid]
                    for r in results
                ])
                fin = np.isfinite(rec_dnu) & np.isfinite(pulls)
                convergence = float(fin.mean())
                median_dnu = float(np.median(rec_dnu[fin])) if fin.any() else np.nan
                bias = median_dnu / dnu_khz - 1.0 if np.isfinite(median_dnu) else np.nan
                median_pull = float(np.median(pulls[fin])) if fin.any() else np.nan
                certify = bool(
                    convergence >= CERT_CONVERGENCE_MIN
                    and np.isfinite(bias) and abs(bias) <= CERT_DNU_TOL
                    and np.isfinite(median_pull) and abs(median_pull) <= CERT_PULL_MAX
                )
                cells.append({
                    "cell": cell, "modulation": m, "dnu_khz": dnu_khz,
                    "a_true": a_true, "convergence": convergence,
                    "median_recovered_dnu_khz": median_dnu,
                    "dnu_fractional_bias": bias, "median_amplitude_pull": median_pull,
                    "median_z_max": float(np.median([r["z_max"] for r in results])),
                    "certify": certify,
                })

            surr = {}
            for a_family in FAMILIES:
                if a_family == family:
                    continue
                pairs = surrogates[a_family]
                powers = op.cross_powers(pairs[:, 0], pairs[:, 1])
                maxz = np.array([scan.zscan(p)["z_max"] for p in powers])
                surr[a_family] = {"p95_max_z": float(np.percentile(maxz, 95)),
                                  "max_z": maxz.tolist()}
            surrogate_p95 = max(v["p95_max_z"] for v in surr.values())
            control_clean = bool(surrogate_p95 <= CONTROL_CLEAN_FACTOR * noise_p95)

            ops.append({
                "family": family, "scale": scale,
                "n_certified": int(sum(c["certify"] for c in cells)),
                "n_certified_213": int(sum(c["certify"] for c in cells if c["dnu_khz"] == 213.0)),
                "certified_ge_drop_dnu": int(sum(
                    c["certify"] for c in cells if c["dnu_khz"] >= DROP_MIN_DNU_KHZ)),
                "noise_p95_max_z": noise_p95,
                "surrogate_p95_max_z": surrogate_p95,
                "control_clean": control_clean,
                "cells": cells,
                "surrogates": surr,
                "null_max_z": null_maxz.tolist(),
                "null_z_curves": [r["z"].tolist() for r in null_results],
            })
            print(json.dumps({
                "family": family, "scale": scale,
                "certified": ops[-1]["n_certified"],
                "control_clean": control_clean,
                "noise_p95": round(noise_p95, 2),
                "surrogate_p95": round(surrogate_p95, 2),
            }), flush=True)

    # frozen operating-point rule (injections + surrogates only)
    operating = {}
    for family in FAMILIES:
        fam_ops = [o for o in ops if o["family"] == family]
        clean = [o for o in fam_ops if o["control_clean"]]
        chosen, reason = None, None
        if not clean:
            reason = "no control-clean scale"
        else:
            best = max(o["n_certified"] for o in clean)
            tied = [o for o in clean if o["n_certified"] == best]
            if len(tied) > 1:
                best213 = max(o["n_certified_213"] for o in tied)
                tied = [o for o in tied if o["n_certified_213"] == best213]
            if len(tied) > 1:
                # less envelope leakage: smaller lambda/ell; larger k_env
                key = (lambda o: -o["scale"]) if family == "M3_delaycut" else (lambda o: o["scale"])
                tied = sorted(tied, key=key)
            chosen = tied[0]
            if chosen["certified_ge_drop_dnu"] == 0:
                reason = f"zero certified cells at dnu >= {DROP_MIN_DNU_KHZ} kHz"
                chosen = None
        operating[family] = {
            "scale": None if chosen is None else chosen["scale"],
            "survives": chosen is not None,
            "drop_reason": reason,
        }

    survivors = [f for f, v in operating.items() if v["survives"]]
    payload = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": _frozen_sha(),
        "operating_points": operating,
        "survivors": survivors,
        "fail_branch": len(survivors) == 0,
        "ops": ops,
        "gates": frozen["e2"],
    }
    _write_json(HERE / "e2_calibration.json", payload)

    FIGDIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        fam_ops = [o for o in ops if o["family"] == family]
        scales = [o["scale"] for o in fam_ops]
        grid = np.array([[c["certify"] for c in o["cells"]] for o in fam_ops], dtype=float)
        im = ax.imshow(grid.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(scales)), [str(s) for s in scales])
        labels = [f"m={c['modulation']}, {c['dnu_khz']:.0f}k" for c in fam_ops[0]["cells"]]
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        title = family
        if operating[family]["survives"]:
            title += f" -> op {operating[family]['scale']}"
        else:
            title += " -> DROPPED"
        ax.set_title(title, fontsize=10)
        for xi, o in enumerate(fam_ops):
            ax.text(xi, -0.8, "cc" if o["control_clean"] else "x", ha="center", fontsize=8)
    fig.suptitle("E2 certification (green = certified cell; cc = control-clean scale)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "e2_certification.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(ops))
    ax.bar(x - 0.2, [o["noise_p95_max_z"] for o in ops], 0.4, label="noise-arm p95 max-z")
    ax.bar(x + 0.2, [o["surrogate_p95_max_z"] for o in ops], 0.4, label="surrogate p95 max-z")
    ax.plot(x, [CONTROL_CLEAN_FACTOR * o["noise_p95_max_z"] for o in ops], "k_", ms=14,
            label=f"control-clean bound (x{CONTROL_CLEAN_FACTOR})")
    ax.set_xticks(x, [f"{o['family'].split('_')[0]}\n{o['scale']}" for o in ops], fontsize=7)
    ax.set_ylabel("max z"); ax.legend(fontsize=8)
    ax.set_title("E2 model-mismatch control")
    fig.tight_layout()
    fig.savefig(FIGDIR / "e2_control.png", dpi=150)
    plt.close(fig)

    print(json.dumps({"survivors": survivors, "fail_branch": len(survivors) == 0}, sort_keys=True))
    return 0 if survivors else 2


# --------------------------------------------------------------------------- #
# e3 -- the real-residual looks + discriminants
# --------------------------------------------------------------------------- #
def cmd_e3(args: argparse.Namespace) -> int:
    frozen = _frozen_or_die()
    e2 = json.loads((HERE / "e2_calibration.json").read_text())
    if e2["frozen_config_sha256"] != _frozen_sha():
        raise SystemExit("frozen config changed after e2 -- new experiment required")
    if e2["fail_branch"]:
        raise SystemExit("E2 fail branch triggered -- the real residual is never scanned")
    products = _load_products()
    assets = _assets()
    e_ref = assets["e_ref"]
    noise_variance = float(assets["noise_variance"])
    channel_width_khz = products.channel_width_mhz * 1e3
    f1, f2 = assets["field_half1"], assets["field_half2"]
    null_pairs = _null_field_pairs(products)

    looks, null_maxz_by_family = {}, {}
    for family in e2["survivors"]:
        scale = e2["operating_points"][family]["scale"]
        op = OpChain(family, scale, products.frequencies, products.good_channels, noise_variance)
        scan = op.build_scan(e_ref, channel_width_khz, null_pairs)
        # THE look: one real on-pulse residual scan per surviving family
        power = op.cross_powers(f1[None], f2[None])[0]
        res = scan.zscan(power)
        a_res = float(res["a_hat"][res["argmax_index"]] - scan.null_mean[res["argmax_index"]])
        implied_m = float(np.sqrt(max(a_res, 0.0)) / p2.BURST_FLUX_FRACTION)
        looks[family] = {
            "scale": scale,
            "z_max": res["z_max"],
            "dnu_khz_argmax": res["dnu_khz_argmax"],
            "a_res_at_argmax": a_res,
            "implied_m": implied_m,
            "z_by_dnu": {str(float(d)): float(z) for d, z in zip(scan.dnu_khz, res["z"])},
            "a_hat_by_dnu": {str(float(d)): float(a) for d, a in zip(scan.dnu_khz, res["a_hat"])},
        }
        null_maxz_by_family[family] = np.array([scan.zscan(p)["z_max"] for p in op.null_powers])
        print(json.dumps({"family": family, "scale": scale, "z_max": round(res["z_max"], 2),
                          "dnu_khz_argmax": round(res["dnu_khz_argmax"], 1),
                          "implied_m": round(implied_m, 3)}), flush=True)

    trials_null = np.max(np.vstack([null_maxz_by_family[f] for f in e2["survivors"]]), axis=0)
    threshold = float(np.percentile(trials_null, CANDIDATE_PERCENTILE))
    best_family = max(looks, key=lambda f: looks[f]["z_max"])
    z_stat = looks[best_family]["z_max"]
    admissible = ADMISSIBLE_M[0] <= looks[best_family]["implied_m"] <= ADMISSIBLE_M[1]
    candidate = bool(z_stat >= threshold and admissible)

    payload = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": _frozen_sha(),
        "looks": looks,
        "trials_threshold": threshold,
        "trials_null_max_z": trials_null.tolist(),
        "best_family": best_family,
        "z_stat": z_stat,
        "amplitude_admissible": bool(admissible),
        "candidate": candidate,
        "gates": frozen["e3"],
    }

    # E3a sub-band scaling (only if powered)
    if candidate and z_stat >= E3A_MIN_Z:
        nu = products.frequencies
        edges = np.linspace(p2.BAND_MHZ[0], p2.BAND_MHZ[1], E3A_N_SUBBANDS + 1)
        family, scale = best_family, looks[best_family]["scale"]
        subbands = []
        for band in range(E3A_N_SUBBANDS):
            sel = (nu >= edges[band]) & (nu < edges[band + 1] + (band == E3A_N_SUBBANDS - 1))
            sub_op = OpChain(family, scale, nu[sel], products.good_channels[sel], noise_variance)
            sub_scan = sub_op.build_scan(
                e_ref[sel], channel_width_khz, null_pairs[:, :, sel],
                template_seed_base=SUBBAND_TEMPLATE_BASE + 100_000 * band,
            )
            sub_power = sub_op.cross_powers(f1[None, sel], f2[None, sel])[0]
            sub_res = sub_scan.zscan(sub_power)
            subbands.append({
                "band": band, "nu_center_mhz": float(0.5 * (edges[band] + edges[band + 1])),
                "z_max": sub_res["z_max"], "dnu_khz_argmax": sub_res["dnu_khz_argmax"],
            })
            print(json.dumps({"e3a_band": band, "z_max": round(sub_res["z_max"], 2),
                              "dnu_khz": round(sub_res["dnu_khz_argmax"], 1)}), flush=True)
        centers = np.array([b["nu_center_mhz"] for b in subbands])
        widths = np.array([b["dnu_khz_argmax"] for b in subbands])
        ok = np.isfinite(widths) & (widths > 0)
        alpha, alpha_err = np.nan, np.nan
        if ok.sum() >= 3:
            coef, cov = np.polyfit(np.log(centers[ok]), np.log(widths[ok]), 1, cov=True)
            alpha, alpha_err = float(coef[0]), float(np.sqrt(cov[0, 0]))
        payload["e3a"] = {
            "ran": True, "subbands": subbands, "alpha": alpha, "alpha_err": alpha_err,
            "pass": bool(np.isfinite(alpha) and E3A_ALPHA_BAND[0] <= alpha <= E3A_ALPHA_BAND[1]),
        }
    else:
        payload["e3a"] = {"ran": False,
                          "reason": "not a candidate" if not candidate else f"z < {E3A_MIN_Z} (underpowered)"}

    # E3b component correlation (only if E0 found >= 2 components)
    e0 = json.loads((HERE / "e0_envelope.json").read_text())
    if candidate and e0["profile"]["e3b_available"]:
        family, scale = best_family, looks[best_family]["scale"]
        op = OpChain(family, scale, products.frequencies, products.good_channels, noise_variance)
        comp_windows = e0["profile"]["component_windows"]
        comp_spectra = []
        for w in comp_windows:
            cf1, cf2 = od.split_ratio_fields(
                products.dynamic, rb.samples_from_window(tuple(w)), products.off_pool,
                allow_unblind=True,  # P4 exploratory: post-unblinding, owner-sanctioned 2026-07-15
            )
            comp_spectra.append(np.nanmean(op.residuals(np.vstack([cf1, cf2])), axis=0))
        good_all = np.all([np.isfinite(s) for s in comp_spectra], axis=0)
        corr = float(np.corrcoef(comp_spectra[0][good_all], comp_spectra[1][good_all])[0, 1])
        null_corr = []
        lengths = [w[1] - w[0] for w in comp_windows[:2]]
        for i in range(N_NULL):
            rng = np.random.default_rng(NULL_SEED_BASE + i)
            perm = rng.permutation(products.off_pool)
            w1 = np.sort(perm[: lengths[0]])
            w2 = np.sort(perm[lengths[0] : lengths[0] + lengths[1]])
            rest = np.sort(perm[lengths[0] + lengths[1] :])
            s1 = np.nanmean(op.residuals(np.vstack(od.split_ratio_fields(products.dynamic, w1, rest))), axis=0)
            s2 = np.nanmean(op.residuals(np.vstack(od.split_ratio_fields(products.dynamic, w2, rest))), axis=0)
            ok = np.isfinite(s1) & np.isfinite(s2)
            null_corr.append(float(np.corrcoef(s1[ok], s2[ok])[0, 1]))
        null_p95 = float(np.percentile(null_corr, 95))
        payload["e3b"] = {"ran": True, "correlation": corr, "null_p95": null_p95,
                          "pass": bool(corr > null_p95), "null_correlations": null_corr}
    else:
        payload["e3b"] = {"ran": False,
                          "reason": "not a candidate" if not candidate else "fewer than 2 components (E0)"}

    # E3c DSA-band consistency (measured constant read from the trusted ledger)
    if candidate and args.dsa_dnu_mhz is not None:
        expected_khz = (
            args.dsa_dnu_mhz * 1e3
            * (np.mean(p2.BAND_MHZ) / args.dsa_freq_mhz) ** E3C_SCALING_INDEX
        )
        ratio = looks[best_family]["dnu_khz_argmax"] / expected_khz
        payload["e3c"] = {
            "ran": True,
            "dsa_dnu_mhz": args.dsa_dnu_mhz, "dsa_freq_mhz": args.dsa_freq_mhz,
            "expected_chime_dnu_khz": float(expected_khz),
            "measured_dnu_khz": looks[best_family]["dnu_khz_argmax"],
            "ratio": float(ratio),
            "pass": bool(1.0 / E3C_FACTOR <= ratio <= E3C_FACTOR),
        }
    elif candidate:
        payload["e3c"] = {"ran": False,
                          "reason": "BLOCKED: --dsa-dnu-mhz not supplied (trusted ledger value required)"}
    else:
        payload["e3c"] = {"ran": False, "reason": "not a candidate"}

    _write_json(HERE / "e3_looks.json", payload)

    FIGDIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for family, look in looks.items():
        dnu = np.array([float(d) for d in look["z_by_dnu"]])
        z = np.array(list(look["z_by_dnu"].values()))
        order = np.argsort(dnu)
        ax.semilogx(dnu[order], z[order], "-o", ms=3, label=f"{family} @ {look['scale']}")
    ax.axhline(threshold, color="crimson", ls="--", lw=1,
               label=f"trials p{CANDIDATE_PERCENTILE:.0f} = {threshold:.2f}")
    ax.set_xlabel("dnu_d [kHz]"); ax.set_ylabel("z (null-calibrated)")
    ax.legend(fontsize=8)
    ax.set_title(f"E3 real-residual looks -- candidate: {candidate}")
    fig.tight_layout()
    fig.savefig(FIGDIR / "e3_looks.png", dpi=150)
    plt.close(fig)

    print(json.dumps({"candidate": candidate, "z_stat": round(z_stat, 2),
                      "threshold": round(threshold, 2),
                      "best_family": best_family,
                      "implied_m": round(looks[best_family]["implied_m"], 3),
                      "amplitude_admissible": bool(admissible)}, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def cmd_verdict(args: argparse.Namespace) -> int:
    _frozen_or_die()
    current = _frozen_sha()
    e2 = json.loads((HERE / "e2_calibration.json").read_text())
    if e2["frozen_config_sha256"] != current:
        raise SystemExit("e2 result does not match the frozen config")
    if e2["fail_branch"]:
        verdict = "DOCUMENTED-FAIL (envelope not separable)"
        detail = {"reason": "all three families dropped in E2",
                  "operating_points": e2["operating_points"]}
    else:
        e3 = json.loads((HERE / "e3_looks.json").read_text())
        if e3["frozen_config_sha256"] != current:
            raise SystemExit("e3 result does not match the frozen config")
        if e3["candidate"]:
            checks = [e3["e3c"].get("pass")]
            if e3["e3a"]["ran"]:
                checks.append(e3["e3a"]["pass"])
            if e3["e3b"]["ran"]:
                checks.append(e3["e3b"]["pass"])
            promoted = all(bool(c) for c in checks)
            verdict = ("exploratory scintillation candidate" if promoted
                       else "exploratory upper limit (candidate failed discriminants)")
            detail = {"z_stat": e3["z_stat"], "best_family": e3["best_family"],
                      "e3a": e3["e3a"], "e3b": e3["e3b"], "e3c": e3["e3c"]}
        else:
            verdict = "exploratory upper limit"
            detail = {"z_stat": e3["z_stat"], "trials_threshold": e3["trials_threshold"],
                      "amplitude_admissible": e3["amplitude_admissible"],
                      "note": "post-subtraction sensitivity quantified by the E2 certification tables"}
    payload = {
        "experiment": EXPERIMENT_ID,
        "frozen_config_sha256": current,
        "verdict": verdict,
        "evidential_class": "exploratory (post-unblinding); no blind-analysis weight",
        "detail": detail,
    }
    _write_json(HERE / "verdict.json", payload)
    print(json.dumps({"verdict": verdict}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "e0", "e2", "verdict"):
        sub.add_parser(name)
    e3p = sub.add_parser("e3")
    e3p.add_argument("--dsa-dnu-mhz", type=float, default=None,
                     help="trusted DSA-band dnu_d for freya [MHz] (manuscript ledger)")
    e3p.add_argument("--dsa-freq-mhz", type=float, default=1405.0,
                     help="DSA reference frequency [MHz]")
    args = parser.parse_args()
    return {
        "freeze": cmd_freeze,
        "e0": cmd_e0,
        "e2": cmd_e2,
        "e3": cmd_e3,
        "verdict": cmd_verdict,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
