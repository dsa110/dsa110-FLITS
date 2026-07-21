"""Adaptive multiresolution arrival-regression measurements.

The policy is uniform across the sample: evaluate the same resolution grid for
every product, retain a dense cluster of science-grade solutions, and report a
weak-only or unconstrained result rather than promoting an isolated optimum.

The stored waterfalls are dedispersed at the DM encoded in their filename.
Residuals therefore attach to that product DM, not necessarily to the catalog
DM carried by the association fixture (notably chromatica).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from dispersion.chime_dm import measure_dm
from dispersion.dm_power_analysis import (
    CHIME_DT_S,
    DSA_DT_S,
    _dm_ref,
    _downsample_for_fit,
    _freq_grid_mhz,
    _orient_waterfall_to_ascending_frequency,
    root_for_telescope,
)


def product_dm_from_filename(filename: str) -> float:
    """Return the actual pre-dedispersion DM encoded in a product filename."""
    match = re.search(r"_I_(\d+)_(\d+)_\d+b_cntr_bpc\.npy$", filename)
    if not match:
        raise ValueError(f"cannot parse product DM from {filename!r}")
    return float(f"{match.group(1)}.{match.group(2)}")


def resolution_grid(config: dict) -> list[dict]:
    """Cartesian candidate grid shared by all products."""
    adaptive = config["adaptive"]
    return [
        {"max_channels": int(ch), "max_time": int(nt), "n_subband": int(ns)}
        for ch in adaptive["max_channels"]
        for nt in adaptive["max_time"]
        for ns in adaptive["n_subbands"]
    ]


def product_path(row: dict, config: dict) -> Path:
    root = root_for_telescope(
        row["telescope"],
        Path(config["chime_full_root"]).expanduser(),
        Path(config["dsa_full_root"]).expanduser(),
    )
    return root / row["filename"]


def evaluate_product(row: dict, config: dict) -> list[dict]:
    """Run the uniform adaptive grid on one product."""
    path = product_path(row, config)
    if not path.exists():
        return []
    telescope = row["telescope"]
    raw = _orient_waterfall_to_ascending_frequency(
        np.asarray(np.load(path, mmap_mode="r"), dtype=float), telescope
    )
    freq = _freq_grid_mhz(telescope, raw.shape[0])
    dt_s = CHIME_DT_S if telescope == "chime" else DSA_DT_S
    product_dm = product_dm_from_filename(row["filename"])
    catalog_dm = _dm_ref(row)
    return evaluate_waterfall(
        raw, freq, dt_s, product_dm, catalog_dm, telescope=telescope, config=config
    )


def evaluate_waterfall(
    raw: np.ndarray,
    freq: np.ndarray,
    dt_s: float,
    product_dm: float,
    catalog_dm: float,
    *,
    telescope: str,
    config: dict,
) -> list[dict]:
    """Evaluate the adaptive grid on an in-memory waterfall (injection seam)."""
    candidates = []
    for spec in resolution_grid(config):
        wf, freq_ds, dt_ds, factors = _downsample_for_fit(
            raw, freq, dt_s, spec["max_channels"], spec["max_time"]
        )
        result = measure_dm(
            wf,
            freq_ds,
            dt_ds,
            product_dm,
            n_subband=spec["n_subband"],
            dm_window=config["windows"][telescope],
            dm_step=float(config["adaptive"]["dm_step"]),
        )
        candidates.append(
            {
                **spec,
                "freq_factor": int(factors["freq"]),
                "time_factor": int(factors["time"]),
                "dt_ms": float(dt_ds * 1e3),
                "product_dm": product_dm,
                "catalog_dm": catalog_dm,
                "dm": result["dm"],
                "sigma": result["dm_err"],
                "residual": None if result["dm"] is None else result["dm"] - product_dm,
                "constrains_dm": bool(result["constrains_dm"]),
                "reason": result["reason"],
                "n_good_subbands": int(result["n_good_subbands"]),
                "peak_snr": float(result["peak_snr"]),
                "chi2_red": result.get("chi2_red"),
                "railed": bool(result["railed"]),
                "subbands": result["subbands"],
                "subbands_dropped": result["subbands_dropped"],
            }
        )
    return candidates


def _resolution_key(candidate: dict) -> tuple[int, int, int]:
    return (
        int(candidate["time_factor"]),
        int(candidate["freq_factor"]),
        int(candidate["n_subband"]),
    )


def select_stable_candidate(
    candidates: list[dict], *, sigma_max: float = 0.5, stability_dm: float = 0.25
) -> dict:
    """Select the strongest member of the densest stable solution cluster.

    A candidate must already pass the estimator gate and the science precision
    ceiling. At least two distinct resolution choices must agree within
    ``stability_dm``; otherwise the product is not promoted.
    """
    constrained = [
        c for c in candidates
        if c.get("constrains_dm") and c.get("dm") is not None and c.get("sigma") is not None
    ]
    from scattering.scat_analysis.burstfit import classify_fit_quality

    for candidate in constrained:
        quality, notes = classify_fit_quality(candidate.get("chi2_red"))
        candidate["fit_quality"] = quality
        candidate["fit_quality_notes"] = notes
    eligible = [
        c for c in constrained
        if c["sigma"] <= sigma_max and not c.get("railed") and c["fit_quality"] != "FAIL"
    ]
    if not eligible:
        weak = min(constrained, key=lambda c: c["sigma"], default=None)
        return {
            "status": "weak-only" if weak else "unconstrained",
            "dm": None,
            "sigma": None,
            "best_weak_dm": None if weak is None else weak["dm"],
            "best_weak_sigma": None if weak is None else weak["sigma"],
            "candidate_count": len(candidates),
            "constrained_candidate_count": len(constrained),
            "stable_candidate_count": 0,
            "distinct_resolution_count": 0,
        }

    def best_cluster(pool: list[dict]) -> tuple[list[dict], int, float]:
        clusters = []
        ordered = sorted(pool, key=lambda candidate: candidate["dm"])
        for start in range(len(ordered)):
            members = []
            for candidate in ordered[start:]:
                if candidate["dm"] - ordered[start]["dm"] > stability_dm:
                    break
                members.append(candidate)
            distinct = len({_resolution_key(c) for c in members})
            spread = float(np.ptp([c["dm"] for c in members])) if len(members) > 1 else 0.0
            clusters.append((members, distinct, spread))
        return max(clusters, key=lambda item: (item[1], len(item[0]), -item[2]))

    # A nearby MARGINAL resolution must not displace a stable cluster whose
    # members all pass the preregistered fit-quality gate.  Fall back to the
    # wider non-FAIL pool only when no stable PASS cluster exists, retaining a
    # marginal result for audit without promoting it to science grade.
    pass_only = [candidate for candidate in eligible if candidate["fit_quality"] == "PASS"]
    if pass_only:
        pass_cluster = best_cluster(pass_only)
    else:
        pass_cluster = ([], 0, 0.0)
    has_stable_pass_cluster = pass_cluster[1] >= 2
    members, distinct, spread = (
        pass_cluster if has_stable_pass_cluster else best_cluster(eligible)
    )
    if distinct < 2:
        return {
            "status": "unstable",
            "dm": None,
            "sigma": None,
            "candidate_count": len(candidates),
            "constrained_candidate_count": len(constrained),
            "stable_candidate_count": len(members),
            "distinct_resolution_count": distinct,
        }

    median_dm = float(np.median([c["dm"] for c in members]))
    # For an even-sized cluster the two middle members are exactly equidistant
    # from the median in real arithmetic, so the raw distance key ties at the
    # ~1e-10 float level and the winner shifts with library version (observed:
    # scipy 1.17 vs 1.18). Quantize far below dm_step so the documented
    # tiebreak (n_good_subbands, then sigma) decides deterministically.
    selected = min(
        members,
        key=lambda c: (round(abs(c["dm"] - median_dm), 6), -c["n_good_subbands"], c["sigma"]),
    )
    return {
        "status": "science-grade" if has_stable_pass_cluster else "marginal-fit",
        "dm": float(selected["dm"]),
        "sigma": float(selected["sigma"]),
        "residual": float(selected["residual"]) if selected.get("residual") is not None else None,
        "selected_resolution": {
            key: selected.get(key)
            for key in ("max_channels", "max_time", "n_subband", "freq_factor", "time_factor", "dt_ms")
        },
        "n_good_subbands": int(selected["n_good_subbands"]),
        "chi2_red": selected.get("chi2_red"),
        "fit_quality": selected.get("fit_quality"),
        "fit_quality_notes": selected.get("fit_quality_notes", []),
        "subbands": selected.get("subbands", []),
        "subbands_dropped": selected.get("subbands_dropped", []),
        "cluster_median_dm": median_dm,
        "cluster_span_dm": spread,
        "candidate_count": len(candidates),
        "constrained_candidate_count": len(constrained),
        "stable_candidate_count": len(members),
        "distinct_resolution_count": distinct,
        "product_dm": selected.get("product_dm"),
        "catalog_dm": selected.get("catalog_dm"),
    }


def combine_event_measurements(burst: str, bands: dict[str, dict]) -> dict:
    """Inverse-variance event summary, with honest support classification.

    For two bands, flag >3-sigma tension and withhold a single event DM rather
    than averaging mutually inconsistent measurements.
    """
    usable = [
        (name, result) for name, result in bands.items()
        if result.get("status") == "science-grade" and result.get("dm") is not None
    ]
    if not usable:
        return {
            "burst": burst, "dm": None, "sigma": None, "support": "none",
            "bands_used": [], "band_measurements": {},
        }
    weights = np.array([1.0 / result["sigma"] ** 2 for _, result in usable])
    values = np.array([result["dm"] for _, result in usable])
    dm = float(np.sum(weights * values) / np.sum(weights))
    sigma = float(np.sum(weights) ** -0.5)
    agreement_z = None
    chi2_red = None
    support = "single-band"
    band_measurements = {
        name: {"dm": float(result["dm"]), "sigma": float(result["sigma"])}
        for name, result in usable
    }
    if len(usable) == 2:
        residuals = values - dm
        chi2_red = float(np.sum(weights * residuals**2))
        agreement_z = float(abs(values[0] - values[1]) / np.hypot(
            usable[0][1]["sigma"], usable[1][1]["sigma"]
        ))
        support = "two-band-tension" if agreement_z > 3.0 else "two-band-consistent"
        if support == "two-band-tension":
            dm = None
            sigma = None
        else:
            sigma *= float(np.sqrt(max(1.0, chi2_red)))
    return {
        "burst": burst,
        "dm": dm,
        "sigma": sigma,
        "support": support,
        "bands_used": [name for name, _ in usable],
        "agreement_z": agreement_z,
        "chi2_red": chi2_red,
        "band_measurements": band_measurements,
    }
