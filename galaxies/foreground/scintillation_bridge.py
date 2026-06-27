"""Wire scintillation interpretation to tau_consistency and multi-scale Δν."""

from __future__ import annotations

import math

import numpy as np

from galaxies.foreground.config import TARGETS
from galaxies.foreground.tau_consistency import (
    ALPHA_CONSISTENCY,
    CHIME_REF_MHZ,
    DSA_REF_MHZ,
    build_tau_consistency_row,
    consistency_status,
)

PLACEHOLDER_Z = 1.0


def _target_by_nickname(nickname: str) -> tuple[str, str, str, float] | None:
    key = nickname.lower()
    for name, ra, dec, z in TARGETS:
        if name.lower() == key:
            return name, ra, dec, float(z)
    return None


def _luminosity_distance_mpc(z: float) -> float:
    if not math.isfinite(z) or z <= 0 or abs(z - PLACEHOLDER_Z) < 1e-6:
        return math.nan
    from astropy.cosmology import Planck18

    return float(Planck18.luminosity_distance(z).to_value("Mpc"))


def build_scintillation_source_block(nickname: str) -> dict[str, float]:
    """Per-burst ``config['source']`` inputs for attach_scintillation_interpretation."""
    target = _target_by_nickname(nickname)
    if target is None:
        return {}
    _name, ra, dec, z = target
    tau_row = build_tau_consistency_row(nickname.lower())
    tau_ms = tau_row.get("tau_consistency_chime_ms")
    if not np.isfinite(tau_ms):
        tau_ms = tau_row.get("tau_joint_1ghz_ms")
    block: dict[str, float] = {
        "nickname": nickname.lower(),
        "ra_deg": math.nan,
        "dec_deg": math.nan,
    }
    sight = _target_by_nickname(nickname)
    if sight:
        from astropy.coordinates import SkyCoord

        c = SkyCoord(sight[1], sight[2], unit=("hourangle", "deg"))
        block["ra_deg"] = float(c.ra.deg)
        block["dec_deg"] = float(c.dec.deg)
    if np.isfinite(tau_ms) and tau_ms > 0:
        block["tau_d_ms"] = float(tau_ms)
    d_mpc = _luminosity_distance_mpc(z)
    if np.isfinite(d_mpc):
        block["distance_mpc"] = d_mpc
    return block


def merge_source_into_config(config: dict, nickname: str | None) -> dict:
    if not nickname:
        return config
    existing = config.get("source") or {}
    if existing.get("tau_d_ms") and existing.get("distance_mpc"):
        return config
    block = build_scintillation_source_block(nickname)
    if not block:
        return config
    out = dict(config)
    out["source"] = {**block, **existing}
    return out


def _band_tau_ms(config: dict, band: str) -> float:
    src = config.get("source") or {}
    if band == "dsa":
        key = "tau_consistency_dsa_ms"
        ref = config.get("analysis", {}).get("fitting", {}).get("reference_frequency_mhz_dsa", DSA_REF_MHZ)
    else:
        key = "tau_consistency_chime_ms"
        ref = config.get("analysis", {}).get("fitting", {}).get("reference_frequency_mhz", CHIME_REF_MHZ)
    tau_row = build_tau_consistency_row(str(src.get("nickname", "")))
    tau = tau_row.get(key)
    if np.isfinite(tau):
        return float(tau)
    tau_1 = tau_row.get("tau_consistency_1ghz_ms")
    if np.isfinite(tau_1):
        from galaxies.foreground.tau_consistency import scale_tau_1ghz_ms

        return scale_tau_1ghz_ms(float(tau_1), float(ref), alpha=ALPHA_CONSISTENCY)
    return float(src.get("tau_d_ms", np.nan))


def consistency_failed_for_component(comp: dict, config: dict, band: str = "chime") -> bool:
    dnu = comp.get("bw_at_ref_mhz")
    if not np.isfinite(dnu):
        return False
    tau = _band_tau_ms(config, band)
    if not np.isfinite(tau):
        return False
    return consistency_status(tau, float(dnu)) == "inconsistent"


def attach_multi_scale_from_acf(
    comp: dict,
    acf_spec: np.ndarray | None,
    channel_width_mhz: float,
) -> dict:
    """Run fit_two_screen_acf and attach wide/narrow Δν to *comp* in place."""
    if acf_spec is None:
        return comp
    from scintillation.scint_analysis.revalidation import fit_two_screen_acf

    fit = fit_two_screen_acf(acf_spec, channel_width_mhz=channel_width_mhz)
    comp["dnu_wide_mhz"] = fit.get("dnu_wide_mhz")
    comp["dnu_narrow_mhz"] = fit.get("dnu_narrow_mhz")
    comp["multi_scale_fit"] = fit
    return comp


def _first_subband_acf(acf_results: dict | None) -> tuple[np.ndarray | None, float]:
    """Return (acf_array, channel_width_mhz) from pipeline ``acf_results``."""
    if not acf_results:
        return None, math.nan
    acfs = acf_results.get("subband_acfs") or []
    if not acfs:
        return None, math.nan
    first = acfs[0]
    if isinstance(first, dict):
        spec = first.get("acf")
        ch_width = first.get("channel_width_mhz", math.nan)
    else:
        spec = first
        widths = acf_results.get("subband_channel_widths_mhz") or []
        ch_width = widths[0] if widths else math.nan
    if spec is None:
        return None, math.nan
    return np.asarray(spec), float(ch_width)


def prepare_multi_scale_components(
    final_results: dict | None,
    config: dict,
    acf_results: dict | None,
) -> None:
    """When τ–Δν is inconsistent, attempt wide+narrow Δν before interpretation attach."""
    if not isinstance(final_results, dict):
        return
    components = final_results.get("components") or {}
    if not components:
        return
    spec, ch_width = _first_subband_acf(acf_results)
    if spec is None or not np.isfinite(ch_width):
        return
    for comp in components.values():
        if not isinstance(comp, dict) or not comp.get("subband_measurements"):
            continue
        if comp.get("dnu_wide_mhz") and comp.get("dnu_narrow_mhz"):
            continue
        if consistency_failed_for_component(comp, config):
            attach_multi_scale_from_acf(comp, np.asarray(spec), ch_width)


def format_two_screen_coherence(comp: dict, distance_mpc: float, ref_freq_mhz: float) -> str:
    dnu_w = comp.get("dnu_wide_mhz")
    dnu_n = comp.get("dnu_narrow_mhz")
    if not (np.isfinite(dnu_w) and np.isfinite(dnu_n) and np.isfinite(distance_mpc)):
        return "N/A — multi-scale dnu or distance missing"
    from scintillation.scint_analysis.analysis import two_screen_coherence_constraint

    result = two_screen_coherence_constraint(
        float(dnu_w), float(dnu_n), float(ref_freq_mhz), float(distance_mpc)
    )
    limit = result.get("d_product_kpc2")
    if not np.isfinite(limit):
        return "N/A — coherence constraint undefined"
    return f"d_Earth,scr1 * d_scr2,src <= {limit:.2g} kpc^2"


def attach_interpretation_with_bridge(
    final_results: dict,
    config: dict,
    nickname: str | None = None,
    acf_results: dict | None = None,
) -> dict:
    """Merge source block, optional multi-scale Δν, then attach_scintillation_interpretation."""
    from scintillation.scint_analysis.analysis import attach_scintillation_interpretation

    nick = nickname or (config.get("source") or {}).get("nickname")
    cfg = merge_source_into_config(config, nick)
    prepare_multi_scale_components(final_results, cfg, acf_results)
    attach_scintillation_interpretation(final_results, cfg)
    return cfg
