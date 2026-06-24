"""Unit tests for the single-system sufficiency harness (attribute_excess).

Offline: build_unified_records is called with enrich=False, so no network. Uses a
real EXCESS sightline name (Wilhelm) so config.TARGETS resolves its z_frb.
"""

import math

import pandas as pd

from galaxies.v2_0 import attribute_excess as ax
from galaxies.v2_0 import sightline_budget as sb
from scattering.scat_analysis.burst_metadata import load_tns_name

_NAME = "Wilhelm"
_Z_FRB = ax._TARGET_BY_NAME[_NAME][3]  # 0.51


def _write_galaxies_csv(results_dir, name, z_frb):
    """One foreground (z<z_frb) + one background (z>z_frb) galaxy near the sightline."""
    _, ra, dec, _ = ax._TARGET_BY_NAME[name]
    sight = sb.SkyCoord(ra, dec, unit=(sb.u.hourangle, sb.u.deg))
    matches = pd.DataFrame(
        {
            "ra": [sight.ra.deg + 0.002, sight.ra.deg - 0.002],
            "dec": [sight.dec.deg + 0.001, sight.dec.deg - 0.001],
            "z": [z_frb - 0.05, z_frb + 0.10],  # foreground, background
            "M_star": [10.0, 10.2],
            "catalog": ["VII/291/gladep", "VII/291/gladep"],
            "impact_kpc": [32.0, 40.0],
        }
    )
    matches.to_csv(results_dir / f"{name.lower()}_galaxies.csv", index=False)


def test_attribute_emits_only_foreground_and_matching_verdict(tmp_path):
    _write_galaxies_csv(tmp_path, _NAME, _Z_FRB)

    attr = ax.attribute(str(tmp_path), budget_df=None)

    # Only the foreground (z<z_frb) row survives; only the one sightline with a CSV.
    assert set(attr["sightline"]) == {_NAME}
    assert len(attr) == 1
    assert (attr["z_gal"] < attr["z_frb"]).all()

    # Verdict must be exactly what _scattering_verdict yields for this row (no
    # measured tau -> NaN tau_obs -> "no scattering measurement ...").
    row = attr.iloc[0]
    pred = row["pred_tau_scat_ms_1GHz"]
    pred_hi = row["pred_tau_scat_ms_1GHz_hi"]
    expected = sb._scattering_verdict(
        math.nan,
        pred if math.isfinite(pred) else 0.0,
        pred_hi if math.isfinite(pred_hi) else 0.0,
        n_fg=1,
    )
    assert row["verdict"] == expected
    assert row["verdict"].startswith("no scattering measurement")


def test_attribute_empty_when_no_csv(tmp_path):
    # No {name}_galaxies.csv on disk -> header-only frame, not a crash.
    attr = ax.attribute(str(tmp_path), budget_df=None)
    assert len(attr) == 0
    assert list(attr.columns) == list(ax._ATTR_COLUMNS)


def test_excess_tau_matches_nickname_or_tns():
    tns = load_tns_name(_NAME)
    assert tns.lower() != _NAME.lower()  # Wilhelm has a real TNS designation

    # Budget CSV emits TNS names (#26): lookup by nickname must still resolve.
    budget_tns = pd.DataFrame({"name": [tns], "tau_obs_ms": [1.23]})
    assert ax._excess_tau_ms(_NAME, budget_tns) == 1.23

    # Legacy nickname-keyed budget also resolves.
    budget_nick = pd.DataFrame({"name": [_NAME.lower()], "tau_obs_ms": [2.0]})
    assert ax._excess_tau_ms(_NAME, budget_nick) == 2.0

    # Unknown sightline / no budget -> NaN.
    assert math.isnan(
        ax._excess_tau_ms(_NAME, pd.DataFrame({"name": ["nope"], "tau_obs_ms": [9.0]}))
    )
    assert math.isnan(ax._excess_tau_ms(_NAME, None))


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
