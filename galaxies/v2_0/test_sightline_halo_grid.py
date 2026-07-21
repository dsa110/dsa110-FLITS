"""Contract tests for the versioned Figure 3 foreground input."""

from __future__ import annotations

import numpy as np

from galaxies.foreground.build_sightline_halo_grid_input import build_frame


def test_figure_input_has_full_host_roster_and_only_confirmed_systems() -> None:
    frame = build_frame()
    hosts = frame[frame.row_kind == "host"]
    systems = frame[frame.row_kind == "system"]
    assert len(hosts) == 12
    assert hosts.nickname.is_unique
    assert set(systems.final_verdict) == {"confirmed"}
    assert not systems[["nickname", "object_id"]].duplicated().any()


def test_geometry_uses_corrected_galaxy_or_sourced_cluster_quantities() -> None:
    frame = build_frame()
    drawn = frame[(frame.row_kind == "system") & (frame.geometry_status == "pass")]
    halos = drawn[drawn.system_type == "halo"]
    clusters = drawn[drawn.system_type == "cluster"]
    assert (halos.radius_definition == "R200c").all()
    assert (clusters.radius_definition == "R500c_catalog").all()
    assert np.isfinite(halos.radius_kpc).all() and (halos.radius_kpc > 0).all()
    assert np.isfinite(halos.mass_msun).all() and (halos.mass_msun > 0).all()
    assert np.isfinite(clusters.radius_kpc).all() and (clusters.radius_kpc > 0).all()
    assert np.isfinite(clusters.mass_msun).all() and (clusters.mass_msun > 0).all()


def test_budget_flag_is_overlay_not_admission_rule() -> None:
    frame = build_frame()
    drawn = frame[(frame.row_kind == "system") & (frame.geometry_status == "pass")]
    assert (~drawn.budget_eligible.astype(bool)).any()
    assert drawn.budget_eligible.astype(bool).any()
