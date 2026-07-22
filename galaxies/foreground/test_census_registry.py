"""Tests for intervening census registry."""

from pathlib import Path

import pandas as pd

from scattering.scat_analysis.burst_metadata import load_tns_name

from galaxies.foreground.census_registry import (
    budget_eligible,
    build_intervening_census_registry,
    registry_to_matches,
    scratch_codetection_dir,
)


def test_verdi_host_redshift_source_is_applied_without_inference():
    data = Path(__file__).parent / "data" / "frozen_census"
    bursts = pd.read_csv(data / "bursts.csv").set_index("nickname")
    source = pd.read_csv(data / "verdi2025_host_redshift_extract.csv").set_index("mapped_nickname")

    assert source.loc["johndoeII", "source_status"] == "reported_repeater"
    assert bursts.loc["johndoeII", "z_spec"] == source.loc["johndoeII", "redshift"]
    assert source.loc["wilhelm", "source_status"] == "missing"
    assert pd.isna(source.loc["wilhelm", "redshift"])
    assert pd.isna(bursts.loc["wilhelm", "z_spec"])


def test_law2024_host_redshifts_bind_the_three_older_sightlines():
    data = Path(__file__).parent / "data" / "frozen_census"
    bursts = pd.read_csv(data / "bursts.csv").set_index("nickname")
    source = pd.read_csv(data / "law2024_host_redshift_extract.csv").set_index("mapped_nickname")

    assert set(source.index) == {"zach", "whitney", "oran"}
    assert set(source["measurement_kind"]) == {"spectroscopic"}
    for nickname, row in source.iterrows():
        assert bursts.loc[nickname, "z_spec"] == row["adopted_redshift"]
        assert abs(row["published_redshift"] - row["adopted_redshift"]) <= 0.0011


def test_registry_uses_owner_approved_verdi_identifiers_and_keeps_whitney_value():
    registry = build_intervening_census_registry()
    expected = {
        "freya": "FRB 20230325C",
        "hamilton": "FRB 20230913G",
        "chromatica": "FRB 20240203D",
    }
    for nickname, tns in expected.items():
        assert load_tns_name(nickname) == tns
        assert set(registry.loc[registry.nickname == nickname, "tns"]) == {tns}
    assert set(registry.loc[registry.nickname == "whitney", "host_z_spec"]) == {0.479}


def test_scratch_codetection_exists():
    assert scratch_codetection_dir().is_dir()


def test_registry_row_count_and_verdicts():
    df = build_intervening_census_registry()
    assert len(df) == 52
    counts = df.final_verdict.value_counts()
    assert counts["confirmed"] == 29
    assert counts["inconclusive"] == 16
    assert counts["refuted"] == 7


def test_v4_extension_is_present_but_budget_ineligible():
    df = build_intervening_census_registry()
    expected = {
        ("isha", "WISEA J044538.83+701843.3"),
        ("oran", "WISEA J211150.32+724807.8"),
        ("phineas", "WHL J115048.0+714428"),
    }
    ext = df[df.provenance_scratch_final == "v4-extension-2026-07-15"]
    assert set(zip(ext.nickname, ext.obj, strict=True)) == expected
    assert not ext.budget_eligible.any()
    assert ext.set_index("obj").loc["WHL J115048.0+714428", "final_verdict"] == "confirmed"


def test_registry_stable_keys_unique():
    df = build_intervening_census_registry()
    keys = df[["nickname", "type", "obj"]].astype(str).agg(tuple, axis=1)
    assert keys.is_unique


def test_budget_eligible_cluster_gate():
    assert budget_eligible("confirmed", "cluster", 0.83) is True
    assert budget_eligible("confirmed", "cluster", 3.93) is False
    assert budget_eligible("confirmed", "halo", float("nan")) is True
    assert budget_eligible("refuted", "halo", 1.0) is False


def test_registry_budget_eligible_counts():
    df = build_intervening_census_registry()
    eligible = df[df.budget_eligible]
    clusters = eligible[eligible.type == "cluster"]
    assert len(clusters) == 1
    assert clusters.iloc[0].nickname == "phineas"


def test_ned_photo_z_without_uncertainty_is_not_budget_eligible():
    df = build_intervening_census_registry()
    row = df[(df.nickname == "chromatica") & (df.obj.astype(str) == "196733128040225775")].iloc[0]
    assert row.final_verdict == "inconclusive"
    assert not bool(row.registry_tier)
    assert not bool(row.budget_eligible)
    assert "no reported uncertainty" in row.final_reason


def test_registry_to_matches_budget_eligible_only():
    reg = build_intervening_census_registry()
    matches = registry_to_matches(reg, "phineas", z_frb=0.271)
    assert len(matches) >= 1
    assert (matches.z < 0.271).all()
    ineligible = registry_to_matches(reg, "phineas", z_frb=0.271)
    assert "catalog" in ineligible.columns


# --- 2026-07-15 census remediation: dedupe, adjudicated masses, geometry -----


def test_registry_to_matches_dedupes_confirmed_pairs():
    """The five confirmed cross-listed pairs collapse to single physical systems."""
    from galaxies.foreground.census_registry import (
        load_intervening_census_registry,
        registry_to_matches,
    )

    reg = load_intervening_census_registry()
    phineas = registry_to_matches(reg, "phineas", z_frb=0.271)
    assert len(phineas) == 6  # 5 physical halos + 1 cluster (was 9 rows)
    casey = registry_to_matches(reg, "casey", z_frb=0.287)
    assert len(casey) == 2  # 2 physical halos (was 4 rows)


def test_whitney_1473_mass_override_applied():
    """Owner adjudication: the WISE-blend mass is superseded by the optical mass."""
    from galaxies.foreground.census_registry import (
        load_intervening_census_registry,
        registry_to_matches,
    )

    reg = load_intervening_census_registry()
    m = registry_to_matches(reg, "whitney", z_frb=0.479)
    assert len(m) == 1
    row = m.iloc[0]
    assert abs(row.logM_adj - 9.604533681319118) < 1e-9
    assert row.mass_source_adj == "desi_ls_sed"


def test_impact_recomputed_for_halos_not_cluster():
    """Halo b comes from the uniform geometry recomputation; the cluster keeps
    its analysis-provenance value (603.6 kpc, b/R500=0.83)."""
    from galaxies.foreground.census_registry import (
        load_intervening_census_registry,
        registry_to_matches,
    )

    reg = load_intervening_census_registry()
    m = registry_to_matches(
        reg, "phineas", z_frb=0.271, sight_ra_deg=177.7813, sight_dec_deg=71.6956
    )
    cluster = m[m.classification == "GClstr"].iloc[0]
    assert abs(cluster.impact_kpc - 603.6) < 0.1
    halos = m[m.classification != "GClstr"]
    # the z=0.1925 physical halo: listed values were 158.8/130.6 (inconsistent
    # duplicate members); the uniform recomputation gives ~144.0
    h = halos[(halos.z - 0.1925).abs() < 1e-3].iloc[0]
    assert abs(h.impact_kpc - 144.0) < 1.0
