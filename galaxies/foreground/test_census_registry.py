"""Tests for intervening census registry."""


from galaxies.foreground.census_registry import (
    budget_eligible,
    build_intervening_census_registry,
    registry_to_matches,
    scratch_codetection_dir,
)


def test_scratch_codetection_exists():
    assert scratch_codetection_dir().is_dir()


def test_registry_row_count_and_verdicts():
    df = build_intervening_census_registry()
    assert len(df) == 49
    counts = df.final_verdict.value_counts()
    assert counts["confirmed"] == 29
    assert counts["inconclusive"] == 13
    assert counts["refuted"] == 7


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


def test_registry_to_matches_budget_eligible_only():
    reg = build_intervening_census_registry()
    matches = registry_to_matches(reg, "phineas", z_frb=0.271)
    assert len(matches) >= 1
    assert (matches.z < 0.271).all()
    ineligible = registry_to_matches(reg, "phineas", z_frb=0.271)
    assert "catalog" in ineligible.columns
