"""Offline tests for catalog query planning."""

from __future__ import annotations

import pytest

from .catalog_plan import (
    CatalogProfile,
    max_cone_radius_arcmin,
    plan_queries,
    profiles_from_adapters,
    solve_plan_greedy,
)
from .domain import Sightline


def _tiny_catalogs() -> list[CatalogProfile]:
    return [
        CatalogProfile(
            catalog_id="allsky-mass",
            access_url="http://example/tap",
            table_id="mass/all",
            query_cost=2,
            provides_mass=True,
            provides_spec_z=False,
            z_max=3.0,
            science_weight=1.0,
        ),
        CatalogProfile(
            catalog_id="local-spec",
            access_url="http://example/tap",
            table_id="local/spec",
            query_cost=1,
            provides_mass=False,
            provides_spec_z=True,
            z_max=0.12,
            host_z_max=0.12,
            science_weight=0.8,
        ),
        CatalogProfile(
            catalog_id="cheap-all",
            access_url="http://example/tap",
            table_id="cheap/all",
            query_cost=1,
            provides_mass=False,
            provides_spec_z=False,
            z_max=3.0,
            science_weight=0.5,
        ),
    ]


@pytest.mark.unit
def test_profiles_from_adapters_includes_builtin_catalogs():
    profiles = profiles_from_adapters()
    ids = {p.catalog_id for p in profiles}
    assert "glade-plus" in ids
    assert "two-mrs" in ids


@pytest.mark.unit
def test_max_cone_radius_is_positive_and_capped():
    r = max_cone_radius_arcmin(Sightline(ra=0, dec=0, redshift=0.15))
    assert 1.0 <= r <= 60.0


@pytest.mark.unit
def test_greedy_covers_every_sightline_within_budget():
    sightlines = [
        Sightline(name="FRB1", ra=10, dec=20, redshift=0.08),
        Sightline(name="FRB2", ra=30, dec=-10, redshift=0.3),
    ]
    plan = solve_plan_greedy(sightlines, _tiny_catalogs(), budget=6, require_mass=True)
    names = {q.sightline_name for q in plan.queries}
    assert names == {"FRB1", "FRB2"}
    assert plan.total_cost <= 6


@pytest.mark.unit
def test_local_catalog_applies_to_high_z_host():
    # A local catalog probes low-z foregrounds in front of a high-z host, so it
    # must remain applicable even when z_host exceeds the catalog's z_max.
    sightlines = [Sightline(name="FRB-high", ra=0, dec=0, redshift=0.5)]
    plan = solve_plan_greedy(sightlines, _tiny_catalogs(), budget=4, require_mass=False)
    catalog_ids = {q.catalog_id for q in plan.queries}
    assert "local-spec" in catalog_ids
    assert len(plan.queries) >= 1


@pytest.mark.unit
def test_plan_queries_greedy_solver():
    sl = Sightline(name="FRB", ra=1, dec=2, redshift=0.1)
    plan = plan_queries([sl], _tiny_catalogs(), budget=4, solver="greedy", require_mass=True)
    assert plan.solver == "greedy"
    assert plan.queries[0].radius_arcmin > 0


@pytest.mark.unit
def test_host_z_max_is_not_a_veto():
    # host_z_max is retained as metadata but must NOT veto applicability: a
    # foreground catalog is applicable whenever the host lies beyond z_min.
    local = CatalogProfile(
        catalog_id="lv",
        access_url="x",
        table_id="t",
        z_max=0.1,
        host_z_max=0.1,
    )
    assert local.applicable_to(Sightline(ra=0, dec=0, redshift=0.08))
    assert local.applicable_to(Sightline(ra=0, dec=0, redshift=0.2))
    assert local.applicable_to(Sightline(ra=0, dec=0, redshift=0.5))


@pytest.mark.unit
def test_applicability_vetoes_only_below_z_min():
    # The sole host-dependent veto is z_host <= z_min (no foreground volume).
    cat = CatalogProfile(
        catalog_id="mid",
        access_url="x",
        table_id="t",
        z_min=0.05,
        z_max=0.3,
    )
    assert not cat.applicable_to(Sightline(ra=0, dec=0, redshift=0.04))
    assert not cat.applicable_to(Sightline(ra=0, dec=0, redshift=0.05))
    assert cat.applicable_to(Sightline(ra=0, dec=0, redshift=0.06))
    # Unknown host redshift -> applicable (cannot rule it out).
    assert cat.applicable_to(Sightline(ra=0, dec=0, redshift=None))


def _glade_catalogs() -> list[CatalogProfile]:
    return [
        CatalogProfile(
            catalog_id="glade-plus",
            access_url="x",
            table_id="t",
            query_cost=2,
            provides_mass=True,
            z_max=3.0,
            science_weight=1.2,
        ),
        CatalogProfile(
            catalog_id="glade2",
            access_url="x",
            table_id="t",
            query_cost=1,
            provides_mass=True,
            z_max=3.0,
            science_weight=1.0,
        ),
    ]


@pytest.mark.unit
def test_greedy_excludes_complementary_glade_pair():
    # With ample budget, greedy would otherwise grab both GLADE catalogs in
    # Pass 3; the exclusivity group must cap it at one per sightline.
    sl = Sightline(name="FRB", ra=0, dec=0, redshift=0.1)
    plan = solve_plan_greedy([sl], _glade_catalogs(), budget=100, require_mass=True)
    ids = [q.catalog_id for q in plan.queries]
    glade = [c for c in ids if c in {"glade-plus", "glade2"}]
    assert len(glade) == 1


@pytest.mark.unit
def test_greedy_complementarity_can_be_disabled():
    sl = Sightline(name="FRB", ra=0, dec=0, redshift=0.1)
    plan = solve_plan_greedy(
        [sl], _glade_catalogs(), budget=100, require_mass=True, exclusive_groups=[]
    )
    ids = {q.catalog_id for q in plan.queries}
    assert ids == {"glade-plus", "glade2"}


@pytest.mark.unit
def test_greedy_pass1_respects_budget():
    # Two sightlines each want a mass catalog (cost 2) but budget only fits one;
    # the Pass-1 guard must not overspend past the budget.
    sightlines = [
        Sightline(name="A", ra=0, dec=0, redshift=0.1),
        Sightline(name="B", ra=1, dec=1, redshift=0.1),
    ]
    mass_only = [
        CatalogProfile(
            catalog_id="allsky-mass",
            access_url="x",
            table_id="t",
            query_cost=2,
            provides_mass=True,
            z_max=3.0,
        )
    ]
    plan = solve_plan_greedy(sightlines, mass_only, budget=2, require_mass=True)
    assert plan.total_cost <= 2
