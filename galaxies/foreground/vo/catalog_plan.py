"""Optimize which galaxy catalogs to query per FRB sightline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from astropy import units as u
from astropy.cosmology import Cosmology, Planck18

from .domain import Sightline
from .halos import ADAPTERS
from .reduce import rdelta_from_mdelta

SolverName = Literal["gurobi", "greedy"]

# Catalog complementarity: within each group, at most one catalog should be
# queried per sightline because the members are redundant for the local
# foreground volume. GLADE+ subsumes GLADE2 locally (same parent compilation,
# GLADE+ is the superset with more complete photometry/redshifts), so querying
# both wastes budget without adding foreground galaxies. ``solve_plan_*`` add a
# per-sightline "<= 1 chosen from group" constraint for each group below.
DEFAULT_EXCLUSIVE_GROUPS: tuple[frozenset[str], ...] = (frozenset({"glade-plus", "glade2"}),)


def _resolve_exclusive_groups(
    exclusive_groups: Sequence[frozenset[str]] | None,
) -> tuple[frozenset[str], ...]:
    """Return the configured exclusivity groups, defaulting to the GLADE pair."""
    if exclusive_groups is None:
        return DEFAULT_EXCLUSIVE_GROUPS
    return tuple(frozenset(g) for g in exclusive_groups)


def _group_of(catalog_id: str, groups: Sequence[frozenset[str]]) -> int | None:
    """Index of the exclusivity group containing ``catalog_id``, else ``None``."""
    for gi, group in enumerate(groups):
        if catalog_id in group:
            return gi
    return None


@dataclass(frozen=True)
class CatalogProfile:
    """Queryable catalog with science/cost metadata for planning."""

    catalog_id: str
    access_url: str
    table_id: str
    query_cost: int = 1
    provides_mass: bool = False
    provides_spec_z: bool = False
    z_max: float = 3.0
    z_min: float = 0.0
    host_z_max: float | None = None
    science_weight: float = 1.0

    def applicable_to(self, sightline: Sightline) -> bool:
        """Return whether this catalog is worth querying along the sightline.

        For *foreground* science the relevant quantity is the catalog's
        redshift coverage [z_min, z_max] intersected with the intervening
        volume in front of the host, i.e. z_foreground < z_host. A local
        catalog (e.g. 2MRS, HECATE) that tops out at z_max ~ 0.1 is still
        fully applicable to a z_host = 0.5 sightline: it catches every
        foreground galaxy out to its z_max. ``host_z_max`` is therefore NOT a
        veto here -- it is retained only as metadata/weighting context (see
        ``profiles_from_adapters``). The only host-dependent veto is that the
        host must lie beyond the catalog's z_min, so there is foreground
        volume to probe at all.
        """
        if sightline.redshift is None:
            return True
        z_host = float(sightline.redshift)
        if z_host <= self.z_min:
            return False
        return self.z_max > self.z_min


def profiles_from_adapters(
    *,
    access_url: str = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
) -> list[CatalogProfile]:
    """Build default catalog profiles from built-in ``ADAPTERS``."""
    specs: list[CatalogProfile] = []
    for key, adapter in ADAPTERS.items():
        provides_mass = adapter.mstar_col is not None or adapter.kmag_col is not None
        provides_spec = adapter.z_type_default == "spec" or adapter.z_type_col is not None
        z_max = 0.12 if key == "two-mrs" else (0.15 if key == "hecate-v2" else 3.0)
        host_z_max = 0.12 if key == "two-mrs" else (0.15 if key == "hecate-v2" else None)
        weight = 1.0
        if key == "glade-plus":
            weight = 1.2
        if provides_spec:
            weight += 0.3
        if provides_mass:
            weight += 0.3
        specs.append(
            CatalogProfile(
                catalog_id=key,
                access_url=access_url,
                table_id=adapter.table_id,
                query_cost=2 if key == "glade-plus" else 1,
                provides_mass=provides_mass,
                provides_spec_z=provides_spec,
                z_max=z_max,
                host_z_max=host_z_max,
                science_weight=weight,
            )
        )
    return specs


def max_cone_radius_arcmin(
    sightline: Sightline,
    *,
    cosmo: Cosmology | None = None,
    m_delta_msun: float = 1e13,
    delta: float = 200.0,
    rvir_multiple: float = 1.2,
    z_floor: float = 0.01,
    default_arcmin: float = 30.0,
    max_arcmin: float = 60.0,
) -> float:
    """Conservative cone radius to catch halos with b <= rvir_multiple * R_delta.

    Uses a typical group/cluster mass (``m_delta_msun`` ~ 1e13 Msun) when no
    candidate-specific M_delta is known yet -- groups dominate the foreground
    cross-section budget, so a 1e12 default undersizes the cone. The angular
    size of a fixed physical radius grows without bound as z -> 0, so the
    nearest foregrounds set the largest cone; ``z_floor = 0.01`` (~40 Mpc)
    bounds that divergence to the regime 2MRS/HECATE actually sample, instead
    of the unphysical 1e-4 floor that blows the cone up to the ``max_arcmin``
    cap for essentially every sightline.

    The returned radius is clamped to ``[1.0, max_arcmin]`` arcmin: for very
    nearby foregrounds the geometric estimate can exceed ``max_arcmin``, in
    which case the cap applies and the returned value is ``max_arcmin``.

    If host redshift is unknown or at/below ``z_floor``, returns
    ``default_arcmin``.
    """
    if sightline.redshift is None or sightline.redshift <= z_floor:
        return default_arcmin
    if cosmo is None:
        cosmo = Planck18

    z_host = float(sightline.redshift)
    zs = np.linspace(z_floor, max(z_floor, z_host * 0.995), 64)
    max_theta_rad = 0.0
    for z in zs:
        try:
            r_delta_kpc = rdelta_from_mdelta(m_delta_msun, delta, float(z), cosmo)
            da_kpc = cosmo.angular_diameter_distance(z).to(u.kpc).value
            if da_kpc <= 0:
                continue
            max_theta_rad = max(max_theta_rad, rvir_multiple * r_delta_kpc / da_kpc)
        except Exception:
            continue
    if max_theta_rad <= 0:
        return default_arcmin
    arcmin = float(np.rad2deg(max_theta_rad) * 60.0)
    return float(min(max(arcmin, 1.0), max_arcmin))


@dataclass
class PlannedQuery:
    """One TAP cone query to execute."""

    sightline_name: str
    catalog_id: str
    access_url: str
    table_id: str
    ra_deg: float
    dec_deg: float
    radius_arcmin: float
    frb_redshift: float | None
    maxrec: int = 5000


@dataclass
class QueryPlan:
    """Optimized catalog query schedule."""

    queries: list[PlannedQuery]
    solver: SolverName
    objective_value: float
    total_cost: int
    budget: int
    metadata: dict = field(default_factory=dict)

    def to_dataframe(self):
        """Return plan rows as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "frb_name": q.sightline_name,
                    "catalog_id": q.catalog_id,
                    "access_url": q.access_url,
                    "table_id": q.table_id,
                    "ra_deg": q.ra_deg,
                    "dec_deg": q.dec_deg,
                    "radius_arcmin": q.radius_arcmin,
                    "frb_z": q.frb_redshift,
                    "maxrec": q.maxrec,
                }
                for q in self.queries
            ]
        )

    def to_json(self) -> str:
        """Serialize plan for reproducible execution."""
        payload = {
            "solver": self.solver,
            "objective_value": self.objective_value,
            "total_cost": self.total_cost,
            "budget": self.budget,
            "metadata": self.metadata,
            "queries": [
                {
                    "frb_name": q.sightline_name,
                    "catalog_id": q.catalog_id,
                    "access_url": q.access_url,
                    "table_id": q.table_id,
                    "ra_deg": q.ra_deg,
                    "dec_deg": q.dec_deg,
                    "radius_arcmin": q.radius_arcmin,
                    "frb_z": q.frb_redshift,
                    "maxrec": q.maxrec,
                }
                for q in self.queries
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _pair_indices(
    sightlines: list[Sightline],
    catalogs: list[CatalogProfile],
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i, sl in enumerate(sightlines):
        for j, cat in enumerate(catalogs):
            if cat.applicable_to(sl):
                pairs.append((i, j))
    return pairs


def _build_queries(
    sightlines: list[Sightline],
    catalogs: list[CatalogProfile],
    chosen_pairs: set[tuple[int, int]],
    *,
    cosmo: Cosmology | None,
    rvir_multiple: float,
    maxrec: int,
) -> list[PlannedQuery]:
    queries: list[PlannedQuery] = []
    for i, j in sorted(chosen_pairs):
        sl = sightlines[i]
        cat = catalogs[j]
        queries.append(
            PlannedQuery(
                sightline_name=sl.name or f"sightline_{i}",
                catalog_id=cat.catalog_id,
                access_url=cat.access_url,
                table_id=cat.table_id,
                ra_deg=float(sl.ra),
                dec_deg=float(sl.dec),
                radius_arcmin=max_cone_radius_arcmin(sl, cosmo=cosmo, rvir_multiple=rvir_multiple),
                frb_redshift=sl.redshift,
                maxrec=maxrec,
            )
        )
    return queries


def solve_plan_greedy(
    sightlines: list[Sightline],
    catalogs: list[CatalogProfile],
    *,
    budget: int,
    require_mass: bool = True,
    # require_spec_z defaults False on purpose: spectroscopic redshifts are
    # not available for every sightline/catalog, and forcing them would make
    # otherwise-feasible plans infeasible. Set True only when a downstream
    # analysis genuinely needs spec-z foregrounds.
    require_spec_z: bool = False,
    cosmo: Cosmology | None = None,
    rvir_multiple: float = 1.2,
    maxrec: int = 5000,
    exclusive_groups: Sequence[frozenset[str]] | None = None,
) -> QueryPlan:
    """Greedy feasibility-first planner (no Gurobi dependency)."""
    pairs = _pair_indices(sightlines, catalogs)
    if not pairs:
        raise ValueError("No applicable (sightline, catalog) pairs")

    groups = _resolve_exclusive_groups(exclusive_groups)

    chosen: set[tuple[int, int]] = set()
    spent = 0

    def conflicts(i: int, j: int) -> bool:
        """True if (i, j) would co-select two catalogs from one group."""
        gi = _group_of(catalogs[j].catalog_id, groups)
        if gi is None:
            return False
        return any(
            pi == i and _group_of(catalogs[pj].catalog_id, groups) == gi for pi, pj in chosen
        )

    def add_pair(i: int, j: int) -> bool:
        nonlocal spent
        if (i, j) in chosen:
            return False
        chosen.add((i, j))
        spent += catalogs[j].query_cost
        return True

    # Pass 1: satisfy per-sightline mass / spec requirements cheaply. Always
    # respect the budget -- the previous "or not chosen" bypass could overspend
    # on the very first sightline and silently exceed the budget cap.
    for i, sl in enumerate(sightlines):
        if require_mass and sl.redshift is not None:
            cands = [
                (catalogs[j].query_cost, -catalogs[j].science_weight, j)
                for _i, j in pairs
                if _i == i and catalogs[j].provides_mass and not conflicts(i, j)
            ]
            if cands:
                cands.sort()
                _cost, _w, j = cands[0]
                if spent + catalogs[j].query_cost <= budget:
                    add_pair(i, j)
        if require_spec_z:
            cands = [
                (catalogs[j].query_cost, -catalogs[j].science_weight, j)
                for _i, j in pairs
                if _i == i and catalogs[j].provides_spec_z and not conflicts(i, j)
            ]
            if cands:
                cands.sort()
                _cost, _w, j = cands[0]
                if spent + catalogs[j].query_cost <= budget:
                    add_pair(i, j)

    # Pass 2: at least one catalog per sightline.
    for i, _sl in enumerate(sightlines):
        if any(pi == i for pi, _ in chosen):
            continue
        cands = [
            (catalogs[j].query_cost, -catalogs[j].science_weight, j) for _i, j in pairs if _i == i
        ]
        if not cands:
            continue
        cands.sort()
        for _cost, _w, j in cands:
            if conflicts(i, j):
                continue
            if spent + catalogs[j].query_cost <= budget:
                add_pair(i, j)
                break

    # Pass 3: spend remaining budget on highest science weight pairs.
    remaining = [
        (catalogs[j].science_weight / max(catalogs[j].query_cost, 1), i, j)
        for i, j in pairs
        if (i, j) not in chosen
    ]
    remaining.sort(reverse=True)
    for _score, i, j in remaining:
        if conflicts(i, j):
            continue
        cost = catalogs[j].query_cost
        if spent + cost > budget:
            continue
        add_pair(i, j)

    queries = _build_queries(
        sightlines, catalogs, chosen, cosmo=cosmo, rvir_multiple=rvir_multiple, maxrec=maxrec
    )
    return QueryPlan(
        queries=queries,
        solver="greedy",
        objective_value=sum(catalogs[j].science_weight for _i, j in chosen),
        total_cost=spent,
        budget=budget,
        metadata={"n_pairs": len(chosen)},
    )


def solve_plan_gurobi(
    sightlines: list[Sightline],
    catalogs: list[CatalogProfile],
    *,
    budget: int,
    require_mass: bool = True,
    require_spec_z: bool = False,
    cosmo: Cosmology | None = None,
    rvir_multiple: float = 1.2,
    maxrec: int = 5000,
    time_limit: float = 30.0,
    exclusive_groups: Sequence[frozenset[str]] | None = None,
) -> QueryPlan:
    """Minimum-cost query plan via Gurobi MIP; falls back to greedy on failure."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        plan = solve_plan_greedy(
            sightlines,
            catalogs,
            budget=budget,
            require_mass=require_mass,
            require_spec_z=require_spec_z,
            cosmo=cosmo,
            rvir_multiple=rvir_multiple,
            maxrec=maxrec,
            exclusive_groups=exclusive_groups,
        )
        plan.metadata["gurobi_fallback"] = f"import error: {exc}"
        return plan

    pairs = _pair_indices(sightlines, catalogs)
    if not pairs:
        raise ValueError("No applicable (sightline, catalog) pairs")

    groups = _resolve_exclusive_groups(exclusive_groups)

    pair_to_idx = {p: k for k, p in enumerate(pairs)}
    m = gp.Model("catalog_query_plan")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit

    y = m.addVars(len(pairs), vtype=GRB.BINARY, name="y")
    costs = [catalogs[j].query_cost for _i, j in pairs]
    weights = [catalogs[j].science_weight for _i, j in pairs]
    m.setObjective(
        gp.quicksum(weights[k] * y[k] for k in range(len(pairs))),
        GRB.MAXIMIZE,
    )
    m.addConstr(gp.quicksum(costs[k] * y[k] for k in range(len(pairs))) <= budget)

    for i, sl in enumerate(sightlines):
        idxs = [pair_to_idx[p] for p in pairs if p[0] == i]
        if idxs:
            m.addConstr(gp.quicksum(y[k] for k in idxs) >= 1, name=f"cover_{i}")
        if require_mass and sl.redshift is not None:
            mass_idxs = [
                pair_to_idx[p] for p in pairs if p[0] == i and catalogs[p[1]].provides_mass
            ]
            if mass_idxs:
                m.addConstr(
                    gp.quicksum(y[k] for k in mass_idxs) >= 1,
                    name=f"mass_{i}",
                )
        if require_spec_z:
            spec_idxs = [
                pair_to_idx[p] for p in pairs if p[0] == i and catalogs[p[1]].provides_spec_z
            ]
            if spec_idxs:
                m.addConstr(
                    gp.quicksum(y[k] for k in spec_idxs) >= 1,
                    name=f"spec_{i}",
                )
        # Complementarity: at most one catalog per exclusive group per
        # sightline (e.g. never query both GLADE+ and GLADE2 locally).
        for gi, group in enumerate(groups):
            grp_idxs = [
                pair_to_idx[p] for p in pairs if p[0] == i and catalogs[p[1]].catalog_id in group
            ]
            if len(grp_idxs) > 1:
                m.addConstr(
                    gp.quicksum(y[k] for k in grp_idxs) <= 1,
                    name=f"excl_{i}_{gi}",
                )

    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL) or m.SolCount < 1:
        reason = f"status {m.Status}" if m.SolCount >= 1 else f"status {m.Status}, no incumbent"
        plan = solve_plan_greedy(
            sightlines,
            catalogs,
            budget=budget,
            require_mass=require_mass,
            require_spec_z=require_spec_z,
            cosmo=cosmo,
            rvir_multiple=rvir_multiple,
            maxrec=maxrec,
            exclusive_groups=exclusive_groups,
        )
        plan.metadata["gurobi_fallback"] = reason
        return plan

    chosen = {pairs[k] for k in range(len(pairs)) if y[k].X > 0.5}
    spent = sum(catalogs[j].query_cost for _i, j in chosen)
    queries = _build_queries(
        sightlines, catalogs, chosen, cosmo=cosmo, rvir_multiple=rvir_multiple, maxrec=maxrec
    )
    return QueryPlan(
        queries=queries,
        solver="gurobi",
        objective_value=float(m.ObjVal),
        total_cost=spent,
        budget=budget,
        metadata={
            "n_pairs": len(chosen),
            "mip_gap": float(getattr(m, "MIPGap", 0.0)),
            "runtime_sec": float(m.Runtime),
        },
    )


def plan_queries(
    sightlines: list[Sightline],
    catalogs: list[CatalogProfile] | None = None,
    *,
    budget: int = 50,
    solver: SolverName = "gurobi",
    require_mass: bool = True,
    require_spec_z: bool = False,
    cosmo: Cosmology | None = None,
    rvir_multiple: float = 1.2,
    maxrec: int = 5000,
    exclusive_groups: Sequence[frozenset[str]] | None = None,
) -> QueryPlan:
    """Plan catalog TAP queries for a list of sightlines."""
    if not sightlines:
        raise ValueError("At least one sightline required")
    if catalogs is None:
        catalogs = profiles_from_adapters()
    if solver == "gurobi":
        return solve_plan_gurobi(
            sightlines,
            catalogs,
            budget=budget,
            require_mass=require_mass,
            require_spec_z=require_spec_z,
            cosmo=cosmo,
            rvir_multiple=rvir_multiple,
            maxrec=maxrec,
            exclusive_groups=exclusive_groups,
        )
    return solve_plan_greedy(
        sightlines,
        catalogs,
        budget=budget,
        require_mass=require_mass,
        require_spec_z=require_spec_z,
        cosmo=cosmo,
        rvir_multiple=rvir_multiple,
        maxrec=maxrec,
        exclusive_groups=exclusive_groups,
    )
