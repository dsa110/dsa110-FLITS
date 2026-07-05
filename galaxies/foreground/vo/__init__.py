"""VO-TAP wide-net foreground halo/cluster discovery along FRB sightlines.

Consolidates the standalone ``los_halos`` (PR #122) and ``frb-foreground-halos``
repos. Complements the curated engines in ``galaxies.foreground``: this layer
casts a wide net across arbitrary VO TAP services (RegTAP discovery ->
table/column inference -> ADQL cone queries -> common schema -> halo masses ->
impact-parameter ranking -> host/foreground classification -> dedupe -> plots).
"""

from .catalog_plan import plan_queries
from .classify import add_intersection_flags, summarize_sightlines
from .dedupe import deduplicate_candidates
from .discover import discover_tables, find_columns
from .domain import ForegroundCandidate, QueryResult, Sightline
from .halos import ADAPTERS, HaloAdapter, add_halo_masses
from .io import read_targets_yaml
from .normalize import SCHEMA_COLUMNS, ColumnMapping, normalize, to_common_schema
from .provenance import make_provenance, parse_provenance
from .query import build_cone_adql, cone_query, safe_search
from .reduce import merge_and_rank
from .registry import discover_tap_services
from .targets import Target, get_cosmology, load_targets

__all__ = [
    "ADAPTERS",
    "ColumnMapping",
    "ForegroundCandidate",
    "HaloAdapter",
    "QueryResult",
    "SCHEMA_COLUMNS",
    "Sightline",
    "Target",
    "add_halo_masses",
    "add_intersection_flags",
    "build_cone_adql",
    "cone_query",
    "deduplicate_candidates",
    "discover_tables",
    "discover_tap_services",
    "find_columns",
    "get_cosmology",
    "load_targets",
    "make_provenance",
    "merge_and_rank",
    "normalize",
    "parse_provenance",
    "plan_queries",
    "read_targets_yaml",
    "safe_search",
    "summarize_sightlines",
    "to_common_schema",
]
