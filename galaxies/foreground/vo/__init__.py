"""VO-TAP wide-net foreground halo/cluster discovery along FRB sightlines.

Consolidates the standalone ``los_halos`` (2026-07-05, PR #122) and
``frb-foreground-halos`` repos. Complements the curated engines in
``galaxies.foreground``: this layer casts a wide net across arbitrary VO TAP
services (RegTAP discovery -> table/column inference -> ADQL cone queries ->
common schema -> halo masses -> impact-parameter ranking -> classification).
"""

from .discover import discover_tables, find_columns
from .normalize import SCHEMA_COLUMNS, ColumnMapping, normalize, to_common_schema
from .provenance import make_provenance, parse_provenance
from .query import build_cone_adql, cone_query, safe_search
from .reduce import merge_and_rank
from .registry import discover_tap_services
from .targets import Target, get_cosmology, load_targets

__all__ = [
    "SCHEMA_COLUMNS",
    "ColumnMapping",
    "Target",
    "build_cone_adql",
    "cone_query",
    "discover_tables",
    "discover_tap_services",
    "find_columns",
    "get_cosmology",
    "load_targets",
    "make_provenance",
    "merge_and_rank",
    "normalize",
    "parse_provenance",
    "safe_search",
    "to_common_schema",
]
