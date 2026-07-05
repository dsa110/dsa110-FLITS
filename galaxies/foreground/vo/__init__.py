"""VO-TAP wide-net foreground halo/cluster discovery along FRB sightlines.

Migrated from the standalone ``los_halos`` repo (github.com/jakobtfaber/los_halos).
Complements the curated engines in ``galaxies.foreground``: this layer casts a wide
net across arbitrary VO TAP services (RegTAP discovery -> table/column inference ->
ADQL cone queries -> common schema -> impact-parameter ranking).
"""

from .discover import discover_tables, find_columns
from .normalize import SCHEMA_COLUMNS, ColumnMapping, normalize, to_common_schema
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
    "merge_and_rank",
    "normalize",
    "safe_search",
    "to_common_schema",
]
