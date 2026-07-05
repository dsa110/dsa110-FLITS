"""CLI for the VO discover -> query -> reduce pipeline (installed as ``flits-halos``).

Ported from the los_halos typer CLI, which defined ``discover`` twice (the
service-printing variant was shadowed); here they are distinct subcommands.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .discover import _normalize_service_url, discover_tables
from .normalize import to_common_schema
from .query import cone_query
from .registry import discover_tap_services
from .targets import load_targets

_SERVICE_ALIASES = {
    "vizier": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap",
    "mast": "https://mast.stsci.edu/vo-tap/",
    "datalab": "https://datalab.noirlab.edu/tap",
}


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def cmd_services(args: argparse.Namespace) -> None:
    kws = tuple(args.keywords) if args.keywords else ("catalog", "galaxy", "group", "cluster")
    df = discover_tap_services(
        keywords=kws,
        max_services=args.max_services,
        regtap_url=args.regtap_url,
        include_anchors=not args.no_anchors,
    )
    print(df.to_string(index=False))


def cmd_tables(args: argparse.Namespace) -> None:
    df = discover_tables(args.access_url, limit=args.limit, cache_dir=args.cache_dir)
    print(df.to_string(index=False))


def cmd_cone(args: argparse.Namespace) -> None:
    df = cone_query(
        args.access_url, args.table, args.ra_col, args.dec_col,
        args.ra, args.dec, args.radius_arcmin / 60.0, columns=args.columns,
    )
    print(df.head().to_string(index=False))


def cmd_run_targets(args: argparse.Namespace) -> None:
    """Cone-query one endpoint/table for every target; print per-target row counts."""
    for t in load_targets(args.targets):
        df = cone_query(
            args.access_url, args.table, args.ra_col, args.dec_col,
            t.ra, t.dec, args.radius_arcmin / 60.0,
        )
        print(f"# {t.name}: {len(df)} rows")


def cmd_discover(args: argparse.Namespace) -> None:
    """Cache the service list and per-service candidate tables."""
    if args.services == "auto":
        df = discover_tap_services(keywords=(), include_anchors=True, max_services=200)
        svc_urls = df["access_url"].tolist()
    else:
        svc_urls = [_SERVICE_ALIASES[k.strip().lower()] for k in args.services.split(",") if k.strip().lower() in _SERVICE_ALIASES]
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    svc_records = []
    for url in svc_urls:
        tables = discover_tables(url, limit=args.limit, cache_dir=cache)
        # hash the normalized URL: discover_tables names its cache file by it, and
        # query/reduce glob tables_{service_hash}_* — the two must agree
        h = _hash(_normalize_service_url(url) or url)
        svc_records.append({"access_url": url, "service_hash": h, "num_tables": len(tables)})
    pd.DataFrame(svc_records).to_parquet(cache / "services.parquet")
    print(f"Wrote services and tables caches for {len(svc_records)} services")


def _cached_tables(cache: Path, service_hash: str) -> pd.DataFrame | None:
    hits = list(cache.glob(f"tables_{service_hash}_lim*.parquet"))
    return pd.read_parquet(hits[0]) if hits else None


def cmd_query(args: argparse.Namespace) -> None:
    """Run cached cone queries for every (service, table, target)."""
    cache = Path(args.cache_dir)
    svc_df = pd.read_parquet(cache / "services.parquet")
    targets = load_targets(args.targets)
    for _, svc in svc_df.iterrows():
        tables = _cached_tables(cache, svc["service_hash"])
        if tables is None:
            continue
        for _, row in tables.iterrows():
            outdir = cache / "queries" / svc["service_hash"] / _hash(row["table"])
            for t in targets:
                try:
                    df = cone_query(
                        svc["access_url"], row["table"], row["ra_col"], row["dec_col"],
                        t.ra, t.dec, args.radius / 60.0, maxrec=args.maxrec,
                    )
                except Exception:
                    continue  # skip failing tables/targets
                outdir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(outdir / f"{t.name}.parquet", index=False)
    print("Queries cached.")


def cmd_reduce(args: argparse.Namespace) -> None:
    """Aggregate cached query results into per-target candidate tables."""
    cache = Path(args.cache_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    svc_df = pd.read_parquet(cache / "services.parquet")
    for t in load_targets(args.targets):
        frb_rows = []
        for _, svc in svc_df.iterrows():
            tables = _cached_tables(cache, svc["service_hash"])
            if tables is None:
                continue
            for _, row in tables.iterrows():
                qpath = cache / "queries" / svc["service_hash"] / _hash(row["table"]) / f"{t.name}.parquet"
                if not qpath.exists():
                    continue
                df = pd.read_parquet(qpath)
                if df.empty:
                    continue
                norm = to_common_schema(
                    df, row["ra_col"], row["dec_col"], row["z_col"],
                    service=svc["access_url"], table=row["table"],
                )
                norm["frb_name"] = t.name
                frb_rows.append(norm)
        if frb_rows:
            frb_df = pd.concat(frb_rows, ignore_index=True)
            frb_outdir = out / t.name
            frb_outdir.mkdir(parents=True, exist_ok=True)
            frb_df.to_parquet(frb_outdir / "candidates.parquet", index=False)
            (frb_outdir / "summary.md").write_text(f"# {t.name} candidates\n\nTotal rows: {len(frb_df)}\n")
    print("Reduction completed.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="flits-halos", description=__doc__)
    p.add_argument("--cache-dir", default=".cache", help="Cache directory (default: .cache)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("services", help="Discover TAP services via RegTAP")
    sp.add_argument("--keywords", nargs="*", default=None)
    sp.add_argument("--max-services", type=int, default=50)
    sp.add_argument("--regtap-url", default="https://dc.g-vo.org/tap")
    sp.add_argument("--no-anchors", action="store_true")
    sp.set_defaults(func=cmd_services)

    sp = sub.add_parser("tables", help="List candidate tables for a TAP endpoint")
    sp.add_argument("access_url")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_tables)

    sp = sub.add_parser("cone", help="Run a single cone query")
    for a in ("access_url", "table", "ra_col", "dec_col"):
        sp.add_argument(a)
    sp.add_argument("ra", type=float)
    sp.add_argument("dec", type=float)
    sp.add_argument("--radius-arcmin", type=float, default=5.0)
    sp.add_argument("--columns", default="*")
    sp.set_defaults(func=cmd_cone)

    sp = sub.add_parser("run-targets", help="Cone-query one endpoint/table for every target")
    for a in ("access_url", "table", "ra_col", "dec_col"):
        sp.add_argument(a)
    sp.add_argument("--targets", type=Path, required=True, help="Path to targets.yaml")
    sp.add_argument("--radius-arcmin", type=float, default=5.0)
    sp.set_defaults(func=cmd_run_targets)

    sp = sub.add_parser("discover", help="Cache services and their candidate tables")
    sp.add_argument("--services", default="auto", help="auto|vizier,mast,datalab")
    sp.add_argument("--limit", type=int, default=500)
    sp.set_defaults(func=cmd_discover)

    sp = sub.add_parser("query", help="Run cached cone queries for all targets")
    sp.add_argument("--targets", type=Path, required=True, help="Path to targets.yaml")
    sp.add_argument("--radius", type=float, default=20.0, help="Search radius (arcmin)")
    sp.add_argument("--maxrec", type=int, default=10000)
    sp.set_defaults(func=cmd_query)

    sp = sub.add_parser("reduce", help="Aggregate cached queries into per-target candidates")
    sp.add_argument("--targets", type=Path, required=True, help="Path to targets.yaml")
    sp.add_argument("--out", type=Path, default=Path("results"))
    sp.set_defaults(func=cmd_reduce)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
