"""CLI for the VO discover -> query -> reduce pipeline (installed as ``flits-halos``).

Two command families share this entry point:

- the los_halos-lineage cache pipeline over arbitrary TAP services
  (``services tables cone run-targets discover query reduce``), and
- the frb-foreground-halos-lineage curated-catalog pipeline with halo masses,
  host exclusion, and intersection classification
  (``rank plan-queries run-plan run-catalog dedupe-candidates plot-candidates``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog_plan import plan_queries
from .classify import add_intersection_flags, empty_sightline_row, summarize_sightlines
from .dedupe import deduplicate_candidates
from .discover import _normalize_service_url, discover_tables
from .halos import ADAPTERS
from .io import normalize_host_redshift, read_table, read_targets_yaml, write_table
from .normalize import ColumnMapping, normalize, to_common_schema
from .plotting import load_targets as load_plot_targets
from .plotting import plot_all_candidate_sky_maps
from .query import cone_query
from .reduce import merge_and_rank
from .registry import discover_tap_services
from .targets import load_targets

VIZIER_TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"

_SERVICE_ALIASES = {
    "vizier": VIZIER_TAP_URL,
    "mast": "https://mast.stsci.edu/vo-tap/",
    "datalab": "https://datalab.noirlab.edu/tap",
}


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _existing_path(value: str) -> Path:
    # parse-time input validation, mirroring click.Path(exists=True) in the ffh original
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path {value!r} does not exist")
    return path


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
    df, meta = cone_query(
        args.access_url,
        args.table,
        args.ra_col,
        args.dec_col,
        args.ra,
        args.dec,
        args.radius_arcmin / 60.0,
        columns=args.columns,
        maxrec=args.maxrec,
    )
    if args.output:
        write_table(df, args.output)
        print(f"Wrote {len(df)} rows to {args.output}")
    else:
        print(df.head().to_string(index=False))
    print(meta["adql"])


def cmd_run_targets(args: argparse.Namespace) -> None:
    """Cone-query one endpoint/table for every target; print per-target row counts."""
    for t in load_targets(args.targets):
        df, _ = cone_query(
            args.access_url,
            args.table,
            args.ra_col,
            args.dec_col,
            t.ra,
            t.dec,
            args.radius_arcmin / 60.0,
        )
        print(f"# {t.name}: {len(df)} rows")


def cmd_discover(args: argparse.Namespace) -> None:
    """Cache the service list and per-service candidate tables."""
    if args.services == "auto":
        df = discover_tap_services(keywords=(), include_anchors=True, max_services=200)
        svc_urls = df["access_url"].tolist()
    else:
        svc_urls = [
            _SERVICE_ALIASES[k.strip().lower()]
            for k in args.services.split(",")
            if k.strip().lower() in _SERVICE_ALIASES
        ]
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
                    df, _ = cone_query(
                        svc["access_url"],
                        row["table"],
                        row["ra_col"],
                        row["dec_col"],
                        t.ra,
                        t.dec,
                        args.radius / 60.0,
                        maxrec=args.maxrec,
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
                qpath = (
                    cache
                    / "queries"
                    / svc["service_hash"]
                    / _hash(row["table"])
                    / f"{t.name}.parquet"
                )
                if not qpath.exists():
                    continue
                df = pd.read_parquet(qpath)
                if df.empty:
                    continue
                norm = to_common_schema(
                    df,
                    ra_col=row["ra_col"],
                    dec_col=row["dec_col"],
                    z_col=row["z_col"],
                    service=svc["access_url"],
                    table=row["table"],
                )
                norm["frb_name"] = t.name
                frb_rows.append(norm)
        if frb_rows:
            frb_df = pd.concat(frb_rows, ignore_index=True)
            frb_outdir = out / t.name
            frb_outdir.mkdir(parents=True, exist_ok=True)
            frb_df.to_parquet(frb_outdir / "candidates.parquet", index=False)
            (frb_outdir / "summary.md").write_text(
                f"# {t.name} candidates\n\nTotal rows: {len(frb_df)}\n"
            )
    print("Reduction completed.")


def _load_target_records(path: Path) -> list[dict]:
    """Load targets from YAML or CSV into a uniform list of records."""
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        sightlines = read_targets_yaml(path)
        return [
            {"name": s.name, "ra": s.ra, "dec": s.dec, "redshift": s.redshift} for s in sightlines
        ]
    if suffix == ".csv":
        df = load_plot_targets(path)
        records: list[dict] = []
        for _, row in df.iterrows():
            z_value = row.get("frb_z_from_targets")
            records.append(
                {
                    "name": str(row["frb_name"]),
                    "ra": float(row["frb_ra"]),
                    "dec": float(row["frb_dec"]),
                    "redshift": float(z_value) if pd.notna(z_value) else None,
                }
            )
        return records
    raise SystemExit(f"Unsupported target file type: {suffix}")


def _merge_mass_provenance(normalized: pd.DataFrame, mass_per_row: pd.Series) -> pd.DataFrame:
    """Inject per-row mass provenance under a 'mass' key in provenance_json."""
    if "provenance_json" not in normalized.columns:
        return normalized
    out = normalized.reset_index(drop=True).copy()
    mass_aligned = mass_per_row.reset_index(drop=True)
    updated: list[str] = []
    for i, base in enumerate(out["provenance_json"]):
        try:
            base_d = json.loads(base) if base else {}
        except (TypeError, ValueError):
            base_d = {}
        mp_raw = mass_aligned.iat[i] if i < len(mass_aligned) else None
        if isinstance(mp_raw, str) and mp_raw and mp_raw != "null":
            try:
                mp_parsed = json.loads(mp_raw)
                if mp_parsed is not None:
                    base_d["mass"] = mp_parsed
            except ValueError:
                pass
        updated.append(json.dumps(base_d, sort_keys=True))
    out["provenance_json"] = updated
    return out


def _load_query_plan(path: Path) -> list[dict]:
    """Load rows from a plan-queries CSV or JSON artifact."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
        rows = payload.get("queries", [])
        if not rows:
            raise SystemExit(f"No queries found in {path}")
        return rows
    if suffix in (".csv", ".parquet"):
        df = read_table(path)
        if df.empty:
            raise SystemExit(f"Empty query plan: {path}")
        return df.to_dict(orient="records")
    raise SystemExit(f"Unsupported plan file type: {suffix}")


def _process_catalog_query(
    *,
    catalog_name: str,
    access_url: str,
    target: dict,
    radius_arcmin: float,
    maxrec: int,
) -> tuple[pd.DataFrame | None, bool, bool]:
    """Cone-query one catalog at one sightline; return ranked frame or None."""
    from astropy.cosmology import Planck18

    if catalog_name not in ADAPTERS:
        raise SystemExit(f"Unknown catalog '{catalog_name}'. Known: {sorted(ADAPTERS)}")
    adapter = ADAPTERS[catalog_name]
    radius_deg = radius_arcmin / 60.0
    raw, meta = cone_query(
        access_url,
        adapter.table_id,
        adapter.ra_col,
        adapter.dec_col,
        target["ra"],
        target["dec"],
        radius_deg,
        maxrec=maxrec,
    )
    if raw.empty:
        return None, False, True

    truncated = bool(meta.get("truncated"))
    prepared = adapter.prepare(raw, cosmo=Planck18)
    mapping = ColumnMapping(
        name=adapter.name_col,
        id=adapter.id_col,
        ra=adapter.ra_col,
        dec=adapter.dec_col,
        z=adapter.z_col,
        z_type="z_type",
        m_delta="m_halo",
        delta_def="delta_def",
    )
    normalized = normalize(
        prepared,
        mapping,
        service=catalog_name,
        table=adapter.table_id,
        adql=meta["adql"],
        extra_provenance={"radius_arcmin": radius_arcmin, "maxrec": maxrec},
    )
    normalized = _merge_mass_provenance(normalized, prepared["mass_provenance"])
    normalized["frb_name"] = target["name"]
    normalized["frb_ra"] = target["ra"]
    normalized["frb_dec"] = target["dec"]
    normalized["frb_z"] = target["redshift"]
    normalized["catalog_id"] = catalog_name
    ranked = merge_and_rank(normalized, frb_ra_deg=target["ra"], frb_dec_deg=target["dec"])
    return ranked, truncated, False


def _classify_foreground_candidates(
    combined: pd.DataFrame,
    *,
    host_theta_arcmin_max: float,
    host_dz_max: float,
    host_dz_spec_max: float,
) -> pd.DataFrame:
    """Add host/foreground flags to a combined candidate table."""
    z_cand = pd.to_numeric(combined["z"], errors="coerce")
    z_host = pd.to_numeric(combined["frb_z"], errors="coerce")
    theta = pd.to_numeric(combined["theta_arcmin"], errors="coerce")
    delta_z = (z_cand - z_host).abs()
    z_type_str = (
        combined["z_type"].astype(str)
        if "z_type" in combined.columns
        else pd.Series(["unknown"] * len(combined), index=combined.index)
    )
    is_spec = z_type_str == "spec"
    dz_threshold = pd.Series(
        np.where(is_spec, host_dz_spec_max, host_dz_max),
        index=combined.index,
        dtype=float,
    )
    out = combined.copy()
    out["z_frb_known"] = z_host.notna()
    out["likely_host"] = (
        z_host.notna()
        & z_cand.notna()
        & theta.notna()
        & (theta <= host_theta_arcmin_max)
        & (delta_z <= dz_threshold)
    ).fillna(False)
    out["is_foreground"] = (
        z_host.notna() & z_cand.notna() & (z_cand < z_host) & ~out["likely_host"]
    ).fillna(False)
    return out


def _finalize_catalog_run(
    *,
    per_target_frames: list[pd.DataFrame],
    target_records: list[dict],
    zero_row_targets: list[str],
    truncated_targets: list[str],
    output: Path,
    summary_output: Path | None,
    maxrec: int,
    host_theta_arcmin_max: float,
    host_dz_max: float,
    host_dz_spec_max: float,
) -> None:
    """Write candidates + per-sightline summary with status priority rules."""
    if per_target_frames:
        combined = pd.concat(per_target_frames, ignore_index=True)
        combined = add_intersection_flags(combined)
        combined = _classify_foreground_candidates(
            combined,
            host_theta_arcmin_max=host_theta_arcmin_max,
            host_dz_max=host_dz_max,
            host_dz_spec_max=host_dz_spec_max,
        )
        n_likely_host = int(combined["likely_host"].sum())
        if n_likely_host:
            print(
                f"  {n_likely_host} candidate(s) flagged likely_host "
                f"(theta <= {host_theta_arcmin_max:g}')"
            )
        write_table(combined, output)
        print(f"Wrote {len(combined)} candidates -> {output}")
        summary = summarize_sightlines(combined)
    else:
        print("No candidates returned; writing summary-only output.")
        summary = summarize_sightlines(pd.DataFrame())

    if zero_row_targets:
        extras = pd.DataFrame(
            [
                empty_sightline_row(name, status="no_candidates_in_cone")
                for name in sorted(set(zero_row_targets))
            ]
        )
        summary = pd.concat([summary, extras], ignore_index=True) if not summary.empty else extras

    if truncated_targets:
        mask = summary["frb_name"].isin(truncated_targets) & (
            summary["status"] != "no_candidates_in_cone"
        )
        summary.loc[mask, "status"] = "possibly_truncated"
    z_unknown_targets = [t["name"] for t in target_records if t["redshift"] is None]
    if z_unknown_targets:
        mask = summary["frb_name"].isin(z_unknown_targets) & (
            summary["status"] != "no_candidates_in_cone"
        )
        summary.loc[mask, "status"] = "z_frb_unknown"
    summary = summary.sort_values("frb_name").reset_index(drop=True)

    summary_path = summary_output or output.with_suffix(".summary.csv")
    write_table(summary, summary_path)
    print(f"Wrote {len(summary)} sightline summaries -> {summary_path}")

    final_status = summary.set_index("frb_name")["status"]
    for status_key, label in (
        ("no_candidates_in_cone", "no_candidates_in_cone"),
        ("z_frb_unknown", "z_frb_unknown"),
        ("possibly_truncated", "possibly_truncated"),
    ):
        names = final_status.index[final_status == status_key].tolist()
        if names:
            print(f"  ({len(names)} sightlines {label}: {', '.join(names)})")


def cmd_rank(args: argparse.Namespace) -> None:
    """Rank an already-normalized candidate table for the first target in a YAML file."""
    targets = read_targets_yaml(args.targets_yaml)
    if not targets:
        raise SystemExit("No targets found")
    target = targets[0]
    df = read_table(args.candidates)
    ranked = merge_and_rank(df, frb_ra_deg=target.ra, frb_dec_deg=target.dec)
    write_table(ranked, args.output)
    print(f"Wrote {len(ranked)} ranked rows to {args.output}")


def cmd_plan_queries(args: argparse.Namespace) -> None:
    """Optimize which catalogs to query per localized FRB under a TAP budget."""
    from .catalog_plan import profiles_from_adapters
    from .domain import Sightline

    target_records = _load_target_records(args.targets)
    if not target_records:
        raise SystemExit(f"No targets found in {args.targets}")

    sightlines = [
        Sightline(
            name=row["name"],
            ra=row["ra"],
            dec=row["dec"],
            redshift=row.get("redshift"),
        )
        for row in target_records
    ]
    plan = plan_queries(
        sightlines,
        profiles_from_adapters(),
        budget=args.budget,
        solver=args.solver,
        require_mass=args.require_mass,
        require_spec_z=args.require_spec_z,
        rvir_multiple=args.rvir_multiple,
        maxrec=args.maxrec,
    )
    df = plan.to_dataframe()
    write_table(df, args.output)
    print(
        f"Solver={plan.solver} cost={plan.total_cost}/{plan.budget} "
        f"queries={len(plan.queries)} -> {args.output}"
    )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(plan.to_json())
        print(f"Wrote plan JSON -> {args.json_output}")
    if plan.metadata.get("gurobi_fallback"):
        print(f"  note: {plan.metadata['gurobi_fallback']}")


def cmd_run_plan(args: argparse.Namespace) -> None:
    """Execute a plan-queries CSV/JSON using per-row radius_arcmin and maxrec."""
    plan_rows = _load_query_plan(args.plan_file)
    per_target_frames: list[pd.DataFrame] = []
    zero_row_by_frb: dict[str, int] = {}
    hits_by_frb: dict[str, int] = {}
    truncated_targets: list[str] = []
    target_records: list[dict] = []
    seen_targets: set[str] = set()
    effective_maxrec = args.maxrec or 5000

    for row in plan_rows:
        name = str(row.get("frb_name") or row.get("sightline_name"))
        catalog_id = str(row["catalog_id"])
        access_url = str(row.get("access_url") or VIZIER_TAP_URL)
        radius = float(row["radius_arcmin"])
        row_maxrec = int(row.get("maxrec") or effective_maxrec)
        if args.maxrec is not None:
            row_maxrec = args.maxrec
        frb_z = normalize_host_redshift(row.get("frb_z"))
        target = {
            "name": name,
            "ra": float(row["ra_deg"]),
            "dec": float(row["dec_deg"]),
            "redshift": frb_z,
        }
        if name not in seen_targets:
            target_records.append(target)
            seen_targets.add(name)
            zero_row_by_frb[name] = 0
            hits_by_frb[name] = 0

        print(f"[{catalog_id}] querying {name} @ {radius:g}' maxrec={row_maxrec} ...")
        ranked, truncated, zero = _process_catalog_query(
            catalog_name=catalog_id,
            access_url=access_url,
            target=target,
            radius_arcmin=radius,
            maxrec=row_maxrec,
        )
        if zero:
            zero_row_by_frb[name] += 1
            print("  -> 0 rows")
            continue
        if truncated:
            truncated_targets.append(name)
            print(f"  WARNING: possibly_truncated at maxrec={row_maxrec}")
        hits_by_frb[name] += 1
        per_target_frames.append(ranked)
        print(f"  -> {len(ranked)} ranked rows")

    zero_row_targets = [
        name for name, misses in zero_row_by_frb.items() if hits_by_frb[name] == 0 and misses > 0
    ]
    _finalize_catalog_run(
        per_target_frames=per_target_frames,
        target_records=target_records,
        zero_row_targets=zero_row_targets,
        truncated_targets=truncated_targets,
        output=args.output,
        summary_output=args.summary_output,
        maxrec=effective_maxrec,
        host_theta_arcmin_max=args.host_theta_arcmin_max,
        host_dz_max=args.host_dz_max,
        host_dz_spec_max=args.host_dz_spec_max,
    )


def cmd_run_catalog(args: argparse.Namespace) -> None:
    """End-to-end: cone-query a catalog at each sightline, derive halo masses, rank, and classify intersections."""
    if args.catalog_name not in ADAPTERS:
        raise SystemExit(f"Unknown catalog '{args.catalog_name}'. Known: {sorted(ADAPTERS)}")
    target_records = _load_target_records(args.targets)
    if not target_records:
        raise SystemExit(f"No targets found in {args.targets}")

    per_target_frames: list[pd.DataFrame] = []
    zero_row_targets: list[str] = []
    truncated_targets: list[str] = []

    for target in target_records:
        print(f"[{args.catalog_name}] querying {target['name']} ...")
        ranked, truncated, zero = _process_catalog_query(
            catalog_name=args.catalog_name,
            access_url=args.access_url,
            target=target,
            radius_arcmin=args.radius_arcmin,
            maxrec=args.maxrec,
        )
        if zero:
            print("  -> 0 rows")
            zero_row_targets.append(target["name"])
            continue
        if truncated:
            truncated_targets.append(target["name"])
            print(
                f"  WARNING: TAP returned rows == maxrec={args.maxrec}; "
                f"status='possibly_truncated'."
            )
        per_target_frames.append(ranked)
        print(f"  -> {len(ranked)} ranked rows")

    _finalize_catalog_run(
        per_target_frames=per_target_frames,
        target_records=target_records,
        zero_row_targets=zero_row_targets,
        truncated_targets=truncated_targets,
        output=args.output,
        summary_output=args.summary_output,
        maxrec=args.maxrec,
        host_theta_arcmin_max=args.host_theta_arcmin_max,
        host_dz_max=args.host_dz_max,
        host_dz_spec_max=args.host_dz_spec_max,
    )


def cmd_dedupe_candidates(args: argparse.Namespace) -> None:
    """Merge duplicate galaxies across catalogs within each sightline."""
    df = read_table(args.input_path)
    before = len(df)
    deduped = deduplicate_candidates(df)
    write_table(deduped, args.output)
    print(f"Deduped {before} -> {len(deduped)} rows -> {args.output}")
    if args.summary_output:
        flagged = add_intersection_flags(deduped)
        summary = summarize_sightlines(flagged)
        z_host = pd.to_numeric(flagged.get("frb_z"), errors="coerce")
        unknown_frbs = set(flagged.loc[z_host.isna(), "frb_name"].dropna().astype(str).unique())
        if unknown_frbs:
            mask = summary["frb_name"].isin(unknown_frbs) & (summary["status"] == "ok")
            summary.loc[mask, "status"] = "z_frb_unknown"
        write_table(summary, args.summary_output)
        print(f"Wrote {len(summary)} sightline summaries -> {args.summary_output}")


def cmd_plot_candidates(args: argparse.Namespace) -> None:
    """Plot foreground-candidate sky maps from a normalized candidate table."""
    candidate_rows = read_table(args.candidates)
    target_rows = load_plot_targets(args.targets_csv)
    outputs = plot_all_candidate_sky_maps(
        candidate_rows,
        target_rows,
        output_dir=args.output_dir,
        search_radius_arcmin=args.search_radius_arcmin,
        label_top_n=args.label_top_n,
    )
    for output in outputs:
        print(output)


def _add_host_exclusion_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--host-theta-arcmin-max",
        type=float,
        default=0.1,
        help="Candidates within this angular separation of the FRB position AND "
        "within the Δz tolerance are flagged likely_host and excluded from "
        "foreground intersection counts (default 0.1' = 6\").",
    )
    sp.add_argument(
        "--host-dz-max",
        type=float,
        default=0.05,
        help="Max |z_cand - z_FRB| for photo/unknown z_type host confusion (GLADE+ photo-z scatter).",
    )
    sp.add_argument(
        "--host-dz-spec-max",
        type=float,
        default=0.005,
        help="Max |z_cand - z_FRB| for spec-z host confusion (spec-z precision).",
    )


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
    sp.add_argument("--maxrec", type=int, default=10000)
    sp.add_argument("--output", type=Path, default=None)
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

    sp = sub.add_parser("rank", help="Rank an already-normalized candidate table (first target)")
    sp.add_argument("targets_yaml", type=_existing_path)
    sp.add_argument("candidates", type=_existing_path)
    sp.add_argument("--output", type=Path, default=Path("results/ranked.csv"))
    sp.set_defaults(func=cmd_rank)

    sp = sub.add_parser(
        "plan-queries", help="Optimize catalog queries per sightline under a budget"
    )
    sp.add_argument("targets", type=_existing_path)
    sp.add_argument("--budget", type=int, default=50, help="Max total query cost")
    sp.add_argument("--solver", type=str.lower, choices=["gurobi", "greedy"], default="gurobi")
    sp.add_argument("--require-mass", action=argparse.BooleanOptionalAction, default=True)
    sp.add_argument("--require-spec-z", action=argparse.BooleanOptionalAction, default=False)
    sp.add_argument(
        "--rvir-multiple",
        type=float,
        default=1.2,
        help="Cone radius sized for b <= this multiple of R_delta (loose bracket)",
    )
    sp.add_argument("--output", type=Path, default=Path("results/query_plan.csv"))
    sp.add_argument("--json-output", type=Path, default=None)
    sp.add_argument(
        "--maxrec",
        type=int,
        default=5000,
        help="TAP row limit recorded in each planned query for run-plan",
    )
    sp.set_defaults(func=cmd_plan_queries)

    sp = sub.add_parser("run-plan", help="Execute a plan-queries CSV/JSON")
    sp.add_argument("plan_file", type=_existing_path)
    sp.add_argument("--maxrec", type=int, default=None, help="Override per-row plan maxrec")
    sp.add_argument("--output", type=Path, default=Path("results/run_plan_candidates.csv"))
    sp.add_argument("--summary-output", type=Path, default=None)
    _add_host_exclusion_args(sp)
    sp.set_defaults(func=cmd_run_plan)

    sp = sub.add_parser("run-catalog", help="End-to-end: cone-query a catalog at each sightline")
    sp.add_argument("targets", type=_existing_path)
    sp.add_argument("catalog_name")
    sp.add_argument("--access-url", default=VIZIER_TAP_URL)
    sp.add_argument("--radius-arcmin", type=float, default=10.0)
    sp.add_argument("--maxrec", type=int, default=200)
    sp.add_argument("--output", type=Path, default=Path("results/run_catalog_candidates.csv"))
    sp.add_argument("--summary-output", type=Path, default=None)
    _add_host_exclusion_args(sp)
    sp.set_defaults(func=cmd_run_catalog)

    sp = sub.add_parser(
        "dedupe-candidates", help="Merge duplicate galaxies across catalogs per sightline"
    )
    sp.add_argument("input_path", type=_existing_path)
    sp.add_argument("--output", type=Path, default=Path("results/candidates_deduped.csv"))
    sp.add_argument("--summary-output", type=Path, default=None)
    sp.set_defaults(func=cmd_dedupe_candidates)

    sp = sub.add_parser("plot-candidates", help="Sky maps from a normalized candidate table")
    sp.add_argument("candidates", type=_existing_path)
    sp.add_argument("targets_csv", type=_existing_path)
    sp.add_argument("--output-dir", type=Path, default=Path("figures"))
    sp.add_argument("--search-radius-arcmin", type=float, default=None)
    sp.add_argument("--label-top-n", type=int, default=5)
    sp.set_defaults(func=cmd_plot_candidates)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
