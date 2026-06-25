"""Main pipeline for finding foreground galaxies."""

import math
import os
import re

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord

from . import scattering_predict as scat
from .build_unified import build_for_target
from .config import (
    CLUSTER_M500_TO_M200,
    CLUSTER_R200_FACTOR,
    DEFAULT_CLUSTER_IMPACT_KPC,
    DEFAULT_IMPACT_KPC,
    DEFAULT_Z_EPS,
    ENABLE_CLUSTER_ENGINE,
    ENABLE_EXTRA_ENGINES,
    FOREGROUND_AMBIGUITY_KMS,
    FOREGROUND_PHOTOZ_FLOOR,
    MAX_SEARCH_RADIUS_DEG,
    MIN_Z_SEARCH,
    PHOTO_Z_CATALOG_SUBSTRINGS,
    SPEC_Z_CATALOG_SUBSTRINGS,
    SPEED_OF_LIGHT_KMS,
    TARGETS,
    VIZIER_CATALOGS,
)
from .engines import VizierEngine, query_ps1_gi_mags
from .engines_extra import ClusterEngine, DesiDr1Engine, NedTapEngine
from .utils import calculate_impact_parameter, get_angular_radius, parse_coord

PHOTO_Z_ERROR_COLUMNS = ("z_phot_err", "e_zphot", "z_best_err")
DUPLICATE_SEPARATION = 10.0 * u.arcsec
DUPLICATE_REDSHIFT_TOLERANCE = 0.02

# Columns that may carry an object classification (NED 'Type', SDSS 'class'/'cl').
CLASSIFICATION_COLUMNS = ("classification", "Type", "class", "cl", "otype")
# Galaxy-cluster classifications: NED 'GClstr', SDSS/SIMBAD 'ClG', bare 'Cl', 'cluster'.
_CLUSTER_RE = re.compile(r"cluster|gclstr|clg|^cl\b", re.IGNORECASE)


def _cluster_mask(df: pd.DataFrame) -> pd.Series:
    """Per-row True where a classification column marks the object a galaxy cluster."""
    is_cluster = pd.Series(False, index=df.index)
    for column in CLASSIFICATION_COLUMNS:
        if column not in df.columns:
            continue
        is_cluster = is_cluster | df[column].astype(str).str.contains(_CLUSTER_RE)
    return is_cluster


def _first_available_numeric(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = pd.Series(math.nan, index=df.index, dtype="float64")
    for column in columns:
        if column not in df.columns:
            continue
        candidate = pd.to_numeric(df[column], errors="coerce")
        values = values.where(values.notna(), candidate)
    return values


def _foreground_mask(
    df: pd.DataFrame,
    z_frb: float,
    z_eps: float,
    impact_kpc: float,
    cluster_impact_kpc: float = DEFAULT_CLUSTER_IMPACT_KPC,
    photoz_floor: float = FOREGROUND_PHOTOZ_FLOOR,
    ambiguity_kms: float = FOREGROUND_AMBIGUITY_KMS,
) -> pd.Series:
    """Per-row foreground mask via per-catalog adjudication (PI decision, not a
    blanket photo-z-error cap).

    Each row is classified spec-z vs photo-z by its ``catalog`` (spec-z catalogs
    carry no per-object photo-z; GLADE+'s generic 0.015 z-error is a floor, not a
    measurement, so it must not be run through the photo-z path). The rules:

    - Background reject (both classes): a galaxy at ``z >= z_frb`` cannot be
      foreground. For spec-z this is subsumed by the velocity-offset test below.
    - Spec-z (NED, GLADE+, SDSS, DESI DR1): foreground iff the recession-velocity
      offset dv = c*(z_frb - z)/(1 + z_frb) clears ``ambiguity_kms``. A neighbour
      inside that window sits at the host's velocity (group member / host peculiar
      velocity) and is host/local-ambiguous, not a clean intervening system.
    - Photo-z (DESI VII/292): foreground iff the *point estimate* is a credible
      foreground (``photoz_floor <= z < z_frb + z_eps``). The photo-z 1-sigma does
      NOT rescue a background point estimate (the old capped-2-sigma cut did, which
      let z > z_frb leak through); impact_kpc/r_vir corroborate marginal cases.
    """
    z = pd.to_numeric(df["z"], errors="coerce")
    impact = pd.to_numeric(df["impact_kpc"], errors="coerce")
    z_limit = z_frb + z_eps

    catalog = (
        df["catalog"].astype(str).str.lower()
        if "catalog" in df.columns
        else pd.Series("", index=df.index)
    )
    is_photoz = _catalog_matches(catalog, PHOTO_Z_CATALOG_SUBSTRINGS)
    is_specz = _catalog_matches(catalog, SPEC_Z_CATALOG_SUBSTRINGS)
    # Rows with no recognised catalog: a positive per-object z-error means photo-z,
    # otherwise treat as spec-z (preserves the no-catalog cluster/spec test rows).
    z_err = _first_available_numeric(df, PHOTO_Z_ERROR_COLUMNS)
    unlabelled = ~is_photoz & ~is_specz
    is_photoz = is_photoz | (unlabelled & z_err.notna() & (z_err > 0.0))
    is_specz = is_specz | (unlabelled & ~(z_err.notna() & (z_err > 0.0)))

    # Spec-z velocity-offset adjudication (drops background dv<0 and host-ambiguous
    # |dv|<ambiguity in one test).
    dv_kms = SPEED_OF_LIGHT_KMS * (z_frb - z) / (1.0 + z_frb)
    spec_foreground = is_specz & (dv_kms >= ambiguity_kms)
    # Photo-z: credible-floor point estimate strictly in the foreground.
    photo_foreground = is_photoz & (z >= photoz_floor) & (z < z_limit)
    foreground = spec_foreground | photo_foreground
    # Clusters get a mass-scaled r200 impact limit; galaxies keep impact_kpc.
    impact_limit = _cluster_impact_limit_kpc(df, impact_kpc, fallback_kpc=cluster_impact_kpc)
    return foreground & (impact <= impact_limit)


def _catalog_matches(catalog: pd.Series, substrings: tuple[str, ...]) -> pd.Series:
    matched = pd.Series(False, index=catalog.index)
    for sub in substrings:
        matched = matched | catalog.str.contains(sub, regex=False)
    return matched


def _cluster_impact_limit_kpc(
    df: pd.DataFrame,
    impact_kpc: float,
    r200_factor: float = CLUSTER_R200_FACTOR,
    fallback_kpc: float = DEFAULT_CLUSTER_IMPACT_KPC,
) -> pd.Series:
    """Per-row impact limit (kpc).

    Galaxies keep ``impact_kpc``. A cluster row with a catalog M500 gets
    ``r200_factor * r200`` (r200 from M200 = CLUSTER_M500_TO_M200 * M500); a cluster
    row without a catalog mass (e.g. NED-Type only) falls back to ``fallback_kpc``.
    """
    limit = pd.Series(float(impact_kpc), index=df.index, dtype="float64")
    is_cluster = _cluster_mask(df)
    m500 = (
        pd.to_numeric(df["m500_msun"], errors="coerce")
        if "m500_msun" in df.columns
        else pd.Series(math.nan, index=df.index)
    )
    z = pd.to_numeric(df["z"], errors="coerce")
    for i in df.index[is_cluster]:
        mi = m500.get(i, math.nan)
        zi = z.get(i, math.nan)
        if pd.notna(mi) and mi > 0.0 and pd.notna(zi) and zi > 0.0:
            r200 = scat.r_delta_kpc(CLUSTER_M500_TO_M200 * float(mi), float(zi), 200)
            limit.at[i] = r200_factor * r200 if math.isfinite(r200) else fallback_kpc
        else:
            limit.at[i] = fallback_kpc
    return limit


def _redshift_error(row: pd.Series) -> float:
    for column in PHOTO_Z_ERROR_COLUMNS:
        if column not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.notna(value) and value > 0.0:
            return float(value)
    return math.inf


def _catalog_rank(catalog: object) -> int:
    catalog_text = str(catalog).lower()
    if "ned" in catalog_text or "sdss" in catalog_text:
        return 0
    if "glade" in catalog_text or "vii/291" in catalog_text:
        return 1
    if "desi" in catalog_text or "vii/292" in catalog_text:
        return 2
    return 3


def _is_duplicate(row_a: pd.Series, row_b: pd.Series, coord_a: SkyCoord, coord_b: SkyCoord) -> bool:
    sep = coord_a.separation(coord_b)
    if sep >= DUPLICATE_SEPARATION:
        return False

    z_a = float(row_a["z"])
    z_b = float(row_b["z"])
    scaled_delta_z = abs(z_a - z_b) / (1.0 + min(z_a, z_b))
    return scaled_delta_z < DUPLICATE_REDSHIFT_TOLERANCE


def _best_duplicate_position(matches: pd.DataFrame, positions: list[int]) -> int:
    rows = [matches.iloc[pos] for pos in positions]
    finite_error_positions = [
        pos for pos, row in zip(positions, rows) if math.isfinite(_redshift_error(row))
    ]
    if len(finite_error_positions) >= 2:
        return min(finite_error_positions, key=lambda pos: _redshift_error(matches.iloc[pos]))
    return min(positions, key=lambda pos: _catalog_rank(matches.iloc[pos].get("catalog")))


def _deduplicate_matches(all_matches: pd.DataFrame) -> pd.DataFrame:
    if len(all_matches) <= 1:
        return all_matches

    coords = SkyCoord(ra=all_matches["ra"].values * u.deg, dec=all_matches["dec"].values * u.deg)
    kept_positions: list[int] = []

    for candidate_pos in range(len(all_matches)):
        candidate_row = all_matches.iloc[candidate_pos]
        duplicate_positions = [
            kept_pos
            for kept_pos in kept_positions
            if _is_duplicate(
                all_matches.iloc[kept_pos],
                candidate_row,
                coords[kept_pos],
                coords[candidate_pos],
            )
        ]

        if not duplicate_positions:
            kept_positions.append(candidate_pos)
            continue

        group_positions = duplicate_positions + [candidate_pos]
        best_pos = _best_duplicate_position(all_matches, group_positions)
        kept_positions = [pos for pos in kept_positions if pos not in duplicate_positions]
        kept_positions.append(best_pos)

    kept_positions.sort()
    return all_matches.iloc[kept_positions].reset_index(drop=True)


def _needs_photometry(row: pd.Series) -> bool:
    """True when a row has no usable catalog stellar mass (so PS1 should fill it)."""
    mstar = pd.to_numeric(pd.Series([row.get("M_star")]), errors="coerce").iloc[0]
    return not (pd.notna(mstar) and mstar > 0.0)


def _enrich_with_ps1_photometry(matches: pd.DataFrame) -> pd.DataFrame:
    """Attach PS1 g & i magnitudes to rows lacking a catalog stellar mass.

    Writes standardized ``gmag``/``imag`` columns (Kron-preferred, see
    ``query_ps1_gi_mags``) plus a ``ps1_sep_arcsec`` provenance column. These are
    consumed downstream by the Taylor+2011 g-i/M_i estimator in
    ``generate_galaxy_plots``. Without this, DESI VII/292 galaxies (which carry no
    photometry) would every one fall back to an assumed L* mass. Rows that
    already have a catalog mass (e.g. GLADE+) are left untouched and not queried.
    """
    if matches.empty:
        return matches

    g_out = pd.to_numeric(
        matches.get("gmag", pd.Series(index=matches.index, dtype="float64")), errors="coerce"
    ).to_numpy(dtype="float64", copy=True)
    i_out = pd.to_numeric(
        matches.get("imag", pd.Series(index=matches.index, dtype="float64")), errors="coerce"
    ).to_numpy(dtype="float64", copy=True)
    sep_out = pd.Series(math.nan, index=matches.index, dtype="float64").to_numpy(copy=True)

    n_filled = 0
    for pos in range(len(matches)):
        row = matches.iloc[pos]
        if not _needs_photometry(row):
            continue
        coord = SkyCoord(ra=float(row["ra"]) * u.deg, dec=float(row["dec"]) * u.deg)
        g_mag, i_mag, sep = query_ps1_gi_mags(coord)
        if g_mag is not None:
            g_out[pos] = g_mag
        if i_mag is not None:
            i_out[pos] = i_mag
        if sep is not None:
            sep_out[pos] = sep
        if g_mag is not None and i_mag is not None:
            n_filled += 1

    matches = matches.copy()
    matches["gmag"] = g_out
    matches["imag"] = i_out
    matches["ps1_sep_arcsec"] = sep_out
    if n_filled:
        print(f"    PS1 cross-match: filled g+i photometry for {n_filled} galaxy(ies).")
    return matches


def run_search(
    impact_kpc: float = DEFAULT_IMPACT_KPC,
    output_dir: str = "results",
    z_eps: float = DEFAULT_Z_EPS,
    build_unified: bool = False,
    cluster_impact_kpc: float = DEFAULT_CLUSTER_IMPACT_KPC,
):
    """Run the galaxy search for all targets."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    engines = [NedTapEngine()]
    for cat_name, cat_id in VIZIER_CATALOGS.items():
        engines.append(VizierEngine(cat_id))
    if ENABLE_EXTRA_ENGINES:
        # Opt-in DESI DR1 zpix spec-z engine. Covers only 3/12 targets
        # (Whitney/Phineas/Casey); returns empty elsewhere and degrades gracefully.
        engines.append(DesiDr1Engine())
    if ENABLE_CLUSTER_ENGINE:
        # All-sky cluster catalogs (PSZ2 + MCXC/MCXC-II) supply M500/R500 so the
        # r200-relative impact cut and beta-model ICM DM have catalog masses.
        engines.append(ClusterEngine())

    summary_data = []

    for i, (name, ra_str, dec_str, z_frb) in enumerate(TARGETS):
        print(f"Processing {name} (Target {i + 1}): {ra_str}, {dec_str} (z={z_frb})")
        coord = parse_coord(ra_str, dec_str)
        # Query the larger of the two impact limits (clusters reach out to ~5 Mpc).
        # Use MIN_Z_SEARCH to capture low-z foreground at larger angular separation, and
        # cap the cone at MAX_SEARCH_RADIUS_DEG so low-z clusters don't time out Vizier.
        radius = min(
            get_angular_radius(min(z_frb, MIN_Z_SEARCH), max(impact_kpc, cluster_impact_kpc)),
            MAX_SEARCH_RADIUS_DEG * u.deg,
        )

        target_matches = []
        for engine in engines:
            df = engine.query(coord, radius)
            engine_name = engine.__class__.__name__
            if isinstance(engine, VizierEngine):
                # Find the catalog name from VIZIER_CATALOGS
                cat_label = engine.catalog_id
                for k, v in VIZIER_CATALOGS.items():
                    if v == engine.catalog_id:
                        cat_label = k
                        break
                engine_name = f"VizierEngine({cat_label})"

            if not df.empty:
                print(f"    {engine_name} returned {len(df)} raw results.")
                # Ensure we have ra, dec, z
                if "ra" not in df.columns or "dec" not in df.columns or "z" not in df.columns:
                    # Try to find them case-insensitively
                    col_map = {c.lower(): c for c in df.columns}
                    if "ra" in col_map:
                        df["ra"] = df[col_map["ra"]]
                    if "dec" in col_map:
                        df["dec"] = df[col_map["dec"]]
                    if "z" in col_map:
                        df["z"] = df[col_map["z"]]

                if "ra" in df.columns and "dec" in df.columns and "z" in df.columns:
                    # Drop rows with NaN in critical columns
                    raw_count = len(df)
                    df = df.dropna(subset=["ra", "dec", "z"])
                    with_z_count = len(df)
                    if df.empty:
                        print(f"      {engine_name}: 0/{raw_count} results have redshifts.")
                        continue

                    df["impact_kpc"] = df.apply(
                        lambda row: calculate_impact_parameter(
                            row["ra"], row["dec"], row["z"], coord.ra.deg, coord.dec.deg
                        ),
                        axis=1,
                    )
                    # Filter for foreground (with buffer) and impact parameter.
                    df_filtered = df[
                        _foreground_mask(df, z_frb, z_eps, impact_kpc, cluster_impact_kpc)
                    ]
                    if not df_filtered.empty:
                        target_matches.append(df_filtered)
                        print(
                            f"      {engine_name}: Found {len(df_filtered)} matches (from {with_z_count} with z)."
                        )
                    else:
                        print(f"      {engine_name}: 0 matches (from {with_z_count} with z).")
            else:
                print(f"    {engine_name} returned 0 results.")

        if target_matches:
            all_matches = pd.concat(target_matches, ignore_index=True)

            all_matches = _deduplicate_matches(all_matches)

            # Backfill stellar-mass photometry from PS1 for catalogs that lack it
            # (notably DESI VII/292, which provides only photo-z).
            all_matches = _enrich_with_ps1_photometry(all_matches)

            out_path = os.path.abspath(os.path.join(output_dir, f"{name.lower()}_galaxies.csv"))
            all_matches.to_csv(out_path, index=False)
            print(f"  Found {len(all_matches)} unique foreground galaxies. Saved to {out_path}")
            if build_unified:
                # Opt-in: derive the {name.lower()}_unified.csv alongside the galaxies CSV.
                build_for_target(name, ra_str, dec_str, z_frb, results_dir=output_dir)
            summary_data.append(
                {
                    "name": name,
                    "target_id": i + 1,
                    "ra": ra_str,
                    "dec": dec_str,
                    "z_frb": z_frb,
                    "num_galaxies": len(all_matches),
                }
            )
        else:
            print("  No foreground galaxies found.")
            summary_data.append(
                {
                    "name": name,
                    "target_id": i + 1,
                    "ra": ra_str,
                    "dec": dec_str,
                    "z_frb": z_frb,
                    "num_galaxies": 0,
                }
            )

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.abspath(os.path.join(output_dir, "search_summary.csv"))
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSearch complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Search for foreground galaxies around FRB targets."
    )
    parser.add_argument(
        "--impact_kpc", type=float, default=100.0, help="Maximum impact parameter in kpc."
    )
    args = parser.parse_args()

    run_search(impact_kpc=args.impact_kpc)
