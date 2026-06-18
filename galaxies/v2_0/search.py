"""Main pipeline for finding foreground galaxies."""

import os
import math
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from .config import TARGETS, DEFAULT_IMPACT_KPC, VIZIER_CATALOGS, DEFAULT_Z_EPS, MIN_Z_SEARCH
from .utils import parse_coord, get_angular_radius, calculate_impact_parameter
from .engines import NedEngine, VizierEngine, query_ps1_gi_mags

PHOTO_Z_ERROR_COLUMNS = ("z_phot_err", "e_zphot", "z_best_err")
DUPLICATE_SEPARATION = 10.0 * u.arcsec
DUPLICATE_REDSHIFT_TOLERANCE = 0.02


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
    n_sigma: float = 2.0,
) -> pd.Series:
    z = pd.to_numeric(df["z"], errors="coerce")
    impact = pd.to_numeric(df["impact_kpc"], errors="coerce")
    z_err = _first_available_numeric(df, PHOTO_Z_ERROR_COLUMNS)
    has_z_err = z_err.notna() & (z_err > 0.0)
    z_limit = z_frb + z_eps

    foreground = (z < z_limit) | (has_z_err & ((z - n_sigma * z_err) < z_limit))
    return foreground & (impact <= impact_kpc)


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

    g_out = pd.to_numeric(matches.get("gmag", pd.Series(index=matches.index, dtype="float64")),
                          errors="coerce").to_numpy(dtype="float64", copy=True)
    i_out = pd.to_numeric(matches.get("imag", pd.Series(index=matches.index, dtype="float64")),
                          errors="coerce").to_numpy(dtype="float64", copy=True)
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


def run_search(impact_kpc: float = DEFAULT_IMPACT_KPC, output_dir: str = "results", z_eps: float = DEFAULT_Z_EPS):
    """Run the galaxy search for all targets."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    engines = [NedEngine()]
    for cat_name, cat_id in VIZIER_CATALOGS.items():
        engines.append(VizierEngine(cat_id))
    
    summary_data = []
    
    for i, (name, ra_str, dec_str, z_frb) in enumerate(TARGETS):
        print(f"Processing {name} (Target {i+1}): {ra_str}, {dec_str} (z={z_frb})")
        coord = parse_coord(ra_str, dec_str)
        # Use MIN_Z_SEARCH to capture low-z foreground galaxies with larger angular separation
        radius = get_angular_radius(min(z_frb, MIN_Z_SEARCH), impact_kpc)
        
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
                if 'ra' not in df.columns or 'dec' not in df.columns or 'z' not in df.columns:
                    # Try to find them case-insensitively
                    col_map = {c.lower(): c for c in df.columns}
                    if 'ra' in col_map: df['ra'] = df[col_map['ra']]
                    if 'dec' in col_map: df['dec'] = df[col_map['dec']]
                    if 'z' in col_map: df['z'] = df[col_map['z']]

                if 'ra' in df.columns and 'dec' in df.columns and 'z' in df.columns:
                    # Drop rows with NaN in critical columns
                    raw_count = len(df)
                    df = df.dropna(subset=['ra', 'dec', 'z'])
                    with_z_count = len(df)
                    if df.empty: 
                        print(f"      {engine_name}: 0/{raw_count} results have redshifts.")
                        continue

                    df['impact_kpc'] = df.apply(
                        lambda row: calculate_impact_parameter(
                            row['ra'], row['dec'], row['z'], coord.ra.deg, coord.dec.deg
                        ), axis=1
                    )
                    # Filter for foreground (with buffer) and impact parameter.
                    df_filtered = df[_foreground_mask(df, z_frb, z_eps, impact_kpc)]
                    if not df_filtered.empty:
                        target_matches.append(df_filtered)
                        print(f"      {engine_name}: Found {len(df_filtered)} matches (from {with_z_count} with z).")
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
            summary_data.append({
                'name': name,
                'target_id': i+1,
                'ra': ra_str,
                'dec': dec_str,
                'z_frb': z_frb,
                'num_galaxies': len(all_matches)
            })
        else:
            print("  No foreground galaxies found.")
            summary_data.append({
                'name': name,
                'target_id': i+1,
                'ra': ra_str,
                'dec': dec_str,
                'z_frb': z_frb,
                'num_galaxies': 0
            })
            
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.abspath(os.path.join(output_dir, "search_summary.csv"))
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSearch complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Search for foreground galaxies around FRB targets.")
    parser.add_argument("--impact_kpc", type=float, default=100.0, help="Maximum impact parameter in kpc.")
    args = parser.parse_args()
    
    run_search(impact_kpc=args.impact_kpc)
