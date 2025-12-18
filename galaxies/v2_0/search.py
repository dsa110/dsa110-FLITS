"""Main pipeline for finding foreground galaxies."""

import os
import pandas as pd
from .config import TARGETS, DEFAULT_IMPACT_KPC, VIZIER_CATALOGS
from .utils import parse_coord, get_angular_radius, calculate_impact_parameter
from .engines import NedEngine, VizierEngine

def run_search(impact_kpc: float = DEFAULT_IMPACT_KPC, output_dir: str = "results"):
    """Run the galaxy search for all targets."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    engines = [
        NedEngine(),
        VizierEngine(VIZIER_CATALOGS["GLADE+"]),
        # Add more engines as needed
    ]
    
    summary_data = []
    
    for i, (ra_str, dec_str, z_frb) in enumerate(TARGETS):
        print(f"Processing Target {i+1}: {ra_str}, {dec_str} (z={z_frb})")
        coord = parse_coord(ra_str, dec_str)
        radius = get_angular_radius(z_frb, impact_kpc)
        
        target_matches = []
        for engine in engines:
            df = engine.query(coord, radius)
            if not df.empty:
                # Ensure we have ra, dec, z
                if 'ra' not in df.columns or 'dec' not in df.columns or 'z' not in df.columns:
                    # Try to find them case-insensitively
                    col_map = {c.lower(): c for c in df.columns}
                    if 'ra' in col_map: df['ra'] = df[col_map['ra']]
                    if 'dec' in col_map: df['dec'] = df[col_map['dec']]
                    if 'z' in col_map: df['z'] = df[col_map['z']]

                if 'ra' in df.columns and 'dec' in df.columns and 'z' in df.columns:
                    # Drop rows with NaN in critical columns
                    df = df.dropna(subset=['ra', 'dec', 'z'])
                    if df.empty: continue

                    df['impact_kpc'] = df.apply(
                        lambda row: calculate_impact_parameter(
                            row['ra'], row['dec'], row['z'], coord.ra.deg, coord.dec.deg
                        ), axis=1
                    )
                    # Filter for foreground and impact parameter
                    df = df[(df['z'] < z_frb) & (df['impact_kpc'] <= impact_kpc)]
                    if not df.empty:
                        target_matches.append(df)
        
        if target_matches:
            all_matches = pd.concat(target_matches, ignore_index=True)
            out_path = os.path.join(output_dir, f"target_{i+1}_galaxies.csv")
            all_matches.to_csv(out_path, index=False)
            print(f"  Found {len(all_matches)} foreground galaxies.")
            summary_data.append({
                'target_id': i+1,
                'ra': ra_str,
                'dec': dec_str,
                'z_frb': z_frb,
                'num_galaxies': len(all_matches)
            })
        else:
            print("  No foreground galaxies found.")
            summary_data.append({
                'target_id': i+1,
                'ra': ra_str,
                'dec': dec_str,
                'z_frb': z_frb,
                'num_galaxies': 0
            })
            
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "search_summary.csv"), index=False)
    print("\nSearch complete. Summary saved to results/search_summary.csv")

if __name__ == "__main__":
    run_search()
