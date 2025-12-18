#!/usr/bin/env python3
"""
Publication-quality visualizations for FRB galaxy search results.

Creates:
1. Sky map of all FRB positions with search apertures
2. Individual field plots showing galaxy candidates
3. Summary statistics figure
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.patches import Circle
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u

# Import shared config
from config import TARGETS, theta_for_impact, R_PHYS_KPC

# Use scienceplots style
plt.style.use(['science', 'notebook'])

# Additional settings
plt.rcParams.update({
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Output directory
RESULTS_DIR = Path("galaxies_100kpc_proper")
OUTPUT_DIR = Path("results/galaxies/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results():
    """Load all CSV results into a dictionary."""
    results = {}
    for csv_file in RESULTS_DIR.glob("*.csv"):
        name = csv_file.stem
        try:
            df = pd.read_csv(csv_file)
            results[name] = df
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")
    return results


def create_sky_map(results):
    """Create all-sky map showing FRB positions and search apertures."""
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'mollweide'})
    
    # Plot each target
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(TARGETS)))
    
    for i, target in enumerate(TARGETS):
        coord = target.coord
        ra_rad = coord.ra.wrap_at(180*u.deg).rad
        dec_rad = coord.dec.rad
        
        # Size based on search radius
        theta = theta_for_impact(target.z_max, R_PHYS_KPC)
        size = max(50, theta * 20)  # Scale for visibility
        
        # Count galaxies found
        n_gal = 0
        for key, df in results.items():
            if key.startswith(f"T{i+1:02d}_"):
                n_gal += len(df)
        
        # Plot marker
        marker = 'o' if n_gal > 0 else 'x'
        ax.scatter(ra_rad, dec_rad, s=size, c=[colors[i]], 
                   marker=marker, edgecolors='black', linewidths=0.5,
                   label=f"{target.name.split()[0]} (z={target.z_max:.2f}, n={n_gal})")
    
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.set_title(f'FRB Galaxy Search: 12 Targets within {R_PHYS_KPC} kpc', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Legend outside
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, 
              title='Target (z, N_gal)')
    
    plt.tight_layout()
    outpath = OUTPUT_DIR / "sky_map_all_targets.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix('.png'))
    print(f"Saved: {outpath}")
    plt.close(fig)
    return outpath


def create_field_plot(target_id, target, results):
    """Create individual field plot showing galaxy candidates around FRB."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    coord = target.coord
    theta = theta_for_impact(target.z_max, R_PHYS_KPC)  # arcmin
    
    # Collect all galaxies for this target
    all_gals = []
    catalogs = []
    for key, df in results.items():
        if key.startswith(f"T{target_id:02d}_"):
            cat_name = key.split("_")[1]
            df = df.copy()
            df['catalog'] = cat_name
            all_gals.append(df)
            catalogs.append(cat_name)
    
    # Plot search aperture
    circle = Circle((0, 0), theta, fill=False, color='red', 
                    linestyle='--', linewidth=2, label=f'{R_PHYS_KPC} kpc aperture')
    ax.add_patch(circle)
    
    # Plot FRB position
    ax.scatter(0, 0, marker='*', s=200, c='red', edgecolors='black', 
               linewidths=1, zorder=10, label='FRB position')
    
    # Plot galaxies
    if all_gals:
        gal_df = pd.concat(all_gals, ignore_index=True)
        
        # Convert to offsets from FRB (arcmin)
        gal_coords = SkyCoord(gal_df['ra'], gal_df['dec'], unit='deg')
        dra = (gal_coords.ra - coord.ra).to(u.arcmin).value * np.cos(coord.dec.rad)
        ddec = (gal_coords.dec - coord.dec).to(u.arcmin).value
        
        # Color by catalog
        catalog_colors = {'NED': 'blue', 'PS1': 'green', 'GLADE+': 'orange', 
                         'SDSS': 'purple', 'GAMA': 'cyan', '2MRS': 'brown'}
        
        for cat in gal_df['catalog'].unique():
            mask = gal_df['catalog'] == cat
            color = catalog_colors.get(cat, 'gray')
            ax.scatter(dra[mask], ddec[mask], s=50, c=color, alpha=0.7,
                      edgecolors='black', linewidths=0.3, label=f'{cat} ({mask.sum()})')
        
        # Annotate galaxies with redshift if available
        for idx in range(len(gal_df)):
            z_val = gal_df.iloc[idx].get('z', np.nan)
            if pd.notna(z_val) and z_val > 0:
                ax.annotate(f'z={z_val:.3f}', (dra[idx], ddec[idx]), 
                           fontsize=7, alpha=0.7, xytext=(3, 3), 
                           textcoords='offset points')
    
    # Formatting
    ax.set_xlim(-theta*1.5, theta*1.5)
    ax.set_ylim(-theta*1.5, theta*1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('ΔRA (arcmin)')
    ax.set_ylabel('ΔDec (arcmin)')
    ax.set_title(f'{target.name}\n'
                f'z = {target.z_max:.3f}, θ = {theta:.2f}\'', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Add scale bar
    scale_kpc = 50  # kpc
    scale_arcmin = theta * (scale_kpc / R_PHYS_KPC)
    ax.plot([-theta*1.2, -theta*1.2 + scale_arcmin], [-theta*1.3, -theta*1.3], 
            'k-', linewidth=2)
    ax.text(-theta*1.2 + scale_arcmin/2, -theta*1.4, f'{scale_kpc} kpc', 
            ha='center', fontsize=8)
    
    plt.tight_layout()
    # Use short name for filename
    short_name = target.name.split()[0].lower()
    outpath = OUTPUT_DIR / f"field_{short_name}.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix('.png'))
    print(f"Saved: {outpath}")
    plt.close(fig)
    return outpath


def create_summary_figure(results):
    """Create summary statistics figure."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Collect statistics
    stats = []
    for i, target in enumerate(TARGETS):
        n_gal = 0
        catalogs_found = []
        for key, df in results.items():
            if key.startswith(f"T{i+1:02d}_"):
                n_gal += len(df)
                catalogs_found.append(key.split("_")[1])
        
        theta = theta_for_impact(target.z_max, R_PHYS_KPC)
        stats.append({
            'target': target.name.split()[0],
            'z': target.z_max,
            'theta_arcmin': theta,
            'n_galaxies': n_gal,
            'catalogs': ', '.join(catalogs_found) if catalogs_found else 'None'
        })
    
    stats_df = pd.DataFrame(stats)
    
    # Panel 1: Number of galaxies vs redshift
    ax1 = axes[0]
    colors = ['green' if n > 0 else 'red' for n in stats_df['n_galaxies']]
    ax1.scatter(stats_df['z'], stats_df['n_galaxies'], c=colors, s=100, 
                edgecolors='black', linewidths=0.5)
    for idx, row in stats_df.iterrows():
        ax1.annotate(row['target'], (row['z'], row['n_galaxies']), 
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax1.set_xlabel('Redshift')
    ax1.set_ylabel('Number of Galaxy Candidates')
    ax1.set_title('Galaxy Candidates vs Redshift')
    ax1.set_yscale('symlog', linthresh=1)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Search radius vs redshift
    ax2 = axes[1]
    ax2.scatter(stats_df['z'], stats_df['theta_arcmin'], c='steelblue', s=100,
                edgecolors='black', linewidths=0.5)
    ax2.set_xlabel('Redshift')
    ax2.set_ylabel('Search Radius (arcmin)')
    ax2.set_title(f'Angular Size of {R_PHYS_KPC} kpc vs Redshift')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Bar chart of galaxies per target
    ax3 = axes[2]
    colors = ['green' if n > 0 else 'red' for n in stats_df['n_galaxies']]
    bars = ax3.bar(stats_df['target'], stats_df['n_galaxies'], color=colors,
                   edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Target')
    ax3.set_ylabel('Number of Galaxy Candidates')
    ax3.set_title('Galaxy Candidates per Target')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_yscale('symlog', linthresh=1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    outpath = OUTPUT_DIR / "summary_statistics.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix('.png'))
    print(f"Saved: {outpath}")
    plt.close(fig)
    return outpath


def create_mosaic_figure(results):
    """Create mosaic of all field plots."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (ax, target) in enumerate(zip(axes, TARGETS)):
        target_id = i + 1
        coord = target.coord
        theta = theta_for_impact(target.z_max, R_PHYS_KPC)
        
        # Collect galaxies
        all_gals = []
        for key, df in results.items():
            if key.startswith(f"T{target_id:02d}_"):
                all_gals.append(df)
        
        # Plot aperture
        circle = Circle((0, 0), theta, fill=False, color='red', 
                        linestyle='--', linewidth=1.5)
        ax.add_patch(circle)
        
        # FRB position
        ax.scatter(0, 0, marker='*', s=100, c='red', edgecolors='black', 
                   linewidths=0.5, zorder=10)
        
        # Galaxies
        n_gal = 0
        if all_gals:
            gal_df = pd.concat(all_gals, ignore_index=True)
            n_gal = len(gal_df)
            
            gal_coords = SkyCoord(gal_df['ra'], gal_df['dec'], unit='deg')
            dra = (gal_coords.ra - coord.ra).to(u.arcmin).value * np.cos(coord.dec.rad)
            ddec = (gal_coords.dec - coord.dec).to(u.arcmin).value
            
            ax.scatter(dra, ddec, s=20, c='blue', alpha=0.6,
                      edgecolors='black', linewidths=0.2)
        
        # Formatting
        ax.set_xlim(-theta*1.5, theta*1.5)
        ax.set_ylim(-theta*1.5, theta*1.5)
        ax.set_aspect('equal')
        short_name = target.name.split()[0]
        ax.set_title(f'{short_name} (z={target.z_max:.2f}, n={n_gal})', fontsize=9)
        
        if i >= 8:  # Bottom row
            ax.set_xlabel("ΔRA (')")
        if i % 4 == 0:  # Left column
            ax.set_ylabel("ΔDec (')")
        
        ax.grid(True, alpha=0.2)
    
    plt.suptitle(f'Galaxy Candidates within {R_PHYS_KPC} kpc of 12 FRB Sight-lines', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    outpath = OUTPUT_DIR / "field_mosaic.pdf"
    fig.savefig(outpath)
    fig.savefig(outpath.with_suffix('.png'))
    print(f"Saved: {outpath}")
    plt.close(fig)
    return outpath


def main():
    """Generate all visualizations."""
    print("Loading results...")
    results = load_all_results()
    print(f"Loaded {len(results)} result files")
    
    print("\nCreating visualizations...")
    
    # Sky map
    create_sky_map(results)
    
    # Individual field plots
    for i, target in enumerate(TARGETS):
        create_field_plot(i + 1, target, results)
    
    # Summary figure
    create_summary_figure(results)
    
    # Mosaic
    create_mosaic_figure(results)
    
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
