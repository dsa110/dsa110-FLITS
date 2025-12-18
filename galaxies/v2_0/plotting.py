import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import List, Optional
import os

from flits.plotting import use_flits_style
from .utils import parse_coord

def plot_impact_vs_redshift(summary_df: pd.DataFrame, all_galaxies_df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Plot impact parameter vs redshift for all identified foreground galaxies.
    """
    use_flits_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot galaxies
    # Use target_name for coloring if available, otherwise name (which might be the same) or target_id
    if 'target_name' in all_galaxies_df.columns:
        color_col = 'target_name'
    elif 'name' in all_galaxies_df.columns:
        color_col = 'name'
    else:
        color_col = 'target_id'
    
    # Create a mapping for categorical colors if using names
    if color_col in ['target_name', 'name']:
        names = all_galaxies_df[color_col].unique()
        name_to_id = {name: i for i, name in enumerate(names)}
        colors = all_galaxies_df[color_col].map(name_to_id)
    else:
        colors = all_galaxies_df['target_id']

    scatter = ax.scatter(
        all_galaxies_df['z'], 
        all_galaxies_df['impact_kpc'], 
        c=colors, 
        cmap='tab20', 
        s=100, 
        edgecolor='k', 
        alpha=0.8,
        label='Foreground Galaxies'
    )
    
    # Plot FRBs as vertical lines or markers at their redshifts
    for _, row in summary_df.iterrows():
        ax.axvline(row['z_frb'], color='gray', linestyle='--', alpha=0.3)
        # Label FRB at the top
        name = row.get('name', f"Target {row['target_id']}")
        ax.text(row['z_frb'], ax.get_ylim()[1], name, rotation=90, verticalalignment='bottom', fontsize=8)
        
    ax.set_xlabel('Redshift ($z$)')
    ax.set_ylabel('Impact Parameter ($b$ [kpc])')
    ax.set_title('Foreground Galaxy Environment')
    
    # Add colorbar or legend
    if color_col == 'name':
        # Legend is better for names
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=name,
                          markerfacecolor=plt.cm.tab20(name_to_id[name]/20), markersize=10)
                          for name in names]
        ax.legend(handles=legend_elements, title="FRB Field", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        cbar = plt.colorbar(scatter)
        cbar.set_label('Target ID')
    
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    return fig, ax

def plot_sightline(target_info: dict, galaxies_df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Plot a sophisticated sightline view for a single FRB.
    """
    use_flits_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    target_coord = parse_coord(target_info['ra'], target_info['dec'])
    ra0, dec0 = target_coord.ra.deg, target_coord.dec.deg
    target_name = target_info.get('name', f"Target {target_info.get('target_id', 'Unknown')}")
    
    # Plot FRB at center
    ax.scatter(0, 0, marker='*', s=400, color='red', edgecolor='k', label=f'FRB {target_name}', zorder=10)
    
    if not galaxies_df.empty:
        # Calculate relative offsets in arcmin
        cos_dec = np.cos(np.radians(dec0))
        galaxies_df['dra'] = (galaxies_df['ra'] - ra0) * 60.0 * cos_dec
        galaxies_df['ddec'] = (galaxies_df['dec'] - dec0) * 60.0
        
        scatter = ax.scatter(
            galaxies_df['dra'], 
            galaxies_df['ddec'], 
            c=galaxies_df['z'], 
            s=150, 
            cmap='viridis', 
            edgecolor='k', 
            alpha=0.8,
            label='Foreground Galaxies'
        )
        
        # Add labels for galaxies
        for _, row in galaxies_df.iterrows():
            label = row['name'] if pd.notna(row['name']) and row['name'] != "" else f"z={row['z']:.3f}"
            ax.annotate(label, (row['dra'], row['ddec']), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
            
        cbar = plt.colorbar(scatter)
        cbar.set_label('Redshift ($z$)')

    # Add concentric circles for physical impact parameters
    if not galaxies_df.empty:
        avg_z = galaxies_df['z'].mean()
    else:
        avg_z = target_info['z_frb'] / 2.0
        
    from .utils import get_angular_radius
    for b_kpc in [100, 250, 500]:
        theta = get_angular_radius(avg_z, b_kpc).to(u.arcmin).value
        circle = plt.Circle((0, 0), theta, color='gray', fill=False, linestyle=':', alpha=0.5)
        ax.add_artist(circle)
        ax.text(0, theta, f"{b_kpc} kpc", color='gray', fontsize=8, ha='center', va='bottom')

    ax.set_xlabel(r'$\Delta$ RA [arcmin]')
    ax.set_ylabel(r'$\Delta$ Dec [arcmin]')
    ax.set_title(f'Sightline Environment: {target_name} ($z={target_info["z_frb"]}$)')
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    # Set limits to show at least the 500 kpc circle
    limit = get_angular_radius(avg_z, 550).to(u.arcmin).value
    ax.set_xlim(limit, -limit) # RA increases to the left
    ax.set_ylim(-limit, limit)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    return fig, ax
