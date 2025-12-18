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
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot galaxies
    scatter = ax.scatter(
        all_galaxies_df['z'], 
        all_galaxies_df['impact_kpc'], 
        c=all_galaxies_df['target_id'], 
        cmap='tab20', 
        s=100, 
        edgecolor='k', 
        alpha=0.8,
        label='Foreground Galaxies'
    )
    
    # Plot FRBs as vertical lines or markers at their redshifts
    for _, row in summary_df.iterrows():
        ax.axvline(row['z_frb'], color='gray', linestyle='--', alpha=0.3)
        
    ax.set_xlabel('Redshift ($z$)')
    ax.set_ylabel('Impact Parameter ($b$ [kpc])')
    ax.set_title('Foreground Galaxy Environment')
    
    # Add colorbar for target IDs
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
    
    # Plot FRB at center
    ax.scatter(0, 0, marker='*', s=400, color='red', edgecolor='k', label='FRB Sightline', zorder=10)
    
    if not galaxies_df.empty:
        # Calculate relative offsets in arcmin
        # Note: This is a simple projection, fine for small fields
        cos_dec = np.cos(np.radians(dec0))
        galaxies_df['dra'] = (galaxies_df['ra'] - ra0) * 60.0 * cos_dec
        galaxies_df['ddec'] = (galaxies_df['dec'] - dec0) * 60.0
        
        scatter = ax.scatter(
            galaxies_df['dra'], 
            galaxies_df['ddec'], 
            c=galaxies_df['z'], 
            s=galaxies_df['impact_kpc'] * 0.5 + 50, # Size scaled by impact parameter (inverted or just scaled)
            cmap='viridis', 
            edgecolor='k', 
            alpha=0.8,
            label='Foreground Galaxies'
        )
        
        # Add labels for galaxies
        for _, row in galaxies_df.iterrows():
            name = row['name'] if pd.notna(row['name']) else f"z={row['z']:.3f}"
            ax.annotate(name, (row['dra'], row['ddec']), xytext=(5, 5), textcoords='offset points', fontsize=8)
            
        cbar = plt.colorbar(scatter)
        cbar.set_label('Redshift ($z$)')

    # Add concentric circles for physical impact parameters
    # We need to convert kpc to arcmin at the galaxy redshifts
    # For simplicity, we'll use the average redshift or a representative one
    if not galaxies_df.empty:
        avg_z = galaxies_df['z'].mean()
    else:
        avg_z = target_info['z_frb'] / 2.0
        
    from .utils import get_angular_radius
    for b_kpc in [100, 250, 500]:
        theta = get_angular_radius(avg_z, b_kpc).to(u.arcmin).value
        circle = plt.Circle((0, 0), theta, color='gray', fill=False, linestyle=':', alpha=0.5)
        ax.add_artist(circle)
        ax.text(0, theta + 0.2, f"{b_kpc} kpc", color='gray', fontsize=8, ha='center')

    ax.set_xlabel(r'$\Delta$ RA [arcmin]')
    ax.set_ylabel(r'$\Delta$ Dec [arcmin]')
    ax.set_title(f"Target {target_info['target_id']} Environment (z={target_info['z_frb']})")
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Set limits to show the largest circle
    max_theta = get_angular_radius(avg_z, 500).to(u.arcmin).value * 1.2
    ax.set_xlim(max_theta, -max_theta) # RA increases to the left
    ax.set_ylim(-max_theta, max_theta)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    return fig, ax
