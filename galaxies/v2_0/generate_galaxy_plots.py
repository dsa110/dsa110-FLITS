#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy import constants as const

# Standard colors for premium look
DARK_BLUE = '#1B365D'
LIGHT_BLUE = '#4A90E2'
ACCENT_ORANGE = '#F5A623'
ACCENT_RED = '#D0021B'
TEXT_DARK = '#333333'
GRID_COLOR = '#E5E5E5'

def estimate_halo_mass(log_mstar):
    """Estimate halo mass from stellar mass using a direct SMHR fit."""
    y = log_mstar
    # Moster et al. 2013 / Behroozi et al. 2019 direct approximation
    log_mh = 12.0 + 0.45 * (y - 10.3) + 0.85 * np.sinh(0.9 * (y - 10.3))
    return 10**log_mh

def get_rvir_and_rs(m_halo_msun, z):
    """Calculate virial radius R_200 and scale radius r_s in kpc."""
    M_h = m_halo_msun * u.Msun
    H_z = cosmo.H(z)
    
    # R_200 definition
    # R_200 = (G * M_h / (100 * H_z^2))^(1/3)
    G = const.G
    r_vir = ((G * M_h / (100 * H_z**2))**(1/3)).to(u.kpc).value
    
    # Concentration c (Dutton & Maccio 2014)
    log_c = 0.905 - 0.101 * np.log10(m_halo_msun / 1e12)
    c = 10**log_c
    
    r_s = r_vir / c
    return r_vir, r_s, c

def nfw_enclosed_mass(r, m_halo, r_vir, r_s, c):
    """Calculate enclosed mass at radius r (kpc) using NFW profile."""
    x = r / r_s
    f_x = np.log(1 + x) - x / (1 + x)
    f_c = np.log(1 + c) - c / (1 + c)
    return m_halo * f_x / f_c

def main():
    # Relative path resolution for repository portability
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "results")
    output_dir = results_dir
    
    # Load targets to map name to z_frb
    targets = [
        ("Zach", 0.0430),
        ("Whitney", 0.4790),
        ("Oran", 0.3005),
        ("Isha", 0.2505),
        ("Wilhelm", 0.5100),
        ("Phineas", 0.2710),
        ("Freya", 1.0000),
        ("Hamilton", 0.3024),
        ("Mahi", 1.0000),
        ("Chromatica", 0.0740),
        ("Casey", 0.2870),
        ("Johndoeii", 1.0000)
    ]
    
    galaxy_data = []
    
    for name, z_frb in targets:
        csv_path = os.path.join(results_dir, f"{name.lower()}_galaxies.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
            
        for _, row in df.iterrows():
            z_gal = row['z']
            impact = row['impact_kpc']
            
            # Estimate stellar mass
            if 'M_star' in row and not np.isnan(row['M_star']) and row['M_star'] > 0:
                log_mstar = row['M_star']
            else:
                log_mstar = 9.5  # Typical L* galaxy stellar mass at z ~ 0.5
                
            m_halo = estimate_halo_mass(log_mstar)
            r_vir, r_s, c = get_rvir_and_rs(m_halo, z_gal)
            
            # Comoving distance along sightline
            d_com = cosmo.comoving_distance(z_gal).to(u.Mpc).value
            
            galaxy_data.append({
                'target': name,
                'z_frb': z_frb,
                'z_gal': z_gal,
                'd_com': d_com,
                'impact': impact,
                'log_mstar': log_mstar,
                'm_halo': m_halo,
                'r_vir': r_vir,
                'r_s': r_s,
                'c': c
            })
            
    if not galaxy_data:
        print("No galaxy matches found to plot.")
        return
        
    df_plot = pd.DataFrame(galaxy_data)
    print(f"Loaded {len(df_plot)} galaxies for visualization.")
    
    # ------------------ Plot 1: Sightline Intersections ------------------
    plt.figure(figsize=(10, 6), dpi=150)
    plt.axhline(0, color=DARK_BLUE, linestyle='-', linewidth=2, label='FRB Sightline ($b=0$)')
    
    # We plot the comoving distance on X, and impact parameter on Y
    for _, row in df_plot.iterrows():
        # Draw the virial sphere as a shaded vertical bar at the galaxy comoving distance
        plt.fill_between([row['d_com']-5, row['d_com']+5], 
                         [row['impact'] - row['r_vir'], row['impact'] - row['r_vir']],
                         [row['impact'] + row['r_vir'], row['impact'] + row['r_vir']],
                         color=LIGHT_BLUE, alpha=0.15)
        
        # Plot the galaxy center
        plt.scatter(row['d_com'], row['impact'], color=DARK_BLUE, edgecolors='white', s=80, zorder=5)
        
        # Draw the impact parameter range
        plt.plot([row['d_com'], row['d_com']], [0, row['impact']], color=ACCENT_ORANGE, linestyle='--', linewidth=1.5)
        
        # Label the galaxy
        label = f"{row['target']}-G (z={row['z_gal']:.3f})"
        plt.text(row['d_com'] + 8, row['impact'] + 2, label, fontsize=8, color=TEXT_DARK, fontweight='bold')
        
    plt.xlabel('Comoving Distance along Sightline (Mpc)', fontsize=12, fontweight='bold', color=TEXT_DARK)
    plt.ylabel('Physical Separation / Impact Parameter $b$ (kpc)', fontsize=12, fontweight='bold', color=TEXT_DARK)
    plt.title('Intervening Galaxies & Virial Radii ($R_{200}$) along FRB Sightlines', fontsize=14, fontweight='bold', color=DARK_BLUE, pad=15)
    
    plt.grid(True, linestyle=':', color=GRID_COLOR, alpha=0.7)
    plt.xlim(0, cosmo.comoving_distance(1.0).value)
    plt.ylim(-50, 250)
    
    # Custom legend elements
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color=DARK_BLUE, linewidth=2, label='FRB Sightline'),
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=DARK_BLUE, markeredgecolor='white', markersize=10, label='Galaxy Center'),
        plt.Line2D([0], [0], color=ACCENT_ORANGE, linestyle='--', linewidth=1.5, label='Impact Parameter ($b$)'),
        Patch(facecolor=LIGHT_BLUE, edgecolor='none', alpha=0.3, label='Virial Radius ($R_{200}$ Halo)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', edgecolor=GRID_COLOR)
    
    plot_path1 = os.path.join(output_dir, 'sightline_intersections.png')
    plt.tight_layout()
    plt.savefig(plot_path1, dpi=300)
    plt.close()
    print(f"Saved sightline intersections plot to {plot_path1}")
    
    # ------------------ Plot 2: NFW Mass Profiles ------------------
    plt.figure(figsize=(10, 6), dpi=150)
    
    r_arr = np.linspace(0.1, 300, 500)
    
    for _, row in df_plot.iterrows():
        # Calculate NFW mass profile
        m_enc = nfw_enclosed_mass(r_arr, row['m_halo'], row['r_vir'], row['r_s'], row['c'])
        
        # Plot the profile
        line, = plt.plot(r_arr, m_enc / 1e11, label=f"{row['target']}-G (z={row['z_gal']:.3f})", linewidth=2)
        color = line.get_color()
        
        # Mark the impact parameter
        plt.axvline(row['impact'], color=color, linestyle=':', alpha=0.5)
        plt.scatter(row['impact'], nfw_enclosed_mass(row['impact'], row['m_halo'], row['r_vir'], row['r_s'], row['c']) / 1e11,
                    color=color, edgecolor='white', s=50, zorder=5)
        
    plt.xlabel('Physical Radius $r$ (kpc)', fontsize=12, fontweight='bold', color=TEXT_DARK)
    plt.ylabel(r'Enclosed NFW Mass $M(<r)$ ($10^{11} M_{\odot}$)', fontsize=12, fontweight='bold', color=TEXT_DARK)
    plt.title('Enclosed Dark Matter Halo Mass Profiles of Intervening Galaxies', fontsize=14, fontweight='bold', color=DARK_BLUE, pad=15)
    
    plt.grid(True, linestyle=':', color=GRID_COLOR, alpha=0.7)
    plt.xlim(0, 300)
    plt.ylim(0, 35)
    plt.legend(loc='upper left', frameon=True, facecolor='white', edgecolor=GRID_COLOR, fontsize=9)
    
    plot_path2 = os.path.join(output_dir, 'galaxy_mass_profiles.png')
    plt.tight_layout()
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"Saved galaxy mass profiles plot to {plot_path2}")

if __name__ == '__main__':
    main()
