"""Burst metadata utilities for the scattering analysis pipeline.

This module provides functions to load burst metadata from external sources
such as CSV files containing TNS names and other burst properties.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


# Cache for burst metadata to avoid re-reading CSV
_BURST_METADATA_CACHE = None


def load_burst_metadata(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load burst metadata from CSV file.
    
    Parameters
    ----------
    csv_path : Path, optional
        Path to CSV file. If None, uses default location.
        
    Returns
    -------
    pd.DataFrame
        Burst metadata with columns: name, TNS, MJD, RA_deg, Dec_deg, etc.
    """
    global _BURST_METADATA_CACHE
    
    if _BURST_METADATA_CACHE is not None:
        return _BURST_METADATA_CACHE
    
    if csv_path is None:
        # Default to chimedsa_burst_specs.csv in repository root
        csv_path = Path(__file__).parent.parent.parent / 'chimedsa_burst_specs.csv'
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Burst metadata CSV not found: {csv_path}")
    
    _BURST_METADATA_CACHE = pd.read_csv(csv_path)
    return _BURST_METADATA_CACHE


def load_tns_name(burst_nickname: str, csv_path: Optional[Path] = None) -> str:
    """Load TNS name for a burst given its nickname.
    
    Parameters
    ----------
    burst_nickname : str
        Burst nickname (e.g., 'casey', 'freya')
    csv_path : Path, optional
        Path to CSV file. If None, uses default location.
        
    Returns
    -------
    str
        TNS name (e.g., 'FRB 20240229A') or nickname if not found
        
    Examples
    --------
    >>> load_tns_name('casey')
    'FRB 20240229A'
    >>> load_tns_name('freya')
    'FRB 20230325A'
    """
    try:
        df = load_burst_metadata(csv_path)
        # Case-insensitive lookup
        nickname_lower = burst_nickname.lower()
        match = df[df['name'].str.lower() == nickname_lower]
        
        if not match.empty:
            return match.iloc[0]['TNS']
        else:
            # Fallback to uppercase nickname if not found
            return burst_nickname.upper()
    except Exception as e:
        # If any error occurs, fallback to uppercase nickname
        print(f"Warning: Could not load TNS name for {burst_nickname}: {e}")
        return burst_nickname.upper()
