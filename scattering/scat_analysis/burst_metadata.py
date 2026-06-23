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


# Canonical nickname -> TNS map. The preferred source, chimedsa_burst_specs.csv,
# is gitignored and absent from clean checkouts, so this committed map is the
# fallback used by load_tns_name. mahi carries a correction verified against a
# TNS cone search (20240119A -> 20240122A). johndoeii uses the DSA-110 archive
# designation FRB 20230814B (a.k.a. "johndoe"): the burst was double-reported to
# TNS, also as 20230814A, but the data producer files it under B.
_FALLBACK_TNS = {
    "zach": "FRB 20220207C", "whitney": "FRB 20220310F", "oran": "FRB 20220506D",
    "isha": "FRB 20221113A", "wilhelm": "FRB 20221203A", "phineas": "FRB 20230307A",
    "freya": "FRB 20230325A", "johndoeii": "FRB 20230814B", "hamilton": "FRB 20230913A",
    "mahi": "FRB 20240122A", "chromatica": "FRB 20240203A", "casey": "FRB 20240229A",
}


def load_tns_name(burst_nickname: str, csv_path: Optional[Path] = None) -> str:
    """Load TNS name for a burst given its nickname.

    Prefers chimedsa_burst_specs.csv when present; otherwise uses the committed
    _FALLBACK_TNS map. Returns the uppercased nickname only if the burst is in
    neither source.
    """
    nickname_lower = burst_nickname.lower()
    try:
        df = load_burst_metadata(csv_path)
        match = df[df["name"].str.lower() == nickname_lower]
        if not match.empty:
            return match.iloc[0]["TNS"]
    except Exception:
        pass
    return _FALLBACK_TNS.get(nickname_lower, burst_nickname.upper())
