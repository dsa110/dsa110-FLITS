"""
results_db.py
=============

Structured storage for FLITS analysis results with comparison capabilities.

Provides:
- Dataclasses for scattering and scintillation results
- SQLite-backed database for persistent storage
- Pandas DataFrame export for analysis
- Filtering and querying capabilities
"""

from __future__ import annotations

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ScatteringResult:
    """Results from scattering (BurstFit) analysis."""
    
    # Identification
    burst_name: str
    telescope: str
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Model selection
    best_model: str = ""  # M0, M1, M2, M3
    bic_m0: Optional[float] = None
    bic_m1: Optional[float] = None
    bic_m2: Optional[float] = None
    bic_m3: Optional[float] = None
    
    # Scattering parameters (from best model)
    tau_1ghz: Optional[float] = None  # ms
    tau_1ghz_err: Optional[float] = None
    alpha: Optional[float] = None  # frequency scaling index
    alpha_err: Optional[float] = None
    
    # Pulse parameters
    t0: Optional[float] = None  # ms, peak time
    t0_err: Optional[float] = None
    zeta: Optional[float] = None  # ms, intrinsic width
    zeta_err: Optional[float] = None
    gamma: Optional[float] = None  # spectral index
    gamma_err: Optional[float] = None
    
    # Fit quality
    chi2_reduced: Optional[float] = None
    n_params: Optional[int] = None
    n_datapoints: Optional[int] = None
    
    # MCMC diagnostics
    n_steps: Optional[int] = None
    n_walkers: Optional[int] = None
    acceptance_fraction: Optional[float] = None
    gelman_rubin_max: Optional[float] = None  # worst R-hat across params
    
    # Data characteristics
    freq_min_ghz: Optional[float] = None
    freq_max_ghz: Optional[float] = None
    time_resolution_ms: Optional[float] = None
    freq_resolution_mhz: Optional[float] = None
    snr_peak: Optional[float] = None
    
    # File references
    config_path: str = ""
    data_path: str = ""
    sampler_path: str = ""  # pickled emcee sampler
    
    # Metadata
    notes: str = ""
    quality_flag: str = "unknown"  # good, marginal, bad, unknown
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScatteringResult":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_pipeline_results(
        cls,
        burst_name: str,
        telescope: str,
        results: Dict[str, Any],
        config_path: str = "",
        data_path: str = "",
    ) -> "ScatteringResult":
        """Create from BurstPipeline results dictionary."""
        best_p = results.get("best_params")
        flat_chain = results.get("flat_chain")
        param_names = results.get("param_names", [])
        gof = results.get("goodness_of_fit", {})
        sampler = results.get("sampler")
        
        # Extract parameter uncertainties from chain if available
        param_stats = {}
        if flat_chain is not None and param_names:
            for i, name in enumerate(param_names):
                median = np.median(flat_chain[:, i])
                std = np.std(flat_chain[:, i])
                param_stats[name] = {"value": median, "err": std}
        
        # Gelman-Rubin from sampler
        rhat_max = None
        if sampler is not None:
            try:
                from scattering.scat_analysis.burstfit import gelman_rubin
                chain = sampler.get_chain(flat=False)
                rhat = gelman_rubin(chain)
                rhat_max = float(np.max(rhat))
            except Exception:
                pass
        
        return cls(
            burst_name=burst_name,
            telescope=telescope,
            best_model=results.get("best_key", ""),
            tau_1ghz=getattr(best_p, "tau_1ghz", None) if best_p else None,
            tau_1ghz_err=param_stats.get("tau_1ghz", {}).get("err"),
            alpha=getattr(best_p, "alpha", None) if best_p else None,
            alpha_err=param_stats.get("alpha", {}).get("err"),
            t0=getattr(best_p, "t0", None) if best_p else None,
            t0_err=param_stats.get("t0", {}).get("err"),
            zeta=getattr(best_p, "zeta", None) if best_p else None,
            zeta_err=param_stats.get("zeta", {}).get("err"),
            gamma=getattr(best_p, "gamma", None) if best_p else None,
            gamma_err=param_stats.get("gamma", {}).get("err"),
            chi2_reduced=gof.get("chi2_reduced"),
            gelman_rubin_max=rhat_max,
            n_steps=sampler.iteration if sampler else None,
            n_walkers=sampler.nwalkers if sampler else None,
            config_path=config_path,
            data_path=data_path,
        )


@dataclass
class ScintillationResult:
    """Results from scintillation analysis."""
    
    # Identification
    burst_name: str
    telescope: str
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Scintillation parameters (frequency-averaged or best sub-band)
    delta_nu_dc: Optional[float] = None  # MHz, decorrelation bandwidth
    delta_nu_dc_err: Optional[float] = None
    modulation_index: Optional[float] = None
    modulation_index_err: Optional[float] = None
    
    # Power-law scaling
    scaling_alpha: Optional[float] = None  # Δν ∝ ν^α
    scaling_alpha_err: Optional[float] = None
    scaling_interpretation: str = ""
    
    # Model selection
    best_model: str = ""  # lorentzian, gaussian, etc.
    n_components: Optional[int] = None
    
    # Sub-band analysis (JSON-encoded arrays)
    subband_freqs_mhz: str = "[]"  # JSON array
    subband_delta_nu: str = "[]"
    subband_delta_nu_err: str = "[]"
    
    # Fit quality
    chi2_reduced: Optional[float] = None
    
    # Data characteristics  
    freq_min_mhz: Optional[float] = None
    freq_max_mhz: Optional[float] = None
    n_subbands: Optional[int] = None
    
    # File references
    config_path: str = ""
    data_path: str = ""
    
    # Metadata
    notes: str = ""
    quality_flag: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScintillationResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_pipeline_results(
        cls,
        burst_name: str,
        telescope: str,
        final_results: Dict[str, Any],
        acf_results: Dict[str, Any],
        config_path: str = "",
        data_path: str = "",
    ) -> "ScintillationResult":
        """Create from ScintillationAnalysis pipeline results."""
        
        # Extract primary component results
        components = final_results.get("components", {})
        primary = components.get("component_1", components.get("scint_scale", {}))
        
        measurements = primary.get("subband_measurements", [])
        
        # Get sub-band arrays
        freqs = [m.get("freq_mhz") for m in measurements]
        bws = [m.get("bw") for m in measurements]
        bw_errs = [m.get("bw_err", 0) for m in measurements]
        
        # Compute weighted average decorrelation bandwidth
        if bws and any(bw is not None for bw in bws):
            valid = [(f, b, e) for f, b, e in zip(freqs, bws, bw_errs) 
                     if b is not None and e is not None and e > 0]
            if valid:
                weights = [1/e**2 for _, _, e in valid]
                avg_bw = sum(w * b for (_, b, _), w in zip(valid, weights)) / sum(weights)
                avg_bw_err = 1 / np.sqrt(sum(weights))
            else:
                avg_bw = np.nanmean([b for b in bws if b is not None])
                avg_bw_err = np.nanstd([b for b in bws if b is not None])
        else:
            avg_bw, avg_bw_err = None, None
        
        return cls(
            burst_name=burst_name,
            telescope=telescope,
            delta_nu_dc=avg_bw,
            delta_nu_dc_err=avg_bw_err,
            best_model=final_results.get("best_model", ""),
            scaling_alpha=primary.get("alpha"),
            scaling_alpha_err=primary.get("alpha_err"),
            scaling_interpretation=primary.get("scaling_interpretation", ""),
            subband_freqs_mhz=json.dumps(freqs),
            subband_delta_nu=json.dumps(bws),
            subband_delta_nu_err=json.dumps(bw_errs),
            n_subbands=len(measurements),
            freq_min_mhz=acf_results.get("subband_center_freqs_mhz", [None])[0],
            freq_max_mhz=acf_results.get("subband_center_freqs_mhz", [None])[-1] if acf_results.get("subband_center_freqs_mhz") else None,
            config_path=config_path,
            data_path=data_path,
        )


class ResultsDatabase:
    """SQLite-backed database for FLITS results."""
    
    def __init__(self, db_path: Union[str, Path] = "flits_results.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()
        
    def _initialize_db(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        
        # Scattering results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scattering_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                burst_name TEXT NOT NULL,
                telescope TEXT NOT NULL,
                analysis_timestamp TEXT,
                best_model TEXT,
                bic_m0 REAL, bic_m1 REAL, bic_m2 REAL, bic_m3 REAL,
                tau_1ghz REAL, tau_1ghz_err REAL,
                alpha REAL, alpha_err REAL,
                t0 REAL, t0_err REAL,
                zeta REAL, zeta_err REAL,
                gamma REAL, gamma_err REAL,
                chi2_reduced REAL,
                n_params INTEGER, n_datapoints INTEGER,
                n_steps INTEGER, n_walkers INTEGER,
                acceptance_fraction REAL, gelman_rubin_max REAL,
                freq_min_ghz REAL, freq_max_ghz REAL,
                time_resolution_ms REAL, freq_resolution_mhz REAL,
                snr_peak REAL,
                config_path TEXT, data_path TEXT, sampler_path TEXT,
                notes TEXT, quality_flag TEXT,
                UNIQUE(burst_name, telescope, analysis_timestamp)
            )
        """)
        
        # Scintillation results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scintillation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                burst_name TEXT NOT NULL,
                telescope TEXT NOT NULL,
                analysis_timestamp TEXT,
                delta_nu_dc REAL, delta_nu_dc_err REAL,
                modulation_index REAL, modulation_index_err REAL,
                scaling_alpha REAL, scaling_alpha_err REAL,
                scaling_interpretation TEXT,
                best_model TEXT, n_components INTEGER,
                subband_freqs_mhz TEXT, subband_delta_nu TEXT, subband_delta_nu_err TEXT,
                chi2_reduced REAL,
                freq_min_mhz REAL, freq_max_mhz REAL, n_subbands INTEGER,
                config_path TEXT, data_path TEXT,
                notes TEXT, quality_flag TEXT,
                UNIQUE(burst_name, telescope, analysis_timestamp)
            )
        """)
        
        conn.commit()
        
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            
    def add_scattering_result(self, result: ScatteringResult) -> int:
        """Add a scattering result to the database."""
        conn = self._get_connection()
        data = result.to_dict()
        
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        
        cursor = conn.execute(
            f"INSERT OR REPLACE INTO scattering_results ({columns}) VALUES ({placeholders})",
            list(data.values())
        )
        conn.commit()
        
        log.info(f"Added scattering result: {result.burst_name}/{result.telescope}")
        return cursor.lastrowid
    
    def add_scintillation_result(self, result: ScintillationResult) -> int:
        """Add a scintillation result to the database."""
        conn = self._get_connection()
        data = result.to_dict()
        
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        
        cursor = conn.execute(
            f"INSERT OR REPLACE INTO scintillation_results ({columns}) VALUES ({placeholders})",
            list(data.values())
        )
        conn.commit()
        
        log.info(f"Added scintillation result: {result.burst_name}/{result.telescope}")
        return cursor.lastrowid
    
    def get_scattering_results(
        self,
        burst_name: Optional[str] = None,
        telescope: Optional[str] = None,
        quality_flag: Optional[str] = None,
    ) -> List[ScatteringResult]:
        """Query scattering results with optional filters."""
        conn = self._get_connection()
        
        query = "SELECT * FROM scattering_results WHERE 1=1"
        params = []
        
        if burst_name:
            query += " AND burst_name = ?"
            params.append(burst_name)
        if telescope:
            query += " AND telescope = ?"
            params.append(telescope)
        if quality_flag:
            query += " AND quality_flag = ?"
            params.append(quality_flag)
            
        query += " ORDER BY burst_name, telescope"
        
        rows = conn.execute(query, params).fetchall()
        return [ScatteringResult.from_dict(dict(row)) for row in rows]
    
    def get_scintillation_results(
        self,
        burst_name: Optional[str] = None,
        telescope: Optional[str] = None,
        quality_flag: Optional[str] = None,
    ) -> List[ScintillationResult]:
        """Query scintillation results with optional filters."""
        conn = self._get_connection()
        
        query = "SELECT * FROM scintillation_results WHERE 1=1"
        params = []
        
        if burst_name:
            query += " AND burst_name = ?"
            params.append(burst_name)
        if telescope:
            query += " AND telescope = ?"
            params.append(telescope)
        if quality_flag:
            query += " AND quality_flag = ?"
            params.append(quality_flag)
            
        query += " ORDER BY burst_name, telescope"
        
        rows = conn.execute(query, params).fetchall()
        return [ScintillationResult.from_dict(dict(row)) for row in rows]
    
    def to_dataframe(
        self,
        table: str = "scattering",
        **filters,
    ) -> pd.DataFrame:
        """
        Export results to pandas DataFrame.
        
        Args:
            table: "scattering" or "scintillation"
            **filters: Passed to get_*_results()
        """
        if table == "scattering":
            results = self.get_scattering_results(**filters)
        elif table == "scintillation":
            results = self.get_scintillation_results(**filters)
        else:
            raise ValueError(f"Unknown table: {table}")
            
        return pd.DataFrame([r.to_dict() for r in results])
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Generate a comparison table joining scattering and scintillation results.
        
        Returns DataFrame with columns from both analyses for each burst/telescope.
        """
        scat_df = self.to_dataframe("scattering")
        scint_df = self.to_dataframe("scintillation")
        
        if scat_df.empty and scint_df.empty:
            return pd.DataFrame()
        
        # Merge on burst_name and telescope
        merged = pd.merge(
            scat_df,
            scint_df,
            on=["burst_name", "telescope"],
            how="outer",
            suffixes=("_scat", "_scint"),
        )
        
        # Select key columns for comparison
        key_cols = [
            "burst_name", "telescope",
            "tau_1ghz", "tau_1ghz_err", "alpha", "alpha_err",  # scattering
            "delta_nu_dc", "delta_nu_dc_err", "scaling_alpha",  # scintillation
            "chi2_reduced_scat", "chi2_reduced_scint",
            "quality_flag_scat", "quality_flag_scint",
        ]
        
        available_cols = [c for c in key_cols if c in merged.columns]
        return merged[available_cols]
    
    def export_latex_table(
        self,
        output_path: Optional[Path] = None,
        caption: str = "FLITS Analysis Results",
    ) -> str:
        """Export comparison table as LaTeX."""
        df = self.get_comparison_table()
        
        if df.empty:
            return "% No results to export"
        
        # Format for publication
        latex = df.to_latex(
            index=False,
            na_rep="--",
            float_format="%.3f",
            caption=caption,
            label="tab:flits_results",
        )
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(latex)
            log.info(f"Exported LaTeX table to {output_path}")
            
        return latex

