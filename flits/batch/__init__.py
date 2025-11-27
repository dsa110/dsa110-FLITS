"""
FLITS Batch Processing Module
=============================

Provides infrastructure for running analysis pipelines across multiple FRBs
and aggregating results for comparative analysis.

Components:
- config_generator: Auto-generate configs from data file naming conventions
- batch_runner: Orchestrate multi-burst analysis with parallel execution
- results_db: Structured storage for analysis results
- joint_analysis: Cross-validate scattering and scintillation measurements
"""

from .config_generator import ConfigGenerator
from .results_db import ResultsDatabase, ScatteringResult, ScintillationResult
from .batch_runner import BatchRunner
from .joint_analysis import JointAnalysis

__all__ = [
    "ConfigGenerator",
    "ResultsDatabase", 
    "ScatteringResult",
    "ScintillationResult",
    "BatchRunner",
    "JointAnalysis",
]

