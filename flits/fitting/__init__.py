"""FLITS fitting module with validation."""

from .diagnostics import ResidualDiagnostics, analyze_residuals
from . import VALIDATION_THRESHOLDS

__all__ = [
    "ResidualDiagnostics",
    "analyze_residuals",
    "VALIDATION_THRESHOLDS",
]
