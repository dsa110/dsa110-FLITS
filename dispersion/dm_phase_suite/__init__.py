"""Controlled, boundary-safe implementation of the DM-phase estimator."""

from .model import CoherenceCurve, DMSearchResult
from .search import search_dm

__all__ = ["CoherenceCurve", "DMSearchResult", "search_dm"]
