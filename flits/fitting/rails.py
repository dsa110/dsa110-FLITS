"""Unified prior-rail classifier (ADR-0008 Gate 3).

A posterior is *railed* when it pins a prior bound rather than measuring the
parameter. Railing — not the value — is the disqualifier: a railed posterior is
model-family rejection (ADR-0007's re-open trigger), never a quotable limit.

This is the **single** rail definition imported anywhere. The three prior
definitions (`grade_beta_campaign.classify_rail`, `sim_gate._railed`,
`gate_joint_committed`'s `RAIL_EDGE` median-only test) are replaced by imports
of :func:`classify_rail`.

Two evaluation paths, strongest first:

- **Posterior-mass (canonical):** given the sampled posterior, railed when
  ≥``EDGE_MASS_FRAC`` of the weighted mass falls within ``EDGE_MASS_WIDTH`` of
  a bound. This catches a tight posterior pinned at the bound that the
  median-only test misses.
- **Summary-only (fallback):** given only the median and asymmetric errors,
  railed when the median sits within 3σ of a bound. Weaker — labeled
  ``method="summary_3sigma"`` — used only when no posterior samples are in-tree
  (the mixed-legacy generation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

EDGE_MASS_WIDTH = 0.05  # beta/alpha window at each bound for the mass test
EDGE_MASS_FRAC = 0.30  # >= this much mass in the window => railed
SUMMARY_SIGMA = 3.0  # median within this many sigma of a bound => railed (fallback)


@dataclass(frozen=True)
class RailVerdict:
    """Rail classification for one sampled parameter (beta or legacy alpha).

    ``cls`` is one of ``interior`` / ``railed-hi`` / ``railed-lo`` /
    ``unconstrained`` (mass at both bounds). ``method`` records which path ran,
    so a summary-only verdict is never mistaken for a posterior-mass one.
    """

    cls: str
    detail: str
    method: str  # "posterior_mass" | "summary_3sigma"
    railed: bool
    railed_hi: bool
    railed_lo: bool
    edge_mass_lo: float | None
    edge_mass_hi: float | None

    def asdict(self) -> dict:
        return {
            "class": self.cls,
            "detail": self.detail,
            "method": self.method,
            "railed": self.railed,
            "railed_hi": self.railed_hi,
            "railed_lo": self.railed_lo,
            "edge_mass_lo": self.edge_mass_lo,
            "edge_mass_hi": self.edge_mass_hi,
        }


def _label(railed_hi: bool, railed_lo: bool) -> str:
    if railed_hi and railed_lo:
        return "unconstrained (mass at both bounds)"
    if railed_hi:
        return "railed-hi (prior-bound pin; model-family rejection; ADR-0007 candidate)"
    if railed_lo:
        return "railed-lo (prior-bound pin; unconstrained-steep)"
    return "interior"


def classify_rail(
    *,
    lo: float,
    hi: float,
    samples: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    median: float | None = None,
    err_minus: float | None = None,
    err_plus: float | None = None,
) -> RailVerdict:
    """Classify one parameter's posterior as railed or interior.

    Pass ``samples`` (and optional ``weights``) for the canonical
    posterior-mass path. Pass ``median``/``err_minus``/``err_plus`` for the
    summary-only 3σ fallback when no samples are in-tree. ``lo``/``hi`` are the
    prior bounds on the *sampled* parameter (beta for the co-model, alpha for
    legacy free-α fits).
    """
    if samples is not None:
        s = np.asarray(samples, float)
        w = np.ones_like(s) if weights is None else np.asarray(weights, float)
        w = w / w.sum()
        mass_lo = float(w[s <= lo + EDGE_MASS_WIDTH].sum())
        mass_hi = float(w[s >= hi - EDGE_MASS_WIDTH].sum())
        railed_lo = mass_lo >= EDGE_MASS_FRAC
        railed_hi = mass_hi >= EDGE_MASS_FRAC
        return RailVerdict(
            cls=_label(railed_hi, railed_lo).split(" ")[0],
            detail=_label(railed_hi, railed_lo),
            method="posterior_mass",
            railed=railed_hi or railed_lo,
            railed_hi=railed_hi,
            railed_lo=railed_lo,
            edge_mass_lo=mass_lo,
            edge_mass_hi=mass_hi,
        )
    if median is None or err_minus is None or err_plus is None:
        raise ValueError("classify_rail needs either samples or (median, err_minus, err_plus)")
    three_sigma_lo = median - SUMMARY_SIGMA * err_minus <= lo
    three_sigma_hi = median + SUMMARY_SIGMA * err_plus >= hi
    return RailVerdict(
        cls=_label(three_sigma_hi, three_sigma_lo).split(" ")[0],
        detail=_label(three_sigma_hi, three_sigma_lo),
        method="summary_3sigma",
        railed=three_sigma_hi or three_sigma_lo,
        railed_hi=three_sigma_hi,
        railed_lo=three_sigma_lo,
        edge_mass_lo=None,
        edge_mass_hi=None,
    )
