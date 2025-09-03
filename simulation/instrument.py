from __future__ import annotations

"""Instrumental configuration for backend noise parameters."""

from dataclasses import dataclass
import astropy.units as u
from astropy import constants as const

K_B_J_PER_K = const.k_B.to(u.J / u.K).value
JY_TO_SI = 1e-26  # 1 Jy in W / m^2 / Hz


@dataclass
class InstrumentalCfg:
    """Backend noise parameters."""

    sefd: u.Quantity | None = None  # direct specification (Jy)
    t_sys: u.Quantity | None = None  # alternative route (K)
    a_eff: u.Quantity | None = None  # for SEFD = 2 k_B T / A_eff (m^2)
    n_pol: int = 2  # usually 1 or 2

    def get_sefd_jy(self) -> float | None:
        """Return SEFD in Jy, calculating from T_sys/A_eff if needed."""
        if self.sefd is not None:
            return self.sefd.to_value(u.Jy)
        if self.t_sys is not None and self.a_eff is not None:
            sefd_si = 2 * K_B_J_PER_K * self.t_sys.to_value(u.K) / self.a_eff.to_value(u.m**2)
            return sefd_si / JY_TO_SI
        return None
