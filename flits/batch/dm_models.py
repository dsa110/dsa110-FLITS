"""
dm_models.py
============

Shared cosmology + modified-NFW halo-gas models for FRB dispersion-measure (DM)
budgeting. Pure numpy (no astropy/scipy dependency).

Contents
--------
- ``Cosmology`` : flat-LCDM distances, R_200, and the Macquart mean DM_cosmic(z).
- ``ModifiedNFW`` : Prochaska & Zheng (2019)-style hot-halo gas profile, giving
  n_e(r), the sightline DM(b), and the cumulative line-of-sight DM.

Calibration note
----------------
The *absolute* halo-DM normalization depends on (f_hot, y0, concentration,
r_max). The defaults here give literature-typical values (DM ~ 54 pc/cm^3
through a 10^12 M_sun halo centre); override them with your calibrated
``sightline_budget`` parameters for production.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

_KPC_CM = 3.0857e21
_MPC_CM = 3.0857e24
_MSUN_G = 1.989e33
_MP_G = 1.6726e-24


@dataclass(frozen=True)
class Cosmology:
    """Flat-LCDM cosmology (numpy-only)."""
    H0: float = 70.0
    Om: float = 0.3
    dm_cosmic_slope: float = 780.0   # pc/cm^3, absorbs (3 c H0 Ob f_d f_e / 8 pi G m_p)

    def E(self, z):
        z = np.asarray(z, float)
        return np.sqrt(self.Om * (1 + z) ** 3 + (1 - self.Om))

    def comoving_mpc(self, z):
        z = float(z); zz = np.linspace(0, max(z, 1e-6), 512)
        return float(np.trapezoid(299792.458 / (self.H0 * self.E(zz)), zz))

    def angular_diameter_kpc(self, z):
        return 1e3 * self.comoving_mpc(z) / (1 + z) if z > 0 else 1.0

    def luminosity_distance_cm(self, z):
        return (1 + z) * self.comoving_mpc(z) * _MPC_CM

    def r200_kpc(self, M_msun, z):
        return 206.0 * (np.asarray(M_msun) / 1e12) ** (1 / 3.) * self.E(z) ** (-2 / 3.)

    def dm_cosmic_mean(self, z):
        """Macquart relation mean <DM_cosmic>(z) [pc/cm^3] (host/observer frame mean)."""
        zz = np.linspace(0, max(float(z), 1e-6), 400)
        return float(self.dm_cosmic_slope * np.trapezoid((1 + zz) / self.E(zz), zz))


@dataclass
class ModifiedNFW:
    """Prochaska & Zheng (2019)-style modified-NFW hot-halo gas.

    rho_gas(r) propto 1 / [ (y0 + r/Rs) (1 + r/Rs)^2 ],  Rs = R200 / concentration,
    normalised so the gas mass within ``rmax_fac * R200`` equals
    ``f_hot * fb * M_200``.
    """
    cosmo: Cosmology = field(default_factory=Cosmology)
    f_hot: float = 0.25
    y0: float = 2.0
    concentration: float = 3.0
    fb: float = 0.158          # cosmic baryon fraction Omega_b / Omega_m
    mu_e: float = 1.17         # mean mass per electron
    rmax_fac: float = 1.0      # integrate/normalise gas out to rmax_fac * R200

    def _rho0_rs_r200(self, M_msun, z):
        R200 = float(self.cosmo.r200_kpc(M_msun, z)); Rs = R200 / self.concentration
        r = np.linspace(1e-3 * R200, self.rmax_fac * R200, 2000); x = r / Rs
        integ = np.trapezoid(4 * np.pi * (r * _KPC_CM) ** 2 / ((self.y0 + x) * (1 + x) ** 2), r * _KPC_CM)
        m_gas_g = self.f_hot * self.fb * M_msun * _MSUN_G
        return m_gas_g / integ, Rs, R200

    def ne(self, M_msun, z, r_kpc):
        """Electron density n_e(r) [cm^-3]."""
        rho0, Rs, _ = self._rho0_rs_r200(M_msun, z); x = np.asarray(r_kpc, float) / Rs
        return rho0 / ((self.y0 + x) * (1 + x) ** 2) / (self.mu_e * _MP_G)

    def dm_of_b(self, M_msun, z, b_kpc):
        """Line-of-sight DM [pc/cm^3] at impact parameter(s) ``b_kpc``."""
        rho0, Rs, R200 = self._rho0_rs_r200(M_msun, z); Rmax = self.rmax_fac * R200
        b = np.atleast_1d(b_kpc).astype(float); out = np.zeros_like(b)
        for i, bb in enumerate(b):
            if bb >= Rmax:
                continue
            l = np.linspace(0, np.sqrt(Rmax ** 2 - bb ** 2), 1500)
            x = np.sqrt(bb ** 2 + l ** 2) / Rs
            out[i] = 2 * np.trapezoid(rho0 / ((self.y0 + x) * (1 + x) ** 2) / (self.mu_e * _MP_G), l * 1e3)
        return out[0] if np.isscalar(b_kpc) or np.ndim(b_kpc) == 0 else out

    def dm_cumulative_los(self, M_msun, z, n=400):
        """Cumulative DM(<r) along a central (b=0) sightline; returns (r/R200, DM[pc/cm^3])."""
        rho0, Rs, R200 = self._rho0_rs_r200(M_msun, z)
        l = np.linspace(0, self.rmax_fac * R200, n); x = l / Rs
        ne = rho0 / ((self.y0 + x) * (1 + x) ** 2) / (self.mu_e * _MP_G)
        cum = 2 * np.concatenate([[0], np.cumsum((ne[1:] + ne[:-1]) / 2 * np.diff(l) * 1e3)])
        return l / R200, cum


if __name__ == "__main__":
    cz = Cosmology(); mn = ModifiedNFW(cosmo=cz)
    print("R200(1e12, z=0.1) = %.0f kpc" % cz.r200_kpc(1e12, 0.1))
    print("DM_cosmic(z=0.19) = %.1f pc/cm^3" % cz.dm_cosmic_mean(0.19))
    for M in (1e12, 1e13, 10**14.5):
        print("DM(b=0, M=%.1e) = %.1f pc/cm^3" % (M, mn.dm_of_b(M, 0.1, 0.0)))
