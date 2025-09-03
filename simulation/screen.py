from __future__ import annotations

"""Scattering screen configuration and realisation utilities."""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import astropy.units as u

from geometry import _array2


@dataclass
class ScreenCfg:
    """Configuration for a single thin scattering screen."""

    N: int = 128
    L: u.Quantity = 1.0 * u.AU
    profile: Literal["gaussian", "powerlaw"] = "gaussian"
    alpha: float = 3.0
    theta0: u.Quantity = 100.0 * u.marcsec
    geometry: Literal["2D", "1D"] = "2D"
    axial_ratio: float = 1.0
    pa: u.Quantity = 0.0 * u.deg
    amp_distribution: Literal["constant", "rayleigh"] = "constant"
    rng_seed: Optional[int] = None
    v_perp: Tuple[float, float] | np.ndarray | None = None

    def __post_init__(self) -> None:
        self.v_perp = _array2(self.v_perp, u.km / u.s)


class Screen:
    """Random physical realisation of a scattering screen."""

    def __init__(self, cfg: ScreenCfg, D_screen_m: float):
        self.cfg = cfg
        self.D_screen_m = D_screen_m
        rng = np.random.default_rng(cfg.rng_seed)

        L_m = cfg.L.to(u.m).value
        sigma_rad = L_m / D_screen_m

        if cfg.geometry == "1D":
            xy = np.zeros((cfg.N, 2))
            xy[:, 0] = rng.uniform(-2 * sigma_rad, 2 * sigma_rad, size=cfg.N)
        else:
            xy = rng.uniform(-2 * sigma_rad, 2 * sigma_rad, size=(cfg.N, 2))

        self.anisotropy_scaling = np.array([1.0, 1.0 / cfg.axial_ratio])
        xy = xy * self.anisotropy_scaling

        if cfg.pa.value != 0.0:
            pa_rad = cfg.pa.to(u.rad).value
            R = np.array([[np.cos(pa_rad), -np.sin(pa_rad)], [np.sin(pa_rad), np.cos(pa_rad)]])
            xy = xy @ R.T
        self.theta = xy

        if cfg.amp_distribution == "constant":
            amps = np.ones(cfg.N)
        else:
            amps = rng.rayleigh(scale=1 / np.sqrt(2), size=cfg.N)
        phases = rng.uniform(0, 2 * np.pi, size=cfg.N)
        field = amps * np.exp(1j * phases)

        if cfg.profile == "gaussian":
            if cfg.pa.value != 0.0:
                pa_rad = cfg.pa.to(u.rad).value
                R_inv = np.array([[np.cos(-pa_rad), -np.sin(-pa_rad)], [np.sin(-pa_rad), np.cos(-pa_rad)]])
                xy_unrotated = xy @ R_inv.T
            else:
                xy_unrotated = xy
            xy_unscaled = xy_unrotated / self.anisotropy_scaling
            r2_unscaled = np.sum(xy_unscaled**2, axis=1)
            w = np.exp(-r2_unscaled / (2 * sigma_rad**2))
        else:
            theta0_rad = cfg.theta0.to(u.rad).value
            r2 = np.sum(self.theta**2, axis=1)
            w = (1 + r2 / theta0_rad**2) ** (-cfg.alpha / 2)

        self.field = field * w
