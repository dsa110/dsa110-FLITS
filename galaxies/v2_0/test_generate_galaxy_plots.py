import matplotlib

matplotlib.use("Agg")

import numpy as np

from galaxies.v2_0.generate_galaxy_plots import _moster_log_mstar, estimate_halo_mass


def test_estimate_halo_mass_clamps_degenerate_stellar_mass():
    # A z~0.001 contaminant yields log M*~5.5, below the SMHM-invertible floor; the
    # clamp must keep brentq from raising a same-sign-bracket ValueError and pin the
    # halo mass at the dwarf-end of the bracket (log M_h ~ 9.5).
    lo = _moster_log_mstar(9.5)
    m_halo = estimate_halo_mass(lo - 4.0)
    assert np.isfinite(m_halo) and m_halo > 0
    assert abs(np.log10(m_halo) - 9.5) < 0.05


def test_estimate_halo_mass_in_range_unchanged():
    # An ordinary log M* ~ 10.5 still inverts to a sensible halo mass.
    m_halo = estimate_halo_mass(10.5)
    assert 1e11 < m_halo < 1e14
