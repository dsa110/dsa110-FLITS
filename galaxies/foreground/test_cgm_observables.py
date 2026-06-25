"""Network-free tests for CGM observable helper functions."""

import math

import pytest

from galaxies.foreground import cgm_observables as cgm


def test_axis_ratio_from_ellipticity_handles_rounder_and_bad_inputs():
    assert cgm.axis_ratio_from_ellipticity(0.0, 0.0) == pytest.approx(1.0)

    q_mild = cgm.axis_ratio_from_ellipticity(0.2, 0.0)
    q_strong = cgm.axis_ratio_from_ellipticity(0.6, 0.0)
    assert q_strong < q_mild < 1.0

    assert cgm.axis_ratio_from_ellipticity(None, 0.0) is None
    assert cgm.axis_ratio_from_ellipticity(math.nan, 0.0) is None


def test_position_angle_deg_folds_to_180_and_rejects_undefined_shapes():
    pa = cgm.position_angle_deg(0.3, 0.4)
    pa_flipped = cgm.position_angle_deg(-0.3, -0.4)
    assert 0.0 <= pa < 180.0
    assert pa_flipped == pytest.approx((pa + 90.0) % 180.0)

    assert cgm.position_angle_deg(0.0, 0.0) is None
    assert cgm.position_angle_deg(None, 0.0) is None
    assert cgm.position_angle_deg(math.nan, 0.0) is None


def test_inclination_deg_handles_face_on_edge_on_and_bad_inputs():
    q0 = 0.15
    assert cgm.inclination_deg(1.0) == pytest.approx(0.0)
    assert cgm.inclination_deg(q0 + 1e-6, q0=q0) == pytest.approx(90.0, abs=0.05)
    assert cgm.inclination_deg(q0, q0=q0) is None

    inc = cgm.inclination_deg(0.5, q0=q0)
    assert 0.0 < inc < 90.0

    assert cgm.inclination_deg(None) is None
    assert cgm.inclination_deg(math.nan) is None


def test_azimuthal_angle_phi_deg_is_folded_and_rejects_bad_inputs():
    phi_major = cgm.azimuthal_angle_phi_deg(10.0, 20.0, 90.0, 10.01, 20.0)
    phi_minor = cgm.azimuthal_angle_phi_deg(10.0, 20.0, 90.0, 10.0, 20.01)
    phi_offset = cgm.azimuthal_angle_phi_deg(10.0, 20.0, 45.0, 10.01, 20.01)

    for phi in (phi_major, phi_minor, phi_offset):
        assert 0.0 <= phi <= 90.0

    assert phi_major < 5.0
    assert phi_minor > 85.0
    assert cgm.azimuthal_angle_phi_deg(None, 20.0, 90.0, 10.01, 20.0) is None
    assert cgm.azimuthal_angle_phi_deg(10.0, math.nan, 90.0, 10.01, 20.0) is None


def test_stellar_mass_desi_gz_is_finite_monotonic_and_rejects_bad_inputs():
    mass = cgm.stellar_mass_desi_gz(g_ab=20.0, z_ab=18.5, z_gal=0.1)
    brighter = cgm.stellar_mass_desi_gz(g_ab=19.0, z_ab=17.5, z_gal=0.1)

    assert math.isfinite(mass)
    assert brighter > mass
    assert cgm.stellar_mass_desi_gz(20.0, 18.5, 0.0) is None
    assert cgm.stellar_mass_desi_gz(20.0, math.nan, 0.1) is None


def test_stellar_mass_wise_w1_is_finite_monotonic_and_uses_fallback_color():
    fallback = cgm.stellar_mass_wise_w1(w1_vega=15.0, z_gal=0.1)
    colored = cgm.stellar_mass_wise_w1(w1_vega=15.0, z_gal=0.1, w1_w2=0.2)
    brighter = cgm.stellar_mass_wise_w1(w1_vega=14.0, z_gal=0.1)

    assert math.isfinite(fallback)
    assert math.isfinite(colored)
    assert brighter > fallback
    assert cgm.stellar_mass_wise_w1(math.nan, 0.1) is None
    assert cgm.stellar_mass_wise_w1(15.0, 0.0) is None


def test_is_star_forming_uses_blue_red_cut_and_mass_tilt_path():
    assert cgm.is_star_forming(0.3) is True
    assert cgm.is_star_forming(0.9) is False
    assert cgm.is_star_forming(0.61, logmstar=8.0) is False
    assert cgm.is_star_forming(math.nan) is False


def test_sfr_wise_w3_is_positive_monotonic_and_rejects_bad_inputs():
    sfr = cgm.sfr_wise_w3(w3_vega=12.0, z_gal=0.1)
    brighter = cgm.sfr_wise_w3(w3_vega=11.0, z_gal=0.1)

    assert math.isfinite(sfr)
    assert sfr > 0.0
    assert brighter > sfr
    assert cgm.sfr_wise_w3(12.0, 0.0) is None
    assert cgm.sfr_wise_w3(math.nan, 0.1) is None


def test_sfr_uv_nuv_is_positive_monotonic_and_rejects_bad_inputs():
    sfr = cgm.sfr_uv_nuv(nuv_ab=20.0, z_gal=0.1)
    corrected = cgm.sfr_uv_nuv(nuv_ab=20.0, z_gal=0.1, ebv=0.1)
    brighter = cgm.sfr_uv_nuv(nuv_ab=19.0, z_gal=0.1)

    assert math.isfinite(sfr)
    assert sfr > 0.0
    assert corrected > sfr
    assert brighter > sfr
    assert cgm.sfr_uv_nuv(20.0, 0.0) is None
    assert cgm.sfr_uv_nuv(math.nan, 0.1) is None


def test_metallicity_mzr_increases_with_mass_and_rejects_bad_inputs():
    low = cgm.metallicity_mzr(8.0)
    high = cgm.metallicity_mzr(11.0, z_gal=0.2)

    assert 7.5 < low < 9.2
    assert 7.5 < high < 9.2
    assert high > low
    assert cgm.metallicity_mzr(None) is None
    assert cgm.metallicity_mzr(math.nan) is None


def test_wise_agn_stern2012_uses_vega_color_cut_and_rejects_bad_inputs():
    assert cgm.wise_agn_stern2012(0.9) is True
    assert cgm.wise_agn_stern2012(0.5) is False
    assert cgm.wise_agn_stern2012(math.nan) is False


def test_azimuthal_angle_phi_deg_folds_to_first_quadrant_and_known_geometry():
    # Galaxy on the equator; major-axis PA=0 (North-South).
    g_ra, g_dec = 150.0, 0.0
    # Sightline due North -> bearing ~0 -> aligned with major axis -> phi ~0.
    north = cgm.azimuthal_angle_phi_deg(g_ra, g_dec, 0.0, g_ra, g_dec + 0.01)
    # Sightline due South -> bearing ~180 -> still along major axis -> phi ~0
    # (exercises the 180 deg wrap that the fold must collapse).
    south = cgm.azimuthal_angle_phi_deg(g_ra, g_dec, 0.0, g_ra, g_dec - 0.01)
    # Sightline due East -> bearing ~90 -> perpendicular (minor axis) -> phi ~90.
    east = cgm.azimuthal_angle_phi_deg(g_ra, g_dec, 0.0, g_ra + 0.01, g_dec)

    assert north == pytest.approx(0.0, abs=1e-3)
    assert south == pytest.approx(0.0, abs=1e-3)
    assert east == pytest.approx(90.0, abs=1e-3)

    # Always folded into [0, 90] for a range of PAs and offsets.
    for pa in (0.0, 30.0, 75.0, 120.0, 179.0):
        for dra, ddec in ((0.02, 0.01), (-0.03, 0.02), (0.01, -0.04)):
            phi = cgm.azimuthal_angle_phi_deg(g_ra, g_dec, pa, g_ra + dra, g_dec + ddec)
            assert phi is not None
            assert 0.0 <= phi <= 90.0

    assert cgm.azimuthal_angle_phi_deg(math.nan, 0.0, 0.0, 1.0, 1.0) is None
