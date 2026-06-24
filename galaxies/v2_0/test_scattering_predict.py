"""Network-free tests for scattering prediction helper functions."""

import math

import pytest

from galaxies.v2_0 import scattering_predict as sp


def test_dm_halo_mnfw_is_positive_decreasing_and_handles_edges():
    dm_inner = sp.dm_halo_mnfw(m_halo_msun=1e12, z_gal=0.1, impact_kpc=25.0)
    dm_mid = sp.dm_halo_mnfw(m_halo_msun=1e12, z_gal=0.1, impact_kpc=50.0)
    dm_outer = sp.dm_halo_mnfw(m_halo_msun=1e12, z_gal=0.1, impact_kpc=100.0)

    assert dm_mid is not None
    assert math.isfinite(dm_mid)
    assert dm_mid > 0.0
    assert dm_inner > dm_mid > dm_outer
    assert sp.dm_halo_mnfw(m_halo_msun=1e12, z_gal=0.1, impact_kpc=5000.0) == 0.0
    assert sp.dm_halo_mnfw(m_halo_msun=None, z_gal=0.1, impact_kpc=50.0) is None
    assert sp.dm_halo_mnfw(m_halo_msun=-1e12, z_gal=0.1, impact_kpc=50.0) is None


def test_dm_cool_scales_with_covering_fraction_and_mgii():
    low_fc = sp.dm_cool(dm_halo=50.0, cool_covering_fraction=0.2)
    high_fc = sp.dm_cool(dm_halo=50.0, cool_covering_fraction=0.8)
    weak_mgii = sp.dm_cool(dm_halo=50.0, cool_covering_fraction=0.5, mgii_wr=0.1)
    strong_mgii = sp.dm_cool(dm_halo=50.0, cool_covering_fraction=0.5, mgii_wr=2.0)

    assert low_fc is not None
    assert low_fc >= 0.0
    assert high_fc > low_fc
    assert strong_mgii > weak_mgii
    assert sp.dm_cool(dm_halo=None, cool_covering_fraction=0.5) is None
    assert sp.dm_cool(dm_halo=50.0, cool_covering_fraction=-0.1) is None


def test_f_tilde_prior_returns_ordered_finite_bracket_and_expected_boosts():
    val0, lo0, hi0 = sp.f_tilde_prior(sfr_msun_yr=0.0)
    val_none, lo_none, hi_none = sp.f_tilde_prior(sfr_msun_yr=None)
    val_sfr, _, _ = sp.f_tilde_prior(sfr_msun_yr=10.0)
    val_agn, _, _ = sp.f_tilde_prior(sfr_msun_yr=10.0, agn=True)
    val_solar, _, _ = sp.f_tilde_prior(sfr_msun_yr=1.0, metallicity_12logOH=8.7)
    val_rich, _, _ = sp.f_tilde_prior(sfr_msun_yr=1.0, metallicity_12logOH=9.1)

    assert 0.0 < lo0 <= val0 <= hi0
    assert 0.0 < lo_none <= val_none <= hi_none
    assert all(math.isfinite(x) for x in (val0, lo0, hi0, val_none, lo_none, hi_none))
    assert val_sfr > val0
    assert val_agn > val_sfr
    assert val_rich > val_solar


def test_g_scatt_has_intervening_geometry_and_zero_boundaries():
    near_front = sp.g_scatt(0.1, 0.5)
    near_source = sp.g_scatt(0.45, 0.5)

    assert sp.g_scatt(0.5, 0.5) == 0.0
    assert sp.g_scatt(0.6, 0.5) == 0.0
    assert sp.g_scatt(0.0, 0.5) == 0.0
    assert sp.g_scatt(None, 0.5) == 0.0
    assert near_front > 0.0
    assert near_source > 0.0
    assert near_source < near_front


def test_tau_scat_ms_scales_with_f_dm_frequency_and_geometry():
    base = sp.tau_scat_ms(f_tilde=0.1, g_scatt_val=50.0, dm_l=20.0, z_lens=0.2)
    high_f = sp.tau_scat_ms(f_tilde=0.2, g_scatt_val=50.0, dm_l=20.0, z_lens=0.2)
    high_dm = sp.tau_scat_ms(f_tilde=0.1, g_scatt_val=50.0, dm_l=40.0, z_lens=0.2)
    low_nu = sp.tau_scat_ms(f_tilde=0.1, g_scatt_val=50.0, dm_l=20.0, z_lens=0.2, nu_ghz=0.5)

    assert base is not None
    assert base >= 0.0
    assert high_f > base
    assert high_dm == pytest.approx(4.0 * base)
    assert low_nu == pytest.approx(16.0 * base)
    assert sp.tau_scat_ms(f_tilde=0.1, g_scatt_val=0.0, dm_l=20.0, z_lens=0.2) == 0.0
    assert sp.tau_scat_ms(f_tilde=None, g_scatt_val=50.0, dm_l=20.0, z_lens=0.2) is None
    assert sp.tau_scat_ms(f_tilde=0.1, g_scatt_val=50.0, dm_l=20.0, z_lens=0.2, nu_ghz=0.0) is None


def test_scint_bandwidth_khz_decreases_with_tau_and_rejects_nonpositive():
    narrow = sp.scint_bandwidth_khz(2.0)
    broad = sp.scint_bandwidth_khz(1.0)

    assert narrow is not None
    assert broad is not None
    assert broad > narrow
    assert sp.scint_bandwidth_khz(0.0) is None
    assert sp.scint_bandwidth_khz(-1.0) is None


def test_predict_mgii_wr_declines_with_impact_and_scales_with_stellar_mass():
    wr_20 = sp.predict_mgii_wr(20.0)
    wr_50 = sp.predict_mgii_wr(50.0)
    wr_100 = sp.predict_mgii_wr(100.0)
    low_mass = sp.predict_mgii_wr(50.0, logmstar=9.5)
    high_mass = sp.predict_mgii_wr(50.0, logmstar=11.5)

    assert wr_100 is not None
    assert wr_20 > wr_50 > wr_100 > 0.0
    assert high_mass > low_mass
    assert sp.predict_mgii_wr(None) is None
    assert sp.predict_mgii_wr(0.0) is None


def test_cool_covering_fraction_returns_ordered_priors_and_expected_trends():
    inner_fc, inner_lo, inner_hi = sp.cool_covering_fraction(0.2, 10.5, True)
    outer_fc, outer_lo, outer_hi = sp.cool_covering_fraction(0.8, 10.5, True)
    sf_fc, _, _ = sp.cool_covering_fraction(0.4, 10.5, True)
    passive_fc, _, _ = sp.cool_covering_fraction(0.4, 10.5, False)
    major_fc, _, _ = sp.cool_covering_fraction(0.4, 10.5, True, phi_deg=0.0)
    minor_fc, _, _ = sp.cool_covering_fraction(0.4, 10.5, True, phi_deg=90.0)
    fallback = sp.cool_covering_fraction(None, math.nan, True)

    assert inner_fc > outer_fc
    assert sf_fc > passive_fc
    assert minor_fc >= major_fc
    for value in (inner_fc, inner_lo, inner_hi, outer_fc, outer_lo, outer_hi, *fallback):
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
    assert inner_lo <= inner_fc <= inner_hi
    assert outer_lo <= outer_fc <= outer_hi


def test_tau_scat_two_phase_adds_clumpy_cool_component():
    # Hot-only baseline.
    base = sp.tau_scat_ms(0.1, 5.0, 100.0, 0.3)
    assert base is not None and base > 0.0

    # Adding a clumpy cool column increases the predicted scattering.
    two = sp.tau_scat_two_phase(0.1, 5.0, 100.0, 30.0, 0.3, cool_clump_boost=10.0)
    assert two > base

    # Zero cool DM -> reduces exactly to the hot-only screen.
    no_cool = sp.tau_scat_two_phase(0.1, 5.0, 100.0, 0.0, 0.3, cool_clump_boost=10.0)
    assert no_cool == pytest.approx(base)

    # A larger clumpiness boost (F_cool/F_hot) raises the cool contribution.
    more = sp.tau_scat_two_phase(0.1, 5.0, 100.0, 30.0, 0.3, cool_clump_boost=30.0)
    assert more > two

    # A missing/NaN cool column degrades to hot-only, not NaN.
    deg = sp.tau_scat_two_phase(0.1, 5.0, 100.0, float("nan"), 0.3, cool_clump_boost=10.0)
    assert deg == pytest.approx(base)

    # Both columns bad -> None.
    assert sp.tau_scat_two_phase(0.1, 5.0, float("nan"), float("nan"), 0.3) is None

    # No geometric leverage -> 0 from each kernel (build_unified maps this to NaN).
    assert sp.tau_scat_two_phase(0.1, 0.0, 100.0, 30.0, 0.3) == 0.0


def test_r_delta_kpc_matches_overdensity_definition():
    # R200 from r_delta must equal the existing get_rvir_and_rs R200 (both 200*rho_c).
    from galaxies.v2_0.generate_galaxy_plots import get_rvir_and_rs

    m200, z = 1.0e14, 0.2
    r200_ref, _, _ = get_rvir_and_rs(m200, z)
    assert abs(sp.r_delta_kpc(m200, z, 200) - r200_ref) / r200_ref < 1e-6
    # Higher overdensity -> smaller radius for the same mass.
    assert sp.r_delta_kpc(m200, z, 500) < sp.r_delta_kpc(m200, z, 200)
    assert math.isnan(sp.r_delta_kpc(-1.0, z, 200))


def test_dm_cluster_beta_model_matches_analytic_projection_untruncated():
    # For a huge truncation the LOS quadrature must match the closed-form Abel
    # projection of the beta-model: Sigma(b) = ne0 rc sqrt(pi) Gamma(3b/2-1/2)/
    # Gamma(3b/2) [1+(b/rc)^2]^((1-3b)/2). This isolates the quadrature.
    m500, z = 5.0e14, 0.25
    r500 = sp.r_delta_kpc(m500, z, 500)
    beta, rc, b = 0.65, 0.15 * r500, 0.05 * r500
    dm = sp.dm_cluster_beta_model(
        m500,
        z,
        b,
        r500_kpc=r500,
        beta=beta,
        rc_over_r500=0.15,
        f_gas=0.13,
        r_trunc_factor=1000.0,
    )
    ne0 = sp._beta_ne0_cm3(m500, z, r500, beta, rc, f_gas=0.13)
    sigma_kpc_cm3 = (
        ne0
        * rc
        * math.sqrt(math.pi)
        * math.gamma(1.5 * beta - 0.5)
        / math.gamma(1.5 * beta)
        * (1.0 + (b / rc) ** 2) ** (0.5 - 1.5 * beta)
    )
    dm_analytic = sigma_kpc_cm3 * 1000.0 / (1.0 + z)
    assert abs(dm - dm_analytic) / dm_analytic < 0.01


def test_dm_cluster_beta_model_zero_beyond_truncation():
    m500, z = 5.0e14, 0.25
    r500 = sp.r_delta_kpc(m500, z, 500)
    assert sp.dm_cluster_beta_model(m500, z, 1.54 * r500 + 1.0) == 0.0
    assert sp.dm_cluster_beta_model(-1.0, z, 100.0) == 0.0


def test_dm_cluster_beta_model_monotonic_and_linear_in_fgas():
    m500, z = 5.0e14, 0.25
    r500 = sp.r_delta_kpc(m500, z, 500)
    near = sp.dm_cluster_beta_model(m500, z, 0.1 * r500)
    far = sp.dm_cluster_beta_model(m500, z, 0.6 * r500)
    assert near > far > 0.0
    d1 = sp.dm_cluster_beta_model(m500, z, 0.2 * r500, f_gas=0.10)
    d2 = sp.dm_cluster_beta_model(m500, z, 0.2 * r500, f_gas=0.20)
    assert abs(d2 / d1 - 2.0) < 1e-6  # DM linear in gas fraction
