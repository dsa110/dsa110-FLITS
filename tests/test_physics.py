import numpy as np
from flits.scintillation.physics import (
    scintillation_bandwidth_to_timescale,
    screen_distance_from_curvature,
    interpret_modulation_index,
    estimate_emission_region_size,
    two_screen_coherence_constraint,
)

def test_scintillation_bandwidth_to_timescale_coefficients():
    # Test standard coefficient C = 1.0
    tau_1 = scintillation_bandwidth_to_timescale(1e6, 1400.0, alpha=4.0, coefficient=1.0)
    assert np.isclose(tau_1, 1.0 / (2.0 * np.pi * 1e6))

    # Test Kolmogorov thin screen coefficient C = 1.16
    tau_kolm = scintillation_bandwidth_to_timescale(1e6, 1400.0, alpha=4.0, coefficient=1.16)
    assert np.isclose(tau_kolm, 1.16 / (2.0 * np.pi * 1e6))
    assert tau_kolm > tau_1

    # Test invalid bandwidth returning NaN
    assert np.isnan(scintillation_bandwidth_to_timescale(-10, 1400.0))

def test_screen_distance_velocity_scaling():
    # Test default V_eff = 100 km/s
    curvature = 1.0  # s^3
    freq_ghz = 1.4
    dist_100 = screen_distance_from_curvature(curvature, freq_ghz, v_eff_kms=100.0)

    # Test unscaled legacy behavior V_eff = 1 m/s (1e-3 km/s)
    dist_legacy = screen_distance_from_curvature(curvature, freq_ghz, v_eff_kms=1e-3)

    # Output distance should scale as V_eff^2
    # V_eff ratio is 100 / 1e-3 = 1e5, so distance ratio is 1e10
    assert np.isclose(dist_100 / dist_legacy, 1e10)

def test_screen_distance_root_selection():
    # Setup values that yield two valid roots
    # Let D_S = 10 Mpc (10^7 pc) and D_eff = 10^5 pc (which is less than D_S/4 = 2.5 * 10^6 pc)
    source_dist_mpc = 10.0
    freq_ghz = 1.4

    # We want to find a curvature that gives exactly d_eff = 10^5 pc
    # d_eff_m = 10^5 * parsec
    # d_eff_m = 2.0 * curvature * (V_eff_mps^2) * (freq_hz^2) / c
    # Solve for curvature:
    import scipy.constants as cons
    d_eff_m = 1e5 * cons.parsec
    v_eff_mps = 100.0 * 1000.0
    freq_hz = freq_ghz * 1e9
    curvature = d_eff_m * cons.c / (2.0 * (v_eff_mps ** 2) * (freq_hz ** 2))

    # Get the smaller root (observer-local)
    d_L_obs = screen_distance_from_curvature(
        curvature, freq_ghz, source_dist_mpc=source_dist_mpc, v_eff_kms=100.0, select_host_root=False
    )

    # Get the larger root (host-galaxy local)
    d_L_host = screen_distance_from_curvature(
        curvature, freq_ghz, source_dist_mpc=source_dist_mpc, v_eff_kms=100.0, select_host_root=True
    )

    assert d_L_obs < d_L_host
    assert np.isclose(d_L_obs + d_L_host, source_dist_mpc * 1e6)

def test_interpret_modulation_index():
    # Test point source
    res1 = interpret_modulation_index(0.98)
    assert res1["resolution_regime"] == "unresolved"
    assert not res1["emission_resolved"]

    # Test marginally resolved
    res2 = interpret_modulation_index(0.85)
    assert res2["resolution_regime"] == "marginally_resolved"
    assert res2["emission_resolved"]

    # Test partially resolved
    res3 = interpret_modulation_index(0.5)
    assert res3["resolution_regime"] == "partially_resolved"

    # Test invalid values
    assert "Invalid" in interpret_modulation_index(float("nan"))["interpretation"]

def test_estimate_emission_region_size():
    # FRB 20221022A parameters from Nimmo et al.
    res = estimate_emission_region_size(
        m=0.78, delta_nu_dc_mhz=0.124, d_source_screen_pc=11000,
        freq_mhz=600, m_err=0.07
    )
    assert res["R_obs_km"] > 0
    assert res["chi_km"] > 0
    assert not res["is_upper_limit"]
    assert "Estimated" in res["physical_context"]

    # Unresolved edge case
    res_unres = estimate_emission_region_size(
        m=1.05, delta_nu_dc_mhz=0.124, d_source_screen_pc=11000,
        freq_mhz=600
    )
    assert res_unres["is_upper_limit"]
    assert res_unres["R_obs_km"] == res_unres["chi_km"]

def test_two_screen_coherence_constraint():
    res = two_screen_coherence_constraint(
        delta_nu_1_mhz=0.006, delta_nu_2_mhz=0.124,
        freq_mhz=600, d_source_mpc=65.189
    )
    assert res["d_product_kpc2"] > 0
    assert len(res["example_constraints"]) == 5
    assert "d_gal_0.64kpc" in res["example_constraints"]

