import json

import numpy as np
import pandas as pd
import pytest

from .halos import (
    ADAPTERS,
    add_halo_masses,
    kband_to_mstar,
    mstar_to_mhalo,
)


@pytest.fixture(scope="module")
def cosmo():
    from astropy.cosmology import Planck18

    return Planck18


@pytest.mark.unit
def test_kband_to_mstar_milky_way_like():
    # K_s ~ 4 (apparent at d=10 Mpc -> M_K ~ -25.5), expect ~10^11 M_sun ballpark.
    mstar = kband_to_mstar(kmag_apparent=4.0, distance_mpc=10.0)
    assert 1e10 < mstar < 1e12


@pytest.mark.unit
def test_mstar_to_mhalo_round_trips_at_z0():
    mstar_in = 1.0e10
    mh = mstar_to_mhalo(mstar_in, z=0.0)
    # invert SHMR forward at z=0 to recover mstar
    M1, N, beta, gamma = (
        10**11.590,
        0.0351,
        1.376,
        0.608,
    )
    ratio = mh / M1
    sm_over_mh = 2.0 * N / (ratio**-beta + ratio**gamma)
    mstar_out = mh * sm_over_mh
    assert abs(mstar_out - mstar_in) / mstar_in < 0.01


@pytest.mark.unit
def test_glade2_adapter_missing_kmag_yields_nan(cosmo):
    df = pd.DataFrame(
        {
            "RAJ2000": [150.0, 150.1],
            "DEJ2000": [2.0, 2.0],
            "z": [0.05, 0.05],
            "Kmag": [12.0, np.nan],
            "Dist": [200.0, 200.0],
        }
    )
    out = add_halo_masses(df, ADAPTERS["glade2"], cosmo=cosmo)
    assert np.isfinite(out.loc[0, "m_halo"])
    assert np.isnan(out.loc[1, "m_halo"])
    prov0 = json.loads(out.loc[0, "mass_provenance"])
    assert prov0["mass_method"] == "kband_bell03_moster13"
    assert prov0["ml_assumption"] == "Bell03_K_0p6"
    assert prov0["mstar_source"] == "Kmag"
    assert prov0["shmr"] == "moster13_table1"
    assert prov0["delta_def"] == 200.0
    assert json.loads(out.loc[1, "mass_provenance"]) is None


@pytest.mark.unit
def test_glade2_adapter_no_kmag_column_does_not_raise(cosmo):
    df = pd.DataFrame({"RAJ2000": [150.0], "DEJ2000": [2.0], "z": [0.05], "Dist": [200.0]})
    out = add_halo_masses(df, ADAPTERS["glade2"], cosmo=cosmo)
    assert np.isnan(out.loc[0, "m_halo"])
    assert json.loads(out.loc[0, "mass_provenance"]) is None


@pytest.mark.unit
def test_glade_plus_direct_mstar_overrides_kband(cosmo):
    df = pd.DataFrame(
        {
            "RAJ2000": [150.0],
            "DEJ2000": [2.0],
            "zhelio": [0.05],
            "M*": [10.0],  # 10 * 1e10 = 1e11 M_sun
            "Kmag": [12.0],
            "dL": [200.0],
        }
    )
    out = add_halo_masses(df, ADAPTERS["glade-plus"], cosmo=cosmo)
    prov = json.loads(out.loc[0, "mass_provenance"])
    assert prov["mass_method"] == "direct_mstar_moster13"
    assert prov["mstar_source"] == "M*"
    assert prov["ml_assumption"] is None


@pytest.mark.unit
def test_glade_plus_falls_back_to_kband_when_mstar_null(cosmo):
    df = pd.DataFrame(
        {
            "RAJ2000": [150.0],
            "DEJ2000": [2.0],
            "zhelio": [0.05],
            "M*": [np.nan],
            "Kmag": [12.0],
            "dL": [200.0],
        }
    )
    out = add_halo_masses(df, ADAPTERS["glade-plus"], cosmo=cosmo)
    assert np.isfinite(out.loc[0, "m_halo"])
    prov = json.loads(out.loc[0, "mass_provenance"])
    assert prov["mass_method"] == "kband_bell03_moster13"


@pytest.mark.unit
def test_hecate_v2_log_mstar(cosmo):
    df = pd.DataFrame(
        {
            "RAJ2000": [150.0, 150.1],
            "DEJ2000": [2.0, 2.0],
            "HRV": [3000.0, 3000.0],  # km/s -> z ~ 0.01
            "logMstarHEC": [10.5, np.nan],
            "Dist": [40.0, 40.0],
            "q_logMstarHEC": [0, 1],  # row 1 flagged unreliable
        }
    )
    out = add_halo_masses(df, ADAPTERS["hecate-v2"], cosmo=cosmo)
    assert np.isfinite(out.loc[0, "m_halo"])
    assert np.isnan(out.loc[1, "m_halo"])
    prov0 = json.loads(out.loc[0, "mass_provenance"])
    assert prov0["mass_method"] == "log_mstar_moster13"


@pytest.mark.unit
def test_two_mrs_kband_with_cz_distance(cosmo):
    df = pd.DataFrame(
        {
            "RAJ2000": [150.0],
            "DEJ2000": [2.0],
            "cz": [3000.0],  # km/s
            "Kcmag": [9.0],
        }
    )
    out = add_halo_masses(df, ADAPTERS["two-mrs"], cosmo=cosmo)
    assert np.isfinite(out.loc[0, "m_halo"])
    prov = json.loads(out.loc[0, "mass_provenance"])
    assert prov["mass_method"] == "kband_bell03_moster13"
    assert prov["mstar_source"] == "Kcmag"


@pytest.mark.unit
def test_glade_plus_z_type_from_f_dL(cosmo):
    """GLADE+'s f_dL is the spec/photo provenance flag (per glade.elte.hu):

      0 -> no measured z/distance (z_col will be NaN -> z_type='none')
      1 -> measured photometric redshift -> 'photo'
      2 -> distance measured first, z back-computed -> falls through to
           z_type_default ('unknown')
      3 -> measured spectroscopic redshift -> 'spec'

    Earlier wiring used f_zcmb (the peculiar-velocity correction flag,
    NOT spec/photo); this test pins the corrected mapping.
    """
    df = pd.DataFrame(
        {
            "RAJ2000": [150.0, 150.05, 150.1, 150.15],
            "DEJ2000": [2.0, 2.05, 2.1, 2.15],
            "zhelio": [0.05, 0.05, 0.05, 0.05],
            "M*": [1.0, 1.0, 1.0, 1.0],
            "dL": [200.0, 200.0, 200.0, 200.0],
            "f_dL": [3, 1, 2, 0],
        }
    )
    out = add_halo_masses(df, ADAPTERS["glade-plus"], cosmo=cosmo)
    assert out.loc[0, "z_type"] == "spec"  # f_dL=3
    assert out.loc[1, "z_type"] == "photo"  # f_dL=1
    # f_dL=2 (distance-derived z) is not 'spec' or 'photo' — falls through
    # to z_type_default, which for glade-plus is "unknown".
    assert out.loc[2, "z_type"] == "unknown"
    # f_dL=0 explicitly means no z; the GLADE+ row would have NaN zhelio
    # in practice. Here we set zhelio=0.05 to isolate the flag mapping;
    # the row gets z_type_default ("unknown") because f_dL=0 doesn't
    # match either spec_value (3) or photo_value (1).
    assert out.loc[3, "z_type"] == "unknown"


@pytest.mark.unit
def test_glade_plus_z_type_does_not_use_f_zcmb(cosmo):
    """f_zcmb (peculiar-velocity correction flag) must NOT drive z_type.

    Regression for the upstream-doc bug: an adapter that wired f_zcmb=0/1 to
    spec/photo would have flipped the labels for every row (because f_zcmb=0
    means 'NOT corrected for peculiar velocity', not 'spectroscopic'). This
    test makes sure that even if f_zcmb is present, z_type still reflects
    f_dL (or falls through to default when f_dL is missing).
    """
    df = pd.DataFrame(
        {
            "RAJ2000": [150.0, 150.05],
            "DEJ2000": [2.0, 2.05],
            "zhelio": [0.05, 0.05],
            "M*": [1.0, 1.0],
            "dL": [200.0, 200.0],
            # f_zcmb values flipped relative to a spec/photo encoding —
            # if the adapter were still using f_zcmb, row 0 would be 'spec'
            # and row 1 would be 'photo'. Under the corrected mapping (f_dL
            # is absent), both rows fall to z_type_default = 'unknown'.
            "f_zcmb": [1, 0],
        }
    )
    out = add_halo_masses(df, ADAPTERS["glade-plus"], cosmo=cosmo)
    assert out.loc[0, "z_type"] == "unknown"
    assert out.loc[1, "z_type"] == "unknown"


@pytest.mark.unit
def test_glade2_z_type_unknown_when_adapter_has_no_flag(cosmo):
    """Adapters with no flag column AND no spec default emit 'unknown'."""
    df = pd.DataFrame(
        {
            "RAJ2000": [150.0],
            "DEJ2000": [2.0],
            "z": [0.05],
            "Kmag": [12.0],
            "Dist": [200.0],
        }
    )
    out = add_halo_masses(df, ADAPTERS["glade2"], cosmo=cosmo)
    assert out.loc[0, "z_type"] == "unknown"


@pytest.mark.unit
def test_two_mrs_z_type_default_is_spec(cosmo):
    """2MRS measures cz spectroscopically for every row; default z_type='spec'.

    Without this default, host-confusion logic would treat 2MRS rows as
    photo-z-equivalent and apply the broader 0.05 Δz tolerance — silently
    excluding real spec-z foreground galaxies that happen to lie within
    photo-z scatter of the FRB host's z.
    """
    df = pd.DataFrame(
        {
            "RAJ2000": [150.0],
            "DEJ2000": [2.0],
            "cz": [3000.0],
            "Kcmag": [9.0],
        }
    )
    out = add_halo_masses(df, ADAPTERS["two-mrs"], cosmo=cosmo)
    assert out.loc[0, "z_type"] == "spec"


@pytest.mark.unit
def test_hecate_v2_z_type_default_is_spec(cosmo):
    """HECATEv2's local-volume redshifts are predominantly spec-z; default 'spec'."""
    df = pd.DataFrame(
        {
            "RAJ2000": [150.0],
            "DEJ2000": [2.0],
            "HRV": [3000.0],
            "logMstarHEC": [10.5],
            "Dist": [40.0],
            "q_logMstarHEC": [0],
        }
    )
    out = add_halo_masses(df, ADAPTERS["hecate-v2"], cosmo=cosmo)
    assert out.loc[0, "z_type"] == "spec"


@pytest.mark.unit
def test_z_type_none_when_redshift_missing(cosmo):
    """A row with NaN in the redshift column gets z_type='none' regardless of
    flag column or default.
    """
    df = pd.DataFrame(
        {
            "RAJ2000": [150.0],
            "DEJ2000": [2.0],
            "zhelio": [np.nan],
            "M*": [1.0],
            "dL": [200.0],
            "f_dL": [3],  # would otherwise be 'spec'
        }
    )
    out = add_halo_masses(df, ADAPTERS["glade-plus"], cosmo=cosmo)
    assert out.loc[0, "z_type"] == "none"
