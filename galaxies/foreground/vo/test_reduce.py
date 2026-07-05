import numpy as np
import pandas as pd
import pytest
from astropy.cosmology import Planck18

from .reduce import (
    angular_separation_arcmin,
    compute_rdelta_from_mdelta,
    impact_parameter_kpc,
    merge_and_rank,
)


@pytest.mark.unit
def test_angular_separation_scalar():
    assert abs(angular_separation_arcmin(150.0, 2.0, 150.0, 2.0)) < 1e-10
    # ~1 degree = 60 arcmin along RA at low dec
    assert abs(angular_separation_arcmin(150.0, 2.0, 151.0, 2.0) - 60.0) < 1.0


@pytest.mark.unit
def test_angular_separation_vectorized():
    res = angular_separation_arcmin(np.array([10.0, 10.01]), np.array([0.0, 0.0]), 10.0, 0.0)
    assert res.shape == (2,)
    assert res[0] < res[1]


@pytest.mark.unit
def test_impact_parameter_scale():
    # 1 arcmin at z=0.1: a few hundred kpc
    b = impact_parameter_kpc(1.0, 0.1, Planck18)
    assert 100 < b < 1000


@pytest.mark.unit
def test_impact_parameter_monotonicity():
    b = impact_parameter_kpc(np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.2, 0.2]), Planck18)
    assert b[0] < b[1] < b[2]


@pytest.mark.unit
def test_rdelta_formula():
    # M200 = 1e14 Msun at z=0.1: halo-scale radius (hundreds of kpc to few Mpc)
    r200 = compute_rdelta_from_mdelta(1e14, 200.0, 0.1, Planck18)
    assert 500 < r200 < 5000
    assert compute_rdelta_from_mdelta(5e14, 500.0, 0.3, Planck18) > 0


def _schema_df(**overrides):
    base = {
        "name": ["G1", "G2"],
        "id": ["1", "2"],
        "ra": [10.0, 10.01],
        "dec": [0.0, 0.0],
        "z": [0.1, 0.2],
        "z_type": ["spec", "spec"],
        "richness": [None, None],
        "m_delta": [None, None],
        "r_delta": [None, None],
        "delta_def": [None, None],
        "service": ["svc", "svc"],
        "table": ["tab", "tab"],
        "provenance_json": ["{}", "{}"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


@pytest.mark.unit
def test_merge_and_rank_b_only():
    out = merge_and_rank(_schema_df(), frb_ra_deg=10.0, frb_dec_deg=0.0, cosmo=Planck18)
    assert np.isfinite(out.loc[0, "b_kpc"]) and out.loc[0, "b_kpc"] <= out.loc[1, "b_kpc"]
    assert out["rank_key"].is_monotonic_increasing
    assert out["theta_arcmin"].iloc[0] < out["theta_arcmin"].iloc[1]


@pytest.mark.unit
def test_merge_and_rank_with_mdelta():
    df = _schema_df(
        name=["C", "D"], id=["3", "4"], ra=[10.02, 10.03], z=[0.3, 0.3],
        m_delta=[1e14, None], delta_def=[200.0, None],
    )
    out = merge_and_rank(df, frb_ra_deg=10.0, frb_dec_deg=0.0, cosmo=Planck18)
    computed = out.loc[out["name"] == "C", "r_delta_computed"].iloc[0]
    assert np.isfinite(computed) and computed > 0


@pytest.mark.unit
def test_merge_and_rank_columns(sample_normalized_result):
    out = merge_and_rank(sample_normalized_result, 150.115, 2.205, Planck18)
    for col in ("theta_arcmin", "b_kpc", "r_delta_computed", "rank_key"):
        assert col in out.columns
