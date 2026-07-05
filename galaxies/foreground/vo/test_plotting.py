import pandas as pd
import pytest

from .plotting import add_offsets


@pytest.mark.unit
def test_add_offsets_joins_targets_and_computes_arcmin_offsets():
    candidates = pd.DataFrame(
        {
            "frb_name": ["FRB_TEST"],
            "ra": [10.1],
            "dec": [20.2],
            "z": [0.1],
            "b_kpc": [100.0],
            "theta_arcmin": [1.0],
            "rank_key": [100.0],
        }
    )
    targets = pd.DataFrame(
        {
            "frb_name": ["FRB_TEST"],
            "frb_ra": [10.0],
            "frb_dec": [20.0],
        }
    )

    out = add_offsets(candidates, targets)

    assert abs(out.loc[0, "ddec_arcmin"] - 12.0) < 1e-10
    assert 5.6 < out.loc[0, "dra_arcmin"] < 5.7
