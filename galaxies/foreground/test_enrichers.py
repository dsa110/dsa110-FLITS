import math

import numpy as np
import pandas as pd

from galaxies.foreground import enrichers as enr


def _matches(n=1, catalog=None):
    catalogs = catalog if catalog is not None else ["NED"] * n
    return pd.DataFrame(
        {
            "ra": [10.0] * n,
            "dec": [20.0] * n,
            "z": [0.1] * n,
            "catalog": catalogs,
            "impact_kpc": [50.0] * n,
            "M_star": [math.nan] * n,
        }
    )


def _dec_offset_arcsec(dec_deg, sep_arcsec):
    return dec_deg + sep_arcsec / 3600.0


def test_enrich_with_desi_ls_attaches_nearest_flux_magnitudes_and_records_far_sep():
    matches = pd.DataFrame(
        {
            "ra": [10.0, 30.0],
            "dec": [20.0, 40.0],
            "z": [0.1, 0.2],
            "catalog": ["NED", "NED"],
            "impact_kpc": [50.0, 60.0],
            "M_star": [math.nan, math.nan],
        }
    )

    def fake_desi(coord, radius):
        sep = 0.5 if math.isclose(coord.ra.deg, 10.0) else 3.0
        return pd.DataFrame(
            {
                "ra": [coord.ra.deg],
                "dec": [_dec_offset_arcsec(coord.dec.deg, sep)],
                "type": ["SER"],
                "flux_g": [10.0],
                "flux_r": [20.0],
                "flux_z": [30.0],
                "flux_i": [-9999],
                "flux_w1": [40.0],
                "flux_w2": [50.0],
                "sersic": [1.5],
                "shape_r": [0.7],
            }
        )

    result = enr.enrich_with_desi_ls(matches, engine=fake_desi)

    assert math.isclose(result.loc[0, "desi_ls_gmag"], 22.5 - 2.5 * math.log10(10.0))
    assert math.isclose(result.loc[0, "desi_ls_rmag"], 22.5 - 2.5 * math.log10(20.0))
    assert math.isclose(result.loc[0, "desi_ls_zmag"], 22.5 - 2.5 * math.log10(30.0))
    assert math.isclose(result.loc[0, "desi_ls_w1mag"], 22.5 - 2.5 * math.log10(40.0))
    assert result.loc[0, "desi_ls_type"] == "SER"
    assert math.isclose(result.loc[0, "desi_ls_sep_arcsec"], 0.5, rel_tol=0.0, abs_tol=1.0e-6)
    assert "desi_ls_imag" not in result.columns
    assert np.isnan(result.loc[1, "desi_ls_gmag"])
    assert math.isclose(result.loc[1, "desi_ls_sep_arcsec"], 3.0, rel_tol=0.0, abs_tol=1.0e-6)


def test_enrich_with_desi_ls_empty_engine_adds_nan_columns_without_crashing():
    result = enr.enrich_with_desi_ls(_matches(), engine=lambda coord, radius: pd.DataFrame())

    for column in enr.DESI_LS_COLUMNS:
        assert column in result.columns
    assert result[list(enr.DESI_LS_COLUMNS)].isna().all().all()


def test_enrich_with_desi_ls_none_engine_adds_nan_columns_without_querying():
    result = enr.enrich_with_desi_ls(_matches(), engine=None)

    for column in enr.DESI_LS_COLUMNS:
        assert column in result.columns
    assert result[list(enr.DESI_LS_COLUMNS)].isna().all().all()


def test_enrich_with_allwise_computes_colors_agn_flag_and_rejects_far_candidate():
    matches = pd.DataFrame(
        {
            "ra": [10.0, 30.0, 50.0],
            "dec": [20.0, 40.0, 60.0],
            "z": [0.1, 0.2, 0.3],
            "catalog": ["NED", "NED", "NED"],
            "impact_kpc": [50.0, 60.0, 70.0],
            "M_star": [math.nan, math.nan, math.nan],
        }
    )

    def fake_wise(coord, radius):
        if math.isclose(coord.ra.deg, 10.0):
            sep, w1, w2 = 0.5, 12.0, 11.1
        elif math.isclose(coord.ra.deg, 30.0):
            sep, w1, w2 = 0.5, 12.0, 11.5
        else:
            sep, w1, w2 = 3.0, 12.0, 11.0
        return pd.DataFrame(
            {
                "RAJ2000": [coord.ra.deg],
                "DEJ2000": [_dec_offset_arcsec(coord.dec.deg, sep)],
                "W1mag": [w1],
                "W2mag": [w2],
                "W3mag": [9.0],
                "W4mag": [8.0],
            }
        )

    result = enr.enrich_with_allwise(matches, engine=fake_wise)

    assert math.isclose(result.loc[0, "wise_W1_W2"], 0.9)
    assert result.loc[0, "wise_agn"] is True
    assert result.loc[1, "wise_agn"] is False
    assert np.isnan(result.loc[2, "W1mag"])
    assert pd.isna(result.loc[2, "wise_agn"])


def test_enrich_with_galex_attaches_magnitudes_and_none_engine_degrades_to_nan():
    matches = _matches()

    def fake_galex(coord, radius):
        return pd.DataFrame(
            {
                "RAJ2000": [coord.ra.deg],
                "DEJ2000": [_dec_offset_arcsec(coord.dec.deg, 1.0)],
                "FUVmag": [21.1],
                "NUVmag": [20.4],
                "E(B-V)": [0.03],
            }
        )

    result = enr.enrich_with_galex(matches, engine=fake_galex)
    degraded = enr.enrich_with_galex(matches, engine=None)

    assert math.isclose(result.loc[0, "galex_fuv"], 21.1)
    assert math.isclose(result.loc[0, "galex_nuv"], 20.4)
    assert math.isclose(result.loc[0, "galex_ebv"], 0.03)
    assert degraded[list(enr.GALEX_COLUMNS)].isna().all().all()


def test_enrich_with_xsc_attaches_shape_columns_and_rejects_far_candidate():
    matches = pd.DataFrame(
        {
            "ra": [10.0, 30.0],
            "dec": [20.0, 40.0],
            "z": [0.1, 0.2],
            "catalog": ["NED", "NED"],
            "impact_kpc": [50.0, 60.0],
            "M_star": [math.nan, math.nan],
        }
    )

    def fake_xsc(coord, radius):
        sep = 1.0 if math.isclose(coord.ra.deg, 10.0) else 6.0
        return pd.DataFrame(
            {
                "RAJ2000": [coord.ra.deg],
                "DEJ2000": [_dec_offset_arcsec(coord.dec.deg, sep)],
                "K.K20e": [13.2],
                "Kb/a": [0.6],
                "Kpa": [45.0],
            }
        )

    result = enr.enrich_with_xsc(matches, engine=fake_xsc)

    assert math.isclose(result.loc[0, "xsc_kmag"], 13.2)
    assert math.isclose(result.loc[0, "xsc_axis_ratio"], 0.6)
    assert math.isclose(result.loc[0, "xsc_pa"], 45.0)
    assert np.isnan(result.loc[1, "xsc_kmag"])
    assert math.isclose(result.loc[1, "xsc_sep_arcsec"], 6.0, rel_tol=0.0, abs_tol=1.0e-6)


def test_enrich_with_desi_emission_joins_only_desi_targetids_and_handles_none_engine():
    matches = pd.DataFrame(
        {
            "ra": [10.0, 30.0],
            "dec": [20.0, 40.0],
            "z": [0.1, 0.2],
            "catalog": ["DESI_DR1", "NED"],
            "impact_kpc": [50.0, 60.0],
            "M_star": [math.nan, math.nan],
            "targetid": [12345, 67890],
        }
    )

    def fake_emission(targetid):
        assert targetid == 12345
        return pd.DataFrame({"oii_3727_flux": [7.5], "is_agn": [True]})

    result = enr.enrich_with_desi_emission(matches, engine=fake_emission)
    degraded = enr.enrich_with_desi_emission(matches, engine=None)

    assert math.isclose(result.loc[0, "desi_oii_flux"], 7.5)
    assert result.loc[0, "desi_is_agn"] is True
    assert result.loc[0, "desi_emission_matched"] is True
    assert np.isnan(result.loc[1, "desi_oii_flux"])
    assert result.loc[1, "desi_emission_matched"] is False
    assert degraded[["desi_oii_flux", "desi_halpha_flux", "desi_is_agn"]].isna().all().all()
    assert degraded["desi_emission_matched"].eq(False).all()


def test_enrich_with_desi_emission_without_targetid_column_is_graceful_noop():
    result = enr.enrich_with_desi_emission(_matches(catalog=["DESI_DR1"]), engine=lambda targetid: pd.DataFrame())

    assert result[["desi_oii_flux", "desi_halpha_flux", "desi_is_agn"]].isna().all().all()
    assert result["desi_emission_matched"].eq(False).all()


def test_enrich_all_catalogs_keeps_other_surveys_when_one_engine_raises():
    matches = _matches()

    def raising_engine(coord, radius):
        raise RuntimeError("offline")

    def fake_galex(coord, radius):
        return pd.DataFrame(
            {
                "RAJ2000": [coord.ra.deg],
                "DEJ2000": [_dec_offset_arcsec(coord.dec.deg, 1.0)],
                "FUVmag": [22.2],
                "NUVmag": [21.8],
                "E(B-V)": [0.04],
            }
        )

    result = enr.enrich_all_catalogs(matches, engines={"desi_ls": raising_engine, "galex": fake_galex})

    for columns in (
        enr.DESI_LS_COLUMNS,
        enr.ALLWISE_COLUMNS,
        enr.GALEX_COLUMNS,
        enr.XSC_COLUMNS,
        enr.DESI_EMISSION_COLUMNS,
    ):
        for column in columns:
            assert column in result.columns
    assert result[list(enr.DESI_LS_COLUMNS)].isna().all().all()
    assert math.isclose(result.loc[0, "galex_fuv"], 22.2)
    assert result["desi_emission_matched"].eq(False).all()
