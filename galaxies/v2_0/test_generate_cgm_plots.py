import base64
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from galaxies.v2_0 import generate_cgm_plots as cgm


def _assert_png_b64(b64):
    assert isinstance(b64, str)
    assert len(b64) > 100
    assert base64.b64decode(b64).startswith(b"\x89PNG")


def _sample_unified_df():
    return pd.DataFrame(
        [
            {
                "z": 0.11,
                "impact_kpc": 35.0,
                "z_source": "VII/291/glade",
                "logM_best": 10.7,
                "mass_source": "glade_catalog",
                "R_vir_kpc": 180.0,
                "b_over_rvir": 0.19,
                "intersects_rvir": True,
                "is_star_forming": True,
                "W1mag": 14.0,
                "W2mag": 13.4,
                "W3mag": 11.6,
                "W4mag": np.nan,
                "wise_W1_W2": 0.6,
                "wise_agn_flag": False,
                "pred_mgii_wr": 1.2,
                "cool_fc": 0.55,
                "cool_fc_lo": 0.35,
                "cool_fc_hi": 0.75,
                "pred_tau_scat_ms_1GHz": 2.0e-3,
                "pred_tau_scat_ms_1GHz_lo": 8.0e-4,
                "pred_tau_scat_ms_1GHz_hi": 5.0e-3,
                "scattering_rank": 1,
                "cgm_extractable_flags": {"stellar_mass": "MEASURED", "wise": "MEASURED"},
            },
            {
                "z": 0.18,
                "impact_kpc": 90.0,
                "z_source": "VII/292/north",
                "logM_best": 10.2,
                "mass_source": "desi_ls_sed",
                "R_vir_kpc": 145.0,
                "b_over_rvir": 0.62,
                "intersects_rvir": True,
                "is_star_forming": False,
                "W1mag": 15.3,
                "W2mag": 14.2,
                "W3mag": 12.0,
                "W4mag": np.nan,
                "wise_W1_W2": 1.1,
                "wise_agn_flag": True,
                "pred_mgii_wr": 0.45,
                "cool_fc": 0.25,
                "cool_fc_lo": 0.12,
                "cool_fc_hi": 0.42,
                "pred_tau_scat_ms_1GHz": 5.0e-4,
                "pred_tau_scat_ms_1GHz_lo": 2.0e-4,
                "pred_tau_scat_ms_1GHz_hi": 1.2e-3,
                "scattering_rank": 2,
                "cgm_extractable_flags": "{'stellar_mass': 'PREDICTED', 'wise': 'MEASURED'}",
            },
            {
                "z": 0.24,
                "impact_kpc": 240.0,
                "z_source": "VII/291/glade",
                "logM_best": 9.8,
                "mass_source": "assumed",
                "R_vir_kpc": 120.0,
                "b_over_rvir": 2.0,
                "intersects_rvir": False,
                "is_star_forming": True,
                "W1mag": np.nan,
                "W2mag": np.nan,
                "W3mag": np.nan,
                "W4mag": np.nan,
                "wise_W1_W2": np.nan,
                "wise_agn_flag": False,
                "pred_mgii_wr": 0.2,
                "cool_fc": 0.08,
                "cool_fc_lo": 0.02,
                "cool_fc_hi": 0.16,
                "pred_tau_scat_ms_1GHz": 1.0e-5,
                "pred_tau_scat_ms_1GHz_lo": 2.0e-6,
                "pred_tau_scat_ms_1GHz_hi": 3.0e-5,
                "scattering_rank": 3,
                "cgm_extractable_flags": None,
            },
        ]
    )


def test_make_figures_return_valid_png_base64():
    df = _sample_unified_df()
    assert cgm._coerce_flags(df.loc[0, "cgm_extractable_flags"])["wise"] == "MEASURED"
    assert cgm._coerce_flags(df.loc[1, "cgm_extractable_flags"])["stellar_mass"] == "PREDICTED"

    figs = [
        cgm.make_tau_rank_fig("Testtarget", 0.3, df),
        cgm.make_covering_fraction_fig("Testtarget", df),
        cgm.make_mgii_fig("Testtarget", df),
        cgm.make_wise_diagnostic_fig("Testtarget", df),
    ]

    for fig, b64 in figs:
        _assert_png_b64(b64)
        plt.close(fig)


def test_wise_diagnostic_degrades_when_wise_is_unavailable():
    df = _sample_unified_df()
    df[["W1mag", "W2mag", "W3mag", "W4mag", "wise_W1_W2"]] = np.nan

    fig, b64 = cgm.make_wise_diagnostic_fig("Nowise", df)

    _assert_png_b64(b64)
    plt.close(fig)


def test_tau_rank_degrades_when_no_galaxy_intersects_rvir():
    df = _sample_unified_df()
    df["intersects_rvir"] = False

    fig, b64 = cgm.make_tau_rank_fig("Nointersect", 0.3, df)

    _assert_png_b64(b64)
    plt.close(fig)


def test_build_cgm_html_embeds_target_name():
    df = _sample_unified_df()
    fig_tau, b64_tau = cgm.make_tau_rank_fig("Testtarget", 0.3, df)
    fig_fc, b64_fc = cgm.make_covering_fraction_fig("Testtarget", df)
    fig_mgii, b64_mgii = cgm.make_mgii_fig("Testtarget", df)
    fig_wise, b64_wise = cgm.make_wise_diagnostic_fig("Testtarget", df)

    html = cgm.build_cgm_html(
        [
            {
                "name": "Testtarget",
                "z_frb": 0.3,
                "b64_tau": b64_tau,
                "b64_fc": b64_fc,
                "b64_mgii": b64_mgii,
                "b64_wise": b64_wise,
                "unified_df": df,
            }
        ]
    )

    assert "<!DOCTYPE" in html or "<html" in html
    assert "Testtarget" in html

    for fig in (fig_tau, fig_fc, fig_mgii, fig_wise):
        plt.close(fig)


def test_builders_do_not_touch_docs_index_html():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    docs_index = os.path.join(repo_root, "docs", "index.html")
    before_exists = os.path.exists(docs_index)
    before_mtime = os.path.getmtime(docs_index) if before_exists else None

    df = _sample_unified_df()
    fig_tau, b64_tau = cgm.make_tau_rank_fig("Testtarget", 0.3, df)
    fig_fc, b64_fc = cgm.make_covering_fraction_fig("Testtarget", df)
    fig_mgii, b64_mgii = cgm.make_mgii_fig("Testtarget", df)
    fig_wise, b64_wise = cgm.make_wise_diagnostic_fig("Testtarget", df)
    cgm.build_cgm_html(
        [
            {
                "name": "Testtarget",
                "z_frb": 0.3,
                "b64_tau": b64_tau,
                "b64_fc": b64_fc,
                "b64_mgii": b64_mgii,
                "b64_wise": b64_wise,
                "unified_df": df,
            }
        ]
    )

    after_exists = os.path.exists(docs_index)
    after_mtime = os.path.getmtime(docs_index) if after_exists else None
    assert after_exists == before_exists
    assert after_mtime == before_mtime

    for fig in (fig_tau, fig_fc, fig_mgii, fig_wise):
        plt.close(fig)
