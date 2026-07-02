"""Tests for FRB-compatible host DM predictions."""

from __future__ import annotations

import math
from pathlib import Path

import astropy.units as u
import pytest
import yaml

from galaxies.host.catalog import HostRecord, host_record_for_target, load_host_catalog
from galaxies.host import em
from galaxies.host.dm_predict import (
    dm_host_from_halpha,
    dm_host_from_ssfr,
    dm_host_halo,
    predict_host_dm,
)


def test_em_dm_roundtrip_is_positive():
    sb = 1e-17 * u.erg / u.s / u.arcsec**2
    em_val = em.em_from_halpha(sb, 0.27)
    dm_src = em.dm_from_em(em_val, 1.0 * u.kpc)
    assert em_val.to(u.pc / u.cm**6).value > 0.0
    assert dm_src.to(u.pc / u.cm**3).value > 0.0


def test_dm_host_halo_matches_frb_mnfw_kernel():
    # Observer-frame mNFW column / 2 (FRB Ne_Rperp/2 omits the (1+z)_DM factor our
    # kernel applies for budget consistency with dm_obs).
    val = dm_host_halo(0.0, 10.5, 0.271)
    assert val is not None
    assert val == pytest.approx(27.810043806301092, rel=1e-4)


def test_dm_host_halo_decreases_with_offset():
    inner = dm_host_halo(0.0, 10.5, 0.271)
    outer = dm_host_halo(50.0, 10.5, 0.271)
    assert inner is not None and outer is not None
    assert inner > outer > 0.0


def test_dm_host_from_halpha_requires_finite_inputs():
    assert dm_host_from_halpha(0.27, 1e-16 * u.erg / u.s, 0.5 * u.arcsec) is not None
    assert dm_host_from_halpha(0.27, 1e-16 * u.erg / u.s, 0.0 * u.arcsec) is None


def test_predict_host_dm_with_stellar_mass():
    rec = HostRecord(nickname="phineas", z=0.271, log10_mstar=10.5)
    pred = predict_host_dm(rec)
    assert pred["host_pred_method"] == "halo_mnfw"
    assert pred["dm_host_halo_pred"] == pytest.approx(27.810043806301092, rel=1e-4)
    assert pred["dm_host_pred"] == pred["dm_host_halo_pred"]


def test_predict_host_dm_skips_placeholder_z():
    rec = HostRecord(nickname="freya", z=1.0, z_is_placeholder=True, log10_mstar=10.5)
    pred = predict_host_dm(rec)
    assert pred["host_pred_method"] == "z_placeholder"
    assert pred["dm_host_halo_pred"] is None


def test_load_host_catalog_merges_yaml(tmp_path: Path):
    path = tmp_path / "hosts.yaml"
    path.write_text(
        yaml.dump(
            {
                "hosts": {
                    "phineas": {
                        "log10_mstar": 10.2,
                        "source": "test",
                    }
                }
            }
        )
    )
    cat = load_host_catalog(path)
    assert cat["phineas"].log10_mstar == pytest.approx(10.2)
    assert cat["phineas"].z == pytest.approx(0.271)
    assert cat["zach"].z == pytest.approx(0.043)


def test_host_record_for_target_unknown_raises():
    with pytest.raises(KeyError):
        host_record_for_target("not_a_burst")
