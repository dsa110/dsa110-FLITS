"""Tests for survey footprint and coverage classification."""

from astropy.coordinates import SkyCoord

from galaxies.foreground.survey_coverage import (
    classify_coverage,
    survey_in_footprint,
)
from galaxies.foreground.survey_footprint_mocs import _full_sky_moc, moc_sky_area_deg2, rasterize_moc


def test_high_latitude_in_northern_surveys():
    coord = SkyCoord("11h51m07.52s +71d41m44.3s", unit=(("hourangle", "deg")))
    assert survey_in_footprint("NED", coord)
    assert survey_in_footprint("GLADE+", coord)
    assert survey_in_footprint("DESI_DR8_NORTH", coord)
    assert survey_in_footprint("SDSS_DR12", coord)
    assert survey_in_footprint("CLUSTERS", coord)


def test_classify_coverage_states():
    assert classify_coverage(in_footprint=False, raw_count=0, foreground_count=0) == "no_footprint"
    assert classify_coverage(in_footprint=True, raw_count=0, foreground_count=0) == "footprint_empty"
    assert classify_coverage(in_footprint=True, raw_count=10, foreground_count=0) == "catalog_hits"
    assert classify_coverage(in_footprint=True, raw_count=10, foreground_count=2) == "foreground"


def test_full_sky_moc_rasterizes():
    moc = _full_sky_moc(order=4)
    assert moc_sky_area_deg2(moc) > 40000
    hmap = rasterize_moc(moc, order=4)
    assert (hmap == 1.0).all()
