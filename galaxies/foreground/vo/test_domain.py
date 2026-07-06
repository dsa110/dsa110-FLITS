import pytest

from .domain import ForegroundCandidate, Sightline


@pytest.mark.unit
def test_sightline_skycoord():
    sightline = Sightline(name="FRB_TEST", ra=150.0, dec=2.0)

    assert sightline.skycoord.ra.deg == 150.0
    assert sightline.skycoord.dec.deg == 2.0


@pytest.mark.unit
def test_candidate_redshift_presence():
    candidate = ForegroundCandidate(ra=150.0, dec=2.0, z=0.12)

    assert candidate.has_redshift()
