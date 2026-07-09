from __future__ import annotations

import numpy as np
import pytest

import flits.batch.codetection_plots as plots
from flits.batch.codetection_plots import _OUTBOARD_GRID_LEFT, BandSpectrum


def _band(f0: float, f1: float, label: str) -> BandSpectrum:
    time_ms = np.linspace(-2.0, 3.0, 6)
    freq_mhz = np.linspace(f0, f1, 4)
    data = np.ones((freq_mhz.size, time_ms.size))
    return BandSpectrum(
        freq_mhz=freq_mhz,
        time_ms=time_ms,
        data=data,
        model=data.copy(),
        sigma=1.0,
        label=label,
    )


def test_strip_label_plan_reserves_spectrum_margin_without_changing_data_window():
    bands = [_band(400.0, 800.0, "CHIME"), _band(1311.0, 1498.75, "DSA")]
    plan_layout = getattr(plots, "_plan_layout", None)

    assert plan_layout is not None
    layout = plan_layout(
        bands,
        np.linspace(-2.0, 3.0, 6),
        symmetric_time_axis=False,
        band_labels=True,
        band_label_style="strip",
    )

    assert layout.xlim == pytest.approx((-2.0, 3.0))
    assert layout.data_xlim == pytest.approx((-2.0, 3.0))
    assert layout.ylim == pytest.approx((400.0, 1498.75))
    assert layout.gaps == pytest.approx([(800.0, 1311.0)])
    assert layout.grid_left == pytest.approx(0.08)
    assert layout.grid_right == pytest.approx(0.90)
    assert layout.labels.style == "strip"
    assert layout.labels.draw_on_spectrum
    assert not layout.labels.draw_on_waterfall


def test_outboard_label_plan_only_expands_left_margin():
    bands = [_band(400.0, 800.0, "CHIME"), _band(1311.0, 1498.75, "DSA")]
    plan_layout = getattr(plots, "_plan_layout", None)

    assert plan_layout is not None
    layout = plan_layout(
        bands,
        np.linspace(-2.0, 3.0, 6),
        symmetric_time_axis=False,
        band_labels=True,
        band_label_style="outboard",
    )

    assert layout.xlim == pytest.approx((-2.0, 3.0))
    assert layout.grid_left == pytest.approx(_OUTBOARD_GRID_LEFT)
    assert layout.grid_right == pytest.approx(0.985)
    assert layout.labels.style == "outboard"
    assert layout.labels.draw_on_waterfall
    assert not layout.labels.draw_on_spectrum


def test_enabled_band_labels_reject_unknown_style():
    bands = [_band(400.0, 800.0, "CHIME")]
    plan_layout = getattr(plots, "_plan_layout", None)

    assert plan_layout is not None
    with pytest.raises(ValueError, match="band_label_style"):
        plan_layout(
            bands,
            np.linspace(-2.0, 3.0, 6),
            symmetric_time_axis=False,
            band_labels=True,
            band_label_style="left-ish",
        )
