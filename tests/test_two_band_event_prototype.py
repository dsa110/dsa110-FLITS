from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_prototype():
    path = (
        Path(__file__).resolve().parents[1]
        / "analysis/scintillation-dsa-lorentzian-2026-07-07/prototype_two_band_event.py"
    )
    spec = importlib.util.spec_from_file_location("two_band_event_prototype", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prototype = _load_prototype()


def _result(band: str, *, diagnostic: bool = False):
    frequencies = (500.0, 700.0) if band == "chime" else (1300.0, 1450.0)
    reference_frequency = 500.0 if band == "chime" else 1300.0
    subbands = []
    for index, frequency in enumerate(frequencies):
        subbands.append(
            {
                "index": index,
                "center_freq_mhz": frequency,
                "selected_components": [
                    {
                        "dnu_mhz": 0.2 * (frequency / reference_frequency) ** 4,
                        "dnu_err": 0.01,
                        "quality_flags": [],
                    }
                ],
            }
        )
    return {
        "subbands": subbands,
        "measurement_status": "diagnostic_only" if diagnostic else "measurement",
    }


def _payload(frequency: float):
    lags = np.linspace(-1.0, 1.0, 41)
    width = 0.2
    modulation = 0.8
    acf = modulation**2 / (1.0 + (lags / width) ** 2)
    return {
        "lags": lags,
        "acf": acf,
        "err": np.full_like(lags, 0.01),
        "summary": {"center_freq_mhz": frequency, "fit_range_mhz": 1.0},
        "fit": {
            "constant": 0.0,
            "components": [{"dnu_mhz": width, "m": modulation}],
        },
    }


def test_component_rows_preserve_quality_and_assign_tracks():
    result = _result("dsa")
    result["subbands"][0]["selected_components"].append(
        {"dnu_mhz": 8.0, "dnu_err": 1.0, "quality_flags": ["broad"]}
    )

    rows = prototype.component_rows(result, "dsa")

    assert [(row["gamma_track"], row["usable"]) for row in rows[:2]] == [
        (1, True),
        (2, False),
    ]


def test_representative_selects_two_payloads_nearest_the_median():
    payloads = [_payload(frequency) for frequency in (400.0, 500.0, 700.0, 800.0)]

    selected = prototype.representative(payloads)

    assert [payload["summary"]["center_freq_mhz"] for payload in selected] == [500.0, 700.0]


def test_build_figure_marks_diagnostic_chime_and_renders(tmp_path):
    chime_payloads = [_payload(frequency) for frequency in (450.0, 550.0, 650.0)]
    dsa_payloads = [_payload(frequency) for frequency in (1250.0, 1350.0, 1450.0)]

    fig, metadata = prototype.build_figure(
        dsa=_result("dsa"),
        dsa_payloads=dsa_payloads,
        chime=_result("chime", diagnostic=True),
        chime_payloads=chime_payloads,
    )

    labels = [text.get_text() for axis in fig.axes for text in axis.texts]
    assert metadata["status"] == "diagnostic_only"
    assert metadata["included_diagnostic_chime"] is True
    assert metadata["acf_panel_count"] == 4
    assert any("not a qualified measurement" in label for label in labels)
    assert any("diagnostic" in label and "alpha" in label for label in labels)

    png = tmp_path / "prototype.png"
    pdf = tmp_path / "prototype.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0
