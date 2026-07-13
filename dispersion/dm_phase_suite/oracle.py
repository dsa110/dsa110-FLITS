from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dispersion.dm_campaign.adapters import ADAPTERS
from dispersion.dm_campaign.injection import standard_bright_case

from .search import search_dm


def build_oracle_report(output_path: Path) -> dict:
    """Run deterministic bright fixtures through released and controlled estimators."""
    rows = []
    for instrument in ("chime", "dsa"):
        waterfall, frequency_ghz, dt_ms, truth = standard_bright_case(instrument, seed=0)
        published = ADAPTERS["dm_phase_published"].measure(
            waterfall,
            frequency_ghz,
            dt_ms,
            truth["dm_ref"],
            truth["window"],
        )
        grid = np.arange(-truth["window"], truth["window"] + 0.125, 0.25)
        controlled = search_dm(
            waterfall=waterfall,
            frequencies_mhz=frequency_ghz * 1e3,
            sample_time_s=dt_ms * 1e-3,
            reference_dm=truth["dm_ref"],
            coarse_grid=grid,
            fine_step=0.02,
            f_cut_hz=(50.0, 1500.0),
        )
        row = {
            "fixture": f"standard_bright_{instrument}",
            "instrument": instrument,
            "truth_dm": truth["dm_true"],
            "published_dm": published.dm,
            "published_sigma": published.sigma,
            "controlled_dm": controlled.absolute_dm,
            "published_error": None if published.dm is None else published.dm - truth["dm_true"],
            "controlled_error": controlled.absolute_dm - truth["dm_true"],
            "controlled_minus_published": (
                None if published.dm is None else controlled.absolute_dm - published.dm
            ),
            "controlled_edge_peak": controlled.edge_peak,
        }
        row["pass"] = bool(
            published.dm is not None
            and not controlled.edge_peak
            and abs(row["controlled_error"]) <= (0.05 if instrument == "chime" else 0.20)
            and abs(row["controlled_minus_published"]) <= (0.10 if instrument == "chime" else 0.35)
        )
        rows.append(row)
    report = {
        "oracle": "external/DM_phase at b7cf5fd61436",
        "intentional_deviations": ["ADR-001", "ADR-002", "ADR-003", "ADR-004", "ADR-005"],
        "rows": rows,
        "pass": all(row["pass"] for row in rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return report
