"""Run the uniform adaptive arrival-regression grid on all 24 products.

Outputs candidate-level results, selected per-band measurements, event-level
inverse-variance summaries, a resolution-stability contact sheet, and a memo.
The event summary is explicitly not a direct cross-band fit: the stored arrays
have independent time origins, so they cannot exploit the inter-telescope
frequency lever arm without restored absolute timing metadata.

Usage (from pipeline/):
    conda run -n flits python -m dispersion.dm_campaign.run_adaptive_arrival
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import yaml  # noqa: E402

from dispersion.dm_campaign.adaptive_arrival import (  # noqa: E402
    combine_event_measurements,
    evaluate_product,
    product_path,
    select_stable_candidate,
)
from dispersion.dm_power_analysis import load_manifest_rows  # noqa: E402

_CONFIG = Path(__file__).parent / "configs" / "battery.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_provenance(rows: list[dict], config: dict, config_path: Path,
                         root: Path, out_path: Path, mode: str) -> None:
    import numpy
    import scipy

    inputs = []
    for row in rows:
        path = product_path(row, config)
        inputs.append({
            "burst": row["burst"], "telescope": row["telescope"],
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    diff = subprocess.run(
        ["git", "diff", "--", "dispersion/dm_campaign", "tests/test_adaptive_arrival.py"],
        cwd=root, check=True, capture_output=True,
    ).stdout
    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "command": "python -m dispersion.dm_campaign.run_adaptive_arrival"
                   + (" --replot" if mode == "replot" else ""),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
        "code_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_files": [
            {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
            for path in (
                root / "dispersion/dm_campaign/adaptive_arrival.py",
                root / "dispersion/dm_campaign/run_adaptive_arrival.py",
                root / "dispersion/dm_campaign/configs/battery.yaml",
                root / "tests/test_adaptive_arrival.py",
            )
        ],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "candidate_grid_sha256": sha256_file(out_path.parent / "candidate_grid.json"),
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": numpy.__version__, "scipy": scipy.__version__,
        },
        "inputs": inputs,
    }
    out_path.write_text(json.dumps(payload, indent=1) + "\n")


def _run_job(job):
    row, config = job
    return row["burst"], row["telescope"], evaluate_product(row, config)


def _select_all(candidate_records: list[dict], config: dict) -> list[dict]:
    adaptive = config["adaptive"]
    selected = []
    for record in candidate_records:
        choice = select_stable_candidate(
            record["candidates"],
            sigma_max=float(adaptive["sigma_max"]),
            stability_dm=float(adaptive["stability_dm"]),
        )
        selected.append({"burst": record["burst"], "telescope": record["telescope"], **choice})
    return selected


def _event_summaries(selected: list[dict]) -> list[dict]:
    bursts = sorted({row["burst"] for row in selected})
    by = {(row["burst"], row["telescope"]): row for row in selected}
    return [
        combine_event_measurements(
            burst,
            {tel: by[(burst, tel)] for tel in ("chime", "dsa") if (burst, tel) in by},
        )
        for burst in bursts
    ]


def plot_resolution_contact_sheet(candidate_records: list[dict], selected: list[dict], path: Path):
    """One panel per product: residual DM versus effective time resolution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    choice = {(row["burst"], row["telescope"]): row for row in selected}
    records = sorted(candidate_records, key=lambda r: (r["burst"], r["telescope"]))
    fig, axes = plt.subplots(6, 4, figsize=(14, 18), squeeze=False)
    colors = {3: "#4477aa", 4: "#228833", 6: "#cc6677", 8: "#aa3377"}
    for ax, record in zip(axes.flat, records, strict=True):
        candidates = [c for c in record["candidates"] if c["dm"] is not None]
        selected_row = choice[(record["burst"], record["telescope"])]
        ticks = sorted({c["dt_ms"] for c in candidates})
        x_position = {value: index for index, value in enumerate(ticks)}
        weak = [c for c in candidates if c["sigma"] > 0.5]
        if weak:
            ax.plot(
                [x_position[r["dt_ms"]] for r in weak],
                [r["dm"] - r["product_dm"] for r in weak],
                ".", ms=2, alpha=0.25, color="0.5", label="weak",
            )
        for nsub in (3, 4, 6, 8):
            rows = [
                c for c in candidates
                if c["n_subband"] == nsub and c["sigma"] <= 0.5
            ]
            if not rows:
                continue
            ax.errorbar(
                [x_position[r["dt_ms"]] for r in rows],
                [r["dm"] - r["product_dm"] for r in rows],
                yerr=[r["sigma"] for r in rows],
                fmt="o", ms=2.5, lw=0.5, alpha=0.55, color=colors[nsub], label=f"{nsub} sb",
            )
        if selected_row["dm"] is not None:
            resolution = selected_row["selected_resolution"]
            ax.plot(
                x_position[resolution["dt_ms"]], selected_row["residual"],
                marker="*" if selected_row["status"] == "science-grade" else "x",
                ms=10 if selected_row["status"] == "science-grade" else 7,
                color="black" if selected_row["status"] == "science-grade" else "#cc3311",
                zorder=5,
            )
        ax.axhline(0, color="0.7", lw=0.6)
        if selected_row["dm"] is not None:
            half_range = max(
                0.1,
                3.0 * selected_row["sigma"],
                1.25 * selected_row["cluster_span_dm"],
            )
            ax.set_ylim(selected_row["residual"] - half_range,
                        selected_row["residual"] + half_range)
        else:
            ax.set_ylim(-20, 20)
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels([f"{tick:.3g}" for tick in ticks])
        ax.set_title(
            f"{record['burst']} / {record['telescope']} - {selected_row['status']}", fontsize=8
        )
        ax.tick_params(labelsize=6)
        ax.set_xlabel("effective dt [ms]", fontsize=7)
        ax.set_ylabel("DM - product DM", fontsize=7)
    axes[0, 0].legend(fontsize=6, frameon=False, ncol=2)
    fig.suptitle(
        "Adaptive arrival regression: identical resolution grid on all products\n"
        "black star = science-grade PASS; red x = MARGINAL candidate; error bars = quoted sigma",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_memo(selected: list[dict], events: list[dict], config: dict, path: Path):
    by_event = {row["burst"]: row for row in events}
    lines = [
        "# Adaptive arrival-regression results", "",
        "The same time/frequency/sub-band grid and stability policy was evaluated",
        "on all 24 products. A per-band result is science-grade only when at least",
        "two distinct resolution choices agree within "
        f"{config['adaptive']['stability_dm']} pc/cm3 and sigma_DM <= "
        f"{config['adaptive']['sigma_max']} pc/cm3. Science-grade clusters contain",
        "only candidates for which the canonical reduced-chi2 classifier returns PASS.",
        "MARGINAL fits retain their candidate DM for audit",
        "but are excluded from event-level summaries.", "",
        "The event-level inverse-variance summaries below are not direct cross-band",
        "fits. The stored CHIME and DSA arrays have independent time origins; a fit",
        "that exploits the 0.4--1.5 GHz lever arm remains blocked until the absolute",
        "per-array time origins are restored and verified.", "",
        "| burst | CHIME | DSA | event support | event DM |", "|---|---|---|---|---:|",
    ]
    by = {(row["burst"], row["telescope"]): row for row in selected}
    for burst in sorted(by_event):
        event = by_event[burst]
        cells = []
        for tel in ("chime", "dsa"):
            row = by[(burst, tel)]
            if row["dm"] is None:
                cells.append(row["status"])
            else:
                cells.append(f"{row['status']}: {row['dm']:.6f} +/- {row['sigma']:.6f}")
        event_dm = "--" if event["dm"] is None else f"{event['dm']:.6f} +/- {event['sigma']:.6f}"
        lines.append(
            f"| {burst} | {cells[0]} | {cells[1]} | {event['support']} | {event_dm} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=_CONFIG)
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(config["adaptive"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_rows(root)

    candidate_path = out_dir / "candidate_grid.json"
    if args.replot:
        candidate_records = json.loads(candidate_path.read_text())
    else:
        candidate_records = []
        with ProcessPoolExecutor(max_workers=int(config["workers"])) as pool:
            jobs = [(row, config) for row in rows]
            for index, (burst, telescope, candidates) in enumerate(pool.map(_run_job, jobs), 1):
                candidate_records.append(
                    {"burst": burst, "telescope": telescope, "candidates": candidates}
                )
                print(f"  {index}/{len(rows)} {burst}/{telescope}")
        candidate_path.write_text(json.dumps(candidate_records, indent=1))

    selected = _select_all(candidate_records, config)
    events = _event_summaries(selected)
    (out_dir / "selected_measurements.json").write_text(json.dumps(selected, indent=1))
    (out_dir / "event_dm_summary.json").write_text(json.dumps(events, indent=1))
    (out_dir / "cross_band_fit_gate.json").write_text(json.dumps({
        "status": "blocked_missing_absolute_time_origins",
        "reason": "The stored arrays do not record a timestamp-to-sample mapping, and the final "
                  "centering/cropping builder was not recovered in the completed host audit",
        "provenance_evidence": [
            "Faber2026/docs/rse/specs/handoff-2026-07-06-22-30-provenance-p0-p2-machine-verification.md",
            "Faber2026/docs/rse/specs/research-trust-reset-revalidation.md",
        ],
        "required_inputs": [
            "verified absolute time origin for each stored CHIME waterfall",
            "verified absolute time origin for each stored DSA waterfall",
            "geometric-delay convention and uncertainty",
        ],
    }, indent=1) + "\n")
    plot_resolution_contact_sheet(
        candidate_records, selected, out_dir / "resolution_stability_contact_sheet.png"
    )
    write_memo(selected, events, config, out_dir / "memo.md")
    write_run_provenance(
        rows, config, args.config, root, out_dir / "run_provenance.json",
        "replot" if args.replot else "full",
    )
    counts = {status: sum(row["status"] == status for row in selected)
              for status in sorted({row["status"] for row in selected})}
    support_kinds = ("two-band-consistent", "two-band-tension", "single-band", "none")
    (out_dir / "figures.manifest.json").write_text(json.dumps({
        "figures": [{
            "path": "resolution_stability_contact_sheet.png",
            "expectation": "24 panels; black stars mark science-grade PASS results; red x marks "
                           "MARGINAL candidates; marginal/weak labels must not be promoted",
        }],
        "per_band_status_counts": counts,
        "event_support_counts": {
            kind: sum(row["support"] == kind for row in events) for kind in support_kinds
        },
        "adoption_verdict": "not_ready",
    }, indent=1) + "\n")
    print(f"adaptive arrival: {counts}; event support: "
          f"{ {kind: sum(row['support'] == kind for row in events) for kind in support_kinds} }")


if __name__ == "__main__":
    main()
