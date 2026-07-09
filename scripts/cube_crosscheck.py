"""P2.3 CHIME cube integrity cross-checks.

Produces a per-burst lag table and thumbnail grid comparing the native
scattering-input CHIME cubes against independently regenerated CHIME profiles
when those references are present locally.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).parents[1]
MANIFEST = ROOT / "data-manifest.csv"
DEFAULT_DATA = pathlib.Path("~/Data/Faber2026").expanduser()
DEFAULT_OUT = ROOT / "analysis/cube_integrity"
INDEPENDENT_CHIME_REFS = {"casey", "freya", "isha", "mahi", "phineas", "whitney"}


def robust_snr(profile: np.ndarray) -> np.ndarray:
    p = np.asarray(profile, dtype=float)
    base = np.nanmedian(p)
    noise = 1.4826 * np.nanmedian(np.abs(p - base))
    return (p - base) / max(float(noise), 1e-12)


def cube_profile(path: pathlib.Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.asarray(arr, dtype=float), axis=0)


def reference_profile(path: pathlib.Path) -> np.ndarray:
    with np.load(path) as z:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(np.asarray(z["power_2d"], dtype=float), axis=0)


def resample_profile(profile: np.ndarray, size: int) -> np.ndarray:
    p = np.asarray(profile, dtype=float)
    if p.size == size:
        return p.copy()
    if p.size == 0:
        raise ValueError("cannot resample an empty profile")
    x_old = np.linspace(0.0, 1.0, p.size)
    x_new = np.linspace(0.0, 1.0, size)
    return np.interp(x_new, x_old, p)


def best_lag_bins(cube: np.ndarray, reference: np.ndarray, max_lag: int = 256) -> int:
    c = robust_snr(cube)
    r = robust_snr(reference)
    if c.size != r.size:
        r = resample_profile(r, c.size)
    c = np.nan_to_num(c - np.nanmean(c))
    r = np.nan_to_num(r - np.nanmean(r))
    max_lag = min(max_lag, c.size - 1)
    best_lag = 0
    best_score = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            cc = c[:lag]
            rr = r[-lag:]
        elif lag > 0:
            cc = c[lag:]
            rr = r[:-lag]
        else:
            cc = c
            rr = r
        denom = np.linalg.norm(cc) * np.linalg.norm(rr)
        score = -np.inf if denom == 0 else float(np.dot(cc, rr) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def direct_verdict(snr: np.ndarray) -> tuple[str, float, str, float | None]:
    n = snr.size
    hot = snr > 5.0
    max_snr = float(np.nanmax(snr))
    if not hot.any():
        return "no-burst-above-5sigma", max_snr, "NA", None
    edge = n // 50
    edge_ok = not bool(hot[:edge].any() or hot[-edge:].any())
    idx = np.flatnonzero(hot)
    centroid = float((snr[idx] * idx).sum() / snr[idx].sum())
    center_ok = bool(0.4 * n < centroid < 0.6 * n)
    verdict = "pass" if edge_ok and center_ok else "fail"
    return verdict, max_snr, str(edge_ok).lower(), centroid


def manifest_chime_rows() -> list[dict[str, str]]:
    with MANIFEST.open() as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r["telescope"] == "chime" and "32000b" in r["filename"]]


def find_one(root: pathlib.Path, filename: str) -> pathlib.Path | None:
    hits = sorted(root.rglob(filename))
    return hits[0] if hits else None


def reference_for(data_root: pathlib.Path, burst: str) -> pathlib.Path | None:
    if burst not in INDEPENDENT_CHIME_REFS:
        return None
    p = data_root / "dsa110" / "scintillation-data" / f"{burst}_chime.npz"
    return p if p.exists() else None


def _rel(path: pathlib.Path, root: pathlib.Path) -> str:
    """Path relative to root, or str(path) if not under root."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_rows(data_root: pathlib.Path, max_lag: int) -> list[dict[str, str]]:
    rows = []
    for row in manifest_chime_rows():
        burst = row["burst"]
        cube_path = find_one(data_root, row["filename"])
        out = {
            "burst": burst,
            "cube_filename": row["filename"],
            "cube_path": _rel(cube_path, data_root) if cube_path else "",
            "reference_path": "",
            "direct_verdict": "missing-cube",
            "max_snr": "",
            "edge_ok": "",
            "centroid_bin": "",
            "cross_lag_bins": "",
            "cross_verdict": "not-run",
        }
        if cube_path is None:
            rows.append(out)
            continue

        prof = cube_profile(cube_path)
        snr = robust_snr(prof)
        verdict, max_snr, edge_ok, centroid = direct_verdict(snr)
        out.update(
            {
                "direct_verdict": verdict,
                "max_snr": f"{max_snr:.6g}",
                "edge_ok": edge_ok,
                "centroid_bin": "" if centroid is None else f"{centroid:.3f}",
            }
        )

        ref_path = reference_for(data_root, burst)
        if ref_path is None:
            out["cross_verdict"] = "no-independent-reference"
            rows.append(out)
            continue

        ref_prof = resample_profile(reference_profile(ref_path), prof.size)
        lag = best_lag_bins(prof, ref_prof, max_lag=max_lag)
        out.update(
            {
                "reference_path": _rel(ref_path, data_root),
                "cross_lag_bins": str(lag),
                "cross_verdict": "pass" if abs(lag) < 5 else "fail",
            }
        )
        rows.append(out)
    return rows


def write_csv(rows: list[dict[str, str]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "burst",
        "cube_filename",
        "cube_path",
        "reference_path",
        "direct_verdict",
        "max_snr",
        "edge_ok",
        "centroid_bin",
        "cross_lag_bins",
        "cross_verdict",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_figure(rows: list[dict[str, str]], path: pathlib.Path, data_root: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ncols = 3
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.4 * nrows), squeeze=False)
    for ax, row in zip(axes.ravel(), rows, strict=False):
        if not row["cube_path"]:
            ax.set_title(f"{row['burst']}: missing cube")
            ax.axis("off")
            continue
        cube = data_root / row["cube_path"]
        x = np.arange(32000)
        cube_snr = robust_snr(cube_profile(cube))
        ax.plot(x, np.clip(cube_snr, -5, 20), lw=0.6, label="cube")
        if row["reference_path"]:
            ref = resample_profile(
                reference_profile(data_root / row["reference_path"]), cube_snr.size
            )
            ref_snr = robust_snr(ref)
            ax.plot(x, np.clip(ref_snr, -5, 20), lw=0.6, alpha=0.8, label="regen")
        title = (
            f"{row['burst']}: {row['direct_verdict']}\n"
            f"lag={row['cross_lag_bins'] or 'NA'} {row['cross_verdict']}"
        )
        ax.set_title(title, fontsize=9)
        ax.axvspan(0, 640, color="tab:red", alpha=0.08)
        ax.axvspan(31360, 32000, color="tab:red", alpha=0.08)
        ax.axvspan(12800, 19200, color="tab:green", alpha=0.05)
        ax.set_xlim(0, 32000)
        ax.set_ylim(-5, 20)
    for ax in axes.ravel()[len(rows) :]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=pathlib.Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--max-lag", type=int, default=256)
    args = parser.parse_args(argv)

    data_root = args.data_root.expanduser()
    rows = build_rows(data_root, max_lag=args.max_lag)
    csv_path = args.out_dir / "cube_crosscheck_lags.csv"
    fig_path = args.out_dir / "cube_crosscheck_thumbnails.png"
    write_csv(rows, csv_path)
    write_figure(rows, fig_path, data_root)
    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")
    failures = [r for r in rows if r["direct_verdict"] != "pass" or r["cross_verdict"] == "fail"]
    if failures:
        print(f"findings: {len(failures)} rows need triage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
