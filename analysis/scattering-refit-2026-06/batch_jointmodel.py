#!/usr/bin/env python
"""Stage local data, dump canonical joint-fit models, and plot 2D data-vs-model panels.

Uses canonical all-exp fits from _a1_fits/ (grade_allexp CANON map) plus joint_json/
base fits for hamilton and whitney. Writes per-burst PNG/SVG/PDF and optional montage.

  FLITS_REPO=... FLITS_RUNS=... python batch_jointmodel.py [--montage OUT]
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("FLITS_REPO", HERE.parents[1]))
RUNS = Path(os.environ.get("FLITS_RUNS", "/central/scratch/jfaber/flits-runs"))
A1 = HERE / "_a1_fits"
JOINT_JSON = HERE / "joint_json"
DATA = RUNS / "data"
CFG = RUNS / "configs"
JOINT = DATA / "joint"
TEL = REPO / "scattering/configs/telescopes.yaml"
SAMP = REPO / "scattering/configs/sampler.yaml"

# grade_allexp CANON + joint_json fallbacks for hamilton/whitney
CANON: dict[str, str] = {
    "casey": "sharedzeta",
    "chromatica": "sharedzeta",
    "freya": "sharedzeta",
    "wilhelm": "sharedzeta",
    "mahi": "C1D1",
    "phineas": "C3D3",
    "oran": "C2D1",
    "isha": "C2D1",
    "johndoeII": "C2D1",
    "zach": "",
    "hamilton": None,
    "whitney": None,
}


def fit_suffix(tag: str | None) -> str:
    if tag is None:
        return ""
    return f"_{tag}_pbf-exp-exp" if tag else "_pbf-exp-exp"


def fit_source(burst: str, tag: str | None) -> Path:
    if tag is None:
        return JOINT_JSON / f"{burst}_joint_fit.json"
    sfx = fit_suffix(tag)
    return A1 / f"{burst}_joint_fit{sfx}.json"


def gen_configs() -> None:
    CFG.mkdir(parents=True, exist_ok=True)
    knobs = dict(
        telcfg_path=str(TEL),
        sampcfg_path=str(SAMP),
        fitting_method="nested",
        outer_trim=0.15,
        nlive=400,
        dlogz=0.5,
        nlive_walks=15,
        alpha_fixed=4.0,
        nproc=8,
    )
    for src in sorted((REPO / "scattering/configs/bursts/chime").glob("*_chime.yaml")):
        burst = src.name[: -len("_chime.yaml")]
        cfg = yaml.safe_load(src.read_text()) or {}
        fname = os.path.basename(cfg["path"])
        local = DATA / fname
        if not local.exists():
            print(f"SKIP chime {burst}: missing {fname}")
            continue
        cfg["path"] = str(local)
        cfg.update(knobs)
        cfg["telescope"] = cfg.get("telescope", "chime")
        (CFG / f"{burst}_chime_run.yaml").write_text(
            yaml.safe_dump(cfg, default_flow_style=False, sort_keys=True)
        )
    dsa_knobs = {**knobs, "telescope": "dsa"}
    for src in sorted((REPO / "scattering/configs/bursts/dsa").glob("*_dsa.yaml")):
        burst = src.name[: -len("_dsa.yaml")]
        hits = glob.glob(str(DATA / "dsa" / f"{burst}_dsa_*.npy"))
        if not hits:
            print(f"SKIP dsa {burst}: no local cube")
            continue
        cfg = yaml.safe_load(src.read_text()) or {}
        cfg["path"] = hits[0]
        cfg.update(dsa_knobs)
        toks = os.path.basename(hits[0]).split("_")
        j = toks.index("I")
        cfg["dm_init"] = float(f"{toks[j + 1]}.{toks[j + 2]}")
        (CFG / f"{burst}_dsa_run.yaml").write_text(
            yaml.safe_dump(cfg, default_flow_style=False, sort_keys=True)
        )


def stage_fits() -> list[tuple[str, str]]:
    JOINT.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[str, str]] = []
    for burst, tag in CANON.items():
        src = fit_source(burst, tag)
        if not src.exists():
            print(f"!! missing fit: {src}")
            continue
        sfx = fit_suffix(tag)
        dst = JOINT / f"{burst}_joint_fit{sfx}.json"
        shutil.copy2(src, dst)
        staged.append((burst, sfx))
        print(f"staged {dst.name}")
    return staged


def dump_all(staged: list[tuple[str, str]]) -> None:
    dump_py = HERE / "dump_jointmodel.py"
    env = os.environ.copy()
    env["FLITS_REPO"] = str(REPO)
    env["FLITS_RUNS"] = str(RUNS)
    for burst, sfx in staged:
        r = subprocess.run(
            [sys.executable, str(dump_py), burst, sfx],
            env=env,
            capture_output=True,
            text=True,
        )
        if r.returncode:
            print(r.stdout, r.stderr, sep="\n")
            raise SystemExit(f"dump failed: {burst}{sfx}")
        print(r.stdout.strip())


def plot_all(out_dir: Path) -> None:
    plot_py = HERE / "plot_jointmodel.py"
    env = os.environ.copy()
    subprocess.check_call(
        [sys.executable, str(plot_py), str(JOINT), str(out_dir), "--vector"],
        env=env,
    )


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "jointmodel_figs"
    gen_configs()
    staged = stage_fits()
    dump_all(staged)
    plot_all(out)
    if "--montage" in sys.argv:
        idx = sys.argv.index("--montage")
        montage_out = Path(sys.argv[idx + 1])
        subprocess.check_call(
            [
                sys.executable,
                str(HERE / "plot_jointmodel_montage.py"),
                str(JOINT),
                str(montage_out),
            ]
        )


if __name__ == "__main__":
    main()
