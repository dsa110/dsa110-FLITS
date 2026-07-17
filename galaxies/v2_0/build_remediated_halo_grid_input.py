"""Build the sightline_halo_grid input: discovery population + budget overlay.

The figure's design intent (matching its caption) is the full discovery-stage
foreground-halo environment -- every z < z_host halo recovered around each
sightline -- NOT the vetted census subset. This builder therefore starts from
the legacy ranked candidate set and applies the 2026-07-15 remediation *to
that population* rather than replacing it:

1. position dedupe within each sightline (<1 arcsec, same z: the same
   physical galaxy cross-listed under two identifiers);
2. an overlay flag ``budget_contributor`` on the systems whose two-phase
   columns actually enter Table tab:budget, located through the SAME chain
   the budget uses (``foreground_unified``: deduped census, adjudicated
   empirical masses, uniformly recomputed halo geometry); contributors absent
   from the discovery set (faint Legacy-only objects) are appended with their
   adjudicated mass and R_vir so the overlay is complete;
3. NO census-catalog clusters are injected: the near-miss and wider cluster
   field belong to the Appendix B figure (fig:clusters_icm); the single
   budget-entering cluster crossing is already the ringed massive halo of the
   discovery set.

Run:  conda run -n flits python -m galaxies.v2_0.build_remediated_halo_grid_input \
          --halo-csv <ranked candidates csv> --out <merged csv>
Then: conda run -n flits python galaxies/v2_0/sightline_halo_grid.py \
          --halo-csv <merged csv> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

# The nine sightlines with spectroscopic host redshifts (V6 positions).
FRB = {
    "zach": ("FRB 20220207C", 310.1995, 72.8823, 0.043),
    "whitney": ("FRB 20220310F", 134.7205, 73.4908, 0.479),
    "oran": ("FRB 20220506D", 318.0448, 72.8273, 0.300),
    "isha": ("FRB 20221113A", 71.4110, 70.3074, 0.251),
    "wilhelm": ("FRB 20221203A", 315.1295, 72.0376, 0.510),
    "phineas": ("FRB 20230307A", 177.7813, 71.6956, 0.271),
    "hamilton": ("FRB 20230913A", 305.0372, 70.7928, 0.302),
    "chromatica": ("FRB 20240203A", 312.6191, 73.9000, 0.074),
    "casey": ("FRB 20240229A", 169.9835, 70.6762, 0.287),
}


def _position_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop same-position same-z rows within each sightline (cross-listings)."""
    keep = np.ones(len(df), dtype=bool)
    for _, idx in df.groupby("frb_name").groups.items():
        rows = df.loc[idx]
        arr = rows[["ra", "dec", "z"]].to_numpy(float)
        for i in range(len(rows)):
            if not keep[df.index.get_loc(idx[i])]:
                continue
            for j in range(i + 1, len(rows)):
                sep = math.hypot(
                    (arr[i, 0] - arr[j, 0]) * math.cos(math.radians(arr[i, 1])),
                    arr[i, 1] - arr[j, 1],
                ) * 3600.0
                if sep < 1.0 and abs(arr[i, 2] - arr[j, 2]) < 1e-3:
                    keep[df.index.get_loc(idx[j])] = False
    return df[keep].copy()


def _budget_contributors() -> list[dict]:
    """The vetted systems whose columns enter tab:budget, via the budget chain."""
    from galaxies.foreground.sightline_budget import foreground_unified

    out: list[dict] = []
    for nick, (tns, fra, fdec, zf) in FRB.items():
        uni = foreground_unified(nick, zf, fra, fdec, results_dir="results", enrich=False)
        for _, r in uni.iterrows():
            dm = (0.0 if pd.isna(r.get("dm_halo")) else float(r["dm_halo"])) + (
                0.0 if pd.isna(r.get("dm_cool")) else float(r["dm_cool"])
            )
            if dm <= 0.0:
                continue
            out.append(
                dict(
                    frb_name=tns, frb_z=zf, frb_dec=fdec,
                    ra=float(r["ra"]), dec=float(r["dec"]), z=float(r["z"]),
                    b_kpc=float(r["impact_kpc"]),
                    m_delta=float(r["M_halo"]),
                    r_delta_computed=float(r["R_vir_kpc"]),
                    is_cluster=str(r.get("classification")) == "GClstr",
                )
            )
    return out


def build(halo_csv: str) -> pd.DataFrame:
    df = pd.read_csv(halo_csv)
    df = _position_dedupe(df)
    df["budget_contributor"] = False

    for c in _budget_contributors():
        sep = (
            np.hypot(
                (df["ra"].astype(float) - c["ra"])
                * np.cos(np.radians(df["dec"].astype(float))),
                df["dec"].astype(float) - c["dec"],
            )
            * 3600.0
        )
        match = (df["frb_name"] == c["frb_name"]) & (sep < 3.0)
        if match.any():
            df.loc[match, "budget_contributor"] = True
        elif not c["is_cluster"]:
            # Faint Legacy-only contributor absent from the discovery set:
            # append with its adjudicated mass/R_vir so the overlay is complete.
            df = pd.concat(
                [df, pd.DataFrame([dict(
                    frb_name=c["frb_name"], frb_z=c["frb_z"], frb_dec=c["frb_dec"],
                    ra=c["ra"], dec=c["dec"], z=c["z"], b_kpc=c["b_kpc"],
                    m_delta=c["m_delta"], logM=math.log10(c["m_delta"]),
                    r_delta_computed=c["r_delta_computed"],
                    is_foreground=True,
                    intersects_strict=bool(c["b_kpc"] <= c["r_delta_computed"]),
                    budget_contributor=True,
                )])],
                ignore_index=True,
            )
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--halo-csv", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    df = build(args.halo_csv)
    df.to_csv(args.out, index=False)
    n_fg = int(df["is_foreground"].astype(str).str.lower().isin(["true", "1"]).sum())
    n_c = int(df["budget_contributor"].sum())
    print(f"wrote {args.out}: {len(df)} rows, {n_fg} foreground halos, "
          f"{n_c} budget-contributor overlays")


if __name__ == "__main__":
    main()
