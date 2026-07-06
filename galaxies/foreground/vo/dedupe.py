"""Cross-catalog deduplication for combined candidate tables."""

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .reduce import angular_separation_arcmin

_MATCH_ARCSEC = 30.0
_MATCH_ARCMIN = _MATCH_ARCSEC / 60.0
_DZ_MAX = 0.01

_Z_TYPE_RANK = {"spec": 0, "photo": 1, "unknown": 2}


def _z_type_rank(z_type: object) -> int:
    return _Z_TYPE_RANK.get(str(z_type), 2)


def _same_galaxy(row_a: pd.Series, row_b: pd.Series) -> bool:
    sep = angular_separation_arcmin(
        float(row_a["ra"]),
        float(row_a["dec"]),
        float(row_b["ra"]),
        float(row_b["dec"]),
    )
    if sep > _MATCH_ARCMIN:
        return False
    za = pd.to_numeric(row_a.get("z"), errors="coerce")
    zb = pd.to_numeric(row_b.get("z"), errors="coerce")
    if pd.notna(za) and pd.notna(zb):
        return abs(float(za) - float(zb)) <= _DZ_MAX
    return True


def _pick_representative(group: pd.DataFrame) -> pd.Series:
    """Choose the best row from a duplicate group."""
    work = group.copy()
    work["_z_rank"] = work["z_type"].map(_z_type_rank) if "z_type" in work.columns else 2
    if "m_delta" in work.columns:
        work["_has_mass"] = pd.to_numeric(work["m_delta"], errors="coerce").notna().astype(int)
    else:
        work["_has_mass"] = 0
    if "rank_key" in work.columns:
        work["_rank_key"] = pd.to_numeric(work["rank_key"], errors="coerce").fillna(np.inf)
    else:
        work["_rank_key"] = np.inf
    idx = work.sort_values(
        ["_z_rank", "_has_mass", "_rank_key"], ascending=[True, False, True]
    ).index[0]
    rep = group.loc[idx].copy()
    sources = sorted(
        {
            str(s)
            for s in group.get("catalog_id", group.get("service", pd.Series(dtype=str)))
            if pd.notna(s)
        }
    )
    rep["catalog_sources"] = ",".join(sources)
    if "provenance_json" in rep.index:
        try:
            prov = json.loads(rep["provenance_json"]) if rep["provenance_json"] else {}
        except (TypeError, ValueError):
            prov = {}
        prov["catalog_sources"] = sources
        rep["provenance_json"] = json.dumps(prov, sort_keys=True)
    return rep.drop(
        labels=[c for c in ("_z_rank", "_has_mass", "_rank_key") if c in rep.index], errors="ignore"
    )


def _bucket_key(ra: float, dec: float, cell_deg: float = 0.01) -> tuple[int, int]:
    return (int(ra / cell_deg), int(dec / cell_deg))


def _neighbor_buckets(key: tuple[int, int]) -> Iterable[tuple[int, int]]:
    bx, by = key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield (bx + dx, by + dy)


def _connected_components(indices: list[int], same: dict[tuple[int, int], bool]) -> list[list[int]]:
    parent = {i: i for i in indices}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (i, j), ok in same.items():
        if ok:
            union(i, j)
    groups: dict[int, list[int]] = {}
    for i in indices:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def deduplicate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the same galaxy seen in multiple catalogs along one sightline."""
    if df.empty:
        out = df.copy()
        out["catalog_sources"] = pd.Series(dtype=str)
        return out

    required = {"frb_name", "ra", "dec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"deduplicate_candidates missing columns: {sorted(missing)}")

    kept: list[pd.Series] = []
    for _frb, group in df.groupby("frb_name", sort=True):
        idxs = list(group.index)
        if len(idxs) == 1:
            row = group.loc[idxs[0]].copy()
            src = row.get("catalog_id", row.get("service"))
            row["catalog_sources"] = str(src) if pd.notna(src) else ""
            kept.append(row)
            continue

        sub = group.loc[idxs]
        buckets: dict[tuple[int, int], list[int]] = {}
        for i in idxs:
            key = _bucket_key(float(sub.at[i, "ra"]), float(sub.at[i, "dec"]))
            buckets.setdefault(key, []).append(i)

        pairs: dict[tuple[int, int], bool] = {}
        seen_pairs: set[tuple[int, int]] = set()
        for i in idxs:
            key = _bucket_key(float(sub.at[i, "ra"]), float(sub.at[i, "dec"]))
            for nb in _neighbor_buckets(key):
                for j in buckets.get(nb, []):
                    if j <= i:
                        continue
                    pair = (i, j)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    pairs[pair] = _same_galaxy(sub.loc[i], sub.loc[j])
        components = _connected_components(idxs, pairs)
        for comp in components:
            rep = _pick_representative(group.loc[comp])
            kept.append(rep)

    out = pd.DataFrame(kept).reset_index(drop=True)
    return out
