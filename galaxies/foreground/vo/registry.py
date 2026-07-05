"""TAP service discovery via RegTAP, with anchor fallbacks."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from urllib.parse import urlparse

import pandas as pd

try:
    from pyvo.dal import TAPService
except Exception:  # pragma: no cover - optional import for offline tests
    TAPService = None  # type: ignore[assignment]
try:
    from pyvo.registry import search as vo_search  # lightweight fallback
except Exception:  # pragma: no cover
    vo_search = None  # type: ignore[assignment]


_ANCHOR_SERVICES = [
    {"ivoid": None, "title": "TAPVizieR", "access_url": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"},
    {"ivoid": None, "title": "MAST TAP", "access_url": "https://mast.stsci.edu/vo-tap/"},
    {"ivoid": None, "title": "NOIRLab Data Lab TAP", "access_url": "https://datalab.noirlab.edu/tap"},
]


def _build_regtap_adql(keywords: Iterable[str], max_services: int) -> str:
    """ADQL finding TAP-capable services via RegTAP.

    Matches rr.capability standard_id for TAP, joins rr.interface for HTTP access
    URLs and rr.resource for titles; keywords filter title/subject/description
    via ivo_hasword.
    """
    kws = [k.replace("'", "''") for k in keywords]
    or_terms = []
    for k in kws:
        or_terms.append(f"ivo_hasword(r.res_title, '{k}')")
        or_terms.append(f"ivo_hasword(r.res_subject, '{k}')")
        or_terms.append(f"ivo_hasword(r.res_description, '{k}')")
    kw_clause = "  AND (" + " OR ".join(or_terms) + ")\n" if or_terms else ""
    return (
        f"SELECT TOP {int(max_services)} i.access_url, MIN(r.res_title) AS title, MIN(r.ivoid) AS ivoid\n"
        f"FROM rr.resource AS r\n"
        f"JOIN rr.capability AS c ON r.ivoid=c.ivoid\n"
        f"JOIN rr.interface AS i ON c.ivoid=i.ivoid AND c.cap_index=i.cap_index\n"
        f"WHERE LOWER(c.standard_id) LIKE 'ivo://ivoa.net/std/tap%'\n"
        f"  AND i.access_url IS NOT NULL\n"
        f"  AND (i.intf_type='vs:paramhttp' OR i.intf_type IS NULL)\n"
        f"{kw_clause}"
        f"GROUP BY i.access_url\n"
        f"ORDER BY title"
    )


def _merge_anchor_services(df: pd.DataFrame) -> pd.DataFrame:
    anchors = pd.DataFrame.from_records(_ANCHOR_SERVICES, columns=["ivoid", "title", "access_url"])
    return _dedupe_endpoints(pd.concat([df, anchors], ignore_index=True))


def _dedupe_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cols = [c for c in ["ivoid", "title", "access_url"] if c in df.columns]
    out = df[cols].copy()
    out.dropna(subset=["access_url"], inplace=True)
    out.drop_duplicates(subset=["access_url"], inplace=True, keep="first")
    out.sort_values(["title", "access_url"], inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    for col in ["ivoid", "title", "access_url"]:
        if col not in out.columns:
            out[col] = None
    return out[["ivoid", "title", "access_url"]]


def _normalize_access_urls(df: pd.DataFrame, resolve_redirects: bool = False) -> pd.DataFrame:
    """Normalize endpoint URLs (strip /sync|/async and '?', upgrade to https).

    Optionally follows one round of redirects (HEAD) to canonicalize URLs.
    Malformed/relative URLs are dropped.
    """
    if df.empty or "access_url" not in df.columns:
        return df

    def _norm_one(u: str) -> str | None:
        if not isinstance(u, str):
            return None
        s = u.strip()
        for suffix in ("/async", "/sync"):
            if s.endswith(suffix):
                s = s[: -len(suffix)]
        if s.endswith("?"):
            s = s[:-1]
        if s.startswith("http://"):
            s = "https://" + s[len("http://") :]
        p = urlparse(s)
        if not p.scheme or not p.netloc:
            return None
        return s

    urls = df["access_url"].map(_norm_one)
    if resolve_redirects:
        try:
            import requests  # local import to keep tests offline
        except Exception:  # pragma: no cover
            requests = None  # type: ignore[assignment]
        if requests is not None:
            canon = []
            for u in urls:
                try:
                    if u is None:
                        canon.append(None)
                        continue
                    r = requests.head(u, allow_redirects=True, timeout=5)
                    canon.append(_norm_one(r.url.strip().rstrip("?")))
                except Exception:
                    canon.append(u)
            urls = pd.Series(canon, index=df.index)
    out = df.copy()
    out["access_url"] = urls
    return out.dropna(subset=["access_url"])


def discover_tap_services(
    keywords: tuple[str, ...] = (),
    max_services: int = 50,
    keyword: str | None = None,
    regtap_url: str = "https://dc.g-vo.org/tap",
    include_anchors: bool = True,
    resolve_redirects: bool = False,
) -> pd.DataFrame:
    """Discover TAP services using RegTAP with optional keyword filtering.

    Returns a DataFrame with columns ``ivoid``, ``title``, ``access_url``; a
    provenance dict (ADQL + parameters) is attached at ``df.attrs['provenance']``.
    Degrades gracefully offline: anchors only.
    """
    final_keywords = list(keywords)
    if keyword:
        final_keywords.append(keyword)

    adql = _build_regtap_adql(final_keywords, max_services)

    if TAPService is None:
        df = pd.DataFrame(columns=["ivoid", "title", "access_url"])
        if include_anchors:
            df = _merge_anchor_services(df)
        df.attrs["provenance"] = {
            "service": regtap_url,
            "adql": adql,
            "keywords": final_keywords,
            "max_services": max_services,
            "anchors_added": include_anchors,
            "resolved_redirects": resolve_redirects,
            "note": "pyvo TAPService unavailable; returned anchors only",
        }
        return df

    try:
        svc = TAPService(regtap_url)
        if hasattr(svc, "run_sync"):
            res = svc.run_sync(adql)
        else:
            res = svc.launch_job_sync(adql)
        table = res.to_table()
        try:
            df = table.to_pandas()
        except Exception:
            df = pd.DataFrame(table.as_array())
        df_raw_len = len(df)
        df = _normalize_access_urls(df, resolve_redirects=resolve_redirects)
        df = _dedupe_endpoints(df)
    except Exception:
        # network or service error -> degrade gracefully
        df = pd.DataFrame(columns=["ivoid", "title", "access_url"])
        df_raw_len = 0

    # Fallback to VO registry simple search if RegTAP returns nothing
    if df.empty and vo_search is not None:
        try:
            kw = " ".join(final_keywords) if final_keywords else None
            results = vo_search(servicetype="tap", keywords=kw) if kw else vo_search(servicetype="tap")
            recs: list[dict] = []
            for res in results[:max_services]:
                try:
                    ivoid = getattr(res, "ivoid", None) or getattr(res, "ivoid_", None)
                    title = getattr(res, "title", "")
                    cap = next(
                        (c for c in getattr(res, "capabilities", []) if getattr(c, "standardid", "").lower().endswith("tap")),
                        None,
                    )
                    access_url = getattr(getattr(cap, "interfaces", [None])[0], "accessurl", None) if cap else None
                    if access_url:
                        recs.append({"ivoid": ivoid, "title": title, "access_url": access_url})
                except Exception:
                    continue
            if recs:
                df = pd.DataFrame.from_records(recs, columns=["ivoid", "title", "access_url"])
                df_raw_len = len(df)
                df = _normalize_access_urls(df, resolve_redirects=resolve_redirects)
                df = _dedupe_endpoints(df)
        except Exception:
            pass

    before = len(df)
    if include_anchors:
        df = _merge_anchor_services(df)

    df.attrs["provenance"] = {
        "service": regtap_url,
        "adql": adql,
        "keywords": final_keywords,
        "max_services": max_services,
        "anchors_added": include_anchors,
        "resolved_redirects": resolve_redirects,
        "grouped_by_access_url": True,
        "num_rows_raw": df_raw_len,
        "num_rows_before_anchors": before,
        "num_rows_after_anchors": len(df),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    return df
