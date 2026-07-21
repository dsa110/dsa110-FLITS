"""Halo-mass estimation for galaxy catalog rows.

Galaxy catalogs differ in what mass-tracer they ship — K-band photometry, direct
stellar mass in linear or log units, redshift versus heliocentric velocity — so
one hard-coded recipe does not generalize. ``HaloAdapter`` captures the
catalog-specific column names and units; ``prepare`` then routes each row
through whichever path its inputs support:

1. Direct stellar mass (M* in linear M_sun, linear 10^10 M_sun, or log10 M_sun)
   → Moster+13 SHMR → M_200c.
2. K-band fallback: Kmag + luminosity distance + Bell+03 M/L_K → stellar mass
   → SHMR → M_200c.
3. Neither path applicable → ``m_halo`` is NaN and ``mass_provenance`` is the
   JSON literal ``null``.

The actual method used is recorded *per row* in ``mass_provenance`` so that a
catalog that mixes paths (e.g. GLADE+ falling back to K-band when M* is null)
remains auditable downstream. The output schema (``m_halo``, ``delta_def``,
``mass_provenance``) is exactly what ``normalize.normalize`` expects to feed
into ``reduce.merge_and_rank``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy import constants as const
from astropy import units as u
from scipy.optimize import brentq

M_K_SOLAR_VEGA = 3.27  # Willmer 2018, 2MASS K_s, Vega
ML_K_BELL = 0.6  # Bell+03 K-band mean stellar M/L
DELTA_DEF = 200.0  # M_200c convention used throughout
SHMR_MODEL = "moster13_table1"

# Moster+13 Table 1 best-fit (M_200c). Each parameter is (z=0 value, slope-in-z/(1+z)).
_M10, _M11 = 11.590, 1.195
_N10, _N11 = 0.0351, -0.0247
_B10, _B11 = 1.376, -0.826
_G10, _G11 = 0.608, 0.329


def _moster_params(z: float) -> tuple[float, float, float, float]:
    a = z / (1.0 + z)
    return (
        10.0 ** (_M10 + _M11 * a),
        _N10 + _N11 * a,
        _B10 + _B11 * a,
        _G10 + _G11 * a,
    )


def kband_absolute_magnitude(kmag_apparent: float, distance_mpc: float) -> float:
    """Apparent K (Vega) → absolute K, given luminosity distance in Mpc."""
    return kmag_apparent - 5.0 * np.log10(distance_mpc) - 25.0


def kband_to_mstar(kmag_apparent: float, distance_mpc: float) -> float:
    """Apparent K (Vega) → stellar mass in M_sun via Bell+03 M/L_K."""
    M_K = kband_absolute_magnitude(kmag_apparent, distance_mpc)
    L_K = 10.0 ** (-0.4 * (M_K - M_K_SOLAR_VEGA))
    return ML_K_BELL * L_K


def mstar_to_mhalo(mstar: float, z: float) -> float:
    """Invert Moster+13 SHMR to recover M_200c from stellar mass at redshift z."""
    M1, N, beta, gamma = _moster_params(z)

    def residual(log_mh: float) -> float:
        ratio = (10.0**log_mh) / M1
        sm_over_mh = 2.0 * N / (ratio**-beta + ratio**gamma)
        return (10.0**log_mh) * sm_over_mh - mstar

    # The expanded-catalog release contract independently checks the inversion
    # to 1e-6 dex. Keep the solver floor well below that scientific tolerance.
    return 10.0 ** brentq(residual, 9.5, 16.0, xtol=1e-10)


@dataclass(frozen=True)
class HaloAdapter:
    """Column-name configuration for one galaxy catalog's mass-derivation route.

    Set ``mstar_col`` for catalogs shipping a direct stellar-mass column, with
    ``mstar_unit`` declaring its convention. Set ``kmag_col`` for the K-band
    fallback (or sole route, for GLADE 2 / 2MRS). Distance comes from
    ``distance_col`` if present, else from ``cz_col`` via the cosmology, else
    from ``z_col`` itself when ``z_velocity_unit`` is ``None``.
    """

    catalog_name: str
    table_id: str
    ra_col: str
    dec_col: str
    z_col: str
    name_col: str | None = None
    id_col: str | None = None
    mstar_col: str | None = None
    mstar_unit: str = "linear_msun"  # linear_msun | linear_1e10_msun | log10_msun
    kmag_col: str | None = None
    distance_col: str | None = None
    cz_col: str | None = None
    z_velocity_unit: str | None = None  # None → redshift; "kms" → cz in km/s
    quality_col: str | None = None  # rows with non-zero finite value here are skipped
    # Redshift-quality signal. ``z_type_col`` names a catalog column whose
    # value distinguishes spectroscopic from photometric redshifts; the two
    # ``*_value`` fields say which raw value maps to which canonical label.
    # Any value not matching either spec or photo (including null) falls
    # through to ``z_type_default``. ``z_type_default`` is also the answer for
    # all rows when the catalog has no flag column at all — set it to "spec"
    # for catalogs whose redshifts are uniformly spec-z'd (2MRS, HECATEv2)
    # so host-confusion logic uses the tight spec tolerance.
    z_type_col: str | None = None
    z_type_spec_value: object = None
    z_type_photo_value: object = None
    z_type_default: str = "unknown"

    def prepare(self, df: pd.DataFrame, *, cosmo) -> pd.DataFrame:
        """Append ``m_halo``, ``delta_def``, ``mass_provenance`` columns; never raise on missing inputs."""
        out = df.copy()
        n = len(out)
        m_halo = np.full(n, np.nan)
        method: list[str] = ["none"] * n
        mstar_source: list[str | None] = [None] * n
        ml_assumption: list[str | None] = [None] * n

        z_series = self._z_series(out)
        distance_series = self._distance_series(out, z_series, cosmo)
        quality_skip = self._quality_skip(out)

        if self.mstar_col and self.mstar_col in out.columns:
            mstar_series = out[self.mstar_col]
            for idx in range(n):
                if (
                    quality_skip.iat[idx]
                    or pd.isna(mstar_series.iat[idx])
                    or pd.isna(z_series.iat[idx])
                ):
                    continue
                mstar = self._mstar_from_value(float(mstar_series.iat[idx]))
                if not np.isfinite(mstar) or mstar <= 0:
                    continue
                try:
                    m_halo[idx] = mstar_to_mhalo(mstar, float(z_series.iat[idx]))
                except (ValueError, RuntimeError):
                    continue
                method[idx] = (
                    "log_mstar_moster13"
                    if self.mstar_unit == "log10_msun"
                    else "direct_mstar_moster13"
                )
                mstar_source[idx] = self.mstar_col

        if self.kmag_col and self.kmag_col in out.columns:
            kmag_series = out[self.kmag_col]
            for idx in range(n):
                if not np.isnan(m_halo[idx]):
                    continue
                if (
                    quality_skip.iat[idx]
                    or pd.isna(kmag_series.iat[idx])
                    or pd.isna(distance_series.iat[idx])
                    or pd.isna(z_series.iat[idx])
                ):
                    continue
                d = float(distance_series.iat[idx])
                if not np.isfinite(d) or d <= 0:
                    continue
                mstar = kband_to_mstar(float(kmag_series.iat[idx]), d)
                if not np.isfinite(mstar) or mstar <= 0:
                    continue
                try:
                    m_halo[idx] = mstar_to_mhalo(mstar, float(z_series.iat[idx]))
                except (ValueError, RuntimeError):
                    continue
                method[idx] = "kband_bell03_moster13"
                mstar_source[idx] = self.kmag_col
                ml_assumption[idx] = "Bell03_K_0p6"

        out["m_halo"] = m_halo
        out["delta_def"] = DELTA_DEF
        out["mass_provenance"] = [
            self._row_provenance(method[i], mstar_source[i], ml_assumption[i], cosmo)
            for i in range(n)
        ]
        out["z_type"] = self._z_type_series(out)
        return out

    def _z_type_series(self, df: pd.DataFrame) -> pd.Series:
        """Return per-row z_type ("spec" | "photo" | "unknown" | "none").

        Initial value is ``z_type_default`` for every row. If ``z_type_col``
        is set and present, rows matching ``z_type_spec_value`` become
        "spec" and rows matching ``z_type_photo_value`` become "photo";
        anything else (including unmapped flag values like GLADE+ f_dL=2
        for distance-derived redshifts) keeps the default. Rows whose
        ``z_col`` is NaN become "none" regardless.
        """
        n = len(df)
        result = pd.Series([self.z_type_default] * n, index=df.index, dtype=object)
        if self.z_type_col and self.z_type_col in df.columns:
            col = df[self.z_type_col]
            if self.z_type_spec_value is not None:
                result.loc[col == self.z_type_spec_value] = "spec"
            if self.z_type_photo_value is not None:
                result.loc[col == self.z_type_photo_value] = "photo"
        if self.z_col in df.columns:
            z_missing = pd.to_numeric(df[self.z_col], errors="coerce").isna()
            result.loc[z_missing] = "none"
        return result

    def _quality_skip(self, df: pd.DataFrame) -> pd.Series:
        if self.quality_col and self.quality_col in df.columns:
            col = pd.to_numeric(df[self.quality_col], errors="coerce")
            return col.notna() & (col.abs() > 0)
        return pd.Series([False] * len(df), index=df.index)

    def _z_series(self, df: pd.DataFrame) -> pd.Series:
        if self.z_col not in df.columns:
            return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
        col = pd.to_numeric(df[self.z_col], errors="coerce")
        if self.z_velocity_unit == "kms":
            c_kms = const.c.to(u.km / u.s).value
            return col / c_kms
        return col

    def _distance_series(self, df: pd.DataFrame, z_series: pd.Series, cosmo) -> pd.Series:
        if self.distance_col and self.distance_col in df.columns:
            return pd.to_numeric(df[self.distance_col], errors="coerce")
        c_kms = const.c.to(u.km / u.s).value
        if self.cz_col and self.cz_col in df.columns:
            cz_kms = pd.to_numeric(df[self.cz_col], errors="coerce")
            redshift = cz_kms / c_kms
        else:
            redshift = z_series
        n = len(df)
        d = np.full(n, np.nan)
        for idx in range(n):
            z = redshift.iat[idx]
            if pd.notna(z) and z > 0:
                try:
                    d[idx] = cosmo.luminosity_distance(float(z)).to(u.Mpc).value
                except Exception:
                    pass
        return pd.Series(d, index=df.index, dtype=float)

    def _mstar_from_value(self, value: float) -> float:
        if self.mstar_unit == "linear_msun":
            return value
        if self.mstar_unit == "linear_1e10_msun":
            return value * 1.0e10
        if self.mstar_unit == "log10_msun":
            return 10.0**value
        raise ValueError(f"Unknown mstar_unit: {self.mstar_unit}")

    def _row_provenance(
        self,
        method: str,
        mstar_source: str | None,
        ml_assumption: str | None,
        cosmo,
    ) -> str:
        if method == "none":
            return json.dumps(None)
        payload = {
            "cosmology": getattr(cosmo, "name", None) or str(cosmo),
            "delta_def": DELTA_DEF,
            "mass_method": method,
            "ml_assumption": ml_assumption,
            "mstar_source": mstar_source,
            "shmr": SHMR_MODEL,
        }
        return json.dumps(payload, sort_keys=True)


def add_halo_masses(
    df: pd.DataFrame,
    adapter: HaloAdapter,
    *,
    cosmo,
) -> pd.DataFrame:
    """Run the adapter's mass pipeline; convenience wrapper around ``adapter.prepare``."""
    return adapter.prepare(df, cosmo=cosmo)


ADAPTERS: dict[str, HaloAdapter] = {
    "glade2": HaloAdapter(
        catalog_name="glade2",
        table_id="VII/281/glade2",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        z_col="z",
        name_col="HyperLEDA",
        id_col="PGC",
        kmag_col="Kmag",
        distance_col="Dist",
    ),
    "glade-plus": HaloAdapter(
        catalog_name="glade-plus",
        table_id="VII/291/gladep",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        z_col="zhelio",
        # GLADE+ has its own row identifier (literal column name "GLADE+", with
        # the plus sign — pyvo preserves it). HyperLEDA is the most useful
        # display name when present; it is sparse, so summaries should show
        # "-" / NaN where the catalog has no LEDA counterpart. PGC is too
        # sparse in GLADE+ to use as id.
        name_col="HyperLEDA",
        id_col="GLADE+",
        mstar_col="M*",
        mstar_unit="linear_1e10_msun",
        kmag_col="Kmag",
        distance_col="dL",
        # GLADE+ records the redshift origin in f_dL (the *distance* flag), per
        # the upstream documentation at glade.elte.hu:
        #   0: galaxy has no measured z or distance (z is NaN; treated as "none")
        #   1: photometric redshift -> z_type "photo"
        #   2: distance measured first, z back-computed -> falls through to
        #      z_type_default ("unknown"), which uses the broader 0.05
        #      host_dz tolerance because the back-computed z's precision
        #      depends on the distance method (TF/SBF/Cepheid; typically
        #      a few percent in d, propagating to ~1-10% in z)
        #   3: spectroscopic redshift -> z_type "spec"
        # The earlier wiring used f_zcmb, which is the peculiar-velocity
        # *correction* flag (0 = uncorrected, 1 = corrected) — not the
        # spec/photo provenance signal. Using f_dL fixes that bug.
        z_type_col="f_dL",
        z_type_spec_value=3,
        z_type_photo_value=1,
    ),
    "hecate-v2": HaloAdapter(
        catalog_name="hecate-v2",
        table_id="J/MNRAS/548/G522/hecatev2",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        z_col="HRV",
        z_velocity_unit="kms",
        mstar_col="logMstarHEC",
        mstar_unit="log10_msun",
        distance_col="Dist",
        quality_col="q_logMstarHEC",
        # HECATEv2 aggregates redshifts predominantly from spec-z surveys
        # (SDSS, 6dFGS, 2dFGRS, and HyperLEDA's curated entries) for
        # local-volume galaxies. The catalog does not ship a per-row
        # spec/photo flag, so default to "spec" — host-exclusion under
        # this default uses the tight --host-dz-spec-max threshold
        # (default 0.005), matching the catalog's typical measurement
        # precision rather than a conservative photo-z-like 0.05 bound.
        z_type_default="spec",
    ),
    "two-mrs": HaloAdapter(
        catalog_name="two-mrs",
        table_id="J/ApJS/199/26/table3",
        ra_col="RAJ2000",
        dec_col="DEJ2000",
        z_col="cz",
        z_velocity_unit="kms",
        kmag_col="Kcmag",
        cz_col="cz",
        # 2MRS is the 2MASS Redshift Survey: every redshift is measured
        # spectroscopically (radial velocity cz from optical/IR
        # spectrographs). No per-row flag; default all rows to "spec".
        z_type_default="spec",
    ),
}
