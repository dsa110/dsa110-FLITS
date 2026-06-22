import os

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "source")
S1 = os.path.join(D, "DSA110_CHIME_Codetection_BurstProperties_Foreground  - Sheet1.csv")
S2 = os.path.join(D, "DSA110_CHIME_Codetection_BurstProperties_Foreground  - Sheet2.csv")
OUT = HERE

NICK = [
    "zach",
    "whitney",
    "oran",
    "isha",
    "wilhelm",
    "phineas",
    "freya",
    "johndoeII",
    "hamilton",
    "mahi",
    "chromatica",
    "casey",
]
NICKSET = set(NICK)


def f(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s in ("", "-", "nan", "NaN", "NA"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------- bursts (top block of Sheet1) ----------
s1 = pd.read_csv(S1, header=None, dtype=str).fillna("")
bursts = []
for i in range(1, 13):  # rows 1..12 are the 12 FRBs
    r = s1.iloc[i]
    tau_std = r[13].strip()  # "(+a, -b)"
    plus = minus = np.nan
    if tau_std.startswith("("):
        try:
            a, b = tau_std.strip("() ").split(",")
            plus = float(a.replace("+", ""))
            minus = float(b.replace("-", "").replace("+", ""))
        except Exception:
            pass
    bursts.append(
        dict(
            nickname=r[0].strip(),
            tns=r[1].strip(),
            mjd=f(r[2]),
            localization=r[3].strip(),
            ra_deg=f(r[6]),
            dec_deg=f(r[7]),
            z_spec=f(r[4]),
            gamma_1p4ghz_mhz=f(r[9]),
            gamma_1p4ghz_err_mhz=f(r[10]),
            gamma_1p4ghz_ne2001_mhz=f(r[11]),
            tau_1ghz_ms=f(r[12]),
            tau_1ghz_err_plus_ms=plus,
            tau_1ghz_err_minus_ms=minus,
        )
    )
bdf = pd.DataFrame(bursts)
zmap = dict(zip(bdf.nickname, bdf.z_spec))
ramap = dict(zip(bdf.nickname, bdf.ra_deg))
decmap = dict(zip(bdf.nickname, bdf.dec_deg))

# ---------- foreground (stacked blocks, Sheet1 = authoritative attribution) ----------
fg = []
cur_b = cur_t = None
for i in range(13, len(s1)):
    r = s1.iloc[i]
    c0 = r[0].strip()
    if c0 in NICKSET:
        cur_b = c0
        hdr = " ".join(r[1:].tolist()).lower()
        if "r500" in hdr or "lam500" in hdr or "obj_name" in hdr:
            cur_t = "cluster"
        elif "obj_id" in hdr or "class" in hdr or "impact" in hdr:
            cur_t = "halo"
        else:
            cur_t = None  # no-foreground marker (isha/johndoeII/mahi)
        continue
    ra = r[6].strip()
    objid = r[1].strip()
    if cur_b and cur_t and (ra or objid):
        rec = dict(
            nickname=cur_b,
            tns=bdf.set_index("nickname").loc[cur_b, "tns"],
            host_z_spec=zmap[cur_b],
            type=cur_t,
            tag=c0,
            obj=objid,
            survey=r[2].strip(),
            impact_kpc_listed=f(r[5]),
            ra_deg=f(r[6]),
            dec_deg=f(r[7]),
            z_phot=f(r[8]),
        )
        if cur_t == "halo":
            rec.update(
                obj_class=r[9].strip(), prob_gal=f(r[10]), prob_star=f(r[11]), prob_qso=f(r[12])
            )
        else:
            rec.update(
                r500_mpc=f(r[9]),
                lam500_richness=f(r[10]),
                m500_1e14msun=f(r[11]),
                n_galaxies=f(r[12]),
            )
        rec["notes"] = r[13].strip()
        fg.append(rec)

fdf = pd.DataFrame(fg)


# ---------- verification (computable from the data itself) ----------
def verify(row):
    flags = []
    nb = row.nickname
    fra, fdec = ramap[nb], decmap[nb]
    sep_as = imp_zphot = imp_zhost = ratio = np.nan
    if np.isfinite(row.ra_deg) and np.isfinite(row.dec_deg) and np.isfinite(fra):
        c1 = SkyCoord(fra * u.deg, fdec * u.deg)
        c2 = SkyCoord(row.ra_deg * u.deg, row.dec_deg * u.deg)
        sep = c1.separation(c2)
        sep_as = sep.arcsec
        if sep.deg > 0.5:
            flags.append("COORD_FAR")
        if np.isfinite(row.z_phot) and row.z_phot > 0:
            da = Planck18.angular_diameter_distance(row.z_phot)
            imp_zphot = (sep.radian * da).to(u.kpc).value
        zh = row.host_z_spec
        if np.isfinite(zh) and zh > 0:
            da = Planck18.angular_diameter_distance(zh)
            imp_zhost = (sep.radian * da).to(u.kpc).value
    if np.isfinite(row.impact_kpc_listed) and np.isfinite(imp_zphot) and imp_zphot > 0:
        ratio = imp_zphot / row.impact_kpc_listed
        if not (0.8 < ratio < 1.25):
            flags.append("IMPACT_MISMATCH_zphot")
    zh = row.host_z_spec
    if not np.isfinite(zh):
        flags.append("HOST_Z_UNKNOWN")
    elif np.isfinite(row.z_phot) and row.z_phot >= zh:
        flags.append(
            "ZPHOT_GE_ZHOST"
        )  # point est. behind/at host; needs photo-z error to adjudicate
    b_over_r500 = np.nan
    if (
        row.type == "cluster"
        and np.isfinite(getattr(row, "r500_mpc", np.nan))
        and row.r500_mpc > 0
        and np.isfinite(row.impact_kpc_listed)
    ):
        b_over_r500 = row.impact_kpc_listed / (row.r500_mpc * 1000.0)
        if b_over_r500 > 1.0:
            flags.append("IMPACT_GT_R500")  # sightline outside cluster R500 -> not pierced
    return pd.Series(
        dict(
            sep_arcsec=sep_as,
            impact_kpc_recomp_zphot=imp_zphot,
            impact_kpc_recomp_zhost=imp_zhost,
            impact_ratio_zphot=ratio,
            b_over_r500=b_over_r500,
            flags="|".join(flags),
        )
    )


vdf = fdf.apply(verify, axis=1)
fdf = pd.concat([fdf, vdf], axis=1)

# counts back onto bursts
nh = fdf[fdf.type == "halo"].groupby("nickname").size()
nc = fdf[fdf.type == "cluster"].groupby("nickname").size()
bdf["n_foreground_halo"] = bdf.nickname.map(nh).fillna(0).astype(int)
bdf["n_foreground_cluster"] = bdf.nickname.map(nc).fillna(0).astype(int)

# ---------- Sheet2 cross-check of counts ----------
s2 = pd.read_csv(S2, header=None, dtype=str).fillna("")
s2_h = s2_c = {}
cur = None
h2 = {}
c2 = {}
for i in range(1, len(s2)):
    r = s2.iloc[i]
    if r[0].strip() in NICKSET:
        cur = r[0].strip()
    if cur:
        if r[13].strip() or r[14].strip():  # halo label or obj_ID
            h2[cur] = h2.get(cur, 0) + 1
        if r[23].strip() or r[24].strip():  # cluster label or obj_name
            c2[cur] = c2.get(cur, 0) + 1

# ---------- write + report ----------
bcols = [
    "nickname",
    "tns",
    "mjd",
    "ra_deg",
    "dec_deg",
    "z_spec",
    "gamma_1p4ghz_mhz",
    "gamma_1p4ghz_err_mhz",
    "gamma_1p4ghz_ne2001_mhz",
    "tau_1ghz_ms",
    "tau_1ghz_err_plus_ms",
    "tau_1ghz_err_minus_ms",
    "n_foreground_halo",
    "n_foreground_cluster",
    "localization",
]
bdf[bcols].to_csv(os.path.join(OUT, "bursts.csv"), index=False)
fcols = [
    "nickname",
    "tns",
    "host_z_spec",
    "type",
    "tag",
    "obj",
    "obj_class",
    "survey",
    "ra_deg",
    "dec_deg",
    "z_phot",
    "impact_kpc_listed",
    "prob_gal",
    "prob_star",
    "prob_qso",
    "r500_mpc",
    "lam500_richness",
    "m500_1e14msun",
    "n_galaxies",
    "sep_arcsec",
    "impact_kpc_recomp_zphot",
    "impact_kpc_recomp_zhost",
    "impact_ratio_zphot",
    "b_over_r500",
    "flags",
    "notes",
]
for c in fcols:
    if c not in fdf.columns:
        fdf[c] = np.nan
fdf[fcols].to_csv(os.path.join(OUT, "foreground.csv"), index=False)

print(
    "BURSTS: %d  (halos=%d, clusters=%d, total fg=%d)"
    % (len(bdf), (fdf.type == "halo").sum(), (fdf.type == "cluster").sum(), len(fdf))
)
print("\nPer-burst counts  [Sheet1 halo/cluster]  vs  [Sheet2 halo/cluster]:")
for n in NICK:
    print(
        "  %-11s S1: %2d / %-2d   S2: %2d / %-2d  %s"
        % (
            n,
            int(bdf.set_index("nickname").loc[n, "n_foreground_halo"]),
            int(bdf.set_index("nickname").loc[n, "n_foreground_cluster"]),
            h2.get(n, 0),
            c2.get(n, 0),
            "<-- MISMATCH"
            if (
                h2.get(n, 0) != int(bdf.set_index("nickname").loc[n, "n_foreground_halo"])
                or c2.get(n, 0) != int(bdf.set_index("nickname").loc[n, "n_foreground_cluster"])
            )
            else "",
        )
    )

print("\n=== VERIFICATION FLAGS (foreground.csv) ===")
flagged = fdf[fdf["flags"] != ""]
print("flagged %d / %d entries" % (len(flagged), len(fdf)))
with pd.option_context("display.width", 200, "display.max_columns", None):
    print(
        fdf[
            [
                "nickname",
                "type",
                "tag",
                "obj",
                "z_phot",
                "host_z_spec",
                "impact_kpc_listed",
                "impact_kpc_recomp_zphot",
                "impact_ratio_zphot",
                "sep_arcsec",
                "flags",
            ]
        ].to_string()
    )
