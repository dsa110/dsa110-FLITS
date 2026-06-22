# Intervening foreground catalog

Foreground halos and galaxy clusters along the sightlines to the 12 CHIME/DSA co-detected FRBs, with each candidate **independently validated against public catalogs** (DESI Legacy Survey DR9 / Zhou+2021 photo-z, DESI DR1 spec-z, NED, PS1-STRM). Every object — confirmed, refuted, and inconclusive — is listed.

!!! note "Verdict summary"

    **49** candidate intervening objects across 12 FRBs: **29 confirmed** foreground · **7 refuted** (background) · **13 inconclusive**. All 49 exist in ≥1 public catalog.

**Verdict definitions** — *confirmed*: best catalog redshift (spec-z, or photo-z ± error) lies below the FRB host redshift; *refuted*: redshift at/above the host (background); *inconclusive*: redshift straddles the host within 1σ, the host has no spec-z, or no trustworthy redshift exists (PS1-STRM `UNSURE` / extrapolated photo-z).

!!! warning "Caveats carried in this table"

    - **14 of 15 clusters lie outside their own $R_{500}$** (`b/R500`>1): real foreground systems, but the sightline does not pierce them.

    - The original spreadsheet `z_phot` column is **unreliable** — for the PS1-STRM halos it is decoupled from the actual catalog value (e.g. zach 0.013→0.469). Trust the `redshift` column here, not the sheet.


| Burst | TNS | Type | Obj ID | Survey | $b$ (kpc) | $b/R_{500}$ | $z$ | $z$ source | Class | Verdict | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zach | FRB 20220207C | halo | 195373100910393540 | WISE/PS1/STRM | 75.9 |  | 0.469 ± 0.047 | PS1-STRM phot | GALAXY | inconclusive | photo-z extrapolated |
| whitney | FRB 20220310F | cluster | J085546.0+732230, 1160094 | DESI/WISE (Wen+) | 2039.4 | 3.93 | 0.128 | DESI spec | cluster | confirmed | DESI spec < host |
| whitney | FRB 20220310F | cluster | J085531.9+732432, 1159975 | DESI/WISE (Wen+) | 5210.8 | 9.03 | 0.402 | DESI spec | cluster | confirmed | DESI spec < host |
| whitney | FRB 20220310F | cluster | J085808.2+731234, 1161367 | DESI/WISE (Wen+) | 5690 | 9.63 | 0.257 | DESI spec | cluster | confirmed | DESI spec < host |
| whitney | FRB 20220310F | halo | 196191347354360083 | WISE/PS1/STRM | 103.3 |  | 0.555 ± 0.045 | LS/Zhou phot | REX | refuted | LS/Zhou phot > host |
| whitney | FRB 20220310F | halo | 1472 | Legacy/Zhou21 | 104 |  | 0.555 ± 0.045 | LS/Zhou phot | REX | refuted | LS/Zhou phot > host |
| whitney | FRB 20220310F | halo | 1473 | Legacy/Zhou21 | 104.8 |  | 0.358 ± 0.113 | LS/Zhou phot | REX | confirmed | LS/Zhou phot < host |
| whitney | FRB 20220310F | halo | 1582 | Legacy/Zhou21 | 183.7 |  | 0.471 ± 0.190 | LS/Zhou phot | EXP | inconclusive | within 1σ of host |
| oran | FRB 20220506D | halo | 195393180643665627 | WISE/PS1/STRM | 73.9 |  | — | PS1-STRM | UNSURE | inconclusive | not a galaxy (STRM UNSURE) |
| wilhelm | FRB 20221203A | halo | 194453151328186646 | WISE/PS1/STRM | 232 |  | — | PS1-STRM | UNSURE | inconclusive | not a galaxy (STRM UNSURE) |
| phineas | FRB 20230307A | cluster | J115120.4+714435, 1254337 | DESI/WISE (Wen+) | 603.6 | 0.83 | 0.200 | DESI spec | cluster | confirmed | DESI spec < host |
| phineas | FRB 20230307A | cluster | J115128.2+713637, 1254415 | DESI/WISE (Wen+) | 1054.7 | 1.25 | 0.192 | DESI spec | cluster | confirmed | DESI spec < host |
| phineas | FRB 20230307A | cluster | J114944.0+714348, 1253496 | DESI/WISE (Wen+) | 1569.2 | 2.96 | 0.244 | DESI spec | cluster | confirmed | DESI spec < host |
| phineas | FRB 20230307A | cluster | J115140.5+712732, 1254506 | DESI/WISE (Wen+) | 2104.7 | 3.32 | 0.176 | DESI spec | cluster | confirmed | DESI spec < host |
| phineas | FRB 20230307A | cluster | J114928.5+712526, 1253366 | DESI/WISE (Wen+) | 3049.6 | 3.96 | -0.000 | DESI spec | cluster | confirmed | DESI spec < host |
| phineas | FRB 20230307A | cluster | J115400.9+713320, 1255773 | DESI/WISE (Wen+) | 3060.1 | 5.44 | 0.155 | DESI spec | cluster | confirmed | DESI spec < host |
| phineas | FRB 20230307A | cluster | J115436.9+713930, 1256077 | DESI/WISE (Wen+) | 3989.8 | 7.12 | 0.263 | DESI spec | cluster | confirmed | DESI spec < host |
| phineas | FRB 20230307A | cluster | J115031.4+715735, 1253898 | DESI/WISE (Wen+) | 4272.8 | 6.12 | 0.270 | DESI spec | cluster | confirmed | DESI spec < host |
| phineas | FRB 20230307A | halo | 194051777813062524 | WISE/PS1/STRM | 101.7 |  | 0.110 | DESI spec | DEV | confirmed | DESI spec < host |
| phineas | FRB 20230307A | halo | 1072 | Legacy/Zhou21 | 105.2 |  | 0.300 ± 0.079 | LS/Zhou phot | EXP | inconclusive | within 1σ of host |
| phineas | FRB 20230307A | halo | 194041777780157594 | WISE/PS1/STRM | 111.8 |  | 0.215 | DESI spec | REX | confirmed | DESI spec < host |
| phineas | FRB 20230307A | halo | 1190 | Legacy/Zhou21 | 121.1 |  | 0.110 | DESI spec | DEV | confirmed | DESI spec < host |
| phineas | FRB 20230307A | halo | 983 | Legacy/Zhou21 | 122.2 |  | 0.165 ± 0.059 | LS/Zhou phot | EXP | confirmed | LS/Zhou phot < host |
| phineas | FRB 20230307A | halo | 194021777634832653 | WISE/PS1/STRM | 130.6 |  | 0.193 ± 0.028 | LS/Zhou phot | DEV | confirmed | LS/Zhou phot < host |
| phineas | FRB 20230307A | halo | 1153 | Legacy/Zhou21 | 135.8 |  | 0.215 | DESI spec | REX | confirmed | DESI spec < host |
| phineas | FRB 20230307A | halo | 832 | Legacy/Zhou21 | 158.8 |  | 0.193 ± 0.028 | LS/Zhou phot | DEV | confirmed | LS/Zhou phot < host |
| phineas | FRB 20230307A | halo | 194031778315722893 | WISE/PS1/STRM | 195.9 |  | 0.884 ± 0.116 | LS/Zhou phot | REX | refuted | LS/Zhou phot > host |
| phineas | FRB 20230307A | halo | 986 | Legacy/Zhou21 | 203.4 |  | 0.208 ± 0.067 | LS/Zhou phot | REX | inconclusive | within 1σ of host |
| phineas | FRB 20230307A | halo | 953 | Legacy/Zhou21 | 242.7 |  | 0.199 | DESI spec | REX | confirmed | DESI spec < host |
| freya | FRB 20230325A | halo | 197030881733398302 | WISE/PS1/STRM | 60.1 |  | 0.305 ± 0.068 | PS1-STRM phot | GALAXY | inconclusive | host z unknown |
| freya | FRB 20230325A | halo | 197040882212782495 | WISE/PS1/STRM | 233.9 |  | 0.618 ± 0.126 | PS1-STRM phot | GALAXY | inconclusive | host z unknown |
| hamilton | FRB 20230913A | halo | 192963050359413614 | WISE/PS1/STRM | 136.1 |  | 0.308 ± 0.073 | PS1-STRM phot | GALAXY | inconclusive | within 1σ of host |
| hamilton | FRB 20230913A | halo | 192943050854547067 | WISE/PS1/STRM | 209.2 |  | — | PS1-STRM | UNSURE | inconclusive | not a galaxy (STRM UNSURE) |
| chromatica | FRB 20240203A | halo | 196723126173351736 | WISE/PS1/STRM | 103.5 |  | 0.475 ± 0.044 | PS1-STRM phot | GALAXY | inconclusive | photo-z extrapolated |
| chromatica | FRB 20240203A | halo | 196673126794497004 | WISE/PS1/STRM | 111.1 |  | — | PS1-STRM | UNSURE | inconclusive | not a galaxy (STRM UNSURE) |
| chromatica | FRB 20240203A | halo | 196733128040225775 | WISE/PS1/STRM | 228.1 |  | 0.054 | NED | galaxy | confirmed | NED < host |
| casey | FRB 20240229A | cluster | J111929.5+705441, 1237905 | DESI/WISE (Wen+) | 3133 | 4.09 | 0.216 | DESI spec | cluster | confirmed | DESI spec < host |
| casey | FRB 20240229A | cluster | J111930.9+702041, 1237924 | DESI/WISE (Wen+) | 3804 | 6.32 | 0.091 | DESI spec | cluster | confirmed | DESI spec < host |
| casey | FRB 20240229A | cluster | J112350.9+704142, 1240175 | DESI/WISE (Wen+) | 4267.4 | 5.03 | 0.213 | DESI spec | cluster | confirmed | DESI spec < host |
| casey | FRB 20240229A | cluster | J112235.5+705438, 1239515 | DESI/WISE (Wen+) | 4284.7 | 7.89 | 0.216 | DESI spec | cluster | confirmed | DESI spec < host |
| casey | FRB 20240229A | halo | 660 | Legacy/Zhou21 | 12.2 |  | 0.373 ± 0.067 | LS/Zhou phot | REX | refuted | LS/Zhou phot > host |
| casey | FRB 20240229A | halo | 192821700026167542 | WISE/PS1/STRM | 170.6 |  | 0.203 | DESI spec | SER | confirmed | DESI spec < host |
| casey | FRB 20240229A | halo | 824 | Legacy/Zhou21 | 178.3 |  | 0.203 | DESI spec | SER | confirmed | DESI spec < host |
| casey | FRB 20240229A | halo | 192821699728654764 | WISE/PS1/STRM | 206.4 |  | 0.375 ± 0.049 | LS/Zhou phot | REX | refuted | LS/Zhou phot > host |
| casey | FRB 20240229A | halo | 796 | Legacy/Zhou21 | 212.2 |  | 0.422 ± 0.095 | LS/Zhou phot | REX | refuted | LS/Zhou phot > host |
| casey | FRB 20240229A | halo | 192831699797402822 | WISE/PS1/STRM | 220.7 |  | 0.240 ± 0.012 | LS/Zhou phot | SER | confirmed | LS/Zhou phot < host |
| casey | FRB 20240229A | halo | 795 | Legacy/Zhou21 | 225.4 |  | 0.375 ± 0.049 | LS/Zhou phot | REX | refuted | LS/Zhou phot > host |
| casey | FRB 20240229A | halo | 827 | Legacy/Zhou21 | 237.2 |  | 0.240 ± 0.012 | LS/Zhou phot | SER | confirmed | LS/Zhou phot < host |
| casey | FRB 20240229A | halo | 825 | Legacy/Zhou21 | 281.4 |  | 0.348 ± 0.203 | LS/Zhou phot | REX | inconclusive | within 1σ of host |

!!! info "Provenance"

    Generated by `scratch/codetection/make_catalog_table.py` from the verified pipeline (`normalize_codetection.py` → `validate_foreground.py` → `ps1_strm_adjudicate.py` → `merge_final.py`). Source spreadsheet: `DSA110_CHIME_Codetection_BurstProperties_Foreground`. Cosmology: Planck18.

