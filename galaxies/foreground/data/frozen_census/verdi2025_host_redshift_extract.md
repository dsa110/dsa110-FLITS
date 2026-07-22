# Verdi host-redshift source extract

Minimal source-bearing extract from the owner-adopted draft table “New DSA-110
FRB Burst Properties and Host-Galaxy Redshifts.” The source archive is the
Overleaf export `Probing_Host_Galaxy_Environments_with_a_New_Sample_of_Localized_FRBs_Detected_with_the_DSA_110.zip`,
downloaded 2026-07-17. SHA-256: `88cab4b89d13dffb9cdaae49edb24455a66dfd99f4c7ca23bebcc86676043621`.
The inner `verdi2025.tex` SHA-256 is
`ea094a20d5cac53d79fde24e696c5c4aca967d82067e3dc7f23c8a6cdb640e90`.

The CSV preserves only the nine rows relevant to this sample. TNS suffix
differences are mapped explicitly; only redshift assignment is adopted. A
source `--` is stored as a missing value. No redshift is inferred.

Zach, Whitney, and Oran predate this source table; their published source rows
are frozen separately in `law2024_host_redshift_extract.csv`.
