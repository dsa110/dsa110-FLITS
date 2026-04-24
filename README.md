# FLITS | FRB Intensity Analysis Pipeline

**F**itting **L**ikelihoods **I**n **T**ime-Frequency **S**pectra: A lightweight, modular, telescope-agnostic toolkit for fitting pulse-broadening and scintillation in Fast Radio Burst (FRB) dynamic spectra, and instrumental effects.

---

## 📚 Documentation

The full documentation is available in the [`docs/`](docs/) directory.

- [**Getting Started**](docs/getting-started/quickstart.md)
- [**Architecture & Data Flow**](docs/architecture/overview.md)
- [**Comprehensive System Reference**](docs/architecture/system_reference.md)
- [**Analysis Inventory**](docs/architecture/inventory.md)
- [**Installation**](docs/getting-started/installation.md)

---

## Consolidated modules (2026-04 monorepo fold)

As of 2026-04, this repository is the canonical home for the CHIME–DSA
co-detection analysis. Satellite code that previously lived scattered across
the Mac-local `chime_dsa_codetections/` tree was folded in on branch
`consolidation/fold-satellites`.

| module | date folded | source (pre-fold) | iCloud data blob | quarantine path for original |
|---|---|---|---|---|
| `notebooks/codetections/interveners.ipynb` | 2026-04-23 | `chime_dsa_codetections/interveners.ipynb` | n/a (notebook only, 560 KB) | `_quarantine/satellites/notebooks/` (Phase 6) |
| `.archive/old_stuff/` | 2026-04-23 | `chime_dsa_codetections/old_stuff/` | `.fil` blobs remain at source (iCloud placeholders) | `_quarantine/satellites/old_stuff/` (Phase 6) |
| `.archive/external/` | 2026-04-23 | `chime_dsa_codetections/{subhalos,halos,dm_budget/FRB,dm_budget/frb_baryon_connor2024}` | n/a (URL + SHA pointers only) | originals remain as standalone repos (plan: "do not touch") |

For the data tier, see [`DATA_LOCATIONS.md`](DATA_LOCATIONS.md) and
[`codetections_manifest.yaml`](codetections_manifest.yaml).

---

## Citing & License

Please cite **Faber et al., _in prep._ (2025)** if you use this code.
