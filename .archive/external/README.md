# External repositories referenced by the CHIME–DSA co-detections analysis

This directory intentionally contains **no source clones**. The plan's original
intent ("clone mirror into `.archive/external/{subhalos,los_halos}/` as
read-only snapshots") would have embedded 500+ MB of `.git` pack files into this
repository. Instead, we record provenance here: repo URL, the exact commit
SHA we were using at the time of the fold, and where the working clone lived
before quarantine.

If you need to reproduce the state at the fold date, clone the repo and check
out the pinned SHA.

## Pinned state at fold time (2026-04-23)

| local path (pre-quarantine) | remote | pinned sha | notes |
|---|---|---|---|
| `chime_dsa_codetections/subhalos/` | `https://github.com/jakobtfaber/subhalos.git` | `3907d4007` | user-owned repo; local clone had its own `.venv/` and `.egg-info/`, excluded here |
| `chime_dsa_codetections/halos/los_halos/` | `https://github.com/jakobtfaber/los_halos.git` | `72bf7a63f` | user-owned repo |
| `chime_dsa_codetections/halos/archive/los_halos_new/` | `https://github.com/jakobtfaber/los_halos.git` | `7dccb1c86` | older snapshot of same repo (superseded) |
| `chime_dsa_codetections/dm_budget/FRB/` | `https://github.com/FRBs/FRB.git` | `4c85e6ae3` | third-party (`FRBs` org) — consumed via clone, no local edits intended |
| `chime_dsa_codetections/dm_budget/frb_baryon_connor2024/` | `https://github.com/liamconnor/frb_baryon_connor2024.git` | `efd391b25` | third-party (Connor et al. 2024) — consumed via clone |

## Rehydration recipe

```bash
mkdir -p external && cd external
git clone https://github.com/jakobtfaber/subhalos.git            && (cd subhalos && git checkout 3907d4007)
git clone https://github.com/jakobtfaber/los_halos.git           && (cd los_halos && git checkout 72bf7a63f)
git clone https://github.com/FRBs/FRB.git                         && (cd FRB && git checkout 4c85e6ae3)
git clone https://github.com/liamconnor/frb_baryon_connor2024.git && (cd frb_baryon_connor2024 && git checkout efd391b25)
```

## Why we don't just fold them

- `subhalos/` and `los_halos/` are separate user-owned repos with their own
  issue trackers, histories, and release cadences. The plan explicitly says
  "do not touch the separate repos."
- `FRB/` and `frb_baryon_connor2024/` are third-party packages — installing
  them via `pip install -e .` in a separate checkout keeps them importable
  without tangling their history with ours.
