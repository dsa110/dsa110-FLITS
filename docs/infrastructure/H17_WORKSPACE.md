# h17 compute workspace

**Host:** `h17` (`lxd110h17`)  
**Root:** `/data/research/astrophysics/frbs/chime-dsa-codetections`  
**Role:** CHIME/DSA co-detection **compute + artifact cache**. Not a third source of truth.

Canonical code lives in GitHub (`jakobtfaber/dsa110-FLITS`, developed on `jakob-mbp`). The manuscript pin is `jakobtfaber/Faber2026` → submodule `pipeline/` → this fork. Data authority is **Google Drive** (`gdrive-jakob:Research/CHIME_DSA_Codetections`, verified drained 2026-07-13); the iacobus staging tree is quarantined move-only at `iacobus:~/Research/_quarantine/CHIME_DSA_Codetections-drained-20260713/` (see [`DATA_LOCATIONS.md`](../../DATA_LOCATIONS.md)).

## Layout

| Path | Tracked? | Role |
|------|----------|------|
| `dsa110-FLITS/` | git clone | Canonical analysis checkout on this host (`main`) |
| `dm_campaign/flits-dm-campaign/` | git worktree | Branch `dm-campaign-2026-07` (do not rsync FLITS trees) |
| `scripts/` | local only | Historical workers; prefer `dsa110-FLITS/scripts/h17_codetections/` |
| `bin/` | local only | Docker wrappers + CANFAR download helpers |
| `metadata/` | local only | Burst fixture used by download/upchan jobs |
| `chime_singlebeam/` | **data** | 12× CHIME singlebeam `.h5` (~14G) |
| `dsa_filterbanks/` | **data** | DSA SIGPROC `.fil` inputs |
| `upchan_codetections/` | **product** | Upchannelized CHIME spectra (current) |
| `results/`, `diagnostics/` | **product** | Small JSON/PNG/CSV — promote into FLITS when citable |
| `manifest_cubes/`, `numpy/` | **product** | Large intermediates — stay here / stage to iacobus |
| `archive/` | cold | June 2026 dump (~36G); not on the active path |
| `baseband-analysis-canfar-src/` | vendor cache | Upstream CHIME baseband-analysis source snapshot |

## Code sync (mbp → h17)

```bash
cd /data/research/astrophysics/frbs/chime-dsa-codetections/dsa110-FLITS
git pull --ff-only origin main
```

DM campaign worktree:

```bash
cd /data/research/astrophysics/frbs/chime-dsa-codetections/dm_campaign/flits-dm-campaign
git fetch origin dm-campaign-2026-07
git checkout origin/dm-campaign-2026-07   # detached OK for compute
```

## What to promote back (h17 → FLITS / Faber)

| Promote into git | Leave on h17 / iacobus / gdrive |
|------------------|----------------------------------|
| Python workers under `scripts/h17_codetections/` | `*.h5`, `*.fil`, multi-GB `*.npy` cubes |
| Small JSON/CSV results, QA PNG diagnostics | `archive/`, docker image layers, `venv/` |
| Regenerated manuscript tables/figures via Faber | Superseded `upchan_codetections/*SUPERSEDED*` |

Checksum large products and list them in manifests (`codetections_manifest.yaml`, Faber `repro_manifest.csv`) instead of committing binaries.

## Path config

Workers default to absolute h17 roots:

- `LOCAL_H5_DIR=/data/research/astrophysics/frbs/chime-dsa-codetections/chime_singlebeam`
- `DEFAULT_OUT_DIR=.../upchan_codetections`

Burst catalog / VOS URIs: workspace `metadata/notebook_reproduction_fixture.json` (mirrors FLITS `crossmatching/notebook_reproduction_fixture.json`). Prefer the FLITS copy when editing; sync to h17 `metadata/` for download scripts that still read the local path.

Docker entrypoint on this host: workspace `bin/baseband_analysis_python.sh` → image `chimefrb/baseband-analysis:latest`.

## SSH topology (verified 2026-07-13)

h17 → jakob-mbp → iacobus works end-to-end: tailnet ACL grant
`tag:hpc → tag:work-laptop` (`tcp:22`) added 2026-07-13 (before that,
jakob-mbp's inbound `PacketFilter` was empty — every data-plane packet
dropped; check `Tailscale debug netmap` first if this recurs). h17's
`ssh iacobus` ProxyJumps through `jakob-mbp` to iacobus's Tailscale IP
(`100.93.229.114`). Drain closeout details:
[`HANDOFF_mbp_tailscale_ssh_iacobus.md`](HANDOFF_mbp_tailscale_ssh_iacobus.md).

## History note (2026-07-13)

`jakobtfaber/dsa110-FLITS` `main` was history-rewritten so commit authors are only Jakob Faber identities (Cursor/Copilot/Devin scrubbed). That rewrite replaced shared commit IDs with `dsa110/dsa110-FLITS`; do **not** merge org upstream by SHA without an intentional reconnect plan. Prefer cherry-picks of specific upstream commits into the fork.

## Related

- [`DATA_LOCATIONS.md`](../../DATA_LOCATIONS.md) — 4-host model  
- [`scripts/h17_codetections/README.md`](../../scripts/h17_codetections/README.md) — promoted workers  
- Faber2026 `PIPELINE.md` — manuscript submodule pin to this fork  
