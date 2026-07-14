# Handoff: unblock h17 → jakob-mbp → iacobus SSH, then drain iacobus → gdrive

**For:** agent running on **jakob-mbp**  
**From:** h17 session (`lxd110h17` / Tailscale `lxd110h17-hpc`)  
**Date:** 2026-07-14  
**Owner:** Jakob Faber (`jakobtfaber`)

## ✅ CLOSED — 2026-07-13 (mbp session)

All goals met. What actually happened, for the record:

1. **Root cause of the SSH timeout was NOT the Mac.** `sshd` was already
   listening on `*:22`, ALF allowed `sshd-auth`/`sshd-keygen-wrapper`,
   shields-up off. The tailnet **ACL** sent `jakob-mbp` an **empty inbound
   `PacketFilter` (`[]`)** — disco `tailscale ping` pongs while every
   data-plane packet (ICMP, tcp/22/445/5000) is dropped. Diagnostic that
   settles it in seconds: `Tailscale debug netmap` → `PacketFilter: []`.
   Phase A's System-Settings iteration was unnecessary; skip it if this
   recurs and check the netmap first.
2. **Fix:** owner added an ACL grant (`tag:hpc` → `tag:work-laptop`,
   `tcp:22`) in the admin console. Verified with nonce round-trips, not
   relayed echoes: h17 → mbp session shows
   `SSH_CONNECTION=100.85.172.12 → 100.121.73.103:22`; h17 → iacobus runs
   `ProxyCommand ssh -W '[100.93.229.114]:22' jakob-mbp` and lands on
   iacobus with client `100.121.73.103`. Note h17's `iacobus` entry now
   targets the **Tailscale IP** (100.93.229.114) through the mbp jump, not
   `iacobus.local` as drafted above.
3. **Drain complete and verified.** `gdrive-jakob` on iacobus confirmed by
   live handshake; metadata sentinel SHA-256 **PASS**; parallel bulk run
   found `burst_pickles` (59.1 GiB) and `burst_npys` (60.8 GiB) already
   100% uploaded by the 2026-06-26 run; `archive` + `rest` completed
   2026-07-13. Final `rclone check --size-only` (excl. `burstprop_paper/**`):
   **0 differences, 5437 matching files**; remote
   `gdrive-jakob:Research/CHIME_DSA_Codetections` = 244.815 GiB / 5438
   objects. The 4 ERROR lines in the archive log are June temp-file races,
   retried successfully 2026-07-02.
4. **Quarantine done (move-only):** source tree now at
   `iacobus:~/Research/_quarantine/CHIME_DSA_Codetections-drained-20260713/`
   with a `PROVENANCE.md` recording evidence and disposal conditions.
   `CHIME_Morphologies`/`burstprop_paper` untouched throughout.
5. **Phase E (h17 compute slice) not started** — optional; pull from the
   quarantine tree or Drive before any disposal of the quarantine.

## Goal

1. Make **SSH from h17 to jakob-mbp over Tailscale** work (TCP port 22).
2. Confirm **ProxyJump** h17 → mbp → `iacobus.local` works.
3. Run **iacobus → Google Drive** drain with **no staging disk on mbp** (rclone runs on iacobus; mbp only relays SSH if needed). Optionally leave a small compute subset on h17 later.

**Out of scope:** `CHIME_Morphologies` (separate project). Do not upload or delete that tree as part of this job.

## Why this is blocked

| Check from h17 | Result |
|----------------|--------|
| `tailscale ping jakob-mbp` | OK (~30 ms) |
| `tailscale ping iacobus-bkp-mbp` | OK |
| `ssh jakob-mbp` → `100.121.73.103:22` | **Connection timed out** |
| `ssh iacobus` (ProxyJump via mbp) | Fails (no first hop) |

Remote Login works **on the home LAN** (`ssh iacobus` from mbp → `iacobus.local` → OK). It does **not** accept SSH from h17’s Tailscale CGNAT address.

h17 SSH config is already set for the intended topology (do not redesign unless broken):

```
h17 --Tailscale--> jakob-mbp (100.121.73.103)
              --LAN--> iacobus.local
```

Relevant h17 `~/.ssh/config` (already written; for your awareness):

```
Host jakob-mbp
    HostName 100.121.73.103
    User jakobfaber
    IdentityFile ~/.ssh/jfaber_key
    IdentitiesOnly yes

Host iacobus
    HostName iacobus.local
    User iacobus
    IdentityFile ~/.ssh/iacobus_key
    IdentitiesOnly yes
    ProxyJump jakob-mbp
```

## Success criteria

On **jakob-mbp**, after fixes:

```bash
# sshd must listen on Tailscale (100.x), not only en0/LAN
sudo lsof -iTCP:22 -sTCP:LISTEN
# expect *:22 or 100.121.73.103:22

# self-test over Tailscale IP
ssh -o BatchMode=yes jakobfaber@100.121.73.103 'hostname; echo MBP_TS_OK'
```

Then ask the human (or an h17 agent) to confirm from h17:

```bash
ssh jakob-mbp 'hostname; echo MBP_OK'
ssh iacobus 'hostname; echo IACOBUS_OK'
ssh iacobus 'rclone listremotes; rclone about gdrive-jakob: | head'
```

All three must succeed before starting the bulk drain.

## Phase A — Fix macOS Remote Login / firewall for Tailscale (mbp)

Do this on **jakob-mbp**:

1. **System Settings → General → Sharing → Remote Login**
   - On
   - Allow access for **All users** (or at least `jakobfaber`)
2. **System Settings → Network → Firewall**
   - If On: **Options** → allow incoming for **Remote Login** / `sshd`
   - Fast test: turn Firewall **Off**, retest Tailscale SSH, then re-enable with an allow rule if that was the cause
3. Confirm Tailscale is **Connected** (MagicDNS / IP `100.121.73.103`)
4. Confirm listen sockets:

```bash
sudo lsof -iTCP:22 -sTCP:LISTEN
sudo pfctl -s rules 2>/dev/null | head   # if pf is in play
```

5. Self-test:

```bash
ssh -v jakobfaber@100.121.73.103 true
# Must NOT hang on "Connecting to 100.121.73.103 port 22"
```

If LAN SSH works but Tailscale IP SSH hangs on the same Mac, the firewall or interface binding is still wrong — keep iterating here; do not start data moves.

Optional (nicer long-term): enable **Tailscale SSH** in the Tailscale admin console for `jakob-mbp` / `iacobus-bkp-mbp` so peers can use Tailscale-authenticated SSH. Not required if classic sshd listens on `100.x:22`.

## Phase B — Verify iacobus hop (from mbp, then from h17)

From **mbp** (already known working via LAN):

```bash
ssh iacobus 'hostname; echo OK; du -sh ~/Research/CHIME_DSA_Codetections; ls ~/Research/CHIME_DSA_Codetections'
```

After Phase A, have h17 run:

```bash
ssh iacobus 'hostname; echo IACOBUS_OK'
```

That uses ProxyJump; second hop is `iacobus.local` on the LAN (correct).

## Phase C — Ensure rclone → gdrive on iacobus

Canonical remote: **`gdrive-jakob`** → account **jakobtfaber@gmail.com**  
Target: `gdrive-jakob:Research/CHIME_DSA_Codetections`  
Source: `iacobus:/Users/iacobus/Research/CHIME_DSA_Codetections` (~220–280 GiB)

On **iacobus** (via `ssh iacobus` from mbp or h17):

```bash
rclone listremotes | grep gdrive-jakob
rclone about gdrive-jakob:
rclone mkdir gdrive-jakob:Research/CHIME_DSA_Codetections
rclone lsd gdrive-jakob:Research/
```

If `gdrive-jakob` is missing, configure once (OAuth needs a browser — do authorize on mbp if needed):

```bash
# On mbp (browser):
rclone authorize "drive"
# On iacobus: rclone config → create remote gdrive-jakob → paste token
```

Script (in FLITS repo): `scripts/migration/iacobus_to_gdrive.sh`  
Docs: `DATA_LOCATIONS.md`, `docs/infrastructure/DEFERRED_MIGRATION.md` (D5).

## Phase D — Drain iacobus → gdrive (no mbp disk)

Prefer orchestrating from a machine that can `ssh iacobus` (h17 once unblocked, or mbp for control only). The script SSHs to iacobus and runs rclone **there**.

From FLITS checkout:

```bash
cd /path/to/dsa110-FLITS   # on mbp: ~/Developer/repos/github.com/jakobtfaber/dsa110-FLITS

# smoke
./scripts/migration/iacobus_to_gdrive.sh --dry-run --subdir metadata
./scripts/migration/iacobus_to_gdrive.sh --subdir metadata

# bulk (~280G) — parallel jobs on iacobus
./scripts/migration/iacobus_to_gdrive.sh --parallel
# monitor:
ssh iacobus 'tail -f ~/logs/gdrive_transfers/iacobus_to_gdrive_*.log'
```

Parallel jobs cover `archive/`, `burst_pickles/`, `burst_npys/`, and the rest.  
**Excludes** `burstprop_paper` (Morphologies — leave alone).

Verify:

```bash
./scripts/migration/iacobus_to_gdrive.sh --verify-only
ssh iacobus 'rclone size gdrive-jakob:Research/CHIME_DSA_Codetections'
ssh iacobus 'du -sh ~/Research/CHIME_DSA_Codetections'
```

Only after verify — **move-only quarantine** on iacobus (project policy: never `rm -rf` research trees):

```bash
ssh iacobus 'mkdir -p ~/Research/_quarantine && mv ~/Research/CHIME_DSA_Codetections ~/Research/_quarantine/CHIME_DSA_Codetections-drained-$(date +%Y%m%d)'
```

## Phase E — Optional: compute slice on h17 (not required to free iacobus)

If h17 still needs job inputs after drain, rsync from iacobus (or later from gdrive) **only**:

- `burst_pickles/` (~61G) — missing on h17  
- remainder of `burst_npys/` (h17 has ~5/218 under `numpy/burst_npys/`)  
- `scattering_results/` (h17 stub)

`dsa_fullstokes_waterfalls/` is already ~complete on h17 (`numpy/dsa_fullstokes_waterfalls/`).

h17 root: `/data/research/astrophysics/frbs/chime-dsa-codetections`  
Layout notes: `docs/infrastructure/H17_WORKSPACE.md` in this repo.

## Do / don’t

| Do | Don’t |
|----|--------|
| Fix mbp `:22` on Tailscale first | Stage ~280G on mbp (disk full) |
| Run rclone **on iacobus** → gdrive | Route bulk bytes through mbp disk |
| Quarantine with `mv` after verify | `rm -rf` codetection trees |
| Skip Morphologies | Upload/delete `CHIME_Morphologies` |

## Report back

Paste:

1. `sudo lsof -iTCP:22 -sTCP:LISTEN` on mbp  
2. Result of `ssh jakobfaber@100.121.73.103 'echo MBP_TS_OK'`  
3. Whether h17 `ssh jakob-mbp` / `ssh iacobus` now work (ask human/h17 agent)  
4. If drain started: log paths + `rclone size` summary  

## Context links

- Repo: https://github.com/jakobtfaber/dsa110-FLITS  
- Manuscript pin: https://github.com/jakobtfaber/Faber2026 (`pipeline/` → this fork)  
- This handoff path: `docs/infrastructure/HANDOFF_mbp_tailscale_ssh_iacobus.md`
