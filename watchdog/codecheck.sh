#!/bin/bash
# §5 code-integrity check — runs every 10th watchdog tick (~10 min).
# VERIFY continuously; the only permitted repair is RESTORE-TO-FROZEN, per-file,
# after archiving the drifted copy to /workspace/archive/drift/ (allowlist item 3).
# Never a blanket discard of a working tree. Every action -> RECOVERY_LOG.md.
# Runs overlapping a drift window are marked SUSPECT in RECOVERY_LOG.md (§5.3).
set -u
W=/workspace
G=$W/Isaac-GR00T
PIN=51d4c89f72fda44cbf77285c6a8114b52676b8a1
GATE=$W/mhh-gate
GATE_MANIFEST=$W/watchdog/gate.sha256   # captured at watchdog install
RLOG=$W/RECOVERY_LOG.md
FLAGS=$W/flags
TS() { date -u +%FT%TZ; }
rlog() { printf -- "- %s — %s\n" "$(TS)" "$*" >> "$RLOG"; }
flag() { printf '{"flag":"CODE_DRIFT","ts":"%s","evidence":"%s"}\n' "$(TS)" "$1" > $FLAGS/CODE_DRIFT; }
suspect_running() {  # mark any in-flight run as suspect
  if [ -f $W/run.current.json ]; then
    rlog "SUSPECT: run $(cat $W/run.current.json | tr -d '\n' | head -c 200) overlapped drift window: $1"
  fi
}

mkdir -p $FLAGS $W/archive/drift

# 1+2. upstream pin + clean tree
HEAD=$(git -C $G rev-parse HEAD 2>/dev/null)
if [ "$HEAD" != "$PIN" ]; then
  flag "Isaac-GR00T HEAD $HEAD != pin $PIN"
  suspect_running "HEAD moved to $HEAD"
  rlog "CODE_DRIFT: HEAD=$HEAD expected=$PIN — archiving diff, restoring pin per-file NOT possible for a commit move; flagged only (a commit move needs eyes: recorded, not auto-checked-out, because the cause matters)"
else
  DIRTY=$(git -C $G status --porcelain --untracked-files=no 2>/dev/null)
  if [ -n "$DIRTY" ]; then
    flag "uncommitted edits in pinned upstream"
    suspect_running "dirty tree: $(echo "$DIRTY" | head -3 | tr '\n' ';')"
    DRIFTDIR=$W/archive/drift/$(date -u +%Y%m%dT%H%M%SZ)
    mkdir -p "$DRIFTDIR"
    echo "$DIRTY" | while read -r st path; do
      [ -z "$path" ] && continue
      # archive the drifted copy, then per-file restore to pinned content (allowlist 3)
      mkdir -p "$DRIFTDIR/$(dirname "$path")"
      cp -a "$G/$path" "$DRIFTDIR/$path" 2>/dev/null
      git -C $G checkout -- "$path" 2>/dev/null \
        && rlog "RESTORED to pin: $path (drifted copy at $DRIFTDIR/$path)" \
        || rlog "RESTORE FAILED for $path — left in place, flagged"
    done
  fi
fi

# 3. mhh-gate SHA manifest
if [ -f "$GATE_MANIFEST" ]; then
  if ! (cd $GATE && sha256sum -c $GATE_MANIFEST --quiet 2>/dev/null); then
    flag "mhh-gate file changed vs install-time manifest"
    suspect_running "mhh-gate sha mismatch"
    (cd $GATE && sha256sum -c $GATE_MANIFEST 2>/dev/null | grep -v OK | head -5) >> "$RLOG"
    rlog "CODE_DRIFT in mhh-gate: NOT auto-restored (gate code has no committed pin on the pod) — NEEDS_HUMAN notified, monitoring continues"
    printf "\n## %s — mhh-gate drift\nGate files changed while watchdog active; see RECOVERY_LOG.md. Runs in flight marked suspect. Watchdog continues.\n" "$(TS)" >> $W/NEEDS_HUMAN.md
  fi
fi

# 4. frozen human values
FS=$(grep -E '^FIXED_STEP_COUNT' $GATE/config.py | grep -o '6000' | head -1)
SHA=$(grep -E '^EVAL_EPISODE_LIST_SHA256' $GATE/config.py | grep -o 'd51371a5[0-9a-f]*' | head -1)
if [ "$FS" != "6000" ] || [ "${SHA:0:8}" != "d51371a5" ]; then
  flag "frozen values changed: FIXED_STEP_COUNT='$FS' SHA_prefix='${SHA:0:8}'"
  suspect_running "frozen config values changed"
  rlog "FROZEN VALUE DRIFT in config.py — never auto-edited (pre-registration); flagged + NEEDS_HUMAN"
  printf "\n## %s — frozen config drift\nconfig.py frozen values differ from pre-registration. NOT auto-fixed. Runs suspect.\n" "$(TS)" >> $W/NEEDS_HUMAN.md
fi

# 4b. eval list hash (spec §6: restore-from-source is sanctioned, but the recorded
# source IS the frozen file itself + git history; here we verify and flag only)
ACTUAL=$(sha256sum $GATE/eval_episodes.json 2>/dev/null | cut -d' ' -f1)
if [ "$ACTUAL" != "d51371a59be544a0d26f4a0f27171fd0d65d39bb94cadf1b2635de67f3b61663" ]; then
  flag "eval_episodes.json sha=$ACTUAL != frozen"
  suspect_running "eval list changed"
  rlog "EVAL LIST DRIFT: sha=$ACTUAL — flagged; restore requires the recorded source (local git), see NEEDS_HUMAN"
  printf "\n## %s — eval list drift\neval_episodes.json no longer matches the frozen sha. Restore from mhh-gate git (commit 4b29e9b) then re-verify.\n" "$(TS)" >> $W/NEEDS_HUMAN.md
fi

# 5. dependency versions vs pins (recorded, drift flagged; reinstall is allowlist 4 — L3)
TORCH=$(timeout 60 $G/.venv/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null)
[ -n "$TORCH" ] && [ "${TORCH%%+*}" != "2.9.0" ] && { flag "torch=$TORCH != 2.9.0"; rlog "DEP DRIFT: torch=$TORCH (pin 2.9.0)"; }
MJ=$(timeout 60 $G/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python -c 'import mujoco; print(mujoco.__version__)' 2>/dev/null)
[ -n "$MJ" ] && [ "$MJ" != "3.3.1" ] && { flag "mujoco=$MJ != 3.3.1"; rlog "DEP DRIFT: mujoco=$MJ (pin 3.3.1)"; }

# 6. full-volume usage sample for DISK_LOW (own timestamp; detector never interpolates)
USED=$(timeout 240 du -sb $W 2>/dev/null | cut -f1)
if [ -n "$USED" ]; then
  printf '{"used_bytes":%s,"ts":"%s"}\n' "$USED" "$(TS)" > $FLAGS/.volume_used_sample.json
fi
