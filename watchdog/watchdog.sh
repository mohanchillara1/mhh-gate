#!/bin/bash
# MHH watchdog loop — Layer 1 heartbeat every 60s; Layer 2 detector each tick;
# §5 code-integrity every 10th tick. Launch: setsid nohup /workspace/watchdog/watchdog.sh
#   > /workspace/watchdog.log 2>&1 &   (from inside tmux)
#
# §10 HARD BOUNDARY (absolute): this process and its children may not delete the pod,
# terminate the pod, release/resize/detach the volume, gpu-reset, unload the driver,
# kill any PID other than the registered training PID, or delete/overwrite any
# checkpoint, run dir, log, eval output, telemetry, recovery log, pre-registration
# file, MANIFEST.md or POD_BRIEF.md. Allowed destructive-shaped actions are exactly:
# (1) kill the supervised training PID per §4b, (2) MOVE files into /workspace/archive/,
# (3) per-file restore to pinned content after archiving the drifted copy,
# (4) reinstall a pinned dependency version, (5) STOP (never terminate) the pod.
set -u
W=/workspace
D=$W/watchdog
LOCK=$W/watchdog.pid

# single-instance lock (spec §7)
if [ -f "$LOCK" ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
  echo "watchdog already running (pid $(cat $LOCK)); exiting"
  exit 0
fi
echo $$ > "$LOCK"

mkdir -p $W/flags $W/outbox $W/archive $W/archive/drift $W/logs
TICK=0
echo "[watchdog] started pid $$ at $(date -u +%FT%TZ)"
while true; do
  python3 $D/heartbeat.py       2>&1 | sed 's/^/[hb] /'
  [ -x $D/detector.py ] || [ -f $D/detector.py ] && python3 $D/detector.py 2>&1 | sed 's/^/[det] /'
  if [ $((TICK % 10)) -eq 0 ] && [ -f $D/codecheck.sh ]; then
    bash $D/codecheck.sh 2>&1 | sed 's/^/[code] /'
  fi
  TICK=$((TICK+1))
  sleep 60
done
