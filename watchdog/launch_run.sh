#!/bin/bash
# The ONLY sanctioned way to start a gate run. Registers the run for the watchdog
# and enforces spec §6: dry-run artifacts NEVER land in the real results dir.
#   bash /workspace/watchdog/launch_run.sh T1 11 [--dry-run]
set -u
ARM=$1; SEED=$2; DRY=${3:-}
W=/workspace
export GROOT_REPO=$W/Isaac-GR00T HF_HOME=$W/hf
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
PY=$GROOT_REPO/.venv/bin/python
if [ "$DRY" = "--dry-run" ]; then
  export MHH_RUNS_DIR=$W/scratch-dryruns    # §6: never the real results path
  LOG=$W/logs/dryrun_${ARM}_s${SEED}.log
else
  export MHH_RUNS_DIR=$W/mhh-runs
  LOG=$W/logs/${ARM}_s${SEED}.log
fi
mkdir -p $MHH_RUNS_DIR $W/logs
cd $W/mhh-gate
nohup $PY run_gate.py --arm $ARM --seed $SEED $DRY > $LOG 2>&1 &
PID=$!
echo $PID > $W/run.pid
printf '{"arm":"%s","seed":%s,"dry":%s,"pid":%s,"log":"%s","max_steps":6000,"runs_dir":"%s","started":"%s"}\n' \
  "$ARM" "$SEED" "$([ -n "$DRY" ] && echo true || echo false)" "$PID" "$LOG" "$MHH_RUNS_DIR" "$(date -u +%FT%TZ)" \
  > $W/run.current.json
echo "launched $ARM seed $SEED pid $PID -> $LOG (runs_dir=$MHH_RUNS_DIR)"
