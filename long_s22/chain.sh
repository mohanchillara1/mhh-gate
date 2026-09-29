#!/bin/bash
# long_s22/chain.sh ARM -- one pod, one arm, LIBERO-Long seed 22, runs with the laptop closed.
#
# Start it ON THE POD, once, inside tmux (the chain itself does everything else):
#   tmux new -d -s mhh "bash /workspace/mhh-gate/long_s22/chain.sh T5 2>&1 | tee -a /workspace/chain_T5.out"
#
# Required env (set as pod environment variables at pod creation, never committed):
#   HF_TOKEN            HuggingFace READ token with access to nvidia/GR00T-N1.7-3B
#   GH_TOKEN            GitHub token that can push to mohanchillara1/mhh-gate (contents: write)
#   RUNPOD_ACCOUNT_KEY  RunPod key for the balance query (read-only is enough)
#   RUNPOD_POD_ID       set by RunPod itself on every pod
#
# What it does, in order. Any failure = push what exists, STOP the pod, exit. It never retries a
# run and never restarts anything on its own (the crash policy's one rerun is a human decision).
#   0 env + balance preflight (must cover this whole arm)       -> push "started"
#   1 setup: Isaac-GR00T @pin, uv venv, HF access, libero_10 data + modality + stats,
#     osmesa libs, LIBERO sim island, LIBERO config.yaml pre-seeded
#   2 Long overlay (make_overlay.py: verifies the eval list byte for byte)
#   3 dry run of this arm (timed steps, all preflight asserts), osmesa render probe
#   4 real run: run_gate.py --arm ARM --seed 22 under nohup
#   5 watch every 60 s: progress, stall (no log growth 90 min), balance guard every 10 min,
#     hourly heartbeat push, push at train->eval, push + stop at the end
# Results leave the pod ONLY by git push to branch runs/long-s22-ARM (never master, never force).
# There is no Discord webhook on record (checked 2026-09-28), so none is used.
set -uo pipefail

ARM=${1:?usage: chain.sh T1|T5}
case $ARM in T1|T5) ;; *) echo "arm must be T1 or T5"; exit 2 ;; esac
SEED=22
MOCK=${MHH_MOCK:-0}
W=${MHH_W:-/workspace}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
LONG=$W/mhh-long
RUNS=$W/mhh-long-runs
LOGS=$W/logs
LOG=$LOGS/long_${ARM}_s${SEED}.log
RCF=$W/rc_long_${ARM}_s${SEED}
STATUS=$W/STATUS-long-s${SEED}-${ARM}.md
OUTBOX=$W/outbox-long-${ARM}
BRANCH=runs/long-s${SEED}-${ARM}
PUSH_URL=${MHH_PUSH_URL:-https://x-access-token:${GH_TOKEN:-}@github.com/mohanchillara1/mhh-gate.git}
TICK=${MHH_TICK:-60}                  # seconds between checks
HEARTBEAT_S=${MHH_HEARTBEAT_S:-3600}  # push a status every hour
GUARD_S=${MHH_GUARD_S:-600}           # balance guard every 10 min
STALL_S=${MHH_STALL_S:-5400}          # 90 min of no log growth = stalled
SETUP_TIMEOUT_S=${MHH_SETUP_TIMEOUT_S:-10800}
export PIN=51d4c89f72fda44cbf77285c6a8114b52676b8a1
export GROOT_REPO=$W/Isaac-GR00T HF_HOME=$W/hf MHH_RUNS_DIR=$RUNS
# osmesa, not EGL: the seed-21 probe evaluated BOTH arms under osmesa after EGL broke on the pod
# (Learnings 2026-08-31). Same renderer as the result being replicated. run_gate uses setdefault,
# so these exports win.
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa MPLBACKEND=agg
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
if [ "$MOCK" = 1 ]; then PY=python3; else PY=$GROOT_REPO/.venv/bin/python; fi
podctl() { python3 "$HERE/podctl.py" "$@"; }

mkdir -p "$LOGS" "$RUNS"
say()   { echo "[chain $ARM $(date -u +%H:%M:%S)] $*"; }
stamp() { echo "- $(date -u '+%Y-%m-%d %H:%M UTC') — $*" >> "$STATUS"; say "$*"; }

# ---------------------------------------------------------------- results off the pod
outbox_init() {
  [ -d "$OUTBOX/.git" ] && return 0
  if git ls-remote --exit-code "$PUSH_URL" "refs/heads/$BRANCH" >/dev/null 2>&1; then
    git clone -q --depth 5 --branch "$BRANCH" "$PUSH_URL" "$OUTBOX" || return 1
  else
    git clone -q --depth 1 --branch master "$PUSH_URL" "$OUTBOX" || return 1
    git -C "$OUTBOX" checkout -q -b "$BRANCH"
  fi
  git -C "$OUTBOX" remote set-url origin https://github.com/mohanchillara1/mhh-gate.git  # no token on disk
}

push() {  # push "<one-line message>"
  local msg=$1 d=$OUTBOX/runs/long-s${SEED}/$ARM
  outbox_init || { say "!! outbox clone failed"; return 1; }
  mkdir -p "$d"
  cp -f "$STATUS" "$d/STATUS.md" 2>/dev/null
  podctl progress "$LOG" "$ARM" > "$d/progress.json" 2>/dev/null
  local rd; rd=$(ls -d "$RUNS/${ARM}_seed${SEED}_att"* 2>/dev/null | tail -1)
  if [ -n "$rd" ]; then
    for f in results.json run_manifest.json; do
      [ -s "$rd/$f" ] && cp -f "$rd/$f" "$d/$f"
      [ -e "$rd/$f" ] && [ ! -s "$rd/$f" ] && echo "$f is 0 bytes (storage truncation?)" >> "$d/WARNINGS.txt"
    done
    [ -d "$rd/eval_primary/client_logs" ] && tar -czf "$d/eval_client_logs.tgz" -C "$rd/eval_primary" client_logs 2>/dev/null
    [ -f "$rd/eval_primary/policy_server.log" ] && tail -n 300 "$rd/eval_primary/policy_server.log" > "$d/policy_server.tail.log"
  fi
  cp -f "$RUNS/attempts.json" "$d/attempts.json" 2>/dev/null
  if [ -f "$LOG" ]; then
    tail -n 400 "$LOG" > "$d/run.tail.log"
    [ "$(stat -c %s "$LOG")" -lt 40000000 ] && gzip -c "$LOG" > "$d/run.log.gz"
  fi
  for f in "$W"/logs/dryrun_long_*.log "$W"/logs/setup_long.log "$W"/logs/render_probe.log; do
    [ -f "$f" ] && tail -n 400 "$f" > "$d/$(basename "$f" .log).tail.log"
  done
  # never publish a secret: redact every token value that is set
  for v in HF_TOKEN GH_TOKEN RUNPOD_ACCOUNT_KEY RUNPOD_API_KEY; do
    local val=${!v:-}
    [ ${#val} -ge 8 ] && grep -rlF -- "$val" "$d" 2>/dev/null | while read -r f; do
      case $f in *.gz|*.tgz) rm -f "$f"; echo "$(basename "$f") dropped: contained a secret" >> "$d/WARNINGS.txt";;
                 *) sed -i "s|$val|<redacted-$v>|g" "$f";; esac; done
  done
  git -C "$OUTBOX" add -A
  git -C "$OUTBOX" -c user.name="Mohan Chillara" -c user.email="mohan.kchill@gmail.com" \
      commit -q -m "long-s${SEED} ${ARM}: ${msg}" || return 0   # nothing changed
  local i
  for i in 1 2 3; do
    git -C "$OUTBOX" push -q "$PUSH_URL" "HEAD:refs/heads/$BRANCH" 2>>"$STATUS.pusherr" && { say "pushed: $msg"; return 0; }
    sleep 20
  done
  stamp "!! push failed 3x ($msg) — results still on the pod volume at $RUNS"
  return 1
}

stop_pod() {
  local r; r=$(podctl stop 2>&1); stamp "pod stop requested: $r"
  if ! echo "$r" | grep -q '"stopped"'; then   # anything but an explicit "stopped" is a failure
    push "STOP FAILED — pod may still be billing: $r"
    # keep trying every 5 min rather than billing silently
    while sleep 300; do r=$(podctl stop 2>&1); echo "$r" | grep -q '"stopped"' && { stamp "stopped on retry: $r"; break; }; done
  fi
}

fatal() {
  stamp "❌ FATAL: $*"
  [ -f "$W/run_long.pid" ] && kill -0 "$(cat "$W/run_long.pid")" 2>/dev/null && stamp "run process still alive; the stop will end it"
  push "FATAL: $*"
  stop_pod
  exit 1
}

# ---------------------------------------------------------------- 0. preflight
touch "$STATUS"
stamp "chain.sh started: arm=$ARM seed=$SEED mock=$MOCK pod=${RUNPOD_POD_ID:-none}"
[ "$MOCK" = 1 ] || [ -n "${HF_TOKEN:-}" ] || { stamp "no HF_TOKEN"; exit 1; }
[ -n "${GH_TOKEN:-}" ] || [ -n "${MHH_PUSH_URL:-}" ] || { stamp "no GH_TOKEN: results could not leave the pod; refusing to start"; exit 1; }
[ -n "${RUNPOD_POD_ID:-}" ] || [ "$MOCK" = 1 ] || { stamp "no RUNPOD_POD_ID: cannot self-stop; refusing to start"; exit 1; }
outbox_init || { stamp "cannot clone/push $BRANCH with GH_TOKEN; refusing to start"; exit 1; }
push "chain started" || { stamp "first push failed; refusing to spend GPU hours with no way out"; exit 1; }

g=$(podctl need "$LOG" "$ARM"); stamp "balance preflight: $g"
case $g in
  *'"verdict": "OK"'*) ;;
  *'"verdict": "BLIND"'*) [ "${MHH_ALLOW_BALANCE_BLIND:-0}" = 1 ] || fatal "balance unreadable (RUNPOD_ACCOUNT_KEY?) and MHH_ALLOW_BALANCE_BLIND!=1" ;;
  *) fatal "balance does not cover this arm: $g" ;;
esac

# ---------------------------------------------------------------- 1. setup
setup() {
  set -e
  [ -d $GROOT_REPO/.git ] || git clone https://github.com/NVIDIA/Isaac-GR00T $GROOT_REPO
  cd $GROOT_REPO && git fetch -q origin && git checkout -q $PIN
  [ "$(git rev-parse HEAD)" = "$PIN" ]
  apt-get update -qq
  apt-get install -y -qq git-lfs ffmpeg libosmesa6 libosmesa6-dev libgl1 libglu1-mesa \
      libegl1 libgles2 libglvnd0 libopengl0 tmux >/dev/null
  git lfs install >/dev/null; git lfs pull
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
  export PATH="$HOME/.local/bin:$PATH"
  $PY -c "import gr00t, flash_attn" 2>/dev/null || uv sync --python 3.12
  $PY -c "import gr00t, flash_attn, torch; assert torch.cuda.is_available(); print('torch', torch.__version__)"
  for repo in nvidia/GR00T-N1.7-3B nvidia/Cosmos-Reason2-2B; do
    code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/models/$repo")
    [ "$code" = 200 ] || { echo "HF access $repo -> $code"; return 1; }
  done
  DATA=$GROOT_REPO/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot
  if [ ! -f $DATA/meta/modality.json ]; then
    $GROOT_REPO/.venv/bin/hf download --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
        --local-dir $DATA --token "$HF_TOKEN"
    cp $GROOT_REPO/examples/LIBERO/modality.json $DATA/meta/
  fi
  [ -f $DATA/meta/stats.json ] || $PY gr00t/data/stats.py --dataset-path $DATA --embodiment-tag LIBERO_PANDA
  [ -f $DATA/meta/stats.json ]
  ls $GROOT_REPO/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python >/dev/null 2>&1 \
      || bash $GROOT_REPO/gr00t/eval/sim/LIBERO/setup_libero.sh
  # LIBERO asks for a dataset path on first import when ~/.libero/config.yaml is missing;
  # headless that is EOFError (Learnings 2026-08-31, 2026-09-25). Answer it once, then check the FILE.
  printf 'N\n' | $GROOT_REPO/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python -c "import libero.libero" || true
  [ -f "$HOME/.libero/config.yaml" ] || { echo "~/.libero/config.yaml still missing"; return 1; }
  set +e
}

if [ "$MOCK" != 1 ]; then
  stamp "setup started"
  ( export -f setup; timeout "$SETUP_TIMEOUT_S" bash -c "PY=$PY; setup" ) > $LOGS/setup_long.log 2>&1 \
      || fatal "setup failed or timed out (see setup_long.tail.log)"
  stamp "✅ setup done"
fi

# ---------------------------------------------------------------- 2. overlay
if [ ! -f "$LONG/config.py" ]; then
  python3 "$HERE/make_overlay.py" "$LONG" >> "$STATUS" 2>&1 || fatal "Long overlay failed verification"
fi
grep -q 'LIBERO_SUITE = "libero_10"' "$LONG/config.py" || fatal "overlay config is not libero_10"
[ "$MOCK" = 1 ] && cp "$HERE/mock_run.py" "$LONG/run_gate.py"   # local test only
stamp "✅ Long overlay in place at $LONG"

# ---------------------------------------------------------------- 3. dry run + render probe
if [ "$MOCK" != 1 ]; then
  ( cd "$LONG" && MHH_RUNS_DIR=$W/scratch-dryruns $PY run_gate.py --arm "$ARM" --seed $SEED --dry-run ) \
      > $LOGS/dryrun_long_${ARM}.log 2>&1 || fatal "dry run $ARM failed (see dryrun_long_${ARM}.tail.log)"
  stamp "✅ dry run $ARM passed: $(grep -iE 's/step|steps/s|peak|GiB' $LOGS/dryrun_long_${ARM}.log | tail -3 | tr '\n' ' ')"
  $GROOT_REPO/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python - > $LOGS/render_probe.log 2>&1 <<'EOF' || fatal "osmesa render probe failed: eval would die after training"
import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()["libero_10"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(20260823); obs = env.reset()
m = float(obs["agentview_image"].mean())
print("render ok", os.environ.get("MUJOCO_GL"), task.name, m)
assert m > 1.0, "black frame"
env.close()
EOF
  stamp "✅ render probe: $(tail -1 $LOGS/render_probe.log)"
fi
push "dry run + render probe passed; launching real run"

# ---------------------------------------------------------------- 4. real run
rm -f "$RCF"
( cd "$LONG" && $PY run_gate.py --arm "$ARM" --seed $SEED \
    --notes "LIBERO-Long seed 22 (exploratory, off-protocol, not pooled). Overlay long_s22/make_overlay.py; osmesa." \
    > "$LOG" 2>&1; echo $? > "$RCF" ) &
echo $! > "$W/run_long.pid"
stamp "🚀 real run launched: $ARM seed $SEED, log $LOG"

# ---------------------------------------------------------------- 5. watch
last_hb=$(date +%s); last_guard=0; phase_seen=""; blind_since=0
while sleep "$TICK"; do
  now=$(date +%s)
  if [ -f "$RCF" ]; then
    rc=$(cat "$RCF")
    rj=$(ls "$RUNS/${ARM}_seed${SEED}_att"*/results.json 2>/dev/null | tail -1)
    if [ "$rc" = 0 ] && [ -s "$rj" ]; then
      stamp "🏁 DONE: $(grep -o '"results_log_line": "[^"]*"' "$rj")"
      push "DONE — results.json off the pod" || { sleep 60; push "DONE (retry)"; } || true
      stop_pod; exit 0
    fi
    fatal "run_gate.py exited rc=$rc without a results.json"
  fi
  p=$(podctl progress "$LOG" "$ARM")
  ph=$(echo "$p" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("phase",""))')
  if [ "$ph" = eval ] && [ "$phase_seen" != eval ]; then
    stamp "✅ TRAINING DONE (checkpoint-6000), eval started"; push "training done, eval started"; phase_seen=eval
  fi
  mt=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
  [ $((now - mt)) -gt "$STALL_S" ] && fatal "stalled: log unchanged for $(( (now - mt) / 60 )) min (phase $ph)"
  if [ $((now - last_guard)) -ge "$GUARD_S" ]; then
    last_guard=$now
    g=$(podctl need "$LOG" "$ARM"); echo "$g" >> "$W/guard_long_${ARM}.jsonl"
    case $g in
      *STOP_FLOOR*|*STOP_SHORT*) fatal "balance guard: $g" ;;
      *BLIND*) [ $blind_since = 0 ] && blind_since=$now
               [ $((now - blind_since)) -ge 1800 ] && { stamp "⚠️ balance unreadable 30+ min: $g"; blind_since=$now; } ;;
      *) blind_since=0 ;;
    esac
  fi
  if [ $((now - last_hb)) -ge "$HEARTBEAT_S" ]; then
    last_hb=$now; stamp "heartbeat: $p"; push "heartbeat ($ph)"
  fi
done
