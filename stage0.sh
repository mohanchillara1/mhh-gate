#!/bin/bash
# stage0.sh — Making History Help, Stage 0 pre-flight. Run ON THE POD under nohup:
#   nohup bash /workspace/mhh-gate/stage0.sh > /workspace/stage0.log 2>&1 &
# Requires: HF_TOKEN in /workspace/hf_token (one line, a HuggingFace READ token).
# Everything is idempotent — safe to re-run after a failure; finished steps are skipped.
set -uo pipefail

W=/workspace
export GROOT_REPO=$W/Isaac-GR00T
export MHH_RUNS_DIR=$W/mhh-runs
export HF_HOME=$W/hf
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PIN=51d4c89f72fda44cbf77285c6a8114b52676b8a1
STATUS=$W/STATUS.md

say() { echo "[stage0 $(date -u +%H:%M:%S)] $*"; }
stamp() { echo "- $(date -u '+%Y-%m-%d %H:%M UTC') — $*" >> "$STATUS"; }
die() { say "FATAL: $*"; stamp "❌ FATAL: $*"; exit 1; }

touch "$STATUS"
stamp "stage0.sh started"

# --- 0. HF token ---------------------------------------------------------
[ -s $W/hf_token ] || die "no /workspace/hf_token — create a READ token at huggingface.co/settings/tokens and: echo TOKEN > /workspace/hf_token"
export HF_TOKEN=$(tr -d ' \n' < $W/hf_token)

# --- 1. Isaac-GR00T at the pin ------------------------------------------
if [ ! -d $GROOT_REPO/.git ]; then
  say "cloning Isaac-GR00T"
  git clone https://github.com/NVIDIA/Isaac-GR00T $GROOT_REPO || die "clone failed"
fi
cd $GROOT_REPO
git fetch -q origin
git checkout -q $PIN || die "cannot checkout pin $PIN"
[ "$(git rev-parse HEAD)" = "$PIN" ] || die "HEAD != pin"
stamp "✅ Isaac-GR00T at pin $PIN"

# --- 2. install gr00t ----------------------------------------------------
# NOTE: import gr00t succeeds from the repo dir even uninstalled (namespace pkg),
# so gate on a real DEPENDENCY instead.
if ! python -c "import huggingface_hub, transformers" 2>/dev/null; then
  say "pip install -e . (long)"
  pip install --break-system-packages -e . || die "gr00t install failed — see log above; likely dependency drift (torchcodec/torch pairing was the Kaggle failure)"
fi
python -c "import gr00t, huggingface_hub" || die "gr00t/huggingface_hub still not importable after install"
stamp "✅ gr00t + deps installed and importable"

# --- 3. gated-model access check (fail fast, before big downloads) -------
for repo in nvidia/GR00T-N1.7-3B nvidia/Cosmos-Reason2-2B; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/models/$repo")
  [ "$code" = "200" ] || die "HF access to $repo returned $code (need 200 — is it granted on this account?)"
  say "access OK: $repo ($code)"
done
stamp "✅ HF access verified for both gated repos"

# --- 4. LIBERO-Spatial data + modality patch ----------------------------
DATA=$GROOT_REPO/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot
if [ ! -f $DATA/meta/modality.json ]; then
  say "downloading LIBERO-Spatial dataset"
  hf download --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
      --local-dir $DATA --token $HF_TOKEN || die "dataset download failed"
  cp $GROOT_REPO/examples/LIBERO/modality.json $DATA/meta/ || die "modality patch copy failed"
fi
stamp "✅ LIBERO-Spatial data present, modality.json patched"

# --- 5. LIBERO simulator island -----------------------------------------
if [ ! -d $GROOT_REPO/gr00t/eval/sim/LIBERO/.venv ] && ! ls $GROOT_REPO/gr00t/eval/sim/LIBERO/*venv* >/dev/null 2>&1; then
  say "running setup_libero.sh (long)"
  bash $GROOT_REPO/gr00t/eval/sim/LIBERO/setup_libero.sh || die "setup_libero.sh failed"
fi
stamp "✅ LIBERO sim island built (or already present)"

# --- 6. dry runs ---------------------------------------------------------
mkdir -p $MHH_RUNS_DIR $W/logs
cd $W/mhh-gate
for ARM in T1 T5; do
  say "dry run $ARM seed 11"
  python run_gate.py --arm $ARM --seed 11 --dry-run 2>&1 | tee $W/logs/dryrun_${ARM}_s11.log
  RC=${PIPESTATUS[0]}
  if [ $RC -eq 0 ]; then stamp "✅ dry run $ARM PASSED — timing in logs/dryrun_${ARM}_s11.log"
  else stamp "❌ dry run $ARM FAILED (rc=$RC) — see logs/dryrun_${ARM}_s11.log"; fi
done

stamp "stage0.sh finished. NEXT (human/Claude): read dry-run logs; if both passed → --verify-eval-determinism on a cheap checkpoint, then Stage 1. If OOM → halve per-device batch, double accum (effective batch unchanged), re-dry-run."
say "done"
