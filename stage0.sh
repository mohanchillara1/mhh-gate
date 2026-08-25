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
# /workspace is a FUSE (fuseblk) volume; HuggingFace's parallel Xet backend trips
# its file locks ("OSError: [Errno 9] Bad file descriptor" in filelock). Disable
# Xet + hf_transfer → plain sequential downloads that the FUSE mount handles fine.
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
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
# The repo ships local wheels (torchcodec) via git-lfs. If cloned before git-lfs
# was installed they are pointer stubs and uv sync dies ("Invalid zip file
# structure" / header 0x73726576 = "vers" of a git-lfs pointer). Materialize them.
apt-get install -y -qq git-lfs >/dev/null 2>&1; git lfs install >/dev/null 2>&1 || true
git lfs pull || die "git lfs pull failed (needed for local torchcodec wheels)"
stamp "✅ Isaac-GR00T at pin $PIN (lfs pulled)"

# --- 2. install gr00t via uv (the REPO'S OWN installer) ------------------
# VERIFIED at source (README): install is `uv sync --python 3.12`, NOT pip.
# pip ignores [tool.uv.sources], so it tries to BUILD flash-attn (needs torch
# at build time) and fails; uv pulls the prebuilt torch2.9 flash-attn wheel.
# uv sync creates $GROOT_REPO/.venv with torch 2.9.0 — ALL later python runs
# use that venv, NOT the base image's torch 2.8.
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
# prereqs the README names: git-lfs (demo parquet) + ffmpeg (torchcodec backend).
# ubuntu2404 ships ffmpeg 6 → in torchcodec's supported 4-7 range.
say "apt prereqs: git-lfs, ffmpeg"
apt-get update -qq && apt-get install -y -qq git-lfs ffmpeg >/dev/null 2>&1
git lfs install >/dev/null 2>&1 || true
VENV=$GROOT_REPO/.venv
if [ ! -x $VENV/bin/python ] || ! $VENV/bin/python -c "import gr00t, flash_attn" 2>/dev/null; then
  say "uv sync --python 3.12 (long — builds venv, torch 2.9, flash-attn)"
  ( cd $GROOT_REPO && uv sync --python 3.12 ) || die "uv sync failed — see log above"
fi
PY=$VENV/bin/python
$PY -c "import gr00t, flash_attn, huggingface_hub, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || die "gr00t venv incomplete after uv sync"
stamp "✅ gr00t installed via uv sync; venv=$VENV"

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
  $VENV/bin/hf download --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
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
  $PY run_gate.py --arm $ARM --seed 11 --dry-run 2>&1 | tee $W/logs/dryrun_${ARM}_s11.log
  RC=${PIPESTATUS[0]}
  if [ $RC -eq 0 ]; then stamp "✅ dry run $ARM PASSED — timing in logs/dryrun_${ARM}_s11.log"
  else stamp "❌ dry run $ARM FAILED (rc=$RC) — see logs/dryrun_${ARM}_s11.log"; fi
done

stamp "stage0.sh finished. NEXT (human/Claude): read dry-run logs; if both passed → --verify-eval-determinism on a cheap checkpoint, then Stage 1. If OOM → halve per-device batch, double accum (effective batch unchanged), re-dry-run."
say "done"
