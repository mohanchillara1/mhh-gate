"""
config.py — frozen constants for the "Making History Help" gate experiment.

EVERY VALUE HERE IS PART OF THE PRE-REGISTRATION.  Changing one after a run has
started invalidates the comparison across arms.  The two values a human MUST
set before the first run are marked ``REQUIRED-HUMAN-INPUT`` and are left as
``None`` on purpose, so ``run_gate.py`` refuses to start rather than guessing.

Source-of-truth for the design: ``mohanvault/01 Projects/Why History Hurts —
Project Card.md``, rounds 3-6.

Every gr00t API/config key referenced in the comments below was read from
NVIDIA/Isaac-GR00T at the pinned commit (see ``GROOT_COMMIT``); file:line
citations are given inline.  Nothing here was inferred from documentation.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# ============ 1. REQUIRED-HUMAN-INPUT — the run will not start without these
# ---------------------------------------------------------------------------

# (1/2) The ONE global step at which every arm and every seed is evaluated.
#       ROUND 5 of the project card: "primary evaluation at exactly global step
#       S = [fix one integer in 5-10K before any run]; best-val/last-checkpoint
#       selection prohibited".  There is deliberately NO default: a default is
#       a post-hoc degree of freedom.
#
#       Constraints run_gate.py enforces mechanically:
#         * FIXED_STEP_COUNT % SAVE_STEPS == 0     (else checkpoint-S never exists)
#         * FIXED_STEP_COUNT <= TRAIN_MAX_STEPS
#         * SAVE_TOTAL_LIMIT is large enough that checkpoint-S is not evicted
FIXED_STEP_COUNT: int | None = 6000          # FROZEN 2026-08-24 (dad in chat; = TRAIN_MAX_STEPS, only other legal value was 5000)

# (2/2) sha256 of the frozen eval-episode file (eval_episodes.json).
#       The file pins, per task, the list of reset seeds used for evaluation.
#       It must be byte-identical for every arm and every seed.  Generate it
#       once with:   python run_gate.py --make-eval-list
#       then paste the printed digest here.  run_gate.py re-hashes the file on
#       every run and hard-fails on mismatch, so the list cannot drift.
EVAL_EPISODE_LIST_SHA256: str | None = "d51371a59be544a0d26f4a0f27171fd0d65d39bb94cadf1b2635de67f3b61663"  # FROZEN 2026-08-24

EVAL_EPISODE_LIST_PATH = os.path.join(os.path.dirname(__file__), "eval_episodes.json")


# ---------------------------------------------------------------------------
# ============ 2. Pins — provenance that is unrecoverable after the run
# ---------------------------------------------------------------------------

# The Isaac-GR00T commit the smoke test was verified against (main, 2026-08-20).
# run_gate.py reads the ACTUAL commit of the installed repo and hard-fails on
# mismatch unless --allow-commit-drift is passed (which is then recorded).
GROOT_COMMIT = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"

# Path to the cloned Isaac-GR00T working tree on the pod.
GROOT_REPO = os.environ.get("GROOT_REPO", "/workspace/Isaac-GR00T")

# The LIBERO evaluation island created by gr00t/eval/sim/LIBERO/setup_libero.sh.
# That script pins mujoco==3.3.1 (setup_libero.sh, final `uv pip install` line)
# because robosuite 1.4.0 calls the pre-3.10.0 mj_fullM signature.  We record
# the version actually importable in that venv, never the pin we expected.
LIBERO_VENV_PYTHON = os.path.join(
    GROOT_REPO, "gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"
)
LIBERO_REPO = os.path.join(GROOT_REPO, "external_dependencies/LIBERO")

BASE_MODEL_PATH = "nvidia/GR00T-N1.7-3B"

# LIBERO-Spatial in GR00T-LeRobot format.  Download + patch per
# examples/LIBERO/README.md ("Fine-tune LIBERO spatial").
DATASET_PATH = os.environ.get(
    "MHH_DATASET_PATH",
    os.path.join(GROOT_REPO, "examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot"),
)

# EmbodimentTag.LIBERO_PANDA.value == "libero_sim"  (embodiment_tags.py:122)
EMBODIMENT_TAG = "LIBERO_PANDA"
EMBODIMENT_VALUE = "libero_sim"


# ---------------------------------------------------------------------------
# ============ 3. Arms — identical in everything except video.delta_indices
# ---------------------------------------------------------------------------
# The libero_sim video modality ships delta_indices=[0] (embodiment_configs.py
# :194-197).  An arm changes ONLY that list; state/action/language are left
# verbatim.  Padding at episode starts is repeat-first-frame, which is exactly
# what allow_padding=True implements:
#     indices_to_load = [max(0, min(idx, len(episode_data) - 1)) ...]
#     -- sharded_single_step_dataset.py:41-42
ARMS: dict[str, list[int]] = {
    "T1": [0],
    "T2": [-1, 0],
    "T5": [-4, -3, -2, -1, 0],
    # The placebo, built at DATA level per the project card (week-one item 1):
    # five copies of the current frame.  Same token/VRAM budget as T5, zero
    # temporal content.
    "STATIC5": [0, 0, 0, 0, 0],
}

# STATIC5 cannot be closed-loop evaluated through the stock GR00T eval path.
# VERIFIED IN SOURCE at the pinned commit:
#   multistep_wrapper.py:279  assert (delta_indices[1] - delta_indices[0]) > 0
# [0,0,...] gives a diff of 0 and trips that assert during wrapper construction.
# run_gate.py therefore refuses `--stage eval` for STATIC5 and says why.
ARMS_WITHOUT_CLOSED_LOOP_EVAL = {"STATIC5"}


# ---------------------------------------------------------------------------
# ============ 4. Training recipe — IDENTICAL across all arms (round-3 item 5)
# ---------------------------------------------------------------------------
# "identical batch size and optimizer settings across ALL arms including T=1;
#  augmentation parameters shared across frames within a sample and RNG logged"
#
# NOTE, stated as a limitation in the paper, not hidden: NVIDIA's official
# LIBERO recipe is 20K steps @ global batch 640 on 8 GPUs
# (examples/LIBERO/README.md).  This is a lighter single-A40 recipe.
TRAIN_MAX_STEPS = 6000          # total optimisation steps to run
GLOBAL_BATCH_SIZE = 16          # pre-accumulation, summed across GPUs (finetune_config.py:124)
GRADIENT_ACCUMULATION_STEPS = 4 # accumulated batch = 16 * 4 = 64
LEARNING_RATE = 1e-4            # finetune_config.py:131 default
WEIGHT_DECAY = 1e-5
WARMUP_RATIO = 0.05
STATE_DROPOUT_PROB = 0.2        # what NVIDIA used for all four LIBERO suites
NUM_GPUS = 1
DATALOADER_NUM_WORKERS = 2
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 1            # QUOTA-ENFORCED 150GB volume: limit 2 gave a 72GB rotation peak that killed T5 att1 mid-save. Limit 1 -> peak 48GB. S=6000 is the final save so it always survives.
SHARD_SIZE = 1024               # data_config.py:76
EPISODE_SAMPLING_RATE = 0.1     # data_config.py:77
TUNE_LLM = False
TUNE_VISUAL = False
TUNE_PROJECTOR = True
TUNE_DIFFUSION_MODEL = True

# data_config.py:94 — the flag that must ARRIVE at extract_step_data.
# FinetuneConfig does NOT expose it (grep of finetune_config.py: zero hits),
# so run_gate.py sets config.data.allow_padding directly and then PROVES
# arrival at the consuming function.  Never set this to False.
ALLOW_PADDING = True

# experiment.py:203/294 uses config.data.seed for both set_seed() and
# TrainingArguments(seed=...).  The dataset sharding RNG also reads it
# (factory.py:67 -> ShardedSingleStepDataset(seed=...)).  One knob, logged.
# Paired design: the SAME seed integer trains T1 and T5, so per-seed
# differences are paired (round-2 fix (a)).
PREREGISTERED_SEEDS = [11, 12, 13, 14, 15]   # N fixed at 5 (ROUND 5, item 1)
ABSTRACT_SEEDS = PREREGISTERED_SEEDS[:3]     # first 3, used regardless of result

# Dataloader workers must be forked so the preflight guard installed in the
# parent process is inherited (see preflight.py).  "spawn" would silently drop
# it.  data_config.py:93 default is already "fork"; we pin it and assert.
MULTIPROCESSING_CONTEXT = "fork"

# Normalisation.  #745: `use_mean_std` is stored and serialized by the
# processor (processing_gr00t_n1d7.py:262, :784) but is NEVER passed to
# StateActionProcessor (processing_gr00t_n1d7.py:251-258 -- the constructor
# call omits it, and StateActionProcessor.__init__ does not accept it,
# state_action_processor.py:63-91).  Setting it True would silently do
# nothing.  Keep it False and let run_gate.py record the mode that is
# actually in force.
USE_MEAN_STD = False
USE_PERCENTILES = True          # q01/q99 min-max, state_action_processor.py:157-162


# ---------------------------------------------------------------------------
# ============ 5. Architecture expectations (issue #755)
# ---------------------------------------------------------------------------
# Gr00tN1d7Config declares select_layer=12 (configs/model/gr00t_n1d7.py:47) and
# diffusion_model_cfg["num_layers"]=16 (:94).  The RELEASED CHECKPOINT ships
# different values, and a finetune override is silently ignored because the
# checkpoint config.json wins.  The smoke test observed 16 LLM layers / 32 DiT
# blocks ON LIVE WEIGHTS (2026-08-23, Kaggle).  Those observed values are
# pinned here so architecture drift between runs is caught, not discovered in
# the writeup.  Set ARCH_DRIFT_IS_FATAL=False only with a written reason.
EXPECTED_LLM_LAYERS = 16
EXPECTED_DIT_BLOCKS = 32
ARCH_DRIFT_IS_FATAL = True


# ---------------------------------------------------------------------------
# ============ 6. Evaluation
# ---------------------------------------------------------------------------
LIBERO_SUITE = "libero_spatial"

# Verbatim from examples/LIBERO/README.md, "Libero Spatial" task list.
# These strings are the gym env ids; the human-language instruction actually
# fed to the model is read at runtime from the LIBERO benchmark and recorded
# in the run manifest (never assumed from the id).
LIBERO_SPATIAL_TASKS = [
    "libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
]

# Episodes per task.  Suite SR = unweighted macro-average of the 10 task SRs
# (ROUND 5, "Aggregation").  20 x 10 tasks = 200 episodes per checkpoint.
EVAL_EPISODES_PER_TASK = 20

# ⚠️ THE ONE PLACE THE PRE-REGISTRATION AND THE CODE DISAGREE — read this.
#
# ROUND 5 says "fixed pre-generated initial-state ID lists per task".  The
# stock GR00T eval path CANNOT do that.  VERIFIED IN SOURCE at the pinned
# commit: LiberoEnv.reset() calls `self._env.seed(int(seed))` then
# `self._env.reset()` (libero_env.py:161-169) and never calls
# `set_init_state()`.  LIBERO's canonical `task_suite.get_task_init_states()`
# is used only in that file's `__main__` demo block (:255-259), not in the
# registered env.
#
# Worse: `_collect_rollout_episodes` seeds ONLY the first reset
# (rollout_policy.py:302-309); every later episode comes from gymnasium's
# unseeded autoreset, whose stream position depends on how many steps the
# policy took -- i.e. it differs BETWEEN ARMS.
#
# The mechanically-enforceable substitute this runner uses: evaluate in shards
# of exactly EVAL_SHARD_SIZE episodes with n_episodes == n_envs == shard size,
# so EVERY episode is a seeded first reset (seeds base..base+shard-1).  The
# per-task list of base seeds is frozen in eval_episodes.json and hash-checked.
# Identical seeds -> identical robosuite RNG -> identical initial states across
# arms, WITHOUT relying on the unseeded autoreset path.
#
# NOT VERIFIED (no GPU here): that robosuite's OffScreenRenderEnv initial state
# is fully determined by .seed(). Confirm on the first dry run by evaluating
# one task twice with the same seeds and checking the episode-length vector is
# identical; run_gate.py --verify-eval-determinism does exactly that.
EVAL_SHARD_SIZE = 5             # == n_envs; VRAM-bound, 5 parallel sim envs
MAX_EPISODE_STEPS = 720         # rollout_policy.py:58 DEFAULT_MAX_EPISODE_STEPS
N_ACTION_STEPS = 8              # examples/LIBERO/README.md eval command
POLICY_SERVER_HOST = "127.0.0.1"
POLICY_SERVER_PORT = 5555
SERVER_STARTUP_TIMEOUT_S = 900  # 3B checkpoint load from disk can be slow

# Master seed used by --make-eval-list to generate the frozen base-seed lists.
EVAL_LIST_MASTER_SEED = 20260823


# ---------------------------------------------------------------------------
# ============ 7. Bookkeeping
# ---------------------------------------------------------------------------
RUNS_DIR = os.environ.get("MHH_RUNS_DIR", "/workspace/mhh-runs")
A40_USD_PER_HOUR = 0.44         # RunPod community-cloud list price, 2026-08.
                                # Estimate only; the manifest records the value
                                # used so the cost line is auditable.
DRY_RUN_TIMED_STEPS = 12        # forward+backward steps timed in --dry-run
DRY_RUN_WARMUP_STEPS = 3
