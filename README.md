# mhh-gate — the gate runner for "Making History Help"

`run_gate.py` executes one pre-registered cell of the gate experiment: **one arm, one
seed**, trained on LIBERO-Spatial and evaluated at exactly one fixed checkpoint.

```
python run_gate.py --arm {T1,T2,T5,STATIC5} --seed N [--dry-run]
```

Its job is not convenience. Its job is to make the wrong experiment **impossible to run
by accident** — every pre-registration guarantee that a human could quietly break is
checked in code, at the place where it can actually fail, and breaks the run rather
than printing a warning nobody reads.

Design authority: `mohanvault/01 Projects/Why History Hurts — Project Card.md`,
rounds 3–6. Where this README and the card disagree, the card wins — except for the
one place the card asks for something the GR00T code cannot do, which is flagged in
**"The one deviation"** below.

---

## Before the first run: two values a human must set

Both live in `config.py`, both are `None`, and the script refuses to start without
them. That is deliberate — a default here is a post-hoc degree of freedom.

### 1. `FIXED_STEP_COUNT`

The single global step at which **every arm and every seed** is evaluated. ROUND 5 calls
best-checkpoint selection "the single most dangerous leak found," so there is exactly one
evaluation step and it is chosen before any run exists.

Pick an integer in 5000–10000. `run_gate.py` then checks, mechanically:

* it is a multiple of `SAVE_STEPS`, so `checkpoint-<S>` is actually written;
* it is ≤ `TRAIN_MAX_STEPS`;
* `SAVE_TOTAL_LIMIT` is large enough that `checkpoint-<S>` is not evicted by later saves.

### 2. `EVAL_EPISODE_LIST_SHA256`

`eval_episodes.json` is already generated and committed — 10 LIBERO-Spatial tasks ×
20 episodes, 200 total, with the reset seeds written out explicitly. Read it, then paste
its digest into `config.py`:

```
EVAL_EPISODE_LIST_SHA256 = "d51371a59be544a0d26f4a0f27171fd0d65d39bb94cadf1b2635de67f3b61663"
```

Pasting the digest **is** the act of freezing. From then on the file is re-hashed on
every run and a single edited byte aborts the run. To regenerate from scratch:
`python run_gate.py --make-eval-list` (refuses to overwrite an existing file).

---

## Running it on a fresh RunPod A40 pod, in order

```bash
# 0. pod: A40 48GB, a CUDA image with python 3.12, /workspace volume
export GROOT_REPO=/workspace/Isaac-GR00T
export MHH_RUNS_DIR=/workspace/mhh-runs
export HF_HOME=/workspace/hf            # keep the 3B checkpoint on the volume
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# 1. Isaac-GR00T at the pinned commit (git clone, then check out the pin;
#    run_gate.py aborts on any other commit unless --allow-commit-drift)
git clone https://github.com/NVIDIA/Isaac-GR00T $GROOT_REPO
cd $GROOT_REPO && git checkout 51d4c89f72fda44cbf77285c6a8114b52676b8a1
pip install -e .

# 2. HuggingFace access. You need BOTH nvidia/GR00T-N1.7-3B and the gated
#    nvidia/Cosmos-Reason2-2B backbone approved on your account.
hf auth login

# 3. LIBERO-Spatial training data + the modality patch (examples/LIBERO/README.md)
hf download --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --local-dir $GROOT_REPO/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/
cp -r $GROOT_REPO/examples/LIBERO/modality.json \
      $GROOT_REPO/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/meta/

# 4. the LIBERO simulator island (a SEPARATE uv venv; this is where MuJoCo lives)
bash $GROOT_REPO/gr00t/eval/sim/LIBERO/setup_libero.sh

# 5. set the two values in config.py (above), then:

# 6. THE ~$1 HARNESS CHECK. Runs every assertion, captures provenance, times one
#    forward+backward at the real batch size and history depth, and STOPS.
#    Do this for T1 and for T5 before spending a single training hour.
python run_gate.py --arm T1 --seed 11 --dry-run
python run_gate.py --arm T5 --seed 11 --dry-run

# 7. ONE-OFF, on the first pod only: does the eval seed actually fix the initial
#    state? (see "The one deviation"). Needs any trained checkpoint; a T1 run at
#    a low step count is fine. Costs ~10 minutes.
python run_gate.py --verify-eval-determinism /workspace/mhh-runs/T1_seed11_att1/train/checkpoint-1000

# 8. the real runs — paired seeds, same seed trains both arms of a pair
python run_gate.py --arm T1 --seed 11
python run_gate.py --arm T5 --seed 11
#   ... then seeds 12, 13, 14, 15, and the T2 arm (collected for all 5 seeds
#   regardless of interim results — ROUND 5)
```

Each run prints a ready-to-paste `#results-log` line at the end and writes the same
content to `results.json` next to `run_manifest.json`.

---

## What each assertion protects against

The three preflight checks are not hygiene. Each one is a specific way this project has
already been shown it can produce a plausible, wrong, unfalsifiable number.

### A — `allow_padding` must **arrive** at `extract_step_data`

`extract_step_data` builds `indices_to_load = [step_index + delta]` and clamps to
`[0, len-1]` **only** when `allow_padding=True`
(`gr00t/data/dataset/sharded_single_step_dataset.py:40-42`). The default is `False`, and
then index `-1` goes into pandas `.iloc`, which returns the episode's **last** frame. At
step 0 of a T=2 arm the model trains on *(final frame of the task, first frame)* as its
"history". T=5 poisons the first four steps of every episode. **No exception is raised.**

`FinetuneConfig` does not expose the flag at all (grep of
`gr00t/configs/finetune_config.py` at the pinned commit: zero hits), so it has to be set
on `config.data.allow_padding` and threaded through `factory.py:68` →
`ShardedSingleStepDataset.allow_padding` → `extract_step_data`. Because issue #745 proves
this repo ships flags that are set, serialized, and never consumed, **setting it is not
evidence.** So the check is made at the consuming function:

* `extract_step_data` is wrapped for the whole run; any call with `allow_padding=False`
  raises and takes the run down — including calls inside forked dataloader workers,
  which is why `multiprocessing_context` is pinned to `"fork"` and asserted;
* a real dataset is built and `get_datapoint` is called at step 0, and the returned
  slot-0 frame is compared **pixel-wise** to episode row 0 (repeat-first-frame) and to
  the episode's last row (the wrap-around signature);
* mid-episode, every slot is compared to its independently-indexed ground-truth row, so
  duplicated or mis-ordered frames fail too.

### B — record the architecture that actually **loaded**

Isaac-GR00T issue #755: the released GR00T-N1.7-3B checkpoint ships a `select_layer` /
DiT depth different from the documented one, a finetune `select_layer` override is
silently ignored (the checkpoint's `config.json` wins), and the ignored value is still
written into `conf.yaml` as if it had been used. The smoke test observed **16 LLM layers /
32 DiT blocks on live weights** (2026-08-23).

So the runner reads the live model object, never a config file, and refuses to run if it
cannot find the layer stacks (it prints the candidate `ModuleList`s instead of guessing).
Drift from the pinned 16/32 is fatal by default — runs on different architectures are not
comparable.

### C — record the normalization mode actually in force

Issue #745, re-verified in source at the pinned commit: the processor stores
`use_mean_std` (`processing_gr00t_n1d7.py:262`), serializes it (`:784`) and accepts it as
an override key (`:874`) — and **never passes it to `StateActionProcessor`**
(`:251-258`), whose `__init__` does not accept the argument at all
(`state_action_processor.py:63-70`). Setting it to `True` would silently keep min/max.

The runner hard-fails if `USE_MEAN_STD=True` is requested, and records the mode that is
genuinely applied per joint group, resolved from the real branch structure
(`state_action_processor.py:234-268`): sin/cos → mean/std → min/max on q01/q99.

### D — the arm's history window reached the processor

`video.delta_indices` is the *only* thing that differs between arms. It is asserted on the
constructed processor after `from_pretrained`'s override merge
(`processing_gr00t_n1d7.py:865-867`), so an arm that silently trained at T=1 is impossible.

### And the ones that are not "assertions" but do the same job

* **Provenance is captured first**, before any heavy work, into `run_manifest.json`:
  Isaac-GR00T commit + dirty flag, LIBERO commit, MuJoCo/robosuite/numpy versions **read
  from the LIBERO venv** (not the training env — that is where MuJoCo lives), torch,
  transformers, GPU model and driver, the dataset's `meta/` file hashes, the exact
  instruction strings for all 10 tasks read out of the LIBERO benchmark, all seeds, the
  eval-list digest, and hashes of `run_gate.py` and `config.py` themselves.
* **No best-checkpoint selection anywhere.** `save_best_eval_metric_name` and
  `eval_strategy` are asserted, and only `checkpoint-<FIXED_STEP_COUNT>` is ever opened.
  If it is missing the run aborts and lists what exists rather than picking a neighbour.
* **Crash policy in code, not in memory.** A ledger at `$MHH_RUNS_DIR/attempts.json`
  grants exactly **one** exact-seed rerun after an infrastructure crash, and **zero**
  after a divergence — a diverged run is recorded as `diverged` and can never be replaced
  by a rerun or a fresh seed. Divergence is detected by scanning every
  `trainer_state.json` for a non-finite loss (`assert_loss_less_than` would not catch NaN,
  since `nan > x` is `False`).

---

## The one deviation from the pre-registration — read this before trusting an SR

ROUND 5 asks for "fixed pre-generated initial-state ID lists per task." **The stock GR00T
eval path cannot do that**, verified in source at the pinned commit:

* `LiberoEnv.reset()` calls `self._env.seed(int(seed))` then `self._env.reset()`
  (`gr00t/eval/sim/LIBERO/libero_env.py:161-169`). It never calls `set_init_state()`.
  LIBERO's canonical `task_suite.get_task_init_states()` appears only in that file's
  `__main__` demo block (`:255-259`), not in the registered environment.
* Only the **first** reset of a rollout is seeded
  (`gr00t/eval/rollout_policy.py:302-309`). Every later episode comes from gymnasium's
  unseeded autoreset, whose RNG position depends on how many steps the policy took —
  i.e. it differs **between arms**, which is precisely the confound the fixed list exists
  to remove.

The substitute this runner uses, which is enforceable with the stock API: evaluate in
shards of exactly `EVAL_SHARD_SIZE` episodes with `n_episodes == n_envs == shard size`,
so **every** episode is a seeded first reset. The per-task seed lists are frozen in
`eval_episodes.json` and hash-checked. Identical seeds → identical robosuite RNG →
identical initial states across arms, without ever touching the unseeded autoreset path.

**The assumption this rests on has not been verified**, and cannot be off-GPU: that
robosuite's `OffScreenRenderEnv` initial state is fully determined by `.seed()`. That is
what `--verify-eval-determinism` exists for — it runs one task's first shard twice with
identical seeds and compares the per-episode length and success vectors. **Run it once,
on the first pod, before 16 GPU-hours ride on it.** If it comes back not-identical, stop
and redesign the eval; do not compare arms.

## STATIC5 cannot be closed-loop evaluated

The placebo arm is built at data level as `delta_indices=[0,0,0,0,0]` — same token and
VRAM budget as T5, zero temporal content. It trains fine. It **cannot** be rolled out,
because `MultiStepWrapper.assert_delta_indices` requires a strictly increasing delta list
(`gr00t/eval/sim/wrapper/multistep_wrapper.py:265-279`, specifically
`assert (delta_indices[1] - delta_indices[0]) > 0`), and a repeated-frame control has a
diff of zero.

`run_gate.py` trains STATIC5 and then refuses the eval stage with that citation, rather
than crashing three hours later inside a vectorised env. The two ways forward — open-loop
only, or a stated patch to the wrapper — both have to be **pre-registered before they are
used**, not chosen after seeing a number.

---

## Files

| file | what it is |
|---|---|
| `run_gate.py` | the runner: CLI, provenance, preflight, train, eval, reporting, crash policy |
| `config.py` | every frozen constant, with the two `REQUIRED-HUMAN-INPUT` values left unset |
| `preflight.py` | assertions A/B/C/D, each made at the consuming site |
| `provenance.py` | environment/version/instruction capture (nothing here can fail a run) |
| `eval_episodes.json` | the frozen 200-episode evaluation set — hash-checked on every run |
| `MANIFEST.md` | **what is verified and what is not.** Read it before believing anything here works. |

## Outputs, per run

```
$MHH_RUNS_DIR/<arm>_seed<N>_att<K>/
  run_manifest.json        provenance + preflight + config + status, written incrementally
  results.json             the reportable result and the #results-log line
  train/                   HF output_dir: checkpoint-*, experiment_cfg/, processor/
  eval_primary/            policy_server.log, per-shard client logs, rollout videos
$MHH_RUNS_DIR/attempts.json   the crash-policy ledger
```

Nothing in `run_manifest.json` is a result until the run reaches `status: completed`.
