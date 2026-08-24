# Verification manifest — mhh-gate

**This code has never been run on a GPU, has never trained anything, and has never
evaluated a LIBERO episode.** It was written on a machine with no CUDA device, no
Isaac-GR00T installation, and no LIBERO simulator. Nothing here is "tested" or "verified
working" in the sense of having executed against the real system.

What follows is the honest split.

---

## 1. Verified — API surface read from real source at a pinned commit

Every `gr00t.*` import, class, function signature, config field and file:line citation in
this repo was read directly from **NVIDIA/Isaac-GR00T at commit
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`** (main, 2026-08-20), fetched from
`raw.githubusercontent.com` during this build. Nothing was recalled from memory and
nothing was invented.

Files read in full or in the cited regions:

| file | what was verified from it |
|---|---|
| `gr00t/data/dataset/sharded_single_step_dataset.py` | `extract_step_data` signature and the `allow_padding` clamp at `:40-42`; `get_datapoint` resolving `extract_step_data` from module globals at `:260-266`; `ShardedSingleStepDataset.__init__` kwargs `:129-138` |
| `gr00t/configs/data/data_config.py` | `DataConfig.allow_padding` at `:94`; `modality_configs` default is the shared `MODALITY_CONFIGS` singleton `:70-72`; `seed` `:92`, `multiprocessing_context` `:93`, `override_pretraining_statistics` `:83` |
| `gr00t/configs/finetune_config.py` | **confirmed: `allow_padding` does not appear anywhere** (211 lines, zero hits) |
| `gr00t/data/dataset/factory.py` | `allow_padding=self.config.data.allow_padding` at `:68`; `seed=self.config.data.seed` at `:67` |
| `gr00t/experiment/launch_finetune.py` | the exact `Config` assembly this runner mirrors, `:61-130` |
| `gr00t/experiment/experiment.py` | `set_seed(config.data.seed)` `:203`; `TrainingArguments(seed=config.data.seed)` `:294`; `BestMetricCheckpointCallback` gated on `save_best_eval_metric_name` `:318-325`; `CheckpointFormatCallback` `:311-316`; `trainer.train(resume_from_checkpoint=...)` `:355` |
| `gr00t/configs/base_config.py` | `Config.load_dict`, `Config.validate` (which strips `modality_configs` to used tags, `:182-193`) |
| `gr00t/configs/training/training_config.py` | field names/defaults used: `bf16=True` `:63`, `eval_strategy="no"` `:85`, `save_best_eval_metric_name=""` `:89`, `assert_loss_less_than` `:124` |
| `gr00t/configs/model/gr00t_n1d7.py` | `select_layer=12` `:47`, `diffusion_model_cfg["num_layers"]=16` `:94`, `use_mean_std=False` `:120`, `max_seq_len=1024` |
| `gr00t/configs/data/embodiment_configs.py` | the verbatim `libero_sim` modality block `:193-210` — video `delta_indices=[0]`, keys `["image","wrist_image"]` |
| `gr00t/model/gr00t_n1d7/setup.py` | `AutoModel.from_pretrained` tune kwargs `:82-95`; `AutoProcessor.from_pretrained(..., modality_configs=...)` override call `:152-183` |
| `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` | the `from_pretrained` override hook `:865-882`; `AutoProcessor.register` `:887`; **`use_mean_std` set `:262`, serialized `:784`, override-listed `:874`, and never forwarded to `StateActionProcessor` `:251-258`** |
| `gr00t/data/state_action/state_action_processor.py` | `__init__` does not accept `use_mean_std` `:63-70`; the real per-joint-group normalization branch `:234-268`; q01/q99-vs-min/max at `:157-162` |
| `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | `AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)` `:627` |
| `gr00t/eval/rollout_policy.py` | `run_gr00t_sim_policy` signature `:538-551`; `RolloutConfig` fields `:625-670`; `DEFAULT_MAX_EPISODE_STEPS = 720` `:58`; **only the first reset is seeded, `:302-309`**; invalid-episode filtering `:485-491` |
| `gr00t/eval/sim/LIBERO/libero_env.py` | **`reset()` seeds robosuite and never calls `set_init_state`, `:161-169`**; `set_init_state` appears only in `__main__` `:255-259`; `register_libero_envs` `:192-213`; `task_description` injected as `annotation.human.action.task_description` `:157` |
| `gr00t/eval/sim/wrapper/multistep_wrapper.py` | **`assert (delta_indices[1] - delta_indices[0]) > 0` at `:279`** — the reason `STATIC5` cannot be rolled out |
| `gr00t/eval/_horizon_contract.py` | `PolicyHorizonSpec.from_policy` reads video delta indices from the policy's modality config `:101-183` |
| `gr00t/eval/run_gr00t_server.py` | `ServerConfig` CLI fields `:54-88`; the model is loaded **before** the socket binds `:105-167` (which is why port-bind is a valid readiness signal) |
| `gr00t/policy/gr00t_policy.py` | `get_modality_config()` `:479-480`; modality configs come from the saved processor `:129-171` |
| `gr00t/data/types.py` | `ModalityConfig` fields incl. `sin_cos_embedding_keys` / `mean_std_embedding_keys` `:85-100` |
| `gr00t/data/embodiment_tags.py` | `LIBERO_PANDA = "libero_sim"` `:122`; `EmbodimentTag.resolve` `:146` |
| `gr00t/eval/sim/LIBERO/setup_libero.sh` | the separate uv venv, and `mujoco==3.3.1` pinned there |
| `examples/LIBERO/README.md` | the 10 LIBERO-Spatial env ids (copied verbatim into `config.py`), the official 20K/640 recipe, and the server+client eval invocation |

Inherited from `mhh-smoke-test/` (which **did** pass all 8 stages on real hardware,
2026-08-23): the `delta_indices` frame-stacking semantics, the processor `T`-override
hook working in practice, `AutoModel.from_pretrained` loading N1.7, and the model
attribute paths `backbone.model.language_model.layers` and
`action_head.model.transformer_blocks` (which is why those are tried first here, with a
fail-loud fallback that enumerates candidates rather than guessing).

## 2. Verified — this repo's own logic, executed locally

These ran on this machine, without gr00t, and passed:

* every file compiles (`compile()` on all four `.py` files);
* `--make-eval-list` generates the 10-task / 200-episode list and prints a digest;
  re-running it refuses to overwrite;
* the eval-list hash check rejects a wrong digest and accepts the right one;
* the `FIXED_STEP_COUNT` consistency checks fire correctly for: not a multiple of
  `SAVE_STEPS`, greater than `TRAIN_MAX_STEPS`, `SAVE_TOTAL_LIMIT` too small to keep
  `checkpoint-S`, and `multiprocessing_context != "fork"`;
* the crash-policy ledger: fresh → attempt 1; after one crash → attempt 2 with a notice;
  after two crashes → refuses; after a divergence → refuses outright; after a completed
  run → refuses unless `--force-new-attempt`;
* the divergence scanner finds a NaN loss in a synthetic `trainer_state.json` and reports
  clean when there is none;
* **the `extract_step_data` monkey-patch mechanism** — tested against a stub module built
  so that `get_datapoint` resolves `extract_step_data` from *module* globals, exactly as
  the real file does. The guard intercepted the `True` call and raised on the `False`
  call. This verifies the *interception mechanism*, not the real repo's behaviour.

## 3. NOT verified — awaits the first run on real hardware

Nothing below has been executed. Each is a place the first dry run may fail.

* **Anything requiring a GPU.** No forward pass, no backward pass, no training step, no
  rollout has been executed by this code. The throughput numbers `--dry-run` prints do
  not exist yet.
* **That the training config assembled here is accepted end to end.** `build_gr00t_config`
  mirrors `launch_finetune.py`, but `Config.validate()`, the model pipeline, DeepSpeed
  selection and `Gr00tTrainer` have never seen this object.
* **Batch size and VRAM.** `GLOBAL_BATCH_SIZE=16 × GRAD_ACCUM=4` on a single A40 with a
  T=5 history (5 frames × 2 cameras = 10 images/sample) is a **guess**. It may OOM. The
  dry run exists to find out for ~$1.
* **The `max_seq_len=1024` positional ceiling at T=5 × 2 cameras** — the card's own
  week-one open question. Unresolved here.
* **The eval determinism assumption.** That robosuite's `OffScreenRenderEnv` initial state
  is fully determined by `.seed()`. The whole frozen-eval-list design rests on it.
  `--verify-eval-determinism` is the check; it has never been run.
* **The client snippet running inside the LIBERO uv venv.** It imports
  `gr00t.eval.rollout_policy` there (the venv exposes gr00t through a `.pth`, per
  `setup_libero.sh`); whether that import resolves cleanly in the island is untested.
* **Server readiness via port bind.** The source ordering says the model loads before the
  bind; the actual zmq bind timing has not been observed.
* **`n_episodes == n_envs` shards.** Reading `_collect_rollout_episodes` says every env
  completes exactly one episode and the loop exits. Never executed.
* **Whether the base checkpoint's `embodiment_id.json` contains `libero_sim`.** The smoke
  test checks this at runtime; here it would surface as a processor build failure in
  preflight.
* **The architecture attribute paths on *this* pod's checkpoint revision.** If they move,
  preflight refuses to run and prints the candidates — by design, but it is still a
  first-run stop.
* **Dependency resolution on the pod image**, torchcodec↔ffmpeg pairing, and any drift
  between the repo's pins and what RunPod ships. The smoke test already hit exactly this
  class of problem on Kaggle.
* **`EXPECTED_LLM_LAYERS=16` / `EXPECTED_DIT_BLOCKS=32`** come from the smoke test's live
  observation, not from this build. If the released checkpoint changes, preflight
  correctly aborts — but that abort has never been triggered here.
* **The `#results-log` line's exact field set** was reconstructed from the task brief's
  description of the template. Compare it against the real template in the project card
  and adjust `results_log_line()` if they differ.

## 4. Known, deliberate deviations from the pre-registration

1. **Eval initial states are reset *seeds*, not LIBERO init-state IDs.** The stock env
   cannot use init-state IDs (source citations in README). Written up rather than
   papered over.
2. **`STATIC5` has no closed-loop eval path** (`multistep_wrapper.py:279`). The runner
   trains it and refuses the eval stage with the citation.

Both need a decision in the project card **before** they are worked around, not after a
number is in hand.

---

*Written 2026-08-23. If you are reading this after the first pod run, the honest thing is
to edit section 3 down as items actually pass — and to add anything that broke.*
