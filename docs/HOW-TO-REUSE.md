# Reusing this harness on something else

Only what `run_gate.py` and `config.py` actually support today. Where the code does not
support something, this file says so rather than describing an intention.

## What the harness is

One pre-registered cell: **one arm, one seed**, trained on LIBERO-Spatial and evaluated
at exactly one fixed checkpoint. It is a gate, not a sweep runner. Running several cells
means invoking it several times.

```
python run_gate.py --arm {T1,T2,T5,STATIC5} --seed N [--dry-run]
```

## Pointing it at a different checkpoint

There is no `--checkpoint` flag for the main path, and that is deliberate: the evaluation
step is `config.FIXED_STEP_COUNT`, frozen before any run exists, and the runner derives
`checkpoint-<FIXED_STEP_COUNT>` from it. Best-checkpoint selection is the leak the design
exists to prevent, so the only supported way to evaluate a different step is to change
`FIXED_STEP_COUNT` and accept that you have started a new pre-registration.

`run_gate.py` mechanically checks that the value you pick:

* is a multiple of `SAVE_STEPS`, so `checkpoint-<S>` is actually written;
* is `<= TRAIN_MAX_STEPS`;
* survives `SAVE_TOTAL_LIMIT` rotation.

One checkpoint path *is* accepted directly, for the one-off determinism probe only:

```
python run_gate.py --verify-eval-determinism <path-to-checkpoint>
```

## Pointing it at a different task suite

`config.py` holds the suite wiring: `GROOT_REPO`, `LIBERO_REPO`, the dataset path under
`examples/LIBERO/…`, and `EMBODIMENT_VALUE`. Changing suite means changing those and
regenerating the episode list:

```
python run_gate.py --make-eval-list      # refuses to overwrite an existing file
```

then pasting the new digest into `EVAL_EPISODE_LIST_SHA256`. **Pasting the digest is the
act of freezing.** Until you paste it, the run will not start.

⚠️ The file:line citations throughout this repo were read against Isaac-GR00T at commit
`51d4c89f72fda44cbf77285c6a8114b52676b8a1`. A different suite on a different commit
invalidates them, and `run_gate.py` will abort on the commit mismatch unless you pass
`--allow-commit-drift`. If you pass it, the assertions may be checking the wrong lines.

## What the pre-registration file must contain

`eval_episodes.json` is generated, not hand-written. It carries:

* `episodes_per_task`, `master_seed`, `max_episode_steps`, `n_action_steps`, `note`
* and the per-task episode entries with their reset seeds written out explicitly.

The whole file is hashed. Reformatting it changes the digest and aborts the run, which is
intended: the digest is over bytes, not over meaning.

## The abort conditions, as implemented

A run stops, rather than warning, when:

1. `FIXED_STEP_COUNT` or `EVAL_EPISODE_LIST_SHA256` is `None`.
2. `eval_episodes.json` does not hash to the frozen digest.
3. The installed Isaac-GR00T commit is not the pinned one (`--allow-commit-drift` overrides).
4. **A: `allow_padding` does not arrive at `extract_step_data`.** The function is wrapped
   for the whole run and any call with `allow_padding=False` raises. This is checked at the
   consuming function, not at the config, because this codebase ships flags that are set,
   serialized and never consumed. `multiprocessing_context` is pinned to `"fork"` and
   asserted, because a `"spawn"` child re-imports a clean module and the guard would
   silently disappear.
5. **B: the architecture that actually loaded** is not the pinned 16 LLM layers / 32 DiT
   blocks. Read from the live model object, never from a config file.
6. **C: `USE_MEAN_STD=True` is requested.** The processor accepts the key and never passes
   it on, so honouring the request is impossible; the runner refuses rather than silently
   keeping min/max.

`--dry-run` executes all of the above, captures provenance, times a forward+backward at
the real batch size and history depth, and stops before training.

## What this harness does not do

* No sweeps, no scheduler, no multi-seed driver.
* No best-checkpoint selection anywhere in the codebase, by design.
* No resume. `--force-new-attempt` starts a fresh attempt directory instead.
* Nothing about the P6 determinism finding: that lives in the separate gate notebook
  (`gr00trobustnessresearch/experiments/eval-determinism-gate/`).
