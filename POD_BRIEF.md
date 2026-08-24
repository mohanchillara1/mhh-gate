---
created: 2026-08-24
related: "[[MHH Hub]]", "[[Plan — Staged Gate Runs]]"
purpose: Copy this file to /workspace/POD_BRIEF.md on the pod. Any Claude session working there reads it FIRST. It exists so pod work is grounded in verified fact, not model memory.
---

# Pod Grounding Brief — Making History Help

You are working on a rented RunPod A40 for a pre-registered research experiment run by
two high schoolers with a Sept 4 deadline and a small budget. Wrong code here costs
real money and real science. These rules are not optional.

## The five rules

1. **Never write code against a gr00t API from memory.** Every import, signature,
   config field, and file:line you use must be read from the checked-out source at
   `/workspace/Isaac-GR00T` IN THIS SESSION before you use it. The repo is pinned at
   `51d4c89f72fda44cbf77285c6a8114b52676b8a1` — if `git rev-parse HEAD` says anything
   else, stop and say so.
2. **`mhh-gate/MANIFEST.md` is the authority on what is verified.** Its section 3 lists
   everything that has never executed. When one of those items passes or fails on this
   pod, EDIT THE MANIFEST in the same session — that is the contract written into it.
3. **The pre-registration is frozen.** You may fix crashes, dependencies, and harness
   bugs. You may NOT change: arms, seeds, `FIXED_STEP_COUNT` once set, the eval episode
   list, endpoints, aggregation, or the decision rule. If a fix seems to require
   touching any of those → stop, post the problem to #results-log via the humans, and
   wait. "It would be easy to just…" is how pre-registrations die.
4. **Report failures verbatim.** Paste the actual traceback, the actual numbers. Never
   summarize an error into what you expect it to be. If a run diverged, it is a failed
   run in the record — never replaced with a fresh seed (crash policy: at most one
   exact-seed rerun for pure infrastructure failure).
5. **This repo ships lies — verify at the consuming site.** Three proven cases, all at
   the pinned commit; treat every config flag as guilty until traced to consumption:
   - **#755:** checkpoint architecture is 16 LLM layers / 32 DiT blocks; documented
     12/16; `select_layer` in a finetune config is silently ignored but still written
     to conf.yaml as if used. Trust only the loaded model object.
   - **#745:** `use_mean_std` is stored, serialized, accepted as an override — and
     never forwarded to `StateActionProcessor` (whose `__init__` doesn't accept it).
   - **Episode wrap-around:** `extract_step_data` clamps indices only when
     `allow_padding=True` (default False); otherwise index −1 hits pandas `.iloc` and
     returns the episode's LAST frame. Silent. The gate wraps the function and asserts.

## Verified facts you may rely on (each with its source)

| fact | source |
|---|---|
| Frame stacking via `delta_indices` override works; T=2/T=5/static verified pixel-wise | smoke test, 8/8 PASS, Kaggle 2026-08-23 |
| Loaded N1.7 arch = 16 LLM / 32 DiT; fp16 forward finite; ~6.1 GiB peak at smoke scale | smoke test, live weights |
| `assert (delta_indices[1]-delta_indices[0]) > 0` blocks STATIC5 rollout | `multistep_wrapper.py:279`, read at pin |
| Eval seeding: only the FIRST reset is seeded; gate uses n_episodes==n_envs shards | `rollout_policy.py:302-309`; README §eval |
| LIBERO island = separate uv venv, mujoco==3.3.1 pinned there | `setup_libero.sh` |
| Kaggle shipped torch 2.10.0+cu128 vs repo pin 2.9.0; torchcodec 0.8.0 native lib failed to load | smoke test env note — expect the same class of drift here |
| Decision (dad, 2026-08-24): STATIC5 → patch wrapper, pre-registered, with T1 inertness check | chat |

## Known-unverified (MANIFEST §3 — first-run landmines)
GPU anything · batch 16×accum 4 VRAM fit at T5 (10 imgs/sample — may OOM; fix by
halving per-device batch and doubling accum, NEVER by changing effective batch) ·
`max_seq_len=1024` ceiling at T5×2cams · eval determinism (run `--verify-eval-determinism`
before trusting the frozen list) · gr00t import inside the LIBERO venv · pod dependency
drift · end-to-end config acceptance.

## Operational
- Training runs: `nohup python run_gate.py --arm X --seed N > logs/X_sN.log 2>&1 &` —
  survives every disconnect. Interactive Claude Code lives in tmux.
- Before launching a run: confirm RunPod balance covers it with margin (pods are
  terminated at $0 and the run is lost).
- Every finished run: paste its printed results line into Discord #results-log
  (a human posts it, or the watcher). **Not logged = didn't happen.**
- Environment: `HF_HOME=/workspace/hf`, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`,
  `GROOT_REPO=/workspace/Isaac-GR00T`, `MHH_RUNS_DIR=/workspace/mhh-runs`.
- Provenance: `run_gate.py` captures it, but if you run ANYTHING outside the runner,
  record commit, mujoco version, torch version, and command line yourself.

## What you never do on this pod
No pushing to any remote · no deleting runs or checkpoints (move aside instead) · no
editing the pre-registration files · no submitting anything anywhere · no "cleaning up"
another session's files. When in doubt, the humans decide.
