- 2026-09-29 05:14 UTC — chain.sh started: arm=T1 seed=22 mock=0 pod=santb25w2jrbls
- 2026-09-29 05:14 UTC — balance preflight: {"phase": "not_started", "hours_left": 12.83, "balance": 24.6035287846, "spend_per_hr": 1.033, "usd_needed": 13.26, "verdict": "OK", "at_utc": "2026-09-29T05:14:48Z"}
- 2026-09-29 05:14 UTC — setup started
- 2026-09-29 05:18 UTC — REPAIR: setup hung (apt stopped by SIGTTOU under tmux+timeout); chain SIGKILLed before any run, relaunched on the no-tty fix
- 2026-09-29 05:18 UTC — chain.sh started: arm=T1 seed=22 mock=0 pod=santb25w2jrbls
- 2026-09-29 05:18 UTC — balance preflight: {"phase": "not_started", "hours_left": 12.83, "balance": 24.5300824226, "spend_per_hr": 1.033, "usd_needed": 13.26, "verdict": "OK", "at_utc": "2026-09-29T05:18:24Z"}
- 2026-09-29 05:18 UTC — setup started
- 2026-09-29 05:30 UTC — ✅ setup done
  FIXED_STEP_COUNT = 6000
  EVAL_SHARD_SIZE = 5
  EVAL_LIST_MASTER_SEED = 20260823
  MAX_EPISODE_STEPS = 720
  N_ACTION_STEPS = 8
  TRAIN_MAX_STEPS = 6000
OVERLAY OK: 10 Long tasks, 200 seeded episodes, sha 5c1fcaa8d90d..., builder reproduces the list byte for byte, 7/7 probe failing seeds in place.
{"dest": "/workspace/mhh-long", "config_sha256": "40b830ed28b1f51fab079dd43d3f453fa972ad254cec40c271c77a143decd12a"}
- 2026-09-29 05:30 UTC — ✅ Long overlay in place at /workspace/mhh-long
- 2026-09-29 05:36 UTC — ✅ dry run T1 passed:   peak_vram_gib: 33.07 
- 2026-09-29 05:37 UTC — ✅ render probe: render ok osmesa LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket 70.59711201985677
- 2026-09-29 05:37 UTC — 🚀 real run launched: T1 seed 22, log /workspace/logs/long_T1_s22.log
- 2026-09-29 06:37 UTC — heartbeat: {"log_bytes": 65508, "log_mtime": 1790663861.0, "phase": "train", "step": 908, "s_per_it": 3.11, "at_utc": "2026-09-29T06:37:43Z"}
- 2026-09-29 07:38 UTC — heartbeat: {"log_bytes": 141098, "log_mtime": 1790667474.0, "phase": "train", "step": 2034, "s_per_it": 3.21, "at_utc": "2026-09-29T07:38:03Z"}
- 2026-09-29 08:38 UTC — heartbeat: {"log_bytes": 207916, "log_mtime": 1790671102.0, "phase": "train", "step": 3027, "s_per_it": 2.65, "at_utc": "2026-09-29T08:38:25Z"}
- 2026-09-29 09:38 UTC — heartbeat: {"log_bytes": 278416, "log_mtime": 1790674725.0, "phase": "train", "step": 4074, "s_per_it": 3.17, "at_utc": "2026-09-29T09:38:47Z"}
- 2026-09-29 10:39 UTC — heartbeat: {"log_bytes": 348577, "log_mtime": 1790678348.0, "phase": "train", "step": 5125, "s_per_it": 4.1, "s_per_it_avg": 3.491, "at_utc": "2026-09-29T10:39:09Z"}
- 2026-09-29 11:30 UTC — ✅ TRAINING DONE (checkpoint-6000), eval started
