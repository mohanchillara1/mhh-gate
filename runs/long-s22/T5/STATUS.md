- 2026-09-29 04:34 UTC — chain.sh started: arm=T5 seed=22 mock=0 pod=x49fp4y4sr5oxi
- 2026-09-29 04:34 UTC — balance preflight: {"phase": "not_started", "hours_left": 21.67, "balance": 24.9947024993, "spend_per_hr": 0.558, "usd_needed": 12.09, "verdict": "OK", "at_utc": "2026-09-29T04:34:05Z"}
- 2026-09-29 04:34 UTC — setup started
- 2026-09-29 05:17 UTC — REPAIR: setup hung (apt stopped by SIGTTOU under tmux+timeout); chain SIGKILLed before any run, relaunched on the no-tty fix
- 2026-09-29 05:17 UTC — chain.sh started: arm=T5 seed=22 mock=0 pod=x49fp4y4sr5oxi
- 2026-09-29 05:17 UTC — balance preflight: {"phase": "not_started", "hours_left": 21.67, "balance": 24.6035287846, "spend_per_hr": 1.033, "usd_needed": 17.8, "verdict": "OK", "at_utc": "2026-09-29T05:17:07Z"}
- 2026-09-29 05:17 UTC — setup started
- 2026-09-29 05:36 UTC — ✅ setup done
  FIXED_STEP_COUNT = 6000
  EVAL_SHARD_SIZE = 5
  EVAL_LIST_MASTER_SEED = 20260823
  MAX_EPISODE_STEPS = 720
  N_ACTION_STEPS = 8
  TRAIN_MAX_STEPS = 6000
OVERLAY OK: 10 Long tasks, 200 seeded episodes, sha 5c1fcaa8d90d..., builder reproduces the list byte for byte, 7/7 probe failing seeds in place.
{"dest": "/workspace/mhh-long", "config_sha256": "40b830ed28b1f51fab079dd43d3f453fa972ad254cec40c271c77a143decd12a"}
- 2026-09-29 05:36 UTC — ✅ Long overlay in place at /workspace/mhh-long
- 2026-09-29 05:45 UTC — ✅ dry run T5 passed:   peak_vram_gib: 33.13 
- 2026-09-29 05:46 UTC — ✅ render probe: render ok osmesa LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket 70.59711201985677
- 2026-09-29 05:46 UTC — 🚀 real run launched: T5 seed 22, log /workspace/logs/long_T5_s22.log
- 2026-09-29 06:46 UTC — heartbeat: {"log_bytes": 34561, "log_mtime": 1790664389.0, "phase": "train", "step": 414, "s_per_it": 7.68, "at_utc": "2026-09-29T06:46:32Z"}
- 2026-09-29 07:46 UTC — heartbeat: {"log_bytes": 66644, "log_mtime": 1790668006.0, "phase": "train", "step": 892, "s_per_it": 7.36, "at_utc": "2026-09-29T07:46:52Z"}
- 2026-09-29 08:47 UTC — heartbeat: {"log_bytes": 98351, "log_mtime": 1790671629.0, "phase": "train", "step": 1360, "s_per_it": 7.14, "at_utc": "2026-09-29T08:47:10Z"}
- 2026-09-29 09:47 UTC — heartbeat: {"log_bytes": 130831, "log_mtime": 1790675242.0, "phase": "train", "step": 1841, "s_per_it": 6.93, "at_utc": "2026-09-29T09:47:28Z"}
- 2026-09-29 10:08 UTC — ❌ FATAL: balance guard: {"log_bytes": 141714, "log_mtime": 1790676512.0, "phase": "train", "step": 2003, "s_per_it": 19.0, "hours_left": 27.1, "balance": 19.53783398, "spend_per_hr": 1.033, "usd_needed": 20.61, "verdict": "STOP_SHORT", "at_utc": "2026-09-29T10:08:36Z"}
- 2026-09-29 10:08 UTC — run process still alive; the stop will end it
