- 2026-09-29 10:18 UTC — chain.sh started: arm=T5 seed=22 mock=0 pod=a5h2opjmy9xnl9
- 2026-09-29 10:18 UTC — balance preflight: {"phase": "not_started", "hours_left": 21.67, "balance": 19.4212756661, "spend_per_hr": 1.074, "usd_needed": 15.66, "verdict": "WARN_SHORT_1", "at_utc": "2026-09-29T10:18:19Z"}
- 2026-09-29 10:18 UTC — ❌ FATAL: balance does not cover this arm: {"phase": "not_started", "hours_left": 21.67, "balance": 19.4212756661, "spend_per_hr": 1.074, "usd_needed": 15.66, "verdict": "WARN_SHORT_1", "at_utc": "2026-09-29T10:18:19Z"}
- 2026-09-29 10:18 UTC — pre-training failure: stopping in 1800s unless /workspace/HOLD appears
- 2026-09-29 10:19 UTC — HOLD found: not stopping; a human has the pod
- 2026-09-29 10:20 UTC — chain.sh started: arm=T5 seed=22 mock=0 pod=a5h2opjmy9xnl9
- 2026-09-29 10:20 UTC — balance preflight: {"phase": "not_started", "hours_left": 18.92, "balance": 19.4212756661, "spend_per_hr": 1.074, "usd_needed": 14.19, "verdict": "OK", "at_utc": "2026-09-29T10:20:18Z"}
- 2026-09-29 10:20 UTC — setup started
- 2026-09-29 10:34 UTC — ✅ setup done
  FIXED_STEP_COUNT = 6000
  EVAL_SHARD_SIZE = 5
  EVAL_LIST_MASTER_SEED = 20260823
  MAX_EPISODE_STEPS = 720
  N_ACTION_STEPS = 8
  TRAIN_MAX_STEPS = 6000
OVERLAY OK: 10 Long tasks, 200 seeded episodes, sha 5c1fcaa8d90d..., builder reproduces the list byte for byte, 7/7 probe failing seeds in place.
{"dest": "/workspace/mhh-long", "config_sha256": "40b830ed28b1f51fab079dd43d3f453fa972ad254cec40c271c77a143decd12a"}
- 2026-09-29 10:34 UTC — ✅ Long overlay in place at /workspace/mhh-long
- 2026-09-29 10:48 UTC — ✅ dry run T5 passed:   peak_vram_gib: 33.14 
- 2026-09-29 10:48 UTC — ✅ render probe: render ok osmesa LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket 70.59711201985677
