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
- 2026-09-29 10:48 UTC — 🚀 real run launched: T5 seed 22, log /workspace/logs/long_T5_s22.log
- 2026-09-29 11:49 UTC — heartbeat: {"log_bytes": 36104, "log_mtime": 1790682545.0, "phase": "train", "step": 437, "s_per_it": 7.0, "s_per_it_avg": 7.636, "at_utc": "2026-09-29T11:49:09Z"}
- 2026-09-29 12:49 UTC — heartbeat: {"log_bytes": 68922, "log_mtime": 1790686158.0, "phase": "train", "step": 926, "s_per_it": 7.79, "s_per_it_avg": 7.505, "at_utc": "2026-09-29T12:49:25Z"}
- 2026-09-29 13:49 UTC — heartbeat: {"log_bytes": 100248, "log_mtime": 1790689780.0, "phase": "train", "step": 1388, "s_per_it": 7.26, "s_per_it_avg": 7.617, "at_utc": "2026-09-29T13:49:41Z"}
- 2026-09-29 14:49 UTC — heartbeat: {"log_bytes": 132031, "log_mtime": 1790693393.0, "phase": "train", "step": 1859, "s_per_it": 7.61, "s_per_it_avg": 7.63, "at_utc": "2026-09-29T14:49:57Z"}
- 2026-09-29 15:50 UTC — heartbeat: {"log_bytes": 163675, "log_mtime": 1790697006.0, "phase": "train", "step": 2329, "s_per_it": 7.88, "s_per_it_avg": 7.642, "at_utc": "2026-09-29T15:50:14Z"}
- 2026-09-29 16:50 UTC — heartbeat: {"log_bytes": 196652, "log_mtime": 1790700625.0, "phase": "train", "step": 2819, "s_per_it": 7.14, "s_per_it_avg": 7.597, "at_utc": "2026-09-29T16:50:32Z"}
- 2026-09-29 17:50 UTC — heartbeat: {"log_bytes": 228682, "log_mtime": 1790704246.0, "phase": "train", "step": 3294, "s_per_it": 7.47, "s_per_it_avg": 7.601, "at_utc": "2026-09-29T17:50:49Z"}
- 2026-09-29 18:51 UTC — heartbeat: {"log_bytes": 261435, "log_mtime": 1790707865.0, "phase": "train", "step": 3780, "s_per_it": 6.98, "s_per_it_avg": 7.581, "at_utc": "2026-09-29T18:51:07Z"}
- 2026-09-29 19:51 UTC — heartbeat: {"log_bytes": 294190, "log_mtime": 1790711481.0, "phase": "train", "step": 4268, "s_per_it": 7.57, "s_per_it_avg": 7.562, "at_utc": "2026-09-29T19:51:24Z"}
- 2026-09-29 20:51 UTC — heartbeat: {"log_bytes": 327055, "log_mtime": 1790715099.0, "phase": "train", "step": 4756, "s_per_it": 7.56, "s_per_it_avg": 7.546, "at_utc": "2026-09-29T20:51:43Z"}
- 2026-09-29 21:52 UTC — heartbeat: {"log_bytes": 360087, "log_mtime": 1790718718.0, "phase": "train", "step": 5239, "s_per_it": 7.1, "s_per_it_avg": 7.542, "at_utc": "2026-09-29T21:52:01Z"}
- 2026-09-29 22:52 UTC — heartbeat: {"log_bytes": 393355, "log_mtime": 1790722332.0, "phase": "train", "step": 5731, "s_per_it": 7.17, "s_per_it_avg": 7.525, "at_utc": "2026-09-29T22:52:18Z"}
- 2026-09-29 23:27 UTC — ✅ TRAINING DONE (checkpoint-6000), eval started
- 2026-09-29 23:52 UTC — heartbeat: {"log_bytes": 507272, "log_mtime": 1790725822.0, "phase": "eval", "shards_started": 4, "tasks_done": [], "at_utc": "2026-09-29T23:52:38Z"}
- 2026-09-30 00:52 UTC — heartbeat: {"log_bytes": 508836, "log_mtime": 1790729443.0, "phase": "eval", "shards_started": 14, "tasks_done": [{"task": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", "n_success": 21, "n": 21}, {"task": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", "n_success": 20, "n": 20}, {"task": "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it", "n_success": 20, "n": 20}], "at_utc": "2026-09-30T00:52:57Z"}
- 2026-09-30 01:53 UTC — heartbeat: {"log_bytes": 509834, "log_mtime": 1790732845.0, "phase": "eval", "shards_started": 21, "tasks_done": [{"task": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", "n_success": 21, "n": 21}, {"task": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", "n_success": 20, "n": 20}, {"task": "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it", "n_success": 20, "n": 20}, {"task": "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", "n_success": 16, "n": 20}, {"task": "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", "n_success": 20, "n": 20}], "at_utc": "2026-09-30T01:53:15Z"}
