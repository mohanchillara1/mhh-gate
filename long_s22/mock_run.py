"""
mock_run.py -- LOCAL TEST ONLY. Stands in for run_gate.py when chain.sh runs with MHH_MOCK=1,
so the chain's control flow (progress parsing, pushes, guard, stop) can be exercised with no
GPU, no data and no spend. It prints the same markers run_gate.py prints. Every number it
prints is fake and labelled MOCK; nothing it writes is a result.

MOCK_FAIL=train  -> exits 2 halfway through training (tests the fatal path)
MOCK_SPEED=0.02  -> seconds per fake step batch
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm"); ap.add_argument("--seed", type=int); ap.add_argument("--notes", default="")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
dt = float(os.environ.get("MOCK_SPEED", "0.02"))
run_dir = Path(os.environ["MHH_RUNS_DIR"]) / f"{a.arm}_seed{a.seed}_att1"
run_dir.mkdir(parents=True, exist_ok=True)
print("MHH GATE  run_id=MOCK", flush=True)
(run_dir / "run_manifest.json").write_text(json.dumps({"MOCK": True, "status": "started"}) + "\n")
for step in range(0, 6001, 500):
    print(f"\r {step*100//6000}%|##| {step}/6000 [00:01<00:01, 2.80s/it]", flush=True)
    time.sleep(dt)
    if step == 3000 and os.environ.get("MOCK_FAIL") == "train":
        print("Traceback: MOCK crash", flush=True)
        sys.exit(2)
print("[5/6] evaluating ONLY at step 6000...", flush=True)
for t in range(10):
    for s in range(4):
        print(f"[eval] libero_sim/MOCK_TASK_{t} shard base_seed={20260823 + t*100000 + s*5}", flush=True)
        time.sleep(dt)
    print(f"[eval] libero_sim/MOCK_TASK_{t}: 20/20", flush=True)
line = f"MOCK RUN arm={a.arm} seed={a.seed} -- NOT A RESULT"
(run_dir / "results.json").write_text(json.dumps({"MOCK": True, "results_log_line": line}, indent=2) + "\n")
print(line, flush=True)
