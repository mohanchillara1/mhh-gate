# long_s22 — LIBERO-Long, seed 22, both arms, one pod per arm

Exploratory second seed for the MHH LIBERO-Long probe (seed 21: T1 199/200 vs T5 194/200 on seeded
episodes). Off-protocol like seed 21, never pooled with the pre-registered Spatial cells. The frozen
Spatial gate at the repo root is not touched.

| file | what it is |
|---|---|
| `make_overlay.py` | builds the Long copy of the harness and proves it: 4 config changes only, the harness's own builder reproduces `eval_episodes_libero10.json` byte for byte (sha256 `5c1fcaa8…c3ac69`), and the probe's 7 failing seeds sit where the rule puts them. `--check` runs it in a temp dir. |
| `eval_episodes_libero10.json` | the Long eval list (10 tasks × 20 seeded episodes, shards of 5). |
| `chain.sh ARM` | the whole pod job: setup → overlay → dry run → osmesa render probe → real run → watch. Pushes to branch `runs/long-s22-ARM` at start, at dry run, every hour, at train→eval, at the end, and on any failure. Then stops the pod (stop, never terminate). Never retries anything. |
| `podctl.py` | balance query, progress parse, balance guard, pod stop. Stdlib only. |
| `mock_run.py` | local test stand-in for `run_gate.py` (`MHH_MOCK=1`). Prints fake numbers labelled MOCK. |

## Launch (per pod, after funding)

Pod env vars: `HF_TOKEN`, `GH_TOKEN` (push to this repo), `RUNPOD_ACCOUNT_KEY` (read-only is enough).
Volume ≥ 150 GB at `/workspace` (T5 checkpoint rotation peaks ~48 GB). Then, over SSH or the web terminal:

```bash
git clone -b master https://github.com/mohanchillara1/mhh-gate.git /workspace/mhh-gate
tmux new -d -s mhh "bash /workspace/mhh-gate/long_s22/chain.sh T5 2>&1 | tee -a /workspace/chain_T5.out"
```

After that the laptop can close. Progress is on GitHub, branch `runs/long-s22-T5`, file
`runs/long-s22/T5/STATUS.md`.

## Guard rules

- Balance preflight must cover the whole arm: this pod's rate × its hours, plus the other pod's rate ×
  min(its hours, `MHH_OTHER_H`, the other arm's planned hours, set per pod), ×1.15, plus a $1.50 floor.
  Without `MHH_OTHER_H` it falls back to assuming every pod bills until this arm ends. Checked every 10 min after that. Short → push, stop.
- No log growth for 90 min → stalled → push, stop.
- `run_gate.py` exits without `results.json` → push logs, stop. The one crash-policy rerun is a human call.
- If the stop call fails, it pushes "STOP FAILED" and retries every 5 min.
