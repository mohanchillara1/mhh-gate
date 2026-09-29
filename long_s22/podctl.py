"""
podctl.py -- the chain's small helpers, stdlib only (runs in any python3 on the pod).

    python3 podctl.py balance                 -> {"balance": 12.3, "spend_per_hr": 0.74} or {"error": ...}
    python3 podctl.py progress LOG ARM        -> {"phase": "train", "step": 1608, "s_per_it": 2.8, ...}
    python3 podctl.py need LOG ARM            -> progress + hours_left + usd_needed + verdict (guard)
    python3 podctl.py stop                    -> stops THIS pod (never terminates)

Balance comes from RunPod GraphQL `myself { clientBalance currentSpendPerHr }` using
RUNPOD_ACCOUNT_KEY (dad's key, a read-only key is enough). Stop uses the pod's own
`runpodctl stop pod $RUNPOD_POD_ID` first, GraphQL podStop with the account key as fallback.
Nothing here can terminate a pod: the only mutation it sends is podStop.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

GRAPHQL = "https://api.runpod.io/graphql"

# Planning rates if the log has no tqdm rate yet (measured on A40, Plan -- Staged Gate Runs.md).
# Seed-22 measurements (2026-09-29): T1 3.04-3.49 s/it; T5 attempt 1 hourly 6.93-7.36, whole-run 7.68 incl. model load.
DEFAULT_S_PER_IT = {"T1": 3.2, "T5": 7.3}
EVAL_SHARDS = 40                      # 10 tasks x 4 shards of 5
DEFAULT_EVAL_H = 6.0                  # T5 Long eval on osmesa took 5.97 h (seed 21)
SETUP_H = 0.75                        # clone + uv sync + downloads + LIBERO island + dry run; measured 0.5 h on 09-29
FLOOR_USD = 1.50                      # hard floor, as in every earlier chain
MARGIN = 1.15                         # projection must clear need x 1.15


def _gql(query: str) -> dict:
    key = os.environ.get("RUNPOD_ACCOUNT_KEY", "")
    if not key:
        return {"error": "RUNPOD_ACCOUNT_KEY unset"}
    req = urllib.request.Request(
        GRAPHQL, data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 # api.runpod.io answers 403 to Python's default "Python-urllib/3.x" agent (seen 2026-09-29)
                 "User-Agent": "mhh-gate-long-s22/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:  # network blips are reported, never swallowed into a number
        return {"error": f"{type(e).__name__}: {e}"}


def balance() -> dict:
    if os.environ.get("MHH_MOCK_BALANCE"):
        v = os.environ["MHH_MOCK_BALANCE"]            # "40,0.74" or "@file" holding that (local tests)
        f = (open(v[1:]).read() if v.startswith("@") else v).strip().split(",")
        return {"balance": float(f[0]), "spend_per_hr": float(f[1]), "n_running": int(f[2]) if len(f) > 2 else 1}
    r = _gql("query { myself { clientBalance currentSpendPerHr pods { desiredStatus } } }")
    try:
        me = r["data"]["myself"]
        n = sum(1 for x in (me.get("pods") or []) if x.get("desiredStatus") == "RUNNING")
        return {"balance": float(me["clientBalance"]), "spend_per_hr": float(me["currentSpendPerHr"]),
                "n_running": max(1, n)}
    except Exception:
        return {"error": r.get("error") or json.dumps(r)[:300]}


TQDM = re.compile(r"(\d+)/(\d+) \[[^\]]*?([\d.]+)(s/it|it/s)")
ELAPSED = re.compile(r"(\d+)/6000 \[([\d:]+)<")


def progress(log: str, arm: str) -> dict:
    try:
        txt = open(log, errors="replace").read()
    except FileNotFoundError:
        return {"phase": "not_started"}
    out: dict = {"log_bytes": len(txt), "log_mtime": os.path.getmtime(log)}
    if "GATE ABORTED" in txt:
        out["phase"] = "aborted"
        return out
    if "[5/6] evaluating" in txt:
        shards = len(re.findall(r"\[eval\] \S+ shard base_seed=", txt))
        tasks = re.findall(r"\[eval\] (libero_sim/\S+): (\d+)/(\d+)", txt)
        out.update(phase="eval", shards_started=shards,
                   tasks_done=[{"task": t.split("/")[-1], "n_success": int(a), "n": int(b)} for t, a, b in tasks])
        return out
    steps = [(int(a), int(b), float(r), u) for a, b, r, u in TQDM.findall(txt) if int(b) == 6000]
    if steps:
        step, _, rate, unit = steps[-1]
        out.update(phase="train", step=step, s_per_it=rate if unit == "s/it" else 1.0 / rate)
        # tqdm's rate is a short moving average: a checkpoint save makes it jump (7 -> 19 s/it at step 2000,
        # 2026-09-29, which falsely stopped T5). For projections use the whole-run average: elapsed / steps.
        el = ELAPSED.findall(txt)
        if el and step > 50:
            s, e = el[-1]
            parts = [int(x) for x in e.split(":")]
            secs = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
            out["s_per_it_avg"] = round(secs / int(s), 3)
    else:
        out.update(phase="setup_or_preflight", step=0)
    return out


def need(log: str, arm: str, eval_h: float = DEFAULT_EVAL_H) -> dict:
    p = progress(log, arm)
    if p["phase"] == "not_started":
        h = SETUP_H + 6000 * DEFAULT_S_PER_IT[arm] / 3600 + eval_h
    elif p["phase"] == "eval":
        h = eval_h * max(0, EVAL_SHARDS - max(0, p["shards_started"] - 1)) / EVAL_SHARDS
    elif p["phase"] in ("train", "setup_or_preflight"):
        rate = p.get("s_per_it_avg") or p.get("s_per_it") or DEFAULT_S_PER_IT[arm]
        h = (6000 - p.get("step", 0)) * rate / 3600 + eval_h
    else:
        h = 0.0
    b = balance()
    p["hours_left"] = round(h, 2)
    if "error" in b:
        p.update(balance_error=b["error"], verdict="BLIND")
        return p
    # This pod bills for all h hours. The other running pod(s) bill for at most min(h, MHH_OTHER_H), the other
    # arm's planned hours (set per pod at creation). Unset -> the old upper bound: everything bills for h.
    # Option A pods are identical, so this pod's share of the spend is spend / n_running.
    own = b["spend_per_hr"] / b["n_running"]
    other_h = float(os.environ.get("MHH_OTHER_H", h))
    usd = own * h + (b["spend_per_hr"] - own) * min(h, other_h)
    p.update(balance=b["balance"], spend_per_hr=b["spend_per_hr"], usd_needed=round(usd, 2))
    if b["balance"] <= FLOOR_USD:
        p["verdict"] = "STOP_FLOOR"
    elif b["balance"] < usd * MARGIN + FLOOR_USD:
        # Two strikes: a projection must be short on two consecutive checks (>= 5 min apart) before it stops a pod.
        sf = os.path.join(os.environ.get("MHH_W", "/workspace"), f"guard_strike_{arm}")
        prev = os.path.getmtime(sf) if os.path.exists(sf) else None
        if prev and time.time() - prev >= 300:
            p["verdict"] = "STOP_SHORT"
        else:
            if not prev:
                open(sf, "w").write(json.dumps(p))
            p["verdict"] = "WARN_SHORT_1"
    else:
        p["verdict"] = "OK"
        sf = os.path.join(os.environ.get("MHH_W", "/workspace"), f"guard_strike_{arm}")
        if os.path.exists(sf):
            os.replace(sf, sf + ".cleared")
    return p


def _finished_ok() -> bool:
    """True only if this pod holds a completed results.json (a real DONE, not a crash or guard stop)."""
    import glob
    runs = os.environ.get("MHH_RUNS_DIR", os.path.join(os.environ.get("MHH_W", "/workspace"), "mhh-long-runs"))
    for f in glob.glob(os.path.join(runs, "*_seed*_att*", "results.json")):
        try:
            if json.load(open(f)).get("status") == "completed":
                return True
        except Exception:
            pass
    return False


def done_hold() -> dict:
    """After a real DONE, keep the pod (GPU + volume) up until the laptop backup finishes, so a closed laptop
    cannot cost the files. Ends on /workspace/BACKUP_DONE, after MHH_DONE_HOLD_S (default 3 h), or when the
    balance reaches the floor. Dad, 2026-09-29: keep pods until the code/results are saved locally."""
    w = os.environ.get("MHH_W", "/workspace")
    limit = int(os.environ.get("MHH_DONE_HOLD_S", "10800"))
    t0 = time.time()
    open(os.path.join(w, "DONE_HOLD_STARTED"), "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    while True:
        if os.path.exists(os.path.join(w, "BACKUP_DONE")):
            return {"hold_end": "BACKUP_DONE", "held_s": int(time.time() - t0)}
        if time.time() - t0 >= limit:
            return {"hold_end": "timeout", "held_s": int(time.time() - t0)}
        b = balance()
        if "error" not in b and b["balance"] <= FLOOR_USD + 0.5:
            return {"hold_end": "balance_floor", "held_s": int(time.time() - t0), "balance": b["balance"]}
        time.sleep(int(os.environ.get("MHH_HOLD_POLL_S", "60")))


def stop() -> dict:
    hold = done_hold() if _finished_ok() else {"hold_end": "none (not a completed run)"}
    r = _stop()
    r["done_hold"] = hold
    return r


def _stop() -> dict:
    pod = os.environ.get("RUNPOD_POD_ID", "")
    if os.environ.get("MHH_MOCK_STOP"):
        return {"stopped": pod or "mock", "via": "mock"}
    if not pod:
        return {"error": "RUNPOD_POD_ID unset (not on a RunPod pod?)"}
    try:
        r = subprocess.run(["runpodctl", "stop", "pod", pod], capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return {"stopped": pod, "via": "runpodctl", "out": r.stdout.strip()[-300:]}
        first = r.stderr.strip()[-300:]
    except Exception as e:
        first = f"{type(e).__name__}: {e}"
    r2 = _gql('mutation { podStop(input: {podId: "%s"}) { id desiredStatus } }' % pod)
    if "data" in r2 and r2["data"].get("podStop"):
        return {"stopped": pod, "via": "graphql", "runpodctl_error": first}
    return {"error": "STOP FAILED", "runpodctl": first, "graphql": json.dumps(r2)[:300]}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "balance":
        res = balance()
    elif cmd == "progress":
        res = progress(sys.argv[2], sys.argv[3])
    elif cmd == "need":
        res = need(sys.argv[2], sys.argv[3], float(os.environ.get("MHH_EVAL_H", DEFAULT_EVAL_H)))
    elif cmd == "stop":
        res = stop()
    else:
        sys.exit(__doc__)
    res["at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(res))
