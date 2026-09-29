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
DEFAULT_S_PER_IT = {"T1": 3.2, "T5": 8.5}
EVAL_SHARDS = 40                      # 10 tasks x 4 shards of 5
DEFAULT_EVAL_H = 6.0                  # T5 Long eval on osmesa took 5.97 h (seed 21)
SETUP_H = 1.5                         # clone + uv sync + downloads + LIBERO island + dry run
FLOOR_USD = 1.50                      # hard floor, as in every earlier chain
MARGIN = 1.15                         # projection must clear need x 1.15


def _gql(query: str) -> dict:
    key = os.environ.get("RUNPOD_ACCOUNT_KEY", "")
    if not key:
        return {"error": "RUNPOD_ACCOUNT_KEY unset"}
    req = urllib.request.Request(
        GRAPHQL, data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:  # network blips are reported, never swallowed into a number
        return {"error": f"{type(e).__name__}: {e}"}


def balance() -> dict:
    if os.environ.get("MHH_MOCK_BALANCE"):
        v = os.environ["MHH_MOCK_BALANCE"]            # "40,0.74" or "@file" holding that (local tests)
        b, s = (open(v[1:]).read() if v.startswith("@") else v).strip().split(",")
        return {"balance": float(b), "spend_per_hr": float(s)}
    r = _gql("query { myself { clientBalance currentSpendPerHr } }")
    try:
        me = r["data"]["myself"]
        return {"balance": float(me["clientBalance"]), "spend_per_hr": float(me["currentSpendPerHr"])}
    except Exception:
        return {"error": r.get("error") or json.dumps(r)[:300]}


TQDM = re.compile(r"(\d+)/(\d+) \[[^\]]*?([\d.]+)(s/it|it/s)")


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
        rate = p.get("s_per_it") or DEFAULT_S_PER_IT[arm]
        h = (6000 - p.get("step", 0)) * rate / 3600 + eval_h
    else:
        h = 0.0
    b = balance()
    p["hours_left"] = round(h, 2)
    if "error" in b:
        p.update(balance_error=b["error"], verdict="BLIND")
        return p
    # Upper bound: every pod on the account keeps billing until THIS arm is done.
    usd = h * b["spend_per_hr"]
    p.update(balance=b["balance"], spend_per_hr=b["spend_per_hr"], usd_needed=round(usd, 2))
    if b["balance"] <= FLOOR_USD:
        p["verdict"] = "STOP_FLOOR"
    elif b["balance"] < usd * MARGIN + FLOOR_USD:
        p["verdict"] = "STOP_SHORT"
    else:
        p["verdict"] = "OK"
    return p


def stop() -> dict:
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
