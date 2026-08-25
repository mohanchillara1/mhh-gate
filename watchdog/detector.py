#!/usr/bin/env python3
"""
Layer 2 detector — deterministic thresholds over telemetry.jsonl. No LLM.
Raises one flag FILE per condition in /workspace/flags/, containing the evidence.
Debounced: a flag re-fires only after its cooldown. Flags are the only interface
to Layer 3 (not built yet — per build order it waits for a clean real run).

§10: this script performs NO destructive action of any kind. It writes flag files,
NEEDS_HUMAN.md (notification, never a gate) and outbox lines. Nothing else.
Build order: BALANCE_CRITICAL was built first; the rest follow it in this file.
"""
import json, os, time

W = "/workspace"
TELE = f"{W}/telemetry.jsonl"
FLAGS = f"{W}/flags"
NEEDS = f"{W}/NEEDS_HUMAN.md"
POD_RATE_USD_HR = 0.46          # deploy-time price of this pod (GPU+disk), constant
COOLDOWN_S = {                   # per-flag re-fire cooldown
    "BALANCE_CRITICAL": 600, "BALANCE_BLIND": 21600, "STALL": 900,
    "LOG_FROZEN": 900, "PROC_GONE": 3600, "OOM_RISK": 600, "DISK_LOW": 1800,
    "LOSS_BAD": 3600, "THROUGHPUT_DROP": 1800,
}

def tail_ticks(n=12):
    try:
        with open(TELE, "rb") as f:
            f.seek(0, 2); size = f.tell()
            f.seek(max(0, size - 200_000))
            lines = f.read().decode(errors="replace").splitlines()
        out = []
        for ln in lines[-n:]:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
        return out
    except Exception:
        return []

def v(tick, key):
    node = tick.get(key)
    return node.get("v") if isinstance(node, dict) else None

def raise_flag(name, evidence):
    """Write flag file unless inside cooldown. Evidence = dict, dumped verbatim."""
    os.makedirs(FLAGS, exist_ok=True)
    path = f"{FLAGS}/{name}"
    now = time.time()
    if os.path.exists(path) and now - os.path.getmtime(path) < COOLDOWN_S.get(name, 900):
        return False
    with open(path, "w") as f:
        json.dump({"flag": name, "ts": time.strftime("%FT%TZ", time.gmtime()),
                   "evidence": evidence}, f, indent=1)
    print(f"FLAG {name}")
    return True

def notify_human(title, body):
    """NEEDS_HUMAN.md is a NOTIFICATION, never a gate (spec §4b). Append-only."""
    stampline = f"\n## {time.strftime('%FT%TZ', time.gmtime())} — {title}\n{body}\n"
    mode = "a" if os.path.exists(NEEDS) else "w"
    with open(NEEDS, mode) as f:
        if mode == "w":
            f.write("# NEEDS_HUMAN — notifications only; the watchdog has already "
                    "continued working (spec §4b)\n")
        f.write(stampline)

def main():
    ticks = tail_ticks()
    if not ticks:
        return
    cur = ticks[-1]
    train = v(cur, "train") or {}
    gpu = v(cur, "gpu") or {}
    prog = v(cur, "progress") or {}
    log = v(cur, "log")
    disk = v(cur, "disk") or {}
    bal = v(cur, "balance")
    meta = v(cur, "run_meta") or {}
    run_active = bool(train.get("alive"))

    # ---- 1. BALANCE_CRITICAL (build-order #2; the run-loser) -------------
    if isinstance(bal, (int, float)):
        runway_h = bal / POD_RATE_USD_HR
        # remaining run time: from steps/sec over last ticks + remaining steps
        remaining_h = None
        steps = [(t["ts"], (v(t, "progress") or {}).get("step")) for t in ticks]
        steps = [(ts, s) for ts, s in steps if s is not None]
        if len(steps) >= 2 and meta.get("max_steps"):
            (t0, s0), (t1, s1) = steps[0], steps[-1]
            dt = (time.mktime(time.strptime(t1, "%Y-%m-%dT%H:%M:%SZ"))
                  - time.mktime(time.strptime(t0, "%Y-%m-%dT%H:%M:%SZ")))
            if dt > 0 and s1 > s0:
                sps = (s1 - s0) / dt
                remaining_h = (meta["max_steps"] - s1) / sps / 3600
        threshold_h = 2 * remaining_h if remaining_h is not None else 4.0  # no run measurable -> flat 4h floor
        if runway_h < threshold_h:
            raise_flag("BALANCE_CRITICAL", {
                "balance_usd": bal, "rate_usd_hr": POD_RATE_USD_HR,
                "runway_h": round(runway_h, 2),
                "remaining_run_h": remaining_h if remaining_h is None else round(remaining_h, 2),
                "threshold_h": round(threshold_h, 2),
                "note": "L3 action when built: force checkpoint + clean STOP (never terminate) before $0",
            })
    elif bal == "UNAVAILABLE":
        if raise_flag("BALANCE_BLIND", {"balance": "UNAVAILABLE",
                "cause": "pod-scoped RUNPOD_API_KEY cannot read clientBalance",
                "fix": "human: create ACCOUNT-scoped read key at runpod.io/console/user/settings, "
                       "write it to /workspace/runpod_account_key (chmod 600)"}):
            notify_human("BALANCE_CRITICAL is blind",
                "The top flag cannot see the account balance. Place an ACCOUNT-scoped RunPod API "
                "key in /workspace/runpod_account_key. Until then the watchdog cannot warn before "
                "a $0 termination. It continues monitoring everything else.")

    # ---- 2. STALL: GPU 0% for 5 ticks while PID alive --------------------
    if run_active and len(ticks) >= 5:
        utils = [(v(t, "gpu") or {}).get("util_pct") for t in ticks[-5:]]
        if all(u is not None and u == 0 for u in utils):
            raise_flag("STALL", {"gpu_util_last5": utils, "pid": train.get("pid"),
                                 "note": "PID alive, GPU idle 5 consecutive ticks"})

    # ---- 3. LOG_FROZEN: mtime unchanged > 10 min while PID alive ---------
    if run_active and log and log.get("mtime_epoch"):
        age = time.time() - log["mtime_epoch"]
        if age > 600:
            raise_flag("LOG_FROZEN", {"log": log.get("path"), "mtime_age_s": int(age),
                                      "pid": train.get("pid")})

    # ---- 4. PROC_GONE: PID dead, no COMPLETE marker ----------------------
    if train.get("pid") is not None and train.get("alive") is False:
        logpath = (meta or {}).get("log")
        complete = False
        if logpath and os.path.exists(logpath):
            try:
                with open(logpath, "rb") as f:
                    f.seek(max(0, os.path.getsize(logpath) - 20000))
                    complete = b"COMPLETE" in f.read()
            except Exception:
                pass
        if not complete:
            raise_flag("PROC_GONE", {"pid": train.get("pid"), "log": logpath,
                                     "complete_marker": complete})

    # ---- 5. OOM_RISK: VRAM > 90% for 3 ticks -----------------------------
    if len(ticks) >= 3:
        frac = []
        for t in ticks[-3:]:
            g = v(t, "gpu") or {}
            if g.get("mem_used_mb") and g.get("mem_total_mb"):
                frac.append(g["mem_used_mb"] / g["mem_total_mb"])
        if len(frac) == 3 and all(f_ > 0.90 for f_ in frac):
            raise_flag("OOM_RISK", {"vram_frac_last3": [round(f_, 3) for f_ in frac]})

    # ---- 6. DISK_LOW: quota-based (df is FUSE-pool garbage here) ---------
    # full-volume usage arrives from codecheck's sample file; own timestamp shown
    used_file = f"{W}/flags/.volume_used_sample.json"
    if os.path.exists(used_file):
        try:
            s = json.load(open(used_file))
            free_gb = disk.get("volume_quota_gb", 150) - s["used_bytes"]/1e9
            if free_gb < 20:
                raise_flag("DISK_LOW", {"free_gb_vs_quota": round(free_gb, 1),
                                        "sample_ts": s.get("ts"),
                                        "note": "relieve by MOVING superseded checkpoints to /workspace/archive — never delete (§10)"})
        except Exception:
            pass

    # ---- 7. LOSS_BAD: NaN/inf ------------------------------------------
    loss = prog.get("loss")
    if isinstance(loss, str) and loss.lower() in ("nan", "inf", "-inf"):
        raise_flag("LOSS_BAD", {"loss": loss, "step": prog.get("step"),
                                "note": "divergence is a SCIENTIFIC RESULT (§4b): record failed, "
                                        "never retried, next seed launches"})

    # ---- 8. THROUGHPUT_DROP: < 70% of first-100-step baseline ------------
    base_file = f"{W}/flags/.throughput_baseline.json"
    steps = [(t["ts"], (v(t, "progress") or {}).get("step")) for t in ticks]
    steps = [(ts, s) for ts, s in steps if s is not None]
    if len(steps) >= 2:
        (t0, s0), (t1, s1) = steps[0], steps[-1]
        dt = (time.mktime(time.strptime(t1, "%Y-%m-%dT%H:%M:%SZ"))
              - time.mktime(time.strptime(t0, "%Y-%m-%dT%H:%M:%SZ")))
        if dt > 0 and s1 > s0:
            sps = (s1 - s0) / dt
            if not os.path.exists(base_file) and s1 <= 150:
                json.dump({"sps": sps, "at_step": s1,
                           "ts": time.strftime("%FT%TZ", time.gmtime())}, open(base_file, "w"))
            elif os.path.exists(base_file):
                base = json.load(open(base_file))
                if base.get("sps") and sps < 0.7 * base["sps"]:
                    raise_flag("THROUGHPUT_DROP", {"current_sps": round(sps, 3),
                                                   "baseline": base})

if __name__ == "__main__":
    main()
