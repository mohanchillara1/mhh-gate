#!/usr/bin/env python3
"""
Layer 1 heartbeat — Making History Help GPU run supervisor.
Pure MEASUREMENT. Never decides, never edits code, never writes to the results path.
One JSON line per tick to telemetry.jsonl; rewrites STATUS.md.

NO-SYNTHETIC RULE (spec §2): every field carries the command that produced it in a
sibling "_src". A field that cannot be measured is null (or "UNAVAILABLE" for balance),
never estimated, never carried forward from a previous tick, never interpolated.
"""
import json, os, re, subprocess, sys, time

W = "/workspace"
TELE = f"{W}/telemetry.jsonl"
STATUS = f"{W}/STATUS.md"
RUN_PIDFILE = f"{W}/run.pid"        # launcher writes the training PID here
RUN_META = f"{W}/run.current.json"  # launcher writes {arm,seed,log,started,pid,...}
ACCT_KEY = f"{W}/runpod_account_key"  # a human drops an ACCOUNT-scoped key here (see NEEDS_HUMAN)

def sh(cmd, timeout=15):
    """Run a shell command; return (stdout stripped, ok). Never raises."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout.strip(), True
        return (r.stderr.strip() or r.stdout.strip()), False
    except Exception as e:
        return f"{type(e).__name__}: {e}", False

def field(value, src):
    return {"v": value, "_src": src}

def gpu():
    cmd = ("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,"
           "temperature.gpu,power.draw --format=csv,noheader,nounits")
    out, ok = sh(cmd)
    if not ok:
        return field(None, f"{cmd} -> FAILED: {out[:120]}")
    try:
        u, mu, mt, t, p = [x.strip() for x in out.split(",")]
        return field({"util_pct": float(u), "mem_used_mb": float(mu),
                      "mem_total_mb": float(mt), "temp_c": float(t),
                      "power_w": float(p)}, cmd)
    except Exception as e:
        return field(None, f"{cmd} -> UNPARSEABLE ({e}): {out[:120]}")

def compute_apps():
    cmd = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits"
    out, ok = sh(cmd)
    if not ok:
        return field(None, f"{cmd} -> FAILED: {out[:120]}")
    apps = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid, mem = [x.strip() for x in line.split(",")]
            apps.append({"pid": int(pid), "used_mb": float(mem)})
        except Exception:
            pass
    return field(apps, cmd)

def read_run_meta():
    try:
        with open(RUN_META) as f:
            return json.load(f)
    except Exception:
        return {}

def train_proc(meta):
    pid = meta.get("pid")
    if pid is None and os.path.exists(RUN_PIDFILE):
        try:
            pid = int(open(RUN_PIDFILE).read().strip())
        except Exception:
            pid = None
    if pid is None:
        return field({"pid": None, "alive": None},
                     f"no {RUN_PIDFILE} / run.current.json.pid -> no run registered")
    alive = os.path.exists(f"/proc/{pid}")  # kill -0 equivalent, no signal
    return field({"pid": pid, "alive": alive},
                 f"os.path.exists(/proc/{pid}) [kill -0 equivalent]")

def logstat(meta):
    path = meta.get("log")
    if not path or not os.path.exists(path):
        return field(None, f"log path {path!r} absent -> no log to stat")
    try:
        st = os.stat(path)
        return field({"path": path, "size_bytes": st.st_size,
                      "mtime_epoch": int(st.st_mtime)}, f"os.stat({path})")
    except Exception as e:
        return field(None, f"os.stat({path}) -> {e}")

STEP_RE = re.compile(r"'?(?:global_)?step'?\s*[=:]\s*(\d+)", re.I)
LOSS_RE = re.compile(r"'?loss'?\s*[=:]\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?|nan|inf|-inf)", re.I)

def progress(meta):
    path = meta.get("log")
    if not path or not os.path.exists(path):
        return field({"step": None, "loss": None}, "no log -> unparseable")
    out, ok = sh(f"tail -c 20000 {path!r}")
    if not ok:
        return field({"step": None, "loss": None}, f"tail failed: {out[:80]}")
    step = loss = None
    for m in STEP_RE.finditer(out):
        step = int(m.group(1))
    for m in LOSS_RE.finditer(out):
        loss = m.group(1)
    return field({"step": step, "loss": loss},
                 f"regex over tail -c 20000 {os.path.basename(path)}")

def disk():
    """FUSE volume: df reports the BACKEND POOL free space (~200TB), not our 150GB
    quota — recorded but labeled unreliable. Real pressure signal = du over the
    growing dir (mhh-runs) vs the deploy-time quota. Full-volume du lives in the
    10-min codecheck (too slow for a 60s tick). Timeouts -> null, never estimated."""
    d = {}
    out, ok = sh("df -P /workspace | tail -1")
    try:
        d["df_avail_gb"] = round(int(out.split()[3])/1024/1024, 1) if ok else None
    except Exception:
        d["df_avail_gb"] = None
    out, ok = sh("timeout 20 du -sb /workspace/mhh-runs 2>/dev/null | cut -f1")
    try:
        d["mhh_runs_bytes"] = int(out) if ok and out else None
    except Exception:
        d["mhh_runs_bytes"] = None
    d["volume_quota_gb"] = 150
    return field(d, "df -P /workspace (FUSE backend pool — unreliable for quota) + "
                    "timeout 20 du -sb /workspace/mhh-runs + quota=150GB (deploy-time constant)")

def balance():
    """Account balance via RunPod GraphQL. The pod-injected RUNPOD_API_KEY is POD-scoped
    and returns UNAUTHORIZED for myself.clientBalance; only an ACCOUNT key placed at
    ACCT_KEY works. Absent that, UNAVAILABLE — never estimated (no-synthetic)."""
    if not os.path.exists(ACCT_KEY):
        return field("UNAVAILABLE",
                     f"no account key at {ACCT_KEY}; pod-scoped RUNPOD_API_KEY cannot read clientBalance (see NEEDS_HUMAN)")
    try:
        key = open(ACCT_KEY).read().strip()
    except Exception as e:
        return field("UNAVAILABLE", f"cannot read {ACCT_KEY}: {e}")
    q = '{"query":"query { myself { clientBalance } }"}'
    out, ok = sh(f"curl -s -H 'Authorization: Bearer {key}' -H 'Content-Type: application/json' "
                 f"-d '{q}' https://api.runpod.io/graphql", timeout=20)
    if not ok:
        return field("UNAVAILABLE", f"curl failed: {out[:100]}")
    try:
        d = json.loads(out)
        bal = d.get("data", {}).get("myself", {}).get("clientBalance")
        if bal is None:
            return field("UNAVAILABLE", f"graphql returned no balance: {out[:120]}")
        return field(float(bal), "runpod graphql myself.clientBalance (account key)")
    except Exception as e:
        return field("UNAVAILABLE", f"parse: {e}: {out[:100]}")

def main():
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta = read_run_meta()
    tick = {
        "ts": ts,
        "gpu": gpu(),
        "compute_apps": compute_apps(),
        "train": train_proc(meta),
        "log": logstat(meta),
        "progress": progress(meta),
        "disk": disk(),
        "balance": balance(),
        "run_meta": {"v": meta or None, "_src": f"{RUN_META}" if meta else "no active run registered"},
    }
    with open(TELE, "a") as f:
        f.write(json.dumps(tick) + "\n")
    write_status(tick)

def val(node):
    """Extract .v from a field dict; may itself be None."""
    return node.get("v") if isinstance(node, dict) else None

def write_status(tick):
    g = val(tick["gpu"]); pr = val(tick["progress"]); tr = val(tick["train"])
    lg = val(tick["log"]); dk = val(tick["disk"]); bal = val(tick["balance"])
    rm = val(tick["run_meta"]) or {}
    def show(x, unit=""):
        return "UNAVAILABLE" if x is None else f"{x}{unit}"
    lines = []
    lines.append(f"# MHH watchdog — STATUS")
    lines.append("")
    lines.append(f"_tick {tick['ts']} — Layer 1 heartbeat. Every value below was measured this tick; UNAVAILABLE means not measurable, never estimated._")
    lines.append("")
    lines.append(f"- **run**: arm `{rm.get('arm','—')}` seed `{rm.get('seed','—')}`  (pid {tr.get('pid') if tr else '—'}, alive {tr.get('alive') if tr else '—'})")
    lines.append(f"- **step / loss**: {show(pr.get('step') if pr else None)} / {show(pr.get('loss') if pr else None)}")
    if g:
        lines.append(f"- **GPU**: util {show(g.get('util_pct'),'%')}, VRAM {show(g.get('mem_used_mb'),'MB')}/{show(g.get('mem_total_mb'),'MB')}, {show(g.get('temp_c'),'°C')}, {show(g.get('power_w'),'W')}")
    else:
        lines.append(f"- **GPU**: UNAVAILABLE ({tick['gpu']['_src'][:80]})")
    if lg:
        age = int(time.time()) - lg.get("mtime_epoch", int(time.time()))
        lines.append(f"- **log**: {lg.get('size_bytes')} bytes, last write {age}s ago (`{os.path.basename(lg.get('path',''))}`)")
    else:
        lines.append(f"- **log**: none registered")
    runs_gb = round(dk['mhh_runs_bytes']/1e9, 2) if dk and dk.get('mhh_runs_bytes') is not None else None
    lines.append(f"- **disk**: mhh-runs {show(runs_gb,' GB')} of {dk.get('volume_quota_gb') if dk else '?'} GB quota (df pool figure: {show(dk.get('df_avail_gb') if dk else None,' GB')}, unreliable on FUSE)")
    lines.append(f"- **balance**: {show(bal)}  ({'account key present' if bal not in (None,'UNAVAILABLE') else 'no account key — BALANCE_CRITICAL is blind, see NEEDS_HUMAN.md'})")
    # active flags
    flags_dir = f"{W}/flags"
    active = sorted(os.listdir(flags_dir)) if os.path.isdir(flags_dir) else []
    lines.append(f"- **active flags**: {', '.join(active) if active else 'none'}")
    if os.path.exists(f"{W}/NEEDS_HUMAN.md"):
        lines.append(f"- 🔔 **NEEDS_HUMAN.md present** (notification, not a gate)")
    lines.append("")
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, STATUS)

if __name__ == "__main__":
    main()
