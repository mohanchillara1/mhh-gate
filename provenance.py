"""
provenance.py — capture the reference BEFORE the run.

Provenance is unrecoverable afterwards: once the pod is torn down nobody can
say which MuJoCo shipped, which LIBERO commit was checked out, or what the
instruction strings actually were.  Everything here is read from the live
environment, never from a config file or a README.

Nothing in this module raises on a missing value; it records
``{"error": "..."}`` instead, so an unreadable version never blocks a run --
but it is visible in the manifest rather than silently absent.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import config as C


# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 120) -> str:
    """Run a command, return stripped stdout, or an ``<error: ...>`` marker."""
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except Exception as exc:  # FileNotFoundError, TimeoutExpired, ...
        return f"<error: {type(exc).__name__}: {exc}>"
    if out.returncode != 0:
        return f"<error: exit {out.returncode}: {(out.stderr or out.stdout).strip()[:400]}>"
    return out.stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj) -> str:
    """Canonical hash of a JSON-able object (sorted keys, no incidental space)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# individual probes
# ---------------------------------------------------------------------------


def git_commit(repo: str) -> dict:
    if not os.path.isdir(repo):
        return {"path": repo, "error": "not a directory"}
    return {
        "path": repo,
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=repo),
        "dirty": _run(["git", "status", "--porcelain"], cwd=repo) != "",
        "describe": _run(["git", "describe", "--always", "--dirty"], cwd=repo),
    }


def training_env_versions() -> dict:
    """Versions inside the interpreter that will run training."""
    out: dict = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
    }
    for mod in ("torch", "numpy", "transformers", "accelerate", "pandas", "torchcodec"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as exc:
            out[mod] = f"<error: {type(exc).__name__}: {exc}>"
    try:
        import torch

        out["cuda_available"] = torch.cuda.is_available()
        out["cuda_version"] = torch.version.cuda
        out["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "capability": list(torch.cuda.get_device_capability(i)),
                "total_mem_gib": round(
                    torch.cuda.get_device_properties(i).total_memory / 2**30, 2
                ),
            }
            for i in range(torch.cuda.device_count())
        ]
        out["bf16_supported"] = bool(
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        )
    except Exception as exc:
        out["cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    out["nvidia_smi"] = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"]
    )
    return out


_LIBERO_PROBE = r"""
import json, sys
out = {}
for mod in ("mujoco", "robosuite", "numpy", "gymnasium", "torch"):
    try:
        out[mod] = __import__(mod).__version__
    except Exception as e:
        out[mod] = "<error: %s: %s>" % (type(e).__name__, e)
try:
    import libero, os
    out["libero_file"] = getattr(libero, "__file__", None)
except Exception as e:
    out["libero_file"] = "<error: %s: %s>" % (type(e).__name__, e)
out["python"] = sys.version.split()[0]
print(json.dumps(out))
"""


def libero_env_versions() -> dict:
    """Versions inside the SEPARATE LIBERO uv venv that runs the simulator.

    MuJoCo lives here, not in the training env -- setup_libero.sh builds this
    island and pins ``mujoco==3.3.1``.  We record what is actually importable.
    """
    py = C.LIBERO_VENV_PYTHON
    if not os.path.exists(py):
        return {"error": f"LIBERO venv python not found at {py} "
                         f"(run gr00t/eval/sim/LIBERO/setup_libero.sh)"}
    raw = _run([py, "-c", _LIBERO_PROBE], timeout=300)
    try:
        return json.loads(raw)
    except Exception:
        return {"error": "probe did not return JSON", "raw": raw[:1000]}


_INSTRUCTION_PROBE = r"""
import json, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from libero.libero import benchmark
suite_name = sys.argv[1]
suite = benchmark.get_benchmark_dict()[suite_name]()
out = {}
for task_id in range(suite.get_num_tasks()):
    task = suite.get_task(task_id)
    out["libero_sim/" + task.name] = {
        "task_id": task_id,
        "language": task.language,
        "bddl_file": task.bddl_file,
        "problem_folder": task.problem_folder,
    }
print(json.dumps(out))
"""


def libero_instruction_strings(suite: str) -> dict:
    """The EXACT instruction strings the policy will be conditioned on.

    ``LiberoEnv`` is registered with ``task_description=task.language``
    (libero_env.py:192-213) and injects it as
    ``annotation.human.action.task_description`` on every observation
    (libero_env.py:157).  Recording it here means a later prompt change --
    like the LIBERO-Plus perturbation-id leak in lerobot PR #4497 -- is
    visible in the diff instead of invisible in the results.
    """
    py = C.LIBERO_VENV_PYTHON
    if not os.path.exists(py):
        return {"error": f"LIBERO venv python not found at {py}"}
    raw = _run([py, "-c", _INSTRUCTION_PROBE, suite], timeout=600)
    try:
        return json.loads(raw)
    except Exception:
        return {"error": "probe did not return JSON", "raw": raw[:2000]}


def dataset_fingerprint(path: str) -> dict:
    """Cheap, reproducible fingerprint of the training dataset directory."""
    if not os.path.isdir(path):
        return {"path": path, "error": "not a directory"}
    meta_dir = os.path.join(path, "meta")
    files: dict = {}
    if os.path.isdir(meta_dir):
        for name in sorted(os.listdir(meta_dir)):
            fp = os.path.join(meta_dir, name)
            if os.path.isfile(fp):
                files[f"meta/{name}"] = {
                    "bytes": os.path.getsize(fp),
                    "sha256": sha256_file(fp),
                }
    n_video = 0
    for root, _dirs, names in os.walk(os.path.join(path, "videos")):
        n_video += sum(1 for n in names if n.endswith(".mp4"))
    return {"path": path, "meta_files": files, "n_mp4": n_video}


# ---------------------------------------------------------------------------


def collect(arm: str, seed: int, delta_indices: list[int], extra: dict) -> dict:
    """Assemble the full provenance block written before training starts."""
    groot = git_commit(C.GROOT_REPO)
    return {
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "arm": arm,
        "seed": seed,
        "video_delta_indices": list(delta_indices),
        "isaac_groot": {
            **groot,
            "pinned_commit": C.GROOT_COMMIT,
            "matches_pin": groot.get("commit") == C.GROOT_COMMIT,
        },
        "libero_repo": git_commit(C.LIBERO_REPO),
        "training_env": training_env_versions(),
        "libero_eval_env": libero_env_versions(),
        "dataset": dataset_fingerprint(C.DATASET_PATH),
        "base_model_path": C.BASE_MODEL_PATH,
        "env_vars_of_interest": {
            k: os.environ.get(k)
            for k in (
                "MUJOCO_GL", "PYOPENGL_PLATFORM", "CUDA_VISIBLE_DEVICES",
                "HF_HOME", "HF_TOKEN_PATH", "GR00T_EVAL_SEED",
                "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "WANDB_MODE",
            )
        },
        **extra,
    }
