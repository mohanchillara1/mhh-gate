"""
make_overlay.py -- build the LIBERO-Long copy of the harness for the seed-22 run.

The frozen gate (config.py at the repo root) is LIBERO-Spatial and stays untouched.
The seed-21 Long probe ran from a hand-edited copy on the pod (/workspace/mhh-probe-long)
that died with the pod. This script makes that copy reproducibly instead:

    python long_s22/make_overlay.py DEST_DIR            # write the Long copy
    python long_s22/make_overlay.py --check             # build in a temp dir, verify, delete

It copies the harness files, changes exactly four things in the copy's config.py and
nothing else, installs the Long eval list, then proves the copy is right:

  1. every substitution matched exactly once (a changed config.py fails loudly);
  2. the harness's OWN build_eval_episode_list(), run on the copy, reproduces
     eval_episodes_libero10.json byte for byte (sha256 5c1fcaa8...c3ac69);
  3. the task names in that list are the 10 LIBERO-Long tasks, read from the file's
     contents, not from its "suite" field (Learnings 2026-08-30: the suite field lied once).

Changes relative to the Spatial gate (all of them):
  LIBERO_SUITE             "libero_spatial" -> "libero_10"
  LIBERO_SPATIAL_TASKS     Spatial list -> the Long list in Isaac-GR00T README order
                           (the variable keeps its name because run_gate.py reads it)
  DATASET_PATH default     libero_spatial_no_noops_1.0.0_lerobot -> libero_10_no_noops_1.0.0_lerobot
  EVAL_EPISODE_LIST_SHA256 d51371a5... -> 5c1fcaa8...
Training recipe, arms, steps (6000), EVAL_SHARD_SIZE (5), master seed, max steps (720)
and n_action_steps (8) are unchanged.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

LONG_SHA256 = "5c1fcaa8d90d1a8e1963053f5f13c9692d254f2dd906de2b94704ffe44c3ac69"
LONG_LIST = HERE / "eval_episodes_libero10.json"

# Isaac-GR00T @51d4c89, examples/LIBERO/README.md, "Libero 10 (Long)", in that order.
# Order matters: base seed = MASTER + task_idx*100000 + shard_idx*5.
LONG_TASKS = [
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
]

# The seven failing seeds the seed-21 probe recorded. Each must sit under the task named.
PROBE_FAILING_SEEDS = {
    "KITCHEN_SCENE8": [21060827, 21060832, 21060836, 21060837, 21060842],
    "LIVING_ROOM_SCENE6": [20860841, 20860842],
}

HARNESS_FILES = ["config.py", "run_gate.py", "preflight.py", "provenance.py",
                 "history_gate.py", "requirements.txt"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sub_once(text: str, old: str, new: str, what: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f"OVERLAY FAILED: {what}: expected exactly 1 match in config.py, found {n}. "
                 "config.py changed since this overlay was written; stop and re-check by hand.")
    return text.replace(old, new)


def patch_config(src: str) -> str:
    out = sub_once(src, 'LIBERO_SUITE = "libero_spatial"', 'LIBERO_SUITE = "libero_10"', "suite")
    out = sub_once(out, "libero_spatial_no_noops_1.0.0_lerobot", "libero_10_no_noops_1.0.0_lerobot",
                   "dataset path")
    out = sub_once(out, "d51371a59be544a0d26f4a0f27171fd0d65d39bb94cadf1b2635de67f3b61663", LONG_SHA256,
                   "eval list sha")
    start = out.index("LIBERO_SPATIAL_TASKS = [")
    end = out.index("]\n", start) + 2
    body = "".join(f'    "{t}",\n' for t in LONG_TASKS)
    new_block = ("# MHH LONG OVERLAY (long_s22/make_overlay.py): LIBERO-Long tasks, README order.\n"
                 "# Name kept because run_gate.py reads LIBERO_SPATIAL_TASKS.\n"
                 f"LIBERO_SPATIAL_TASKS = [\n{body}]\n")
    return out[:start] + new_block + out[end:]


def build(dest: Path) -> dict:
    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"{dest} exists and is not empty; refusing to write into it.")
    dest.mkdir(parents=True, exist_ok=True)
    for f in HARNESS_FILES:
        shutil.copy2(REPO / f, dest / f)
    for d in ("watchdog", "demo"):
        if (REPO / d).is_dir():
            shutil.copytree(REPO / d, dest / d)
    (dest / "config.py").write_text(patch_config((REPO / "config.py").read_text()))
    shutil.copy2(LONG_LIST, dest / "eval_episodes.json")
    return verify(dest)


def verify(dest: Path) -> dict:
    listed = dest / "eval_episodes.json"
    if sha256(listed) != LONG_SHA256:
        sys.exit(f"eval list sha mismatch: {sha256(listed)}")

    # Run the harness's own builder on the copy.
    sys.path.insert(0, str(dest))
    for m in ("config", "provenance", "run_gate"):
        sys.modules.pop(m, None)
    C = importlib.import_module("config")
    RG = importlib.import_module("run_gate")
    rebuilt = json.dumps(RG.build_eval_episode_list(), indent=2, sort_keys=True) + "\n"
    if rebuilt.encode() != listed.read_bytes():
        sys.exit("harness builder does NOT reproduce eval_episodes_libero10.json byte for byte")
    if C.EVAL_EPISODE_LIST_SHA256 != LONG_SHA256 or C.LIBERO_SUITE != "libero_10":
        sys.exit("patched config.py does not carry the Long values")
    for k in ("FIXED_STEP_COUNT", "EVAL_SHARD_SIZE", "EVAL_LIST_MASTER_SEED", "MAX_EPISODE_STEPS",
              "N_ACTION_STEPS", "TRAIN_MAX_STEPS"):
        print(f"  {k} = {getattr(C, k)}")
    assert C.FIXED_STEP_COUNT == 6000 and C.EVAL_SHARD_SIZE == 5
    assert C.ARMS["T1"] == [0] and C.ARMS["T5"] == [-4, -3, -2, -1, 0]

    # Read contents, not metadata.
    data = json.loads(listed.read_text())
    names = list(data["tasks"])
    if sorted(names) != sorted(LONG_TASKS):
        sys.exit(f"eval list tasks are not the LIBERO-Long tasks: {names}")
    for scene, seeds in PROBE_FAILING_SEEDS.items():
        key = next(n for n in names if f"/{scene}_" in n)
        missing = [s for s in seeds if s not in data["tasks"][key]["episode_seeds"]]
        if missing:
            sys.exit(f"probe failing seeds {missing} not under {scene}")
    n_eps = sum(len(v["episode_seeds"]) for v in data["tasks"].values())
    print(f"OVERLAY OK: {len(names)} Long tasks, {n_eps} seeded episodes, sha {LONG_SHA256[:12]}..., "
          "builder reproduces the list byte for byte, 7/7 probe failing seeds in place.")
    return {"dest": str(dest), "config_sha256": sha256(dest / "config.py")}


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        with tempfile.TemporaryDirectory() as td:
            build(Path(td) / "mhh-long")
    elif len(sys.argv) == 2:
        print(json.dumps(build(Path(sys.argv[1]).resolve())))
    else:
        sys.exit(__doc__)
