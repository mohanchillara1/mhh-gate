#!/usr/bin/env python3
"""Demonstrate the GR00T frame-history loader bug on a synthetic episode.

No GPU, no checkpoint, no dataset, no Isaac-GR00T install. Only pandas + numpy.

The bug (NVIDIA/Isaac-GR00T issue #771, unmerged fix in PR #772, both opened
2026-09-05): extract_step_data builds

    indices_to_load = [step_index + delta for delta in config.delta_indices]

and clamps that list to [0, len-1] ONLY when allow_padding=True
(gr00t/data/dataset/sharded_single_step_dataset.py:40-42). The default is False.
So at step 0 of a five-frame arm the history indices are -4..-1, pandas .iloc
accepts negative positions, and the loader silently returns frames from the END
of the episode as "history". No exception is raised.

Run:  python3 demo/loader_bug_demo.py
"""
import numpy as np
import pandas as pd

EPISODE_LEN = 20
DELTA_INDICES = [-4, -3, -2, -1, 0]   # the five-frame (T5) arm


def episode():
    """Each row is labelled with its own index, so a wrong frame is unmistakable."""
    return pd.DataFrame({"video.cam": list(np.arange(EPISODE_LEN, dtype=float))})


def load(df, step_index, allow_padding):
    """The released indexing logic, reproduced exactly."""
    idx = [step_index + d for d in DELTA_INDICES]
    if allow_padding:
        idx = [max(0, min(i, len(df) - 1)) for i in idx]
    return idx, list(df["video.cam"].iloc[idx])


def main():
    df = episode()
    print(f"synthetic episode: {EPISODE_LEN} frames, frame k has value k")
    print(f"delta_indices    : {DELTA_INDICES}  (five-frame arm)\n")

    for step in (0, 2):
        raw_idx, raw = load(df, step, allow_padding=False)
        fix_idx, fix = load(df, step, allow_padding=True)
        print(f"--- step_index = {step} ---")
        print(f"  allow_padding=False (RELEASED DEFAULT)")
        print(f"    indices -> {raw_idx}")
        print(f"    frames  -> {raw}")
        wrapped = [i for i in raw_idx if i < 0]
        if wrapped:
            print(f"    ^ {len(wrapped)} negative index/indices reached .iloc and returned")
            print(f"      frames from the END of the episode. No exception raised.")
        print(f"  allow_padding=True  (the fix we applied before our runs)")
        print(f"    indices -> {fix_idx}")
        print(f"    frames  -> {fix}")
        print(f"    ^ history before the episode start is clamped to frame 0.\n")

    raw_idx, raw = load(df, 0, allow_padding=False)
    assert raw[0] == EPISODE_LEN - 4, "expected the wrap-around signature at step 0"
    assert load(df, 0, allow_padding=True)[1][0] == 0.0, "expected clamp-to-first-frame"
    print("Both assertions hold: the bug reproduces, and the fix removes it.")
    print("\nNote: PR #772 fixes this differently. It keeps allow_padding=False as the")
    print("default and clamps negative HISTORY indices to 0 while raising IndexError on")
    print("any other out-of-range index. Either route removes the silent wrap-around.")


if __name__ == "__main__":
    main()
