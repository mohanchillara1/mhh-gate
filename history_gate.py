"""transition-gated history — a training-free mitigation for the frame-history penalty.

STATUS: prototype. Implemented and unit-tested on synthetic sequences (CPU only, `python
history_gate.py`). NOT yet wired into the GR00T eval path and NOT yet evaluated on LIBERO.
Nothing in this file has produced a success-rate number. Do not cite it as a result.

THE HYPOTHESIS IT TESTS
-----------------------
The LIBERO-Long probe (seed 21, exploratory) found the five-frame penalty concentrated in 2 of
10 tasks: KITCHEN_SCENE8 19/20 -> 16/20 and LIVING_ROOM_SCENE6 20/20 -> 18/21. Six additional
failed episodes, all on long-horizon multi-stage tasks. LIBERO-Spatial, which is essentially one
continuous reach, showed a paired delta of exactly 0.0 across three seeds.

That pattern is the signature of *copycat* / causal confusion (de Haan et al.; Wen et al.): a
behavioural-cloning policy handed its own recent past can lower training loss by extrapolating its
previous actions rather than reading the scene. Extrapolation is nearly free accuracy while motion
is smooth, and it is exactly wrong at a sub-goal transition, where the correct action is
discontinuous with the previous one. Spatial has ~one phase; Long has many.

THE MITIGATION
--------------
If history is useful *within* a phase and harmful *across* a boundary, then gate it on the
boundary rather than removing it globally:

    when the current observation is dissimilar enough from recent history to look like a phase
    change, collapse the history window to the current frame (i.e. fall back to single-frame
    conditioning) and let it refill as the phase stabilises.

Why this design and not plain history-dropout: the single-frame arm scored **99.5%** on this suite.
Single-frame is not a degraded mode, it is a *measured-good* fallback. So the gate's worst case —
firing on every step — is bounded below by T1 behaviour, and its best case keeps history everywhere
it was already helping. That bound is the reason to prefer gating over ablation.

It is training-free: it runs at inference on the existing T5 checkpoint, so testing it costs eval
time and no new training.

FALSIFIABLE PREDICTION (pre-committed before any run)
-----------------------------------------------------
If the copycat/transition account is right, gating recovers KITCHEN_SCENE8 and LIVING_ROOM_SCENE6
toward T1 levels *without* degrading the eight tasks that were already perfect in both arms.
If the two tasks do not recover, the hypothesis is wrong and we report that.
"""

from collections import deque

import numpy as np


def frame_distance(a, b, downsample=8):
    """Mean absolute difference between two frames, on a coarse grayscale grid.

    Deliberately cheap: this runs once per environment step inside the rollout loop, so it must
    cost far less than a policy forward pass. Coarse downsampling also makes it insensitive to
    pixel noise and to small object motion, which is what we want — we are looking for a scene
    reconfiguration (gripper closes, object leaves the table, drawer opens), not for jitter.

    Frames are HxW, HxWxC uint8 or float. Returns a float in [0, 1].
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"frame shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim == 3:
        a = a.mean(axis=2)
        b = b.mean(axis=2)
    if a.max() > 1.0 or b.max() > 1.0:
        a = a / 255.0
        b = b / 255.0
    a = a[::downsample, ::downsample]
    b = b[::downsample, ::downsample]
    return float(np.abs(a - b).mean())


class TransitionGatedHistory:
    """Maintains the history window a history-conditioned policy is fed, and collapses it on a
    detected phase transition.

    Usage inside a rollout, one call per env step:

        gate = TransitionGatedHistory(history_len=5, tau=0.06)
        gate.reset()
        for obs in episode:
            window = gate.step(obs["agentview_image"])   # list of `history_len` frames, oldest first
            action = policy(window, ...)

    `history_len` must match the arm's `video.delta_indices` length (5 for the T5 arm), so the
    tensor shape the processor receives is unchanged. Gating only changes *which* frames occupy
    the slots, never how many.
    """

    def __init__(self, history_len=5, tau=0.06, hold_steps=3, downsample=8):
        if history_len < 1:
            raise ValueError("history_len must be >= 1")
        if hold_steps < 0:
            raise ValueError("hold_steps must be >= 0")
        self.history_len = history_len
        self.tau = tau
        self.hold_steps = hold_steps
        self.downsample = downsample
        self.reset()

    def reset(self):
        """Clear state. Must be called at every episode reset.

        Note the episode-start case is exactly the bug this project found in the released loader:
        with no history available, the only correct fill is to repeat the *current* frame. Filling
        with anything else (notably a wrapped-around end-of-episode frame) is the defect documented
        in mhh-gate/README.md section A.
        """
        self._buf = deque(maxlen=self.history_len)
        self._hold = 0
        self.n_steps = 0
        self.n_gated = 0
        self.transitions = []      # step indices where the gate fired
        self.distances = []        # per-step distance, for calibration and post-hoc analysis

    def step(self, frame):
        """Feed the current observation, get back the history window to condition on."""
        gated = False
        if not self._buf:
            # episode start: repeat-current-frame fill (see reset() docstring)
            d = 0.0
        else:
            d = frame_distance(self._buf[-1], frame, self.downsample)
            if d > self.tau:
                gated = True
                self._hold = self.hold_steps
            elif self._hold > 0:
                gated = True
                self._hold -= 1

        self.distances.append(d)

        if gated or not self._buf:
            # collapse: every slot becomes the current frame -> single-frame conditioning,
            # which is the arm measured at 99.5% on this suite.
            self._buf.clear()
            for _ in range(self.history_len):
                self._buf.append(frame)
            if gated:
                self.n_gated += 1
                self.transitions.append(self.n_steps)
        else:
            self._buf.append(frame)

        self.n_steps += 1
        return list(self._buf)

    @property
    def gate_rate(self):
        """Fraction of steps on which the gate fired. Sanity band: a task with a handful of
        sub-goals should land in the low single-digit percents. Near 0 means tau is too high and
        the gate is inert; near 1 means tau is too low and we have silently rebuilt the T1 arm.
        """
        return self.n_gated / self.n_steps if self.n_steps else 0.0


def calibrate_tau(episodes, quantile=0.98, downsample=8):
    """Pick tau from data instead of by hand.

    `episodes` is an iterable of frame sequences (e.g. rollouts already recorded on the pod, or
    training-set episodes). Returns the given quantile of consecutive-frame distances.

    Choose the quantile BEFORE looking at any success rate. Tuning tau against SR would turn this
    mitigation into the same post-hoc degree of freedom the harness exists to prevent, and
    `run_gate.py` treats that class of choice as a pre-registration violation.
    """
    dists = []
    for ep in episodes:
        for prev, cur in zip(ep, ep[1:]):
            dists.append(frame_distance(prev, cur, downsample))
    if not dists:
        raise ValueError("no frame pairs to calibrate on")
    return float(np.quantile(dists, quantile))


# --------------------------------------------------------------------------------------------
# self-test: synthetic sequences, no GPU, no LIBERO. `python history_gate.py`
# --------------------------------------------------------------------------------------------

def _synthetic_episode(n_phases=3, steps_per_phase=20, size=64, seed=0):
    """Frames that drift slowly within a phase and jump discontinuously between phases."""
    rng = np.random.default_rng(seed)
    frames = []
    for p in range(n_phases):
        base = rng.uniform(0.2, 0.8, size=(size, size)).astype(np.float32)
        base[:] = np.clip(base + p * 0.0, 0, 1)
        cur = base.copy()
        for _ in range(steps_per_phase):
            cur = np.clip(cur + rng.normal(0, 0.002, cur.shape), 0, 1).astype(np.float32)
            frames.append(cur.copy())
    return frames, [i * steps_per_phase for i in range(1, n_phases)]


def _run_self_test():
    frames, true_boundaries = _synthetic_episode()
    tau = calibrate_tau([frames], quantile=0.98)
    print(f"calibrated tau = {tau:.5f}  (98th pct of consecutive-frame distance)")

    gate = TransitionGatedHistory(history_len=5, tau=tau, hold_steps=3)
    for f in frames:
        window = gate.step(f)
        assert len(window) == 5, "history window must always be history_len long"

    print(f"steps={gate.n_steps}  gated={gate.n_gated}  gate_rate={gate.gate_rate:.3f}")
    print(f"true phase boundaries : {true_boundaries}")
    print(f"gate fired at         : {gate.transitions}")

    # every true boundary should be caught within a couple of steps
    for b in true_boundaries:
        assert any(abs(t - b) <= 2 for t in gate.transitions), f"missed boundary at step {b}"

    # and it must not fire constantly — that would just be the T1 arm wearing a costume
    assert gate.gate_rate < 0.25, f"gate too trigger-happy: {gate.gate_rate:.3f}"

    # within-phase drift alone must not trip it
    calm, _ = _synthetic_episode(n_phases=1, steps_per_phase=60, seed=1)
    calm_gate = TransitionGatedHistory(history_len=5, tau=tau, hold_steps=3)
    for f in calm:
        calm_gate.step(f)
    assert calm_gate.n_gated == 0, f"fired {calm_gate.n_gated} times on a single-phase episode"
    print(f"single-phase control  : gated {calm_gate.n_gated} times (expected 0)")

    # degenerate config: history_len=1 is the T1 arm and must be a no-op passthrough
    one = TransitionGatedHistory(history_len=1, tau=tau)
    for f in frames[:10]:
        assert len(one.step(f)) == 1
    print("history_len=1 passthrough OK")

    print("\nself-test PASSED — prototype behaves as designed on synthetic data.")
    print("NOT evidence of any effect on LIBERO. Next step is wiring it into the eval path.")


if __name__ == "__main__":
    _run_self_test()
