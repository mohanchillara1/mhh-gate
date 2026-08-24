"""
preflight.py — the three landmines, checked so they HARD FAIL.

Each check answers a question the project has already been burned by, and each
one is deliberately made at the CONSUMING site rather than at the config site,
because this repo demonstrably ships flags that are written, serialized, and
never read.

  A. allow_padding actually ARRIVES at extract_step_data       (issue #745 class)
  B. the architecture that actually LOADED, not the one declared  (issue #755)
  C. the normalization mode actually IN FORCE                     (issue #745)
  D. the arm's video.delta_indices actually reached the processor

Failure raises ``PreflightFailure``; ``run_gate.py`` turns that into a non-zero
exit before a single training step runs.
"""

from __future__ import annotations

import copy
import numpy as np

import config as C


class PreflightFailure(RuntimeError):
    """Raised when a pre-registration guarantee cannot be proven."""


# ---------------------------------------------------------------------------
# arm construction
# ---------------------------------------------------------------------------


def build_modality_configs(delta_indices: list[int]) -> dict:
    """The full ``{embodiment: {modality: ModalityConfig}}`` dict for one arm.

    Deep-copies ``MODALITY_CONFIGS`` first.  That object is a module-level
    singleton (embodiment_configs.py) and is the default value of
    ``DataConfig.modality_configs`` (data_config.py:70-72); mutating it in
    place would leak the arm's history window into every later construction in
    the same process, including the eval server.

    Only ``video.delta_indices`` differs between arms -- state, action and
    language are left exactly as shipped (embodiment_configs.py:193-210).
    """
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    if C.EMBODIMENT_VALUE not in MODALITY_CONFIGS:
        raise PreflightFailure(
            f"embodiment '{C.EMBODIMENT_VALUE}' not in MODALITY_CONFIGS "
            f"(have: {sorted(MODALITY_CONFIGS)})"
        )
    cfgs = copy.deepcopy(MODALITY_CONFIGS)
    cfgs[C.EMBODIMENT_VALUE]["video"].delta_indices = list(delta_indices)
    return cfgs


# ---------------------------------------------------------------------------
# A. allow_padding must ARRIVE at extract_step_data
# ---------------------------------------------------------------------------
#
# Why this is not paranoia.  extract_step_data builds
#     indices_to_load = [step_index + delta_index for ...]     (:40)
# and clamps to [0, len-1] ONLY when allow_padding is True     (:41-42).
# With the default False, step 0 of a T=2 arm asks for index -1, and pandas
# `.iloc[-1]` returns the episode's LAST row -- a silent teleport-backward
# pair, no exception.  T=5 poisons the first four steps of every episode.
#
# FinetuneConfig does not expose the flag at all (grep of
# gr00t/configs/finetune_config.py at the pinned commit: zero hits), so the
# value has to be written onto ``config.data.allow_padding`` (data_config.py:94)
# and is then threaded:
#     factory.py:68                allow_padding=self.config.data.allow_padding
#     sharded_single_step_dataset.py:137,146   -> self.allow_padding
#     sharded_single_step_dataset.py:260-266   -> extract_step_data(..., self.allow_padding)
# Every link in that chain is a chance for a refactor to drop it.  We check the
# END of the chain.

_GUARD_STATE: dict = {"installed": False, "calls": 0, "false_calls": 0,
                      "last": None, "original": None}


def install_padding_guard(fatal: bool = True):
    """Wrap ``extract_step_data`` so a False ``allow_padding`` cannot pass.

    ``ShardedSingleStepDataset.get_datapoint`` resolves ``extract_step_data``
    from its own module globals at call time (sharded_single_step_dataset.py
    :260), so replacing the module attribute intercepts every real call --
    including the ones made inside forked dataloader workers, which inherit
    this patched module image.  That inheritance is why
    ``config.data.multiprocessing_context`` must stay ``"fork"``; under
    ``"spawn"`` the child re-imports a clean module and the guard silently
    disappears.  ``run_gate.py`` asserts the context.
    """
    import gr00t.data.dataset.sharded_single_step_dataset as ssd

    if _GUARD_STATE["installed"]:
        return

    original = ssd.extract_step_data

    def guarded(episode_data, step_index, modality_configs, embodiment_tag,
                allow_padding=False):
        _GUARD_STATE["calls"] += 1
        if not allow_padding:
            _GUARD_STATE["false_calls"] += 1
            msg = (
                "PRE-REGISTRATION VIOLATION: extract_step_data was called with "
                f"allow_padding={allow_padding!r} at step_index={step_index}. "
                "Episode-boundary indices are NOT clamped, so negative indices "
                "wrap to the episode's last frame via pandas .iloc "
                "(sharded_single_step_dataset.py:40-45). Training data would be "
                "silently boundary-poisoned. Aborting."
            )
            if fatal:
                raise PreflightFailure(msg)
            print("!! " + msg)
        out = original(episode_data, step_index, modality_configs,
                       embodiment_tag, allow_padding)
        _GUARD_STATE["last"] = {
            "step_index": int(step_index),
            "allow_padding": bool(allow_padding),
            "video_delta_indices": list(
                modality_configs["video"].delta_indices
            ),
        }
        return out

    ssd.extract_step_data = guarded
    _GUARD_STATE.update(installed=True, original=original)


def guard_report() -> dict:
    return {
        "installed": _GUARD_STATE["installed"],
        "extract_step_data_calls_observed": _GUARD_STATE["calls"],
        "calls_with_allow_padding_false": _GUARD_STATE["false_calls"],
        "last_observed_call": _GUARD_STATE["last"],
    }


def assert_padding_arrives(delta_indices: list[int]) -> dict:
    """Prove, in-process and on real data, that padding reaches the consumer.

    Builds the dataset exactly the way ``DatasetFactory`` does (factory.py
    :61-68), calls ``get_datapoint`` once at step 0, and checks three things:

      1. the guard observed ``allow_padding=True`` at the call site;
      2. ``ShardedSingleStepDataset.allow_padding`` is True on the object;
      3. the returned slot-0 frame is PIXEL-IDENTICAL to episode row 0
         (repeat-first-frame) and NOT to the episode's last row (wrap-around).

    (3) is the one that cannot be faked by a config that looks right.
    """
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
    from gr00t.data.embodiment_tags import EmbodimentTag

    if not C.ALLOW_PADDING:
        raise PreflightFailure(
            "config.ALLOW_PADDING is False. The pre-registered padding scheme is "
            "repeat-first-frame, which is what allow_padding=True implements "
            "(sharded_single_step_dataset.py:41-42). Refusing to run."
        )

    install_padding_guard(fatal=True)
    tag = EmbodimentTag.resolve(C.EMBODIMENT_TAG)
    modality_configs = build_modality_configs(delta_indices)

    ds = ShardedSingleStepDataset(
        dataset_path=C.DATASET_PATH,
        embodiment_tag=tag,
        modality_configs=modality_configs[C.EMBODIMENT_VALUE],
        shard_size=C.SHARD_SIZE,
        episode_sampling_rate=C.EPISODE_SAMPLING_RATE,
        seed=C.PREREGISTERED_SEEDS[0],
        allow_padding=C.ALLOW_PADDING,
    )
    if ds.allow_padding is not True:
        raise PreflightFailure(
            f"ShardedSingleStepDataset.allow_padding is {ds.allow_padding!r}, not True."
        )

    # A callable stub: get_datapoint only asserts the processor exists and then
    # calls it (sharded_single_step_dataset.py:259, :269). We do not need the
    # 3B processor to prove index clamping, and loading it here would make this
    # check depend on the very thing it is guarding.
    captured: dict = {}

    def stub_processor(messages):
        captured["messages"] = messages
        return {"stub": True}

    ds.processor = stub_processor

    episode = ds.episode_loader[0]
    n_rows = len(episode)
    ds.get_datapoint(episode, 0)

    step = captured["messages"][0]["content"]          # VLAStepData
    video_key = modality_configs[C.EMBODIMENT_VALUE]["video"].modality_keys[0]
    frames = [np.asarray(f) for f in step.images[video_key]]

    if len(frames) != len(delta_indices):
        raise PreflightFailure(
            f"expected {len(delta_indices)} video slots for arm deltas "
            f"{delta_indices}, got {len(frames)}"
        )

    col = f"video.{video_key}"
    first_row = np.asarray(episode[col].iloc[0])
    last_row = np.asarray(episode[col].iloc[n_rows - 1])

    n_pad = sum(1 for d in delta_indices if d < 0)     # slots clamped at step 0
    for i in range(n_pad):
        if not np.array_equal(frames[i], first_row):
            raise PreflightFailure(
                f"padding scheme violated: at step 0 slot {i} (delta="
                f"{delta_indices[i]}) is not the episode's FIRST frame. "
                "Expected repeat-first-frame clamping."
            )
        if np.array_equal(frames[i], last_row) and not np.array_equal(
            first_row, last_row
        ):
            raise PreflightFailure(
                f"WRAP-AROUND DETECTED at step 0 slot {i}: the frame equals the "
                "episode's LAST row. allow_padding did not take effect."
            )

    if _GUARD_STATE["last"] is None or _GUARD_STATE["last"]["allow_padding"] is not True:
        raise PreflightFailure(
            "the guard did not observe allow_padding=True at extract_step_data; "
            f"last observed call: {_GUARD_STATE['last']!r}"
        )

    # Mid-episode sanity: with a real history window the slots must be ordered
    # and temporally distinct, i.e. not the same frame five times.
    mid_note = None
    if len(delta_indices) > 1 and n_rows > (abs(min(delta_indices)) + 2):
        s = min(60, n_rows // 2)
        ds.get_datapoint(episode, s)
        mid = captured["messages"][0]["content"]
        mid_frames = [np.asarray(f) for f in mid.images[video_key]]
        for i, d in enumerate(delta_indices):
            gt = np.asarray(episode[col].iloc[s + d])
            if not np.array_equal(mid_frames[i], gt):
                raise PreflightFailure(
                    f"frame stacking wrong mid-episode: slot {i} (delta={d}) at "
                    f"step {s} does not equal episode row {s + d}."
                )
        distinct = len({f.tobytes() for f in mid_frames})
        is_static_arm = len(set(delta_indices)) == 1
        if not is_static_arm and distinct == 1:
            raise PreflightFailure(
                "all history slots are pixel-identical mid-episode on a "
                "non-static arm -- frames are duplicated, not stacked."
            )
        mid_note = {"probe_step": s, "distinct_frames": distinct}

    return {
        "verdict": "PASS",
        "checked_at": "extract_step_data (sharded_single_step_dataset.py:27-79)",
        "dataset_allow_padding_attr": bool(ds.allow_padding),
        "guard_observed_allow_padding": True,
        "episode_0_rows": int(n_rows),
        "padded_slots_verified_pixelwise": n_pad,
        "video_key_checked": video_key,
        "mid_episode_stack_check": mid_note,
    }


# ---------------------------------------------------------------------------
# B. the architecture that actually loaded
# ---------------------------------------------------------------------------


def _count_module_list(model, dotted: str):
    obj = model
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return len(obj)


def load_base_model(for_training: bool):
    """Load the base checkpoint the way the training pipeline does.

    ``import gr00t.model`` registers ``Gr00tN1d7`` with ``AutoModel``
    (gr00t_n1d7.py:627), which is what makes ``AutoModel.from_pretrained``
    resolve.  With ``for_training=True`` we pass the same tune flags the
    pipeline passes (setup.py:82-95) so the trainable-parameter set -- and
    therefore the dry-run's memory and step time -- matches the real run.
    """
    from transformers import AutoModel
    import gr00t.model  # noqa: F401  (registration side effect)

    kwargs = {}
    if for_training:
        kwargs = dict(
            tune_llm=C.TUNE_LLM,
            tune_visual=C.TUNE_VISUAL,
            tune_projector=C.TUNE_PROJECTOR,
            tune_diffusion_model=C.TUNE_DIFFUSION_MODEL,
            state_dropout_prob=C.STATE_DROPOUT_PROB,
        )
    return AutoModel.from_pretrained(C.BASE_MODEL_PATH, **kwargs)


def dump_architecture(model) -> dict:
    """Record what loaded. Never trust conf.yaml or the README (issue #755).

    The released GR00T-N1.7-3B checkpoint's ``config.json`` wins over any
    finetune ``select_layer`` override, which is written into ``conf.yaml`` as
    if it had been used.  The declared code defaults are select_layer=12
    (configs/model/gr00t_n1d7.py:47) and diffusion num_layers=16 (:94); the
    smoke test measured 16 and 32 on live weights.  We read the live object.
    """
    import torch.nn as nn

    info: dict = {"source": "live model object, not conf.yaml"}
    cfg = getattr(model, "config", None)
    for key in ("select_layer", "max_seq_len", "use_alternate_vl_dit",
                "backbone_embedding_dim", "action_horizon", "hidden_size",
                "attend_text_every_n_blocks", "use_vlln", "model_name"):
        info[f"config.{key}"] = getattr(cfg, key, "<absent>")
    dmc = getattr(cfg, "diffusion_model_cfg", None)
    if isinstance(dmc, dict):
        info["config.diffusion_model_cfg.num_layers"] = dmc.get("num_layers", "<absent>")
        info["config.diffusion_model_cfg.num_attention_heads"] = dmc.get(
            "num_attention_heads", "<absent>"
        )

    llm_layers = None
    for path in ("backbone.model.language_model.layers",
                 "backbone.model.model.language_model.layers",
                 "backbone.model.layers"):
        try:
            llm_layers = _count_module_list(model, path)
            info["llm_layers_path"] = path
            break
        except AttributeError:
            continue

    dit_blocks = None
    for path in ("action_head.model.transformer_blocks",
                 "action_head.transformer_blocks",
                 "action_head.model.blocks"):
        try:
            dit_blocks = _count_module_list(model, path)
            info["dit_blocks_path"] = path
            break
        except AttributeError:
            continue

    if llm_layers is None or dit_blocks is None:
        # Do not guess. Enumerate what IS there so the human can pin the path.
        candidates = [
            f"{name} (len={len(mod)})"
            for name, mod in model.named_modules()
            if isinstance(mod, nn.ModuleList) and len(mod) > 1
        ]
        raise PreflightFailure(
            "could not locate the LLM layer stack and/or the DiT block stack on "
            f"the loaded model (llm={llm_layers}, dit={dit_blocks}). "
            "Refusing to run without recording the real architecture "
            "(Isaac-GR00T issue #755). ModuleList candidates found:\n  "
            + "\n  ".join(candidates[:40])
        )

    info["llm_layers_loaded"] = llm_layers
    info["dit_blocks_loaded"] = dit_blocks
    info["total_parameters"] = sum(p.numel() for p in model.parameters())
    info["trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    info["expected_llm_layers"] = C.EXPECTED_LLM_LAYERS
    info["expected_dit_blocks"] = C.EXPECTED_DIT_BLOCKS
    info["matches_expected"] = (
        llm_layers == C.EXPECTED_LLM_LAYERS and dit_blocks == C.EXPECTED_DIT_BLOCKS
    )
    info["declared_select_layer_is_ignored"] = (
        getattr(cfg, "select_layer", None) != llm_layers
    )

    if not info["matches_expected"]:
        msg = (
            "ARCHITECTURE DRIFT: loaded LLM layers="
            f"{llm_layers} DiT blocks={dit_blocks}, but this gate is pinned to "
            f"{C.EXPECTED_LLM_LAYERS}/{C.EXPECTED_DIT_BLOCKS} (the values the "
            "smoke test measured on live weights, 2026-08-23). Runs on different "
            "architectures are not comparable."
        )
        if C.ARCH_DRIFT_IS_FATAL:
            raise PreflightFailure(msg)
        info["drift_warning"] = msg
        print("!! " + msg)
    return info


# ---------------------------------------------------------------------------
# C. the normalization mode actually in force
# ---------------------------------------------------------------------------


def build_processor(delta_indices: list[int]):
    """Build the processor exactly as ``Gr00tN1d7Pipeline._create_dataset`` does.

    The override hook is ``Gr00tN1d7Processor.from_pretrained``: it pops
    ``modality_configs`` from kwargs and merges per-embodiment entries over the
    checkpoint's own (processing_gr00t_n1d7.py:865-867), then applies a fixed
    allow-list of scalar overrides (:868-882).  ``AutoProcessor`` routes to it
    via ``AutoProcessor.register("Gr00tN1d7", Gr00tN1d7Processor)`` (:887).
    """
    from transformers import AutoProcessor
    import gr00t.model  # noqa: F401

    modality_configs = build_modality_configs(delta_indices)
    return AutoProcessor.from_pretrained(
        C.BASE_MODEL_PATH,
        modality_configs=modality_configs,
        use_percentiles=C.USE_PERCENTILES,
        use_mean_std=C.USE_MEAN_STD,
        state_dropout_prob=C.STATE_DROPOUT_PROB,
    )


def assert_arm_override(processor, delta_indices: list[int]) -> dict:
    """D. The arm's history window really reached the processor."""
    got = list(processor.modality_configs[C.EMBODIMENT_VALUE]["video"].delta_indices)
    if got != list(delta_indices):
        raise PreflightFailure(
            f"arm override did not take: processor has video.delta_indices={got}, "
            f"expected {delta_indices}."
        )
    return {
        "verdict": "PASS",
        "video_delta_indices": got,
        "video_modality_keys": list(
            processor.modality_configs[C.EMBODIMENT_VALUE]["video"].modality_keys
        ),
        "action_horizon": len(
            processor.modality_configs[C.EMBODIMENT_VALUE]["action"].delta_indices
        ),
    }


def record_normalization_mode(processor) -> dict:
    """Resolve and record the normalization actually applied to state/action.

    Issue #745, VERIFIED IN SOURCE at the pinned commit: the processor stores
    ``self.use_mean_std`` (processing_gr00t_n1d7.py:262) and serializes it
    (:784) and accepts it as an override key (:874) -- but it is NOT passed to
    ``StateActionProcessor`` (:251-258), whose ``__init__`` does not even
    accept the argument (state_action_processor.py:63-70).  It is dead config.

    What is actually in force, per joint group
    (state_action_processor.py:234-268):
      sin/cos   if apply_sincos_state_encoding and key in sin_cos_embedding_keys
      mean/std  elif key in mean_std_embedding_keys
      min/max   otherwise -- on q01/q99 when use_percentiles else min/max (:157-162),
                clipped to [-1,1] when clip_outliers.
    """
    sap = getattr(processor, "state_action_processor", None)
    if sap is None:
        raise PreflightFailure(
            "processor has no .state_action_processor; cannot record the "
            "normalization mode in force."
        )

    if C.USE_MEAN_STD:
        raise PreflightFailure(
            "config.USE_MEAN_STD is True, but that flag is DEAD in this repo at "
            "the pinned commit (issue #745): it is stored and serialized but "
            "never reaches StateActionProcessor. Requesting mean/std this way "
            "produces min/max silently. Refusing to run."
        )

    mc = processor.modality_configs[C.EMBODIMENT_VALUE]
    per_group: dict = {}
    for modality in ("state", "action"):
        cfg = mc.get(modality)
        if cfg is None:
            continue
        sincos = set(getattr(cfg, "sin_cos_embedding_keys", None) or [])
        meanstd = set(getattr(cfg, "mean_std_embedding_keys", None) or [])
        for key in cfg.modality_keys:
            if getattr(sap, "apply_sincos_state_encoding", False) and key in sincos:
                mode = "sin_cos"
            elif key in meanstd:
                mode = "mean_std"
            else:
                mode = "minmax_q01q99" if getattr(sap, "use_percentiles", False) \
                    else "minmax_min_max"
            per_group[f"{modality}.{key}"] = mode

    return {
        "verdict": "RECORDED",
        "processor.use_mean_std_flag_value": bool(getattr(processor, "use_mean_std", None)),
        "processor.use_mean_std_is_consumed": False,
        "processor.use_mean_std_note": (
            "dead config at commit "
            f"{C.GROOT_COMMIT[:9]}: set at processing_gr00t_n1d7.py:262, serialized "
            ":784, never forwarded to StateActionProcessor (:251-258)"
        ),
        "state_action_processor.use_percentiles": bool(
            getattr(sap, "use_percentiles", None)
        ),
        "state_action_processor.clip_outliers": bool(getattr(sap, "clip_outliers", None)),
        "state_action_processor.apply_sincos_state_encoding": bool(
            getattr(sap, "apply_sincos_state_encoding", None)
        ),
        "state_action_processor.use_relative_action": bool(
            getattr(sap, "use_relative_action", None)
        ),
        "effective_mode_per_joint_group": per_group,
        "statistics_source": (
            "checkpoint statistics.json, overridden by dataset statistics when "
            "config.data.override_pretraining_statistics is True (data_config.py:83)"
        ),
    }
