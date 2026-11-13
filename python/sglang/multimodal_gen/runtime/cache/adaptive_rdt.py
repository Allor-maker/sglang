# SPDX-License-Identifier: Apache-2.0
"""Simplified AlignCache-style error/risk calibration for native H3/Wan DBCache.

Risk accumulates adjacent middle-input motion, normalized by the exact middle
output and scaled by anchor motion gain. Counterfactual output error is measured
before the exact cache commit. Device state is consumed at the existing Python similarity synchronization point.
"""

import math

import torch
import torch.distributed as dist


class AdaptiveRDT:
    min_rdt = 0.01
    max_rdt = 1.0
    max_decrease = 0.50
    max_increase = 0.10
    # Numerical protection only; no narrow band tying RDT to its start value.
    min_ratio = 1e-4
    max_ratio = 1e4
    epsilon = 1e-12

    def __init__(
        self,
        initial_rdt: float,
        decay: float,
        *,
        shadow: bool = False,
        probe_interval: int = 16,
    ):
        if not self.min_rdt <= initial_rdt <= self.max_rdt:
            raise ValueError("Adaptive RDT requires initial RDT in [0.01, 1.0]")
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("SGLANG_CACHE_DIT_RDT_EMA must be in [0, 1)")
        if probe_interval < 2:
            raise ValueError("SGLANG_CACHE_DIT_RDT_PROBE_INTERVAL must be >= 2")
        self.context_name = ""
        self.branch = "main"
        self.initial_rdt = initial_rdt
        self.decay = decay
        self.shadow = shadow
        self.probe_interval = probe_interval
        self.reset()

    def reset(self):
        self.rdt = None
        self.desired_rdt = None
        self.ema = None
        self.risk = None
        self.risk_valid = None
        self.risk_samples = 0
        self.error = None
        self.ratio = None
        self.step_risk = None
        self.has_history = False
        self.last_anchor_step = -1
        self.last_feedback_step = 0
        self.compared = False
        self.budget_triggered = False
        self.feedback_eligible = False
        self.reason = "bootstrap"
        self.records = []
        self.previous_feature = None
        self.current_feature = None
        self.anchor_feature = None
        self.anchor_output = None
        self.gain = None
        self.output_scale = None
        self.calibrated = None
        self.feedback_attempts = 0

    def initialize(self, reference):
        if self.rdt is None:
            self.rdt = reference.new_tensor(self.initial_rdt, dtype=torch.float32)
            self.desired_rdt = self.rdt
            # E/B=1 means the proxy is calibrated: scale=1, RDT=base RDT.
            self.ema = torch.ones_like(self.rdt)
            self.gain = torch.ones_like(self.rdt)
            self.output_scale = torch.ones_like(self.rdt)
            self.calibrated = self.rdt.new_tensor(False, dtype=torch.bool)
            self.clear_risk()

    def clear_risk(self):
        self.risk = torch.zeros_like(self.rdt)
        self.risk_valid = self.rdt.new_tensor(True, dtype=torch.bool)
        self.risk_samples = 0

    def add_risk(self, difference):
        # Similarity already reduced this feature difference over TP/SP.
        difference = difference.detach().float()
        valid = torch.isfinite(difference) & (difference >= 0)
        self.risk_valid = self.risk_valid & valid
        self.risk = self.risk + torch.where(valid, difference, 0.0)
        self.risk_valid = self.risk_valid & torch.isfinite(self.risk)
        self.risk_samples += 1

    @staticmethod
    def reduce_means(values, group):
        stats = torch.stack(values)
        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.AVG, group=group)
        return stats

    def add_feature_risk(self, group):
        """AlignCache adjacent motion of the actual cached-segment input."""
        feature = self.current_feature
        previous = self.previous_feature
        if previous is None or previous.shape != feature.shape:
            delta = feature.new_tensor(float("inf"), dtype=torch.float32)
        else:
            delta = (feature.float() - previous.float()).abs().mean()
        delta = self.reduce_means([delta], group)[0]
        self.add_risk(self.gain * delta / self.output_scale.clamp_min(self.epsilon))
        self.previous_feature = feature

    def update_motion(self, actual, group):
        # Fixed-size FP32 collectives, including invalid/missing local history.
        feature = self.current_feature
        invalid = self.rdt.new_tensor(float("nan"))
        scale = actual.detach().float().abs().mean() if actual is not None else invalid
        pair = (
            actual is not None and self.anchor_output is not None
            and self.anchor_output.shape == actual.shape
            and self.anchor_feature.shape == feature.shape
        )
        output_motion = (
            (actual.detach().float() - self.anchor_output.float()).abs().mean()
            if pair else invalid
        )
        input_motion = (
            (feature.float() - self.anchor_feature.float()).abs().mean()
            if pair else invalid
        )
        stats = self.reduce_means([scale, output_motion, input_motion], group)
        self.output_scale = torch.where(
            torch.isfinite(stats[0]) & (stats[0] > self.epsilon),
            stats[0], self.output_scale,
        )
        candidate = stats[1] / stats[2].clamp_min(self.epsilon)
        # Degenerate anchors do not prove that a TS forecast has zero risk.
        valid = (stats[2] > self.epsilon) & torch.isfinite(candidate) & (candidate > 0)
        self.gain = torch.where(valid, candidate, self.gain)
        self.anchor_feature = feature
        self.previous_feature = feature
        self.anchor_output = actual.detach().clone() if actual is not None else None

    def update(self, error):
        """Calibrate E/B, leaving EMA and RDT unchanged for invalid windows."""
        valid = (
            self.risk_valid
            & (self.risk > self.epsilon)
            & torch.isfinite(error)
            & (error >= 0)
        )
        ratio = (error / (self.risk + self.epsilon)).clamp(
            self.min_ratio, self.max_ratio
        )
        ratio = torch.nan_to_num(ratio, nan=self.max_ratio)
        candidate = (self.decay * self.ema + (1 - self.decay) * ratio).clamp(
            self.min_ratio, self.max_ratio
        )
        # Do not dilute the first observation with an arbitrary neutral prior.
        candidate = torch.where(self.calibrated, candidate, ratio)
        self.ema = torch.where(valid, candidate, self.ema)
        self.calibrated = self.calibrated | valid
        desired = (self.initial_rdt / self.ema).clamp(self.min_rdt, self.max_rdt)
        self.desired_rdt = desired
        limited = desired.clamp(
            self.rdt * (1 - self.max_decrease),
            self.rdt * (1 + self.max_increase),
        ).clamp(self.min_rdt, self.max_rdt)
        self.rdt = torch.where(valid, limited, self.rdt)
        self.error = torch.where(valid, error, float("nan"))
        self.ratio = torch.where(valid, ratio, float("nan"))

    def observe(self, prediction, actual, group):
        # E is relative MSE of the actual cached-segment output.
        if actual is None:
            moments = self.rdt.new_tensor([float("nan"), 1.0])
        elif prediction is None:
            # Participate even if one shard has invalid prediction history.
            moments = self.rdt.new_tensor([float("nan"), 1.0])
        else:
            actual = actual.detach().float()
            delta = prediction.detach().float() - actual
            moments = torch.stack((delta.square().sum(), actual.square().sum()))
        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(moments, op=dist.ReduceOp.SUM, group=group)
        self.update(moments[0] / moments[1].clamp_min(self.epsilon))
        self.feedback_attempts += 1

    def should_probe(self, step):
        return (
            not self.shadow
            and self.has_history
            and self.risk_samples > 0
            and (
                (self.feedback_attempts == 0 and self.risk_samples >= 2)
                or step - self.last_feedback_step >= self.probe_interval
            )
        )

    def anchor(self, step):
        self.step_risk = self.risk
        self.clear_risk()
        self.has_history = True
        self.last_anchor_step = step

    def record(self, step, exact, probe):
        missing = self.rdt.new_tensor(float("nan"))
        values = torch.stack((
            self.step_risk if exact else self.risk,
            self.error if self.error is not None else missing,
            self.ratio if self.ratio is not None else missing,
            self.ema, self.rdt, self.desired_rdt,
        ))
        self.records.append((
            step, exact, probe, self.reason, self.feedback_eligible, values
        ))

    def flush(self, logger):
        if not self.records:
            return
        values = torch.stack([record[5] for record in self.records]).cpu().tolist()
        for record, scalars in zip(self.records, values):
            step, exact, probe, reason, feedback, _ = record
            risk, error, ratio, ema, rdt, desired = scalars
            logger.info(
                "Adaptive RDT step=%d mode=%s shadow=%s probe=%s "
                "reason=%s feedback_eligible=%s metric=middle_output_relative_mse "
                "risk=%g error=%g ratio=%g ratio_ema=%g rdt=%g rdt_desired=%g "
                "context=%s branch=%s",
                step, "exact" if exact else "reuse", self.shadow, probe, reason, feedback,
                risk, error, ratio, ema, rdt, desired, self.context_name, self.branch,
            )
        self.records.clear()


def enable_adaptive_rdt(
    transformer, initial_rdt, logger, *, feature_difference=None
):
    """Route on accumulated motion risk; calibrate before TS cache commits."""
    from sglang.multimodal_gen import envs

    shadow = envs.SGLANG_CACHE_DIT_ADAPTIVE_RDT_SHADOW
    if not envs.SGLANG_CACHE_DIT_ADAPTIVE_RDT and not shadow:
        return False

    controller = AdaptiveRDT(
        initial_rdt,
        envs.SGLANG_CACHE_DIT_RDT_EMA,
        shadow=shadow,
        probe_interval=envs.SGLANG_CACHE_DIT_RDT_PROBE_INTERVAL,
    )
    # cache-dit swaps native blocks only inside forward. Its persistent manager
    # and the native blocks are both accessible here, before compilation.
    manager = transformer._context_manager
    context_names = transformer._context_names
    if len(context_names) != 1:
        raise ValueError("Adaptive RDT requires one cache context per transformer")
    manager.set_context(context_names[0])
    config = manager.current_context.cache_config
    if (
        getattr(config, "steps_computation_mask", None) is not None
        and config.steps_computation_policy == "static"
    ):
        raise ValueError("Adaptive RDT requires dynamic DBCache feature checks")
    if manager.is_calibrator_enabled():
        from cache_dit.caching.cache_contexts.calibrators.taylorseer import (
            TaylorSeerCalibrator,
        )

        calibrator, _ = manager.get_calibrator()
        if not isinstance(calibrator, TaylorSeerCalibrator):
            raise ValueError("Adaptive RDT supports only no calibrator or TaylorSeer")
    if not manager.is_cache_residual():
        raise ValueError("Adaptive RDT requires residual caching")
    separate_cfg = manager.enable_separate_cfg()
    controllers = [controller]
    if separate_cfg:
        controllers.append(AdaptiveRDT(
            initial_rdt, envs.SGLANG_CACHE_DIT_RDT_EMA,
            shadow=shadow, probe_interval=envs.SGLANG_CACHE_DIT_RDT_PROBE_INTERVAL,
        ))
    for index, state in enumerate(controllers):
        state.context_name = context_names[0]
        state.branch = "cfg" if index else "main"

    blocks = transformer.blocks
    first = manager.Fn_compute_blocks()
    stop = len(blocks) - manager.Bn_compute_blocks()
    if first < 1:
        raise ValueError("Adaptive RDT input-risk capture requires FN >= 1")
    if not first < stop <= len(blocks):
        raise ValueError("Adaptive RDT requires a nonempty cached middle segment")

    transformer._sglang_adaptive_rdt_controllers = tuple(controllers)
    manager._sglang_adaptive_rdt = controller
    original_can_cache = manager.can_cache
    original_set_buffer = manager.set_Bn_buffer
    active_context = None
    prediction = predicted_output = actual_output = None
    routing_input = None
    probe = False

    def group():
        return (
            getattr(manager, "_sglang_tp_sp_group", None)
            or getattr(manager, "_sglang_sp_group", None)
            or getattr(manager, "_sglang_tp_group", None)
        )

    def owns_context():
        # Wan 2.2 experts share a manager. Delegate other expert contexts to
        # the original hook chain; never mix their histories or thresholds.
        return getattr(manager.current_context, "name", context_names[0]) == context_names[0]

    def record(exact):
        controller.record(manager.get_current_step(), exact, probe)
        last_step = manager.current_context.cache_config.num_inference_steps - 1
        if manager.get_current_step() == last_step:
            controller.flush(logger)

    @torch.compiler.disable
    def capture_routing_input(_module, _args, output):
        nonlocal routing_input
        # The final Fn block runs on exact AND reuse steps, before can_cache.
        # Clone now: later blocks may update the activation in place.
        routing_input = output.detach().clone()

    @torch.compiler.disable
    def capture_input(_module, args):
        nonlocal prediction, predicted_output
        if controller.feedback_eligible and prediction is not None:
            hidden_states = args[0]
            if prediction.shape == hidden_states.shape:
                # Exactly the hypothetical residual reuse performed by
                # Pattern_3.apply_cache, on the *current* middle input.
                predicted_output = prediction + hidden_states
            prediction = None

    @torch.compiler.disable
    def capture_output(_module, _args, output):
        nonlocal actual_output
        actual_output = output.detach()

    hooks = []
    try:
        hooks.append(blocks[first - 1].register_forward_hook(capture_routing_input))
        hooks.append(blocks[first].register_forward_pre_hook(capture_input))
        hooks.append(blocks[stop - 1].register_forward_hook(capture_output))
    except Exception:
        for hook in hooks:
            hook.remove()
        raise
    transformer._sglang_adaptive_rdt_hooks = tuple(hooks)

    @torch.compiler.disable
    def can_cache(states_tensor, parallelized=False, threshold=None, prefix="Fn"):
        nonlocal active_context, prediction, predicted_output, actual_output, probe, routing_input
        nonlocal controller
        if not owns_context():
            return original_can_cache(
                states_tensor, parallelized=parallelized, threshold=threshold, prefix=prefix
            )
        context = manager.current_context
        if context is not active_context:
            for state in controllers:
                state.reset()
            context.cache_config.residual_diff_threshold = initial_rdt
            active_context = context
        cfg_branch = separate_cfg and manager.is_separate_cfg_step()
        controller = controllers[int(cfg_branch)]
        manager._sglang_adaptive_rdt = controller
        controller.initialize(states_tensor)
        if routing_input is None:
            raise RuntimeError("Adaptive RDT did not capture the current middle input")
        controller.current_feature = routing_input
        routing_input = None
        controller.error = controller.ratio = controller.step_risk = None
        controller.compared = controller.budget_triggered = False
        controller.feedback_eligible = False
        prediction = predicted_output = actual_output = None
        probe = False
        warmup = manager.is_in_warmup()
        mask_due = (
            manager.is_steps_computation_mask_enabled()
            and manager.is_in_full_compute_steps()
        )
        max_cached = manager.get_max_cached_steps()
        cached_steps = manager.get_cfg_cached_steps() if cfg_branch else manager.get_cached_steps()
        total_due = max_cached >= 0 and len(cached_steps) >= max_cached
        mc = manager.get_max_continuous_cached_steps()
        continuous = (
            manager.get_cfg_continuous_cached_steps() if cfg_branch
            else manager.get_continuous_cached_steps()
        )
        mc_due = mc >= 0 and continuous >= mc
        reuse = original_can_cache(
            states_tensor, parallelized=parallelized, threshold=threshold, prefix=prefix
        )
        if warmup:
            controller.reason = "warmup"
        elif mask_due:
            controller.reason = "mask"
        elif total_due:
            controller.reason = "max_cached"
        elif reuse:
            controller.reason = "reuse"
        elif controller.compared:
            controller.reason = "budget" if controller.budget_triggered else "threshold"
        elif mc_due:
            controller.reason = "mc"
        else:
            controller.reason = "guard" if controller.has_history else "bootstrap"

        # Include early-exact endpoints in the same accumulated motion risk.
        if not controller.compared and not warmup and controller.has_history:
            controller.add_feature_risk(group())

        # Count time since feedback, not since any anchor: frequent MC exacts
        # must not prevent deliberately scheduled calibration probes forever.
        if (
            not warmup
            and controller.reason in ("reuse", "mc")
            and controller.should_probe(manager.get_current_step())
        ):
            probe = True
            reuse = False
            controller.reason = "probe"
        controller.feedback_eligible = (
            not reuse
            and not warmup
            and controller.reason in ("budget", "probe", "mc")
            and controller.has_history
            and controller.risk_samples > 0
        )
        if reuse:
            record(exact=False)
        elif controller.feedback_eligible:
            cache_prefix = prefix.rsplit("_Fn_", 1)[0]
            prediction = manager.get_Bn_buffer(prefix=f"{cache_prefix}_Bn_residual")
        return reuse

    @torch.compiler.disable
    def set_buffer(buffer, prefix="Bn"):
        nonlocal prediction, predicted_output, actual_output
        if not owns_context():
            return original_set_buffer(buffer, prefix=prefix)
        if controller.feedback_eligible:
            measured = actual_output
            if measured is None:
                # Invalid capture participates in the same reduction on all
                # ranks and cannot update EMA. Never substitute residual MSE.
                predicted_output = None
                controller.reason += "_output_unavailable"
            if predicted_output is not None and predicted_output.shape != measured.shape:
                predicted_output = None
            controller.observe(predicted_output, measured, group())
            controller.last_feedback_step = manager.get_current_step()
        # Commit the true residual/TaylorSeer derivatives only after measuring
        # the counterfactual middle output against the real output.
        controller.update_motion(actual_output, group())
        result = original_set_buffer(buffer, prefix=prefix)
        controller.anchor(manager.get_current_step())
        record(exact=True)
        prediction = predicted_output = actual_output = None
        return result

    manager.can_cache = can_cache
    manager.set_Bn_buffer = set_buffer
    return True
