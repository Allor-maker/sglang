# SPDX-License-Identifier: Apache-2.0
"""Simplified AlignCache-style error/risk calibration for native H3 DBCache.

Risk uses the feature differences actually evaluated by DBCache. Error is
measured on counterfactual residuals before the exact cache commit. Device
state is consumed at the existing Python similarity synchronization point.
"""

import math

import torch
import torch.distributed as dist


class AdaptiveRDT:
    min_rdt = 0.01
    max_rdt = 0.99
    max_decrease = 0.50
    max_increase = 0.10
    min_ratio = 1e-4
    max_ratio = 1e4
    epsilon = 1e-12

    def __init__(
        self,
        initial_rdt: float,
        target: float,
        decay: float,
        *,
        shadow: bool = False,
        probe_interval: int = 16,
    ):
        if not self.min_rdt <= initial_rdt <= self.max_rdt:
            raise ValueError("Adaptive RDT requires initial RDT in [0.01, 0.99]")
        if not math.isfinite(target) or target <= 0:
            raise ValueError("SGLANG_CACHE_DIT_TARGET_ERROR must be finite and > 0")
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("SGLANG_CACHE_DIT_RDT_EMA must be in [0, 1)")
        if probe_interval < 2:
            raise ValueError("SGLANG_CACHE_DIT_RDT_PROBE_INTERVAL must be >= 2")
        self.initial_rdt = initial_rdt
        self.target = target
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
        self.records = []

    def initialize(self, reference):
        if self.rdt is None:
            self.rdt = reference.new_tensor(self.initial_rdt, dtype=torch.float32)
            self.desired_rdt = self.rdt
            # Neutral prior: target / EMA starts at the configured base RDT.
            self.ema = self.rdt.new_tensor(self.target / self.initial_rdt)
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
        self.risk_samples += 1

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
        self.ema = torch.where(valid, candidate, self.ema)
        desired = (self.target / self.ema).clamp(self.min_rdt, self.max_rdt)
        self.desired_rdt = desired
        limited = desired.clamp(
            self.rdt * (1 - self.max_decrease),
            self.rdt * (1 + self.max_increase),
        ).clamp(self.min_rdt, self.max_rdt)
        self.rdt = torch.where(valid, limited, self.rdt)
        self.error = torch.where(valid, error, float("nan"))
        self.ratio = torch.where(valid, ratio, float("nan"))

    def observe(self, prediction, actual, group):
        # E is relative MSE of the residual, not of the hidden-state output.
        actual = actual.detach().float()
        if prediction is None:
            # Participate even if one shard has invalid prediction history.
            moments = actual.new_tensor([float("nan"), 1.0])
        else:
            delta = prediction.detach().float() - actual
            moments = torch.stack((delta.square().sum(), actual.square().sum()))
        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(moments, op=dist.ReduceOp.SUM, group=group)
        self.update(moments[0] / moments[1].clamp_min(self.epsilon))

    def should_probe(self, step):
        return (
            not self.shadow
            and self.has_history
            and self.risk_samples > 0
            and step - self.last_anchor_step >= self.probe_interval
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
        self.records.append((step, exact, probe, values))

    def flush(self, logger):
        if not self.records:
            return
        values = torch.stack([record[3] for record in self.records]).cpu().tolist()
        for (step, exact, probe, _), (risk, error, ratio, ema, rdt, desired) in zip(
            self.records, values
        ):
            logger.info(
                "Adaptive RDT step=%d mode=%s shadow=%s probe=%s "
                "risk=%g error=%g ratio=%g ratio_ema=%g rdt=%g rdt_desired=%g",
                step, "exact" if exact else "reuse", self.shadow, probe,
                risk, error, ratio, ema, rdt, desired,
            )
        self.records.clear()


def enable_adaptive_rdt(transformer, initial_rdt, logger):
    """Instrument the mounted H3 wrapper without changing cache commits or MC."""
    from sglang.multimodal_gen import envs

    shadow = envs.SGLANG_CACHE_DIT_ADAPTIVE_RDT_SHADOW
    if not envs.SGLANG_CACHE_DIT_ADAPTIVE_RDT and not shadow:
        return False

    controller = AdaptiveRDT(
        initial_rdt,
        envs.SGLANG_CACHE_DIT_TARGET_ERROR,
        envs.SGLANG_CACHE_DIT_RDT_EMA,
        shadow=shadow,
        probe_interval=envs.SGLANG_CACHE_DIT_RDT_PROBE_INTERVAL,
    )
    # cache-dit 1.5.1 swaps model.blocks only inside transformer.forward.
    # Its persistent manager is available at mount time; instrument that
    # instance instead of assuming blocks[0] is already a CachedBlocks wrapper.
    manager = transformer._context_manager
    context_names = transformer._context_names
    if len(context_names) != 1:
        raise ValueError("Adaptive RDT requires one H3 cache context")
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
    if manager.enable_separate_cfg() or not manager.is_cache_residual():
        raise ValueError("Adaptive RDT requires H3's single-stream residual cache")

    manager._sglang_adaptive_rdt = controller
    original_can_cache = manager.can_cache
    original_set_buffer = manager.set_Bn_buffer
    active_context = None
    prediction = None
    probe = False

    def record(exact):
        controller.record(manager.get_current_step(), exact, probe)
        last_step = manager.current_context.cache_config.num_inference_steps - 1
        if manager.get_current_step() == last_step:
            controller.flush(logger)

    @torch.compiler.disable
    def can_cache(states_tensor, parallelized=False, threshold=None, prefix="Fn"):
        nonlocal active_context, prediction, probe
        context = manager.current_context
        if context is not active_context:
            controller.reset()
            context.cache_config.residual_diff_threshold = initial_rdt
            active_context = context
        controller.initialize(states_tensor)
        controller.error = None
        controller.ratio = None
        controller.step_risk = None
        prediction = None
        probe = False
        reuse = original_can_cache(
            states_tensor, parallelized=parallelized, threshold=threshold, prefix=prefix
        )
        if (
            reuse
            and not manager.is_in_warmup()
            and controller.should_probe(manager.get_current_step())
        ):
            probe = True
            reuse = False
        if reuse:
            record(exact=False)
        elif (
            controller.has_history
            and controller.risk_samples > 0
            and not manager.is_in_warmup()
        ):
            # Read the same residual stream that Pattern_3.apply_cache would
            # use, before the middle forward and before its exact commit.
            cache_prefix = prefix.rsplit("_Fn_", 1)[0]
            prediction = manager.get_Bn_buffer(prefix=f"{cache_prefix}_Bn_residual")
        return reuse

    @torch.compiler.disable
    def set_buffer(buffer, prefix="Bn"):
        if (
            controller.has_history
            and controller.risk_samples > 0
            and not manager.is_in_warmup()
        ):
            group = (
                getattr(manager, "_sglang_tp_sp_group", None)
                or getattr(manager, "_sglang_sp_group", None)
                or getattr(manager, "_sglang_tp_group", None)
            )
            valid_prediction = prediction
            if prediction is not None and prediction.shape != buffer.shape:
                valid_prediction = None
            # Pattern_3 supplies its actual middle residual here. Calibrate
            # before the original setter updates TaylorSeer derivatives/cache.
            controller.observe(valid_prediction, buffer, group)
        result = original_set_buffer(buffer, prefix=prefix)
        controller.anchor(manager.get_current_step())
        record(exact=True)
        return result

    manager.can_cache = can_cache
    manager.set_Bn_buffer = set_buffer
    return True
