# SPDX-License-Identifier: Apache-2.0
"""Collect per-rank Cache-DiT statistics for MiniMax H3."""

import torch
import torch.distributed as dist


class AdaptiveRDT:
    def __init__(self, initial_rdt: float):
        self.initial_rdt = initial_rdt
        self.rdt = initial_rdt
        self.prev_x = None
        self.current_x = None
        self.delta_x = None
        self.current_y = None
        self.anchor_x = None
        self.anchor_y = None
        self.current_g = None
        self.eps = 1e-9
        self.current_scale = None
        self.B = 0
        self.last_B = None
        self.current_error = None
        self.ratio = None
        self.ema_lambda = 0.9
        self.ema_ratio = None
        self.records = []
        self.decisions = []
        self.tp_sp_group = None

    def capture_input(self, module, args, output):

        self.current_x = output.detach().clone()
        if self.prev_x is not None:
            self.delta_x = self.global_mean_abs(self.prev_x.float() - self.current_x.float())

            if self.current_g is not None and self.current_scale is not None:
                self.B += (self.current_g * self.delta_x) / (self.current_scale + self.eps)
        self.prev_x = self.current_x.clone()

    def capture_output(self, module, args, output):
        
        self.current_y = output.detach().clone()
        if (self.anchor_x is not None and self.anchor_y is not None):
            dy = self.global_mean_abs(self.current_y.float() - self.anchor_y.float())
            dx = self.global_mean_abs(self.current_x.float() - self.anchor_x.float())
            self.current_g = dy / (dx + self.eps)

        self.current_scale = self.global_mean_abs(self.current_y.float())
        self.anchor_y = self.current_y
        self.anchor_x = self.current_x
        self.last_B = self.B
        self.B = 0

    def global_mean_abs(self, tensor):
        sum_elem = tensor.abs().sum()
        num_elem = torch.tensor(tensor.numel(),device=tensor.device)
        if self.tp_sp_group is not None:
            dist.all_reduce(sum_elem, op=dist.ReduceOp.SUM, group=self.tp_sp_group)
            dist.all_reduce(num_elem, op=dist.ReduceOp.SUM, group=self.tp_sp_group)

        return sum_elem / num_elem

    def global_sum(self, tensor):
        if self.tp_sp_group is not None:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.tp_sp_group)
        return tensor
    
    def observe_prediction(self, r_hat, buffer, step):
        if r_hat is None or not torch.is_tensor(self.last_B):
            return
        assert r_hat.shape == buffer.shape == self.current_y.shape

        error = self.global_sum((r_hat.float() - buffer.float()).pow(2).sum())
        norm_y = self.global_sum(self.current_y.float().pow(2).sum())
        self.current_error = (error / (norm_y + self.eps))

        self.ratio = torch.where(
            self.last_B > self.eps,
            self.current_error / (self.last_B + self.eps),
            torch.full_like(self.last_B, float("nan")),
        )
        if self.ema_ratio is None:
            self.ema_ratio = (
                self.ema_lambda * torch.ones_like(self.ratio)
                + (1 - self.ema_lambda) * self.ratio
            )
        else:
            average = self.ema_lambda * self.ema_ratio + (1 - self.ema_lambda) * self.ratio
            average = torch.where(torch.isfinite(self.ema_ratio), average, self.ratio)
            self.ema_ratio = torch.where(torch.isfinite(self.ratio), average, self.ema_ratio)

        desired_rdt = (self.initial_rdt / (self.ema_ratio + self.eps)).clamp(min=0.01, max=1.0)
        self.rdt = torch.where(
            torch.isfinite(self.ema_ratio), desired_rdt,
            torch.full_like(self.ema_ratio, self.initial_rdt),
        )
        self.records.append((step, self.last_B, self.current_error, self.ratio, self.ema_ratio, self.rdt))

    def flush(self, logger):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.decisions:
            values = torch.stack([torch.stack(row[3:]) for row in self.decisions]).cpu().tolist()
            for (step, mode, reason, *_), (B, rdt) in zip(self.decisions, values):
                print(
                    f"[AdaptiveRDT decision] rank={rank} step={step} mode={mode} "
                    f"reason={reason} B={B:g} rdt_used={rdt:g}",
                    flush=True,
                )
            self.decisions.clear()
        if not self.records:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        values = torch.stack([torch.stack(row[1:]) for row in self.records]).cpu().tolist()
        for (step, *_), (B, E, ratio, ema, rdt) in zip(self.records, values):
            print(
                f"[AdaptiveRDT] rank={rank} step={step} "
                f"B={B:g} E={E:g} ratio={ratio:g} ema={ema:g} rdt={rdt:g}",
                flush=True,
            )
        self.records.clear()

    def reset(self):
        self.prev_x = None
        self.current_x = None
        self.delta_x = None
        self.current_y = None
        self.anchor_x = None
        self.anchor_y = None
        self.current_g = None
        self.current_scale = None
        self.B = 0
        self.last_B = None
        self.current_error = None
        self.ratio = None
        self.ema_ratio = None
        self.rdt = self.initial_rdt
        self.records = []
        self.decisions = []
