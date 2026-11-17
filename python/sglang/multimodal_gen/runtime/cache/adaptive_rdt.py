# SPDX-License-Identifier: Apache-2.0
"""Starting point for a per-generation adaptive Cache-DiT threshold.

This controller is not connected to Cache-DiT yet.
"""


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
        self.last_B = 0
        self.current_error = None
        self.ratio = None
        self.ema_lambda = 0.9
        self.ema_ratio = None

    def capture_input(self, module, args, output):

        self.current_x = output.detach().clone()
        if self.prev_x is not None:
            self.delta_x = (self.prev_x.float() - self.current_x.float()).abs().mean()

            if self.current_g is not None and self.current_scale is not None:
                self.B += (self.current_g * self.delta_x) / (self.current_scale + self.eps)
        self.prev_x = self.current_x.clone()

    def capture_output(self, module, args, output):
        
        self.current_y = output.detach().clone()
        if (self.anchor_x is not None and self.anchor_y is not None):
            dy = (self.current_y.float() - self.anchor_y.float()).abs().mean()
            dx = (self.current_x.float() - self.anchor_x.float()).abs().mean()
            self.current_g = dy / (dx + self.eps)

        self.current_scale = self.current_y.float().abs().mean()
        self.anchor_y = self.current_y
        self.anchor_x = self.current_x
        self.last_B = self.B
        print(f"[AdaptiveRDT] B={self.last_B.item() if hasattr(self.last_B, 'item') else self.last_B:.6f}")
        self.B = 0

    def observe_prediction(self, r_hat, buffer):
        if r_hat is None or self.last_B <= self.eps:
            return
    
        error = (r_hat.float() - buffer.float()).pow(2).sum()
        norm_y = self.current_y.float().pow(2).sum()
        self.current_error = (error / (norm_y + self.eps))

        self.ratio = self.current_error / (self.last_B + self.eps)

        if self.ema_ratio is None:
            self.ema_ratio = self.ratio
        else:
            self.ema_ratio = self.ema_lambda * self.ema_ratio + (1 - self.ema_lambda) * self.ratio
        
        self.rdt = (
            self.initial_rdt / (self.ema_ratio + self.eps)
        ).clamp(min=0.01, max=1.0)

        print(
            f"[AdaptiveRDT] "
            f"E={self.current_error.item():.6f} "
            f"ratio={self.ratio.item():.6f} "
            f"ema={self.ema_ratio.item():.6f}"
        )
