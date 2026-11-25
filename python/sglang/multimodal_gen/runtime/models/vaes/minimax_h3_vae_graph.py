# SPDX-License-Identifier: Apache-2.0
"""Opt-in H3 tile decode graphs; tiling, collectives and output processing stay eager."""

from contextlib import contextmanager, nullcontext

from torch import nn

from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    BaseBreakableCudaGraphRunner,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class _TileDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, z):
        return self.decoder(self.post_quant_conv(z))


class _TileGraphRunner(BaseBreakableCudaGraphRunner):
    capture_enabled = False

    def _should_capture_on_call(self, key):
        return self.capture_enabled

    def _tp_graph_capture(self):
        # Each rank decodes independent full tiles. No TP collectives here.
        return nullcontext()

    def _capture(self, kwargs, key):
        entry = super()._capture(kwargs, key)
        # The decoder has a single-shape RoPE cache. A later tile shape replaces
        # it, while earlier graphs still reference its storage.
        entry.offload_keepalive.append(self.transformer.decoder._rotary_pos_emb_cache)
        return entry


def _weight_signature(module):
    # A graph references storage, not the current Python parameter object.
    return tuple(
        (name, tensor.data_ptr(), tuple(tensor.shape), tensor.dtype, tensor.device)
        for name, tensor in (
            *module.named_parameters(),
            *module.named_buffers(),
        )
    )


@contextmanager
def use_minimax_h3_vae_npu_graph(vae, *, enabled, is_warmup, residency):
    if not enabled:
        yield
        return
    if not current_platform.is_npu() or type(vae) is not MiniMaxH3VideoVAE:
        raise ValueError("VAE NPU graphs require the native MiniMax H3 video VAE")
    if residency != "resident":
        raise ValueError("VAE NPU graphs require --component-residency video_vae=resident")
    if vae.training:
        raise ValueError("VAE NPU graphs require eval mode")

    # Set up after residency acquisition and decode-dtype preparation. Keeping
    # the runner outside nn.Module's child tree avoids duplicate state_dict keys.
    adapter = _TileDecoder(vae)
    signature = _weight_signature(adapter)
    state = vae.__dict__.get("_npu_tile_graph_state")
    if state is not None and state[0] != signature:
        state[1].reset(disabled_reason="VAE weight storage changed")
        state = None
    if state is None:
        device = next(adapter.parameters()).device
        runner = _TileGraphRunner(adapter, device)
        # A deliberately small bound: each graph owns full tile activations.
        runner.max_entries = 4
        state = (signature, runner)
        vae.__dict__["_npu_tile_graph_state"] = state
        logger.info("MiniMax H3 VAE NPU graphs enabled; attention stays eager")
    runner = state[1]
    runner.capture_enabled = bool(is_warmup)
    # All native tile/window scheduling keeps
    # calling the same entry point, but only the tile's neural network is graphed.
    previous_decode = vae.__dict__.get("decode")
    vae.decode = lambda z: runner(z=z)
    try:
        yield
    finally:
        if previous_decode is None:
            del vae.decode
        else:
            vae.decode = previous_decode
        runner.capture_enabled = False
        if is_warmup:
            logger.info(
                "MiniMax H3 VAE NPU graphs: %d captured tile shapes", len(runner.entries)
            )
            if not runner.entries:
                logger.warning("No VAE tile graphs captured; decode will run eager")
