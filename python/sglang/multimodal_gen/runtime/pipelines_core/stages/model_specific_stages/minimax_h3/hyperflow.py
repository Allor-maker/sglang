# SPDX-License-Identifier: Apache-2.0
"""HyperFlow's file contract and fixed H3 sampling schedule.

Reference: https://github.com/Video-Rebirth/hyperflow (Apache-2.0).
The adapter needs interval conditioning in addition to ordinary PEFT weights.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

_DEFAULT_SIGMAS = (
    1.0,
    0.931506,
    0.839236,
    0.703462,
    0.5,
    0.296538,
    0.160764,
    0.068494,
    0.0,
)
_KEY = re.compile(r"^transformer\.(.+)\.lora_([AB])\.weight$")


@dataclass(frozen=True)
class MiniMaxH3HyperFlow:
    version: str
    gate: float
    sigmas: tuple[float, ...]
    video_shift: float
    audio_shift: float
    rank: int
    alpha: int
    keys: tuple[str, ...]

    @classmethod
    def from_metadata(
        cls, metadata: Mapping[str, str], keys: tuple[str, ...]
    ) -> MiniMaxH3HyperFlow:
        if metadata.get("hyperflow", "").lower() != "true":
            raise ValueError("Expected hyperflow=true in the adapter header")
        if metadata.get("base_model") != "MiniMaxAI/MiniMax-H3":
            raise ValueError("HyperFlow requires the full-size MiniMaxAI/MiniMax-H3")
        version = metadata.get("hyperflow_version", "")
        if not version:
            raise ValueError("HyperFlow header is missing hyperflow_version")
        gate = float(metadata["hyperflow_gate"])
        raw = json.loads(metadata.get("hyperflow_sigmas", json.dumps(_DEFAULT_SIGMAS)))
        if not isinstance(raw, list) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw
        ):
            raise ValueError("hyperflow_sigmas must be a numeric JSON array")
        sigmas = tuple(float(v) for v in raw)
        if (
            len(sigmas) != 9
            or sigmas[0] != 1.0
            or sigmas[-1] != 0.0
            or not all(math.isfinite(v) for v in sigmas)
            or not all(a > b for a, b in zip(sigmas, sigmas[1:]))
        ):
            raise ValueError("HyperFlow requires nine decreasing sigmas from 1 to 0")
        video_shift = float(metadata.get("hyperflow_video_shift", 12.0))
        audio_shift = float(metadata.get("hyperflow_audio_shift", 3.0))
        if not math.isfinite(gate) or any(
            not math.isfinite(v) or v <= 0 for v in (video_shift, audio_shift)
        ):
            raise ValueError("HyperFlow gate/shifts must be finite, shifts positive")
        rank = float(metadata.get("lora_rank", 256))
        alpha = float(metadata["lora_alpha"])
        if any(
            not math.isfinite(v) or v <= 0 or not v.is_integer() for v in (rank, alpha)
        ):
            raise ValueError("HyperFlow rank/alpha must be positive integers")
        modules: dict[str, set[str]] = {}
        for key in keys:
            match = _KEY.fullmatch(key)
            if match is None:
                raise ValueError(f"Unexpected HyperFlow weight: {key}")
            modules.setdefault(match[1], set()).add(match[2])
        if not modules or any(matrices != {"A", "B"} for matrices in modules.values()):
            raise ValueError("HyperFlow requires complete LoRA A/B pairs")
        for prefix in ("time_embedder", "endpoint_time_embedder"):
            for linear in ("linear_1", "linear_2"):
                if f"{prefix}.{linear}" not in modules:
                    raise ValueError(f"HyperFlow is missing {prefix}.{linear}")
        return cls(
            version, gate, sigmas, video_shift, audio_shift, int(rank), int(alpha), keys
        )

    def shifted_sigmas(
        self, num_steps: int, *, warmup: bool = False
    ) -> dict[str, list[float]]:
        if (not warmup and num_steps != 8) or not 1 <= num_steps <= 8:
            raise ValueError("HyperFlow requires num_inference_steps=8")
        base = torch.tensor(self.sigmas, dtype=torch.float32)
        result = {}
        for modality, shift in (
            ("video", self.video_shift),
            ("audio", self.audio_shift),
        ):
            shifted = shift * base / (1.0 + (shift - 1.0) * base)
            result[modality] = shifted[: num_steps + 1].tolist()
        return result


def read_hyperflow(
    path: str | None, weight_name: str | None = None
) -> tuple[str, MiniMaxH3HyperFlow] | None:
    """Detect a local adapter before the pipeline's ordinary LoRA initialization.

    Remote ordinary LoRAs retain their existing download path. HyperFlow startup
    uses a local file/directory so its architecture is known before LoRA wrapping.
    """
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_dir():
        if weight_name is not None:
            candidate = candidate / weight_name
        elif (candidate / "hyperflow.json").is_file():
            manifest = json.loads((candidate / "hyperflow.json").read_text())
            candidate = candidate / manifest["default"]
        else:
            files = list(candidate.glob("*.safetensors"))
            if len(files) != 1:
                return None
            candidate = files[0]
    if not candidate.is_file() or candidate.suffix != ".safetensors":
        return None
    with safe_open(str(candidate), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = tuple(handle.keys())
        if metadata.get("hyperflow", "").lower() != "true":
            if any("endpoint_time_embedder." in key for key in keys):
                raise ValueError("Endpoint LoRA weights require a HyperFlow header")
            return None
        config = MiniMaxH3HyperFlow.from_metadata(metadata, keys)
        for key in keys:
            shape = handle.get_slice(key).get_shape()
            rank_axis = 0 if ".lora_A." in key else 1
            if len(shape) != 2 or shape[rank_axis] != config.rank:
                raise ValueError(f"HyperFlow rank/shape mismatch for {key}: {shape}")
    return str(candidate.resolve()), config
