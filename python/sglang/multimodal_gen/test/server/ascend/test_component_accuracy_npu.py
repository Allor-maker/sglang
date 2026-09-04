"""NPU component accuracy: sglang transformer output vs diffusers' own
reference implementation, for cases whose model_path is already resolved to
this ascend host's local (ModelScope-cached) weights.

test_component_accuracy_1_gpu.py / test_component_accuracy_2_gpu.py source
their case list from test/server/gpu_cases.py, whose model_path values are
bare HF Hub repo ids (e.g. "Qwen/Qwen-Image") -- fine for CUDA CI runners
with HF Hub access, but on an ascend host that means falling through to an
unauthenticated multi-GB HF download even when the weights already exist
locally under a different resolved path. This file reuses the same
AccuracyEngine / comparison machinery unchanged and only swaps the case
source to TWO_NPU_CASES (testcase_configs_npu.py), whose model_path values
already point at the local ModelScope cache.
"""

import pytest

from sglang.multimodal_gen.test.server.ascend.testcase_configs_npu import TWO_NPU_CASES
from sglang.multimodal_gen.test.single_test_file.component_accuracy.config import (
    ComponentType,
    get_skip_reason,
    should_skip_component,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.engine import (
    AccuracyEngine,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    run_native_component_accuracy_case,
)

ACCURACY_NPU_CASES = [
    case for case in TWO_NPU_CASES if case.run_component_accuracy_check
]


@pytest.mark.parametrize("case", ACCURACY_NPU_CASES, ids=lambda case: case.id)
class TestComponentAccuracyNPU:
    """Component accuracy suite for ascend cases with local weights."""

    def test_transformer_accuracy(self, case):
        if should_skip_component(case, ComponentType.TRANSFORMER):
            pytest.skip(get_skip_reason(case, ComponentType.TRANSFORMER))
        run_native_component_accuracy_case(
            AccuracyEngine,
            case,
            ComponentType.TRANSFORMER,
            "diffusers",
            case.server_args.num_gpus,
        )