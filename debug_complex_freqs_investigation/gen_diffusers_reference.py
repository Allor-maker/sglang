"""Generate a reference image with upstream diffusers directly (not sglang),
for the exact prompt/seed/settings qwen_image_t2i_2npu uses, as an
independent ground truth for the complex_freqs accuracy investigation.

Prompt/settings match python/sglang/multimodal_gen/test/server/testcase_configs.py
T2I_sampling_params + QwenImageSamplingParams defaults:
  prompt="Doraemon is eating dorayaki", negative_prompt=" ",
  1024x1024, guidance_scale(true_cfg_scale)=4.0, num_inference_steps=50, seed=42.

CAUTION: diffusers' QwenImagePipeline uses `true_cfg_scale` for real
classifier-free guidance (this model has guidance_embeds=False, so it's real
CFG, not embedded/distilled guidance) -- verify this still matches the
installed diffusers version with `help(QwenImagePipeline.__call__)` if the
output looks unexpected; pipeline signatures do drift across versions.

Run: python gen_diffusers_reference.py

If this still OOMs even with enable_model_cpu_offload() (e.g. another
process is also holding NPU memory, or the offload hook doesn't fully
release the transformer's peak activation memory), switch the offload call
below to pipe.enable_sequential_cpu_offload(device=device) instead --
moves weights layer-by-layer rather than component-by-component, much
lower peak memory but noticeably slower.
"""

import torch
from diffusers import QwenImagePipeline

MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen-Image"  # adjust if different
OUTPUT_PATH = "./diffusers_reference.png"

PROMPT = "Doraemon is eating dorayaki"
NEGATIVE_PROMPT = " "
WIDTH = 1024
HEIGHT = 1024
TRUE_CFG_SCALE = 4.0
NUM_INFERENCE_STEPS = 50
SEED = 42


def main() -> None:
    pipe = QwenImagePipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    device = "npu:0"
    # text_encoder (~14GB) + transformer (~38GB) + VAE together exceed a
    # single NPU's memory if all resident at once. This keeps only the
    # currently-active component on-device, moving others to CPU RAM between
    # pipeline stages -- same idea as sglang's own component residency
    # manager, via accelerate's offload hooks instead.
    pipe.enable_model_cpu_offload(device=device)

    generator = torch.Generator(device=device).manual_seed(SEED)
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        true_cfg_scale=TRUE_CFG_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=generator,
    )
    image = result.images[0]
    image.save(OUTPUT_PATH)
    print(f"Saved diffusers reference image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
