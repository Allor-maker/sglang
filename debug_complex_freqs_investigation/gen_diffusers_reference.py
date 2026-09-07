"""Generate a reference image with upstream diffusers directly (not sglang),
for the exact prompt/seed/settings the given NPU consistency test case uses,
as an independent ground truth for the complex_freqs accuracy investigation.

Settings per model come from the sglang test config:
  python/sglang/multimodal_gen/test/server/testcase_configs.py (T2I_sampling_params)
  python/sglang/multimodal_gen/configs/sample/{qwenimage,flux}.py (per-model defaults)

qwen_image uses real classifier-free guidance (guidance_embeds=False in its
config): negative_prompt=" ", true_cfg_scale=4.0.
flux (FLUX.1-dev) uses embedded/distilled guidance (guidance_embeds=True):
no negative_prompt, guidance_scale=3.5 -- passing a negative_prompt or
true_cfg_scale here would silently mismatch what sglang's flux_image_t2i_npu
case actually runs.

Run: python gen_diffusers_reference.py --model qwen_image
     python gen_diffusers_reference.py --model flux

If this OOMs even with enable_model_cpu_offload() (e.g. another process is
also holding NPU memory, or the offload hook doesn't fully release peak
activation memory), switch to pipe.enable_sequential_cpu_offload(device=device)
instead -- moves weights layer-by-layer rather than component-by-component,
much lower peak memory but noticeably slower.
"""

import argparse

import torch

MODEL_CONFIGS = {
    "qwen_image": {
        "pipeline_class": "QwenImagePipeline",
        "model_path": "/root/.cache/modelscope/hub/models/Qwen/Qwen-Image",
        "prompt": "Doraemon is eating dorayaki",
        "negative_prompt": " ",
        "width": 1024,
        "height": 1024,
        "guidance_kwarg": "true_cfg_scale",
        "guidance_scale": 4.0,
        "num_inference_steps": 50,
        "seed": 42,
    },
    "flux": {
        "pipeline_class": "FluxPipeline",
        "model_path": "/root/.cache/modelscope/hub/models/black-forest-labs/FLUX.1-dev",
        "prompt": "Doraemon is eating dorayaki",
        "negative_prompt": None,  # FLUX.1-dev: embedded guidance, no real CFG
        "width": 1024,
        "height": 1024,
        "guidance_kwarg": "guidance_scale",
        "guidance_scale": 3.5,
        "num_inference_steps": 50,
        "seed": 42,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--out", default=None, help="Defaults to ./<model>_diffusers_reference.png")
    parser.add_argument("--model-path", default=None, help="Override the default local path")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    out_path = args.out or f"./{args.model}_diffusers_reference.png"
    model_path = args.model_path or cfg["model_path"]

    import diffusers

    pipeline_cls = getattr(diffusers, cfg["pipeline_class"])
    pipe = pipeline_cls.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    device = "npu:0"
    # text_encoder + transformer + VAE together can exceed a single NPU's
    # memory if all resident at once. This keeps only the currently-active
    # component on-device -- same idea as sglang's own component residency
    # manager, via accelerate's offload hooks instead.
    pipe.enable_model_cpu_offload(device=device)

    generator = torch.Generator(device=device).manual_seed(cfg["seed"])
    call_kwargs = {
        "prompt": cfg["prompt"],
        "width": cfg["width"],
        "height": cfg["height"],
        "num_inference_steps": cfg["num_inference_steps"],
        "generator": generator,
        cfg["guidance_kwarg"]: cfg["guidance_scale"],
    }
    if cfg["negative_prompt"] is not None:
        call_kwargs["negative_prompt"] = cfg["negative_prompt"]

    result = pipe(**call_kwargs)
    image = result.images[0]
    image.save(out_path)
    print(f"Saved {args.model} diffusers reference image to {out_path}")


if __name__ == "__main__":
    main()
