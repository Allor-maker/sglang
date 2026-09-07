"""Generate an image through sglang itself, using the same prompt/seed/
settings as gen_diffusers_reference.py, for the given model.

Both models run num_gpus=2, ring_degree=2 -- held identical on purpose so
the same test config is exercised across models (flux_image_t2i_npu's own
test case defaults to num_gpus=1; overridden here). Launch both via
torch.distributed.run --nproc_per_node=2, not plain `python`.

Run twice per model to get both variants for the A/B comparison:

    python gen_sglang_variant.py --model qwen_image --out ./qwen_old.png --disable-complex-freqs
    python gen_sglang_variant.py --model qwen_image --out ./qwen_new.png

    python gen_sglang_variant.py --model flux --out ./flux_old.png --disable-complex-freqs
    python gen_sglang_variant.py --model flux --out ./flux_new.png

--disable-complex-freqs sets the model-specific env var before importing
sglang, reproducing the pre-complex_freqs behavior via the temporary toggle
added to the model's DiT/pipeline config code.
"""

import argparse
import os

MODEL_CONFIGS = {
    "qwen_image": {
        "model_path": "/root/.cache/modelscope/hub/models/Qwen/Qwen-Image",
        "prompt": "Doraemon is eating dorayaki",
        "negative_prompt": " ",
        "width": 1024,
        "height": 1024,
        "guidance_scale": 4.0,
        "num_inference_steps": 50,
        "seed": 42,
        "num_gpus": 2,
        "ulysses_degree": 1,
        "ring_degree": 2,
        "disable_env_var": "SGLANG_QWEN_IMAGE_DISABLE_COMPLEX_FREQS",
    },
    "flux": {
        "model_path": "/root/.cache/modelscope/hub/models/black-forest-labs/FLUX.1-dev",
        "prompt": "Doraemon is eating dorayaki",
        "negative_prompt": None,
        "width": 1024,
        "height": 1024,
        "guidance_scale": 3.5,
        "num_inference_steps": 50,
        "seed": 42,
        # Held identical to qwen_image (not flux_image_t2i_npu's own
        # num_gpus=1 default) on purpose: same SP config across models so
        # ring-degree sharding isn't a confounding variable when comparing.
        "num_gpus": 2,
        "ulysses_degree": 1,
        "ring_degree": 2,
        "disable_env_var": "SGLANG_FLUX_DISABLE_COMPLEX_FREQS",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--disable-complex-freqs", action="store_true")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    if args.disable_complex_freqs:
        os.environ[cfg["disable_env_var"]] = "1"

    # Import after setting the env var: the toggle is read at pipeline-config
    # call time, not import time, but setting it first avoids any doubt.
    # Note: DiffGenerator is exported from sglang.multimodal_gen, not the
    # top-level sglang package (that's the LLM/srt runtime's namespace).
    from sglang.multimodal_gen import DiffGenerator

    from_pretrained_kwargs = {
        "model_path": args.model_path or cfg["model_path"],
        "num_gpus": cfg["num_gpus"],
    }
    if cfg["ulysses_degree"] is not None:
        from_pretrained_kwargs["ulysses_degree"] = cfg["ulysses_degree"]
    if cfg["ring_degree"] is not None:
        from_pretrained_kwargs["ring_degree"] = cfg["ring_degree"]

    gen = DiffGenerator.from_pretrained(**from_pretrained_kwargs)

    out_dir, _out_name = os.path.split(os.path.abspath(args.out))
    sampling_params_kwargs = {
        "prompt": cfg["prompt"],
        "height": cfg["height"],
        "width": cfg["width"],
        "guidance_scale": cfg["guidance_scale"],
        "num_inference_steps": cfg["num_inference_steps"],
        "seed": cfg["seed"],
        "save_output": True,
        "output_path": out_dir or ".",
    }
    if cfg["negative_prompt"] is not None:
        sampling_params_kwargs["negative_prompt"] = cfg["negative_prompt"]

    result = gen.generate(sampling_params_kwargs=sampling_params_kwargs)
    print(f"model={args.model} disable_complex_freqs={args.disable_complex_freqs}")
    print(f"Saved to: {result.output_file_path}")
    print(f"(requested --out was {args.out} -- rename/move if the actual "
          f"saved filename differs from what you expected)")


if __name__ == "__main__":
    main()
