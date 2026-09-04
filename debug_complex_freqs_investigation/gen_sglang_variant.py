"""Generate a qwen_image_t2i_2npu image through sglang itself, using the
same prompt/seed/settings as gen_diffusers_reference.py.

Run twice to get both variants for the A/B comparison:

    python gen_sglang_variant.py --out ./sglang_old.png \
        --disable-complex-freqs

    python gen_sglang_variant.py --out ./sglang_new.png

--disable-complex-freqs sets SGLANG_QWEN_IMAGE_DISABLE_COMPLEX_FREQS=1
before importing sglang, reproducing the pre-complex_freqs behavior via the
temporary toggle added to configs/pipeline_configs/qwen_image.py.
"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output image path")
    parser.add_argument(
        "--model-path",
        default="/root/.cache/modelscope/hub/models/Qwen/Qwen-Image",
    )
    parser.add_argument("--disable-complex-freqs", action="store_true")
    args = parser.parse_args()

    if args.disable_complex_freqs:
        os.environ["SGLANG_QWEN_IMAGE_DISABLE_COMPLEX_FREQS"] = "1"

    # Import after setting the env var: the toggle is read at pipeline-config
    # call time, not import time, but setting it first avoids any doubt.
    # Note: DiffGenerator is exported from sglang.multimodal_gen, not the
    # top-level sglang package (that's the LLM/srt runtime's namespace).
    from sglang.multimodal_gen import DiffGenerator

    gen = DiffGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=2,
        ulysses_degree=1,
        ring_degree=2,
    )
    out_dir, out_name = os.path.split(os.path.abspath(args.out))
    result = gen.generate(
        sampling_params_kwargs={
            "prompt": "Doraemon is eating dorayaki",
            "negative_prompt": " ",
            "output_size": "1024x1024",
            "guidance_scale": 4.0,
            "num_inference_steps": 50,
            "seed": 42,
            "save_output": True,
            "output_path": out_dir or ".",
        }
    )
    print(f"disable_complex_freqs={args.disable_complex_freqs}")
    print(f"Saved to: {result.output_file_path}")
    print(f"(requested --out was {args.out} -- rename/move if the actual "
          f"saved filename differs from what you expected)")


if __name__ == "__main__":
    main()
