"""Score both sglang variants (complex_freqs on/off) against the diffusers
reference image, using the exact same metric functions the consistency
check itself uses (test_utils.py), so this is directly comparable to those
thresholds (clip>=0.92, ssim>=0.95, psnr>=28.0, mean_abs_diff<=8.0).

Run after gen_diffusers_reference.py and both gen_sglang_variant.py calls:

    python compare_outputs.py \
        --reference ./diffusers_reference.png \
        --old ./sglang_old.png \
        --new ./sglang_new.png
"""

import argparse

import numpy as np
from PIL import Image

from sglang.multimodal_gen.test.test_utils import (
    compute_clip_embedding,
    compute_clip_similarity,
    compute_mean_abs_diff,
    compute_psnr,
    compute_ssim,
)


def load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def score(name: str, candidate: np.ndarray, ref: np.ndarray, ref_emb) -> None:
    if candidate.shape != ref.shape:
        raise ValueError(
            f"{name}: shape mismatch, candidate={candidate.shape} ref={ref.shape}"
        )
    cand_emb = compute_clip_embedding(candidate)
    clip_sim = compute_clip_similarity(cand_emb, ref_emb)
    ssim = compute_ssim(candidate, ref)
    psnr = compute_psnr(candidate, ref)
    mean_abs_diff = compute_mean_abs_diff(candidate, ref)
    print(
        f"{name}: clip={clip_sim:.4f} ssim={ssim:.4f} psnr={psnr:.4f} "
        f"mean_abs_diff={mean_abs_diff:.4f}"
    )
    print(
        f"  (thresholds: clip>=0.92, ssim>=0.95, psnr>=28.0, mean_abs_diff<=8.0)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--old", required=True, help="complex_freqs disabled")
    parser.add_argument("--new", required=True, help="complex_freqs enabled")
    args = parser.parse_args()

    ref = load_rgb(args.reference)
    ref_emb = compute_clip_embedding(ref)

    old = load_rgb(args.old)
    new = load_rgb(args.new)

    print("Scoring against diffusers reference:")
    score("old (complex_freqs off)", old, ref, ref_emb)
    score("new (complex_freqs on) ", new, ref, ref_emb)


if __name__ == "__main__":
    main()
