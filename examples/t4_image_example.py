#!/usr/bin/env python3
"""Run SAM 3 text-prompted image segmentation on a 16 GB NVIDIA T4."""

import argparse
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
import torch
from huggingface_hub.errors import GatedRepoError
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=root / "assets/images/truck.jpg")
    parser.add_argument("--prompt", default="truck")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("sam3_t4_output.png"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--attention",
        choices=("flash", "sdpa"),
        default="flash",
        help="attention backend to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--random-weights",
        action="store_true",
        help="skip checkpoint loading for a GPU compatibility smoke test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_start = perf_counter()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this example.")

    capability = torch.cuda.get_device_capability()
    print(
        f"GPU: {torch.cuda.get_device_name()} "
        f"(compute capability {capability[0]}.{capability[1]})"
    )

    try:
        model_start = perf_counter()
        model = build_sam3_image_model(
            checkpoint_path=str(args.checkpoint) if args.checkpoint else None,
            load_from_HF=args.checkpoint is None and not args.random_weights,
            compile=False,
            use_flash_attention=args.attention == "flash",
        )
    except GatedRepoError as error:
        raise SystemExit(
            "SAM 3 weights are gated. Run `hf auth login` after receiving access at "
            "https://huggingface.co/facebook/sam3, or pass --checkpoint /path/to/sam3.pt."
        ) from error

    torch.cuda.synchronize()
    model_seconds = perf_counter() - model_start
    processor = Sam3Processor(model, confidence_threshold=args.threshold)
    print(f"Inference dtype: {processor.inference_dtype}")
    if args.attention == "flash":
        import flash_attn

        print(f"Attention: FlashAttention {flash_attn.__version__}")
    else:
        print("Attention: PyTorch SDPA")

    image = Image.open(args.image).convert("RGB")
    for _ in range(args.warmup):
        warmup_state = processor.set_image(image)
        processor.set_text_prompt(args.prompt, warmup_state)

    image_times = []
    prompt_times = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        image_start = perf_counter()
        state = processor.set_image(image)
        torch.cuda.synchronize()
        image_times.append(perf_counter() - image_start)

        prompt_start = perf_counter()
        result = processor.set_text_prompt(args.prompt, state)
        torch.cuda.synchronize()
        prompt_times.append(perf_counter() - prompt_start)

    overlay = np.asarray(image).copy()
    masks = result["masks"].squeeze(1).cpu().numpy()
    if len(masks):
        combined_mask = masks.any(axis=0)
        color = np.array([30, 144, 255], dtype=np.uint8)
        overlay[combined_mask] = (
            overlay[combined_mask].astype(np.float32) * 0.45 + color * 0.55
        ).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(args.output)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Found {len(masks)} mask(s); wrote {args.output}")
    print(f"Peak CUDA memory: {peak_memory:.2f} GiB")
    print(f"Model load: {model_seconds:.3f} s")
    total_times = [a + b for a, b in zip(image_times, prompt_times)]
    print(f"Measured runs: {args.repeats} after {args.warmup} warm-up run(s)")
    print(f"Image encode median: {median(image_times):.3f} s")
    print(f"Prompt inference median: {median(prompt_times):.3f} s")
    print(f"Inference median: {median(total_times):.3f} s")
    print(f"Inference mean: {mean(total_times):.3f} s")
    print(f"End-to-end: {perf_counter() - total_start:.3f} s")


if __name__ == "__main__":
    main()
