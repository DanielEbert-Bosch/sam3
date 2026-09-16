#!/usr/bin/env python3
"""Run production-style SAM3 model calls for the AutoGT semantic classes."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


COLORS = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 212],
        [0, 128, 128],
    ],
    dtype=np.float32,
)

SAM_CLASSES = {
    "road": ("road", "pavement"),
    "Bicycle": ("bicycle",),
    "Bobbycar": ("ride-on toy car", "toy car"),
    "Box": ("box", "rectangular box"),
    "Bush": ("bush", "vegetation"),
    "Ceiling Beam": ("ceiling beam", "ceiling"),
    "Cone": ("traffic cone",),
    "Curbstone": ("curb",),
    "Fence": ("fence",),
    "Motorcycle": ("motorcycle",),
    "Pedestrian": ("person",),
    "Pillar": ("pillar",),
    "Pole": ("pole",),
    "Foam Cube": ("foam cube", "black box"),
    "Speedbump": ("speed bump", "black and white strip"),
    "Tirestopper": (
        "parking wheel stop",
        "parking curb",
        "wheel stop",
        "tire stopper",
        "low concrete block on ground",
    ),
    "Tree": ("tree",),
    "U Barrier": ("yellow barrier",),
    "Vehicle": ("vehicle", "car"),
    "Wall": ("wall", "building"),
}


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="input PNG path")
    parser.add_argument("--checkpoint", type=Path, default=root / "sam3.pt")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python environment.")
    if not args.checkpoint.is_file():
        raise SystemExit("Checkpoint not found: {}".format(args.checkpoint))
    if not args.image.is_file() or args.image.suffix.lower() != ".png":
        raise SystemExit("Input must be an existing PNG: {}".format(args.image))

    prompts = [
        (class_id, class_name, prompt)
        for class_id, (class_name, class_prompts) in enumerate(
            SAM_CLASSES.items(), start=1
        )
        for prompt in class_prompts
    ]
    output = args.output or args.image.with_name(args.image.stem + "_colored.png")

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    print("PyTorch: {}".format(torch.__version__))
    print(
        "GPU: {} (compute capability {}.{})".format(
            torch.cuda.get_device_name(device), capability[0], capability[1]
        )
    )
    print("Loading checkpoint: {}".format(args.checkpoint))

    torch.cuda.reset_peak_memory_stats(device)
    model = build_sam3_image_model(
        device="cuda",
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        compile=False,
    )
    processor = Sam3Processor(model, device=device, confidence_threshold=args.threshold)
    # Reuse the processor's dtype choice: `torch.cuda.is_bf16_supported()` is True
    # even on pre-Ampere cards, where bf16 has no tensor-core support and every
    # GEMM falls back to a slow MAGMA kernel.
    amp_dtype = processor.inference_dtype
    print("Autocast dtype: {}".format(amp_dtype))
    context = torch.autocast("cuda", dtype=amp_dtype)
    with context, torch.inference_mode():
        # Every caption is padded to the text encoder's fixed context length, so one
        # batched forward is equivalent to (and much cheaper than) N single ones.
        batched_text = model.backbone.forward_text(
            [prompt for _, _, prompt in prompts], device=device
        )
        # "language_features"/"language_embeds" are seq-first, "language_mask" is
        # batch-first; slice each back out to the single-prompt layout.
        text_outputs = [
            {
                "language_features": batched_text["language_features"][:, i : i + 1],
                "language_embeds": batched_text["language_embeds"][:, i : i + 1],
                "language_mask": batched_text["language_mask"][i : i + 1],
            }
            for i in range(len(prompts))
        ]

    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    model_seconds = 0.0
    mask_count = 0
    with context, torch.inference_mode():
        state = processor.set_image(image)
        state["geometric_prompt"] = model._get_dummy_prompt()
        labels = torch.zeros((height, width), dtype=torch.int32, device=device)
        scores = torch.zeros((height, width), dtype=torch.float32, device=device)
        for (class_id, class_name, prompt), text_output in zip(prompts, text_outputs):
            state["backbone_out"].update(text_output)
            torch.cuda.synchronize(device)
            model_started = time.perf_counter()
            result = processor._forward_grounding(state)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - model_started
            model_seconds += elapsed

            masks = result["masks"]
            mask_count += len(masks)
            if len(result["scores"]):
                best_scores = (
                    masks.squeeze(1) * result["scores"][:, None, None]
                ).amax(dim=0).float()
                update = best_scores > scores
                labels[update] = class_id
                scores[update] = best_scores[update]
            print(
                "{} / {!r}: {} mask(s), model call {:.3f} seconds".format(
                    class_name, prompt, len(masks), elapsed
                )
            )

    overlay = np.asarray(image).copy()
    labels = labels.cpu().numpy()
    for class_id, class_name in enumerate(SAM_CLASSES, start=1):
        color = COLORS[(class_id - 1) % len(COLORS)]
        selected = labels == class_id
        overlay[selected] = (
            overlay[selected].astype(np.float32) * 0.4 + color * 0.6
        ).astype(np.uint8)
        print(
            "Class {!r}: {} pixel(s), color #{:02x}{:02x}{:02x}".format(
                class_name, int(selected.sum()), *color.astype(np.uint8)
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output)

    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    print("Classes: {}".format(len(SAM_CLASSES)))
    print("Prompts: {}".format(len(prompts)))
    print("Masks: {}".format(mask_count))
    print("Grounding model calls: {:.3f} seconds".format(model_seconds))
    print("Peak CUDA memory: {:.2f} GiB".format(peak_gib))
    print("Output: {}".format(output))


if __name__ == "__main__":
    main()
