# NVIDIA T4 Support

- Image inference now selects FP16 on pre-Ampere GPUs and consistently applies autocast.
- FlashAttention 1.x support was added and wired through the image model attention paths.
- Fused ViT MLPs now honor the autocast dtype instead of forcing slow BF16 on T4.
- Added the `einops` dependency and a synchronized T4 benchmark example.

On a Tesla T4, warm end-to-end inference improved from about 0.99 s to 0.45 s.
