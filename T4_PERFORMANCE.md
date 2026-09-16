# SAM 3 Performance on NVIDIA T4

This document records the work required to run SAM 3 image inference on an
NVIDIA Tesla T4 and establishes a baseline for future performance work.

## Test Environment

- GPU: NVIDIA Tesla T4, 16 GiB, compute capability 7.5
- Driver: 535.274.02
- PyTorch: 2.9.1+cu128
- Python: 3.10.11
- Precision: FP16
- FlashAttention: 1.0.9, compiled for `sm_75`
- Model: `sam3.pt`
- Input: `assets/images/truck.jpg`
- Model resolution: 1008 x 1008
- Compilation: disabled

The T4 does not have native BF16 tensor cores. Image inference therefore
selects FP16 on pre-Ampere CUDA devices and BF16 on Ampere or newer devices.

## Changes

- Added automatic FP16 selection for T4 image inference.
- Added internal autocast handling to `Sam3Processor`.
- Added the missing `einops` runtime dependency.
- Added a FlashAttention 1.x adapter for dense batched Q/K/V tensors.
- Fixed propagation of the custom attention flag through the common
  `MultiheadAttention` path.
- Added FlashAttention configuration to the image model builder.
- Added `examples/t4_image_example.py` with output visualization, synchronized
  timing, warm-up, repeated measurements, and backend selection.

## FlashAttention Installation

The default CUDA compiler in the test environment did not expose the required
CUDA headers. FlashAttention 1.0.9 was built using the installed CUDA 12.2
toolkit and an explicit T4 architecture target:

```bash
CUDA_HOME=/usr/local/cuda-12.2 \
TORCH_CUDA_ARCH_LIST=7.5 \
MAX_JOBS=2 \
pip install "flash-attn==1.0.9" --no-build-isolation
```

The old `flash_attn_func` compatibility API in FlashAttention 1.0.9 is not
compatible with the current PyTorch custom-op signature. The repository uses
`flash_attn_unpadded_func`, which was tested directly on the T4.

## Reproducing the Benchmark

FlashAttention:

```bash
python examples/t4_image_example.py \
  --checkpoint ./sam3.pt \
  --attention flash \
  --warmup 3 \
  --repeats 15 \
  --output /tmp/sam3_flash.png
```

PyTorch SDPA:

```bash
python examples/t4_image_example.py \
  --checkpoint ./sam3.pt \
  --attention sdpa \
  --warmup 3 \
  --repeats 15 \
  --output /tmp/sam3_sdpa.png
```

Run these commands sequentially. Running both simultaneously on one GPU causes
contention and produces invalid results.

CUDA operations are asynchronous. The example calls `torch.cuda.synchronize()`
at timing boundaries so measurements include completed GPU work rather than
only kernel launch time.

## Baseline Results

| Backend | Image encode median | Prompt median | Total median | Total mean |
| --- | ---: | ---: | ---: | ---: |
| FlashAttention 1.0.9 | 0.884 s | 0.093 s | 0.979 s | 0.977 s |
| PyTorch SDPA | 0.888 s | 0.104 s | 0.992 s | 0.992 s |

FlashAttention improved median end-to-end inference by approximately 1.3%.
Peak allocated CUDA memory was 4.33 GiB for both backends during the repeated
benchmark.

The generated overlays differed at only 24 pixels. This is consistent with
minor FP16 numerical differences at the mask threshold.

Model loading took approximately 11 seconds but is excluded from inference
latency. A service should construct and retain the model rather than reload it
per request.

## Interpretation

Attention is not the dominant end-to-end cost for this image workload. The
1008-pixel ViT image encoder accounts for about 90% of warm inference time, but
that stage also includes projections, MLPs, normalization, positional encoding,
and feature-neck work. Replacing only attention kernels cannot substantially
accelerate those operations.

PyTorch SDPA is also not necessarily using its slow mathematical backend. On a
T4 it can dispatch to a memory-efficient CUDA implementation, reducing the
difference from external FlashAttention 1.x.

## Next Optimization Work

Prioritize experiments in this order and measure accuracy as well as latency:

1. Profile with `torch.profiler` and export a Chrome trace. Rank CUDA kernels by
   total time before changing more code.
2. Measure `torch.compile` on the image backbone after a sufficient warm-up.
   Record compilation time separately and test whether generated Triton kernels
   support `sm_75` reliably.
3. Test lower input resolutions. Vision token count and attention cost grow
   rapidly with spatial resolution, so this is likely the largest latency lever,
   but it can reduce small-object accuracy.
4. Cache image embeddings when applying multiple prompts to one image. A new
   prompt currently costs about 0.1 seconds versus about 0.9 seconds to encode a
   new image.
5. Test batching for throughput-oriented workloads. Report images per second
   and p50/p95 latency independently.
6. Profile host-to-device preprocessing and consider pinned memory, nonblocking
   copies, and GPU-native resize/normalization if input throughput becomes a
   bottleneck.
7. Evaluate TensorRT or ONNX graph conversion for stable production shapes.
   Validate unsupported operators and output quality before investing in a full
   conversion.
8. Consider structured model reduction or quantization only after kernel and
   graph-level opportunities are measured. T4 INT8 tensor cores may improve
   throughput, but calibration and segmentation quality require careful testing.

## Benchmarking Rules

- Keep GPU clocks, power limits, and concurrent workload consistent.
- Run backends sequentially in fresh processes.
- Use at least three warm-up iterations and ten measured iterations.
- Synchronize CUDA at every timing boundary.
- Report median, mean, p95, peak allocated memory, and exact software versions.
- Separate model load, image encode, prompt inference, and postprocessing.
- Compare masks or metrics after every numerical optimization.
- Repeat benchmarks after machine restart or environment changes before drawing
  conclusions from differences below 5%.

## Current Status

The pretrained image example runs successfully on the T4 in FP16, returns one
truck mask for the bundled example, and passes all nine repository unit tests.
FlashAttention 1.0.9 is active and remains the example default, although its
current end-to-end benefit is small.
