# Hardware and Development Environment

## Target machine

- AMD Ryzen 7 9800X3D
- NVIDIA GeForce RTX 5060 Ti, 16GB GDDR7
- 64GB system DRAM

This machine is sufficient for Phase A–C of APC if experiments are designed around a single 16GB GPU.

## What is realistic

### Comfortable
- synthetic task generation and reference interpreters;
- 10M–100M parameter models;
- low-rank primitive banks;
- temporary expansion experiments in the tens/hundreds of millions of parameters when activation sizes are controlled;
- multiple sequential continual-learning experiments;
- CPU-side replay buffers and experiment logging.

### Possible with careful configuration
- roughly several-hundred-million-parameter training experiments using mixed precision, gradient accumulation/checkpointing, or lower-memory optimizers;
- pretrained 0.5B–2B language-model adaptation with LoRA/QLoRA in later phases;
- CPU-hosted inactive primitive storage in later prototypes.

### Not a good initial target
- training a 1B+ language model from scratch to useful general-language quality;
- full-parameter fine-tuning of multi-billion-parameter models as the default workflow;
- large hyperparameter sweeps;
- multi-million-expert experiments whose main question is systems throughput;
- reproducing frontier-scale mechanistic-interpretability pipelines.

## Why the GPU is adequate

The RTX 5060 Ti is a Blackwell GPU with a 16GB memory option and 4608 CUDA cores. APC Phase A is deliberately sized so the research bottleneck is the learning architecture, not raw training scale.

The 16GB VRAM limit should be treated as a design constraint: every default config must fit it, and the project should report peak VRAM as a first-class metric.

## Recommended software stack (August 2026)

Recommended baseline:
- Ubuntu 24.04 LTS (native Linux or WSL2 if the host is Windows);
- recent NVIDIA driver;
- Python 3.12;
- PyTorch 2.12 or newer stable;
- CUDA 13.0-compatible PyTorch wheel for Blackwell.

PyTorch 2.7 introduced Blackwell support with CUDA 12.8. PyTorch 2.12 deprecated its standard CUDA 12.8 wheel and recommends CUDA 13.0+ for newer GPUs such as Blackwell. The PyTorch 2.12 release notes state minimum driver versions for CUDA 13.0 of 580.65.06 on Linux and 580.88 on Windows.

Always verify the current installation selector before installing because package matrices change.

Minimum verification script:

```python
import torch

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("torch cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
```

The RTX 5060 Ti reports Blackwell compute capability `sm_120` in current PyTorch discussions.

## Memory engineering rules

1. Use BF16 when supported and numerically stable; otherwise FP16.
2. Use FP32 for small numerically sensitive statistics if needed.
3. Keep sequence lengths short in Phase A.
4. Use gradient accumulation instead of increasing batch size to the VRAM limit.
5. Add activation checkpointing only when measurements justify it; complexity has a cost.
6. Do not use CPU optimizer offload in the first prototype unless required. It can obscure whether the architecture is efficient.
7. Keep replay data in system RAM/disk, not VRAM.
8. Keep experiment artifacts off the GPU.

## Rough scale guidance

These are planning estimates, not guarantees; activation memory depends strongly on sequence length, batch size, optimizer and implementation.

### 30–100M model
Expected to be straightforward for full training on 16GB with mixed precision.

### 100–500M experimental model
Feasible with controlled batch/context sizes and memory-conscious optimizer/configuration. Use only after smaller experiments validate the loop.

### 0.5B–2B pretrained LM
Use parameter-efficient adaptation (LoRA/QLoRA) rather than full fine-tuning. This belongs to Phase D, not Phase A.

### 7B-class model
Quantized inference/adaptation may fit depending on configuration, but it is unnecessary for validating APC and should not become an early dependency.

## CPU and RAM usage

The 64GB DRAM pool is useful for:
- replay buffers;
- generated synthetic corpora;
- inactive primitive snapshots;
- checkpoint staging;
- experiment aggregation.

The CPU is adequate for data generation and evaluation. The GPU should remain the training bottleneck in normal runs.

## Windows note

Native Windows can run current Blackwell-compatible PyTorch builds. For research repositories that may later use Linux-first custom kernels or shell tooling, WSL2/Linux is still the preferred development target. Keep Phase A free of custom kernels so the repository remains portable.

## Sources

- NVIDIA RTX 5060 family specifications: https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/
- PyTorch 2.7 release / Blackwell support: https://pytorch.org/blog/pytorch-2-7/
- PyTorch 2.12 release / CUDA 13 recommendation for Blackwell: https://pytorch.org/blog/pytorch-2-12-release-blog/
- PyTorch install selector: https://pytorch.org/get-started/locally/
