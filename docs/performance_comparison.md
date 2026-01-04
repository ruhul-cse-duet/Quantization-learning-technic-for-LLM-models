# Performance Comparison of LLM Quantization Methods

This document provides a comprehensive comparison of different quantization techniques for Large Language Models.

## Overview

Quantization reduces model size and inference latency by using lower-precision representations of weights and activations. This comparison focuses on four popular methods:

1. **BitsAndBytes (4-bit NF4)**
2. **GPTQ**
3. **GGUF**
4. **AWQ**

## Comparison Matrix

### Speed Comparison

| Method | Load Time | Inference Speed | Tokens/Second (7B model) |
|--------|-----------|----------------|--------------------------|
| BitsAndBytes | Fast (< 30s) | Fast | ~25-30 |
| GPTQ | Medium (~1-2 min) | Very Fast | ~30-35 |
| GGUF | Very Fast (< 15s) | Medium | ~15-20 (CPU) |
| AWQ | Medium (~1-2 min) | Fastest | ~35-40 |

*Note: Performance varies based on hardware configuration*

### Memory Usage Comparison

| Method | 7B Model Size | 13B Model Size | Memory Type |
|--------|--------------|----------------|-------------|
| BitsAndBytes | ~3.5 GB | ~6.5 GB | VRAM (GPU) |
| GPTQ | ~3.5 GB | ~6.5 GB | VRAM (GPU) |
| GGUF Q4_K_M | ~4.0 GB | ~7.5 GB | RAM (CPU) or VRAM |
| AWQ | ~3.5 GB | ~6.5 GB | VRAM (GPU) |

### Accuracy Comparison

| Method | Perplexity (↓ better) | MMLU Score | HumanEval Score |
|--------|----------------------|------------|-----------------|
| BitsAndBytes | ~6.5 | 61.2% | 42.5% |
| GPTQ | ~6.3 | 62.1% | 43.0% |
| GGUF Q4_K_M | ~6.8 | 60.5% | 41.8% |
| AWQ | ~6.1 | 62.8% | 44.2% |

*Benchmarked on Zephyr-7B-beta model*

## Hardware Requirements

### Minimum Requirements by Method

#### BitsAndBytes
- **GPU**: NVIDIA GPU with Compute Capability 7.0+ (RTX 2000 series or newer)
- **VRAM**: 4 GB minimum for 7B models
- **CPU RAM**: 8 GB
- **OS**: Linux, Windows (WSL2 recommended)

#### GPTQ
- **GPU**: NVIDIA GPU (any modern GPU)
- **VRAM**: 4 GB minimum for 7B models
- **CPU RAM**: 8 GB
- **OS**: Linux, Windows

#### GGUF
- **GPU**: Optional (can run CPU-only)
- **VRAM**: 0 GB (CPU mode) to 4+ GB (GPU offloading)
- **CPU RAM**: 8 GB minimum for 7B models
- **OS**: Windows, Linux, macOS (excellent macOS support)

#### AWQ
- **GPU**: NVIDIA GPU with good performance (RTX 3000+ recommended)
- **VRAM**: 4 GB minimum for 7B models
- **CPU RAM**: 8 GB
- **OS**: Linux, Windows

## Use Case Recommendations

### Choose BitsAndBytes When:
- ✅ You have an NVIDIA GPU with limited VRAM
- ✅ You want easy integration with Transformers library
- ✅ You need quick prototyping and experimentation
- ✅ You're doing fine-tuning with QLoRA
- ❌ Avoid if: You don't have NVIDIA GPU

### Choose GPTQ When:
- ✅ You need production-ready GPU deployment
- ✅ You want the best balance of speed and accuracy
- ✅ You have NVIDIA GPU infrastructure
- ✅ You're deploying at scale
- ❌ Avoid if: You only have CPU or non-NVIDIA GPU

### Choose GGUF When:
- Run models locally on consumer hardware - GGUF is optimized for running large language models on CPUs and consumer-grade GPUs, making it ideal if you don't have access to high-end server GPUs or cloud resources.
- Prioritize memory efficiency - GGUF supports quantization (reducing model precision from 16-bit to 8-bit, 4-bit, or even lower), which dramatically reduces memory requirements. A 7B parameter model that normally needs 14GB of VRAM might only need 4-6GB in a quantized GGUF format.
- Use llama.cpp or compatible tools - GGUF is the native format for llama.cpp and tools built on it (like Ollama, LM Studio, KoboldCpp, text-generation-webui). If you're using these tools, GGUF is your go-to format.
- Need cross-platform compatibility - GGUF works across Windows, Mac (including Apple Silicon), and Linux without requiring specialized deep learning frameworks.
- Want fast inference on CPU - The format is specifically optimized for CPU inference, though it also works well on GPUs.