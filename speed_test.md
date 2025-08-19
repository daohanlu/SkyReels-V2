# JAX vs PyTorch Performance Comparison Report

## Executive Summary

This document summarizes comprehensive performance testing of JAX vs PyTorch implementations for the SkyReels-V2 video generation model. Key findings show that while JAX successfully matches PyTorch's memory efficiency through custom optimizations, it remains 6-8x slower than PyTorch's highly optimized CUDA implementation.

## Test Configuration

- **GPU**: NVIDIA RTX 6000 Ada (48GB)
- **Model**: SkyReels-V2-I2V-1.3B-540P
- **Resolution**: 540P (960x544)
- **Inference Steps**: 5 (for most tests)
- **Test Frames**: 9, 17, and 57 frames

## Memory Usage Comparison

### Standard Attention (JAX's default dot_product_attention)

| Frames | Memory Required | Status | Notes |
|--------|----------------|--------|--------|
| 9 frames | ~4GB | ✅ Works | Fits in memory |
| 17 frames | 9.3GB | ❌ OOM | Exceeds available memory |
| 57 frames | 41.86GB | ❌ OOM | Nearly entire GPU memory |

### Memory-Efficient Attention (Chunked, 256 tokens)

| Frames | Memory Required | Status | Notes |
|--------|----------------|--------|--------|
| 9 frames | ~4GB | ✅ Works | Same as standard |
| 17 frames | ~6GB | ✅ Works | Linear scaling |
| 57 frames | ~10GB | ✅ Works | Efficient memory use |

### PyTorch (Flash Attention v2)

| Frames | Memory Required | Status | Notes |
|--------|----------------|--------|--------|
| 57 frames | ~8GB | ✅ Works | Native Flash Attention |

## Speed Comparison

### 17 Frames Generation (5 inference steps)

| Implementation | Total Time | Denoising Only | Speed vs PyTorch |
|----------------|------------|----------------|------------------|
| **PyTorch** | ~7s | ~5s | 1.0x (baseline) |
| **JAX (Standard Attention)** | OOM | - | N/A |
| **JAX (Memory-Efficient)** | 46.48s | ~43s | 6.6x slower |
| **JAX (Memory-Efficient + XLA)** | 34.19s | ~30s | 4.9x slower |

### 57 Frames Generation (5 inference steps)

| Implementation | Total Time | Denoising Only | Speed vs PyTorch |
|----------------|------------|----------------|------------------|
| **PyTorch** | ~20s | ~15s | 1.0x (baseline) |
| **JAX (Standard Attention)** | OOM | - | N/A |
| **JAX (Memory-Efficient)** | 192.19s | ~181s | 9.6x slower |
| **JAX (Memory-Efficient + XLA)** | 130.39s | 118.63s | 6.5x slower |

### 9 Frames Generation (5 inference steps) - Detailed Comparison

| Implementation | Total Time | Per Step | Notes |
|----------------|------------|----------|--------|
| **PyTorch** | ~5s | ~1s | Baseline |
| **JAX (Standard, No JIT)** | 26.80s | 4.94s | 5.4x slower |
| **JAX (Standard, With JIT)** | 26.41s | 2.24s* | *After compilation |
| **JAX (Memory-Efficient)** | 27.46s | 5.09s | Nearly same as standard |

## JIT Compilation Analysis

### XLA Compilation Overhead (First Run)

| Metric | 9 frames | 57 frames |
|--------|----------|-----------|
| First step (with compilation) | 14.41s | 52.22s |
| Subsequent steps (compiled) | ~2.24s | ~16.60s |
| Compilation overhead | 12.17s | 35.61s |
| Speed improvement per step | 2.2x | 2.0x |

### Break-Even Analysis

| Inference Steps | No JIT | With JIT | Speedup | Worth It? |
|-----------------|--------|----------|---------|-----------|
| 5 steps | 26.8s | 26.4s | 1.01x | ❌ No benefit |
| 10 steps | ~52s | ~34s | 1.53x | ✅ Moderate |
| 30 steps | ~152s | ~79s | 1.92x | ✅ Significant |
| 50 steps | ~252s | ~124s | 2.03x | ✅ Very good |

## Key Optimizations Implemented

### 1. Memory-Efficient Attention
- **File**: `jax_implementation/modules/attention_memory_efficient.py`
- **Method**: Chunked attention processing (256 token chunks)
- **Impact**: Enables 57+ frame generation on 48GB GPU
- **Overhead**: Negligible (~2% slower than standard for small sequences)

### 2. XLA Memory Configuration
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
export JAX_MEMORY_EFFICIENT_ATTENTION=true
```

### 3. JIT Compilation
- **File**: `test_jax_generation_jit.py`
- **Impact**: 2.2x faster per step after compilation
- **Trade-off**: 12-35s compilation overhead on first run

## Performance Bottlenecks

### Why JAX is Slower

1. **Attention Implementation**
   - PyTorch: Optimized Flash Attention v2 in CUDA
   - JAX: Chunked attention in Python/XLA (no native Flash Attention)

2. **Data Transfer Overhead**
   - VAE and CLIP models remain in PyTorch
   - Constant tensor conversion between PyTorch ↔ JAX

3. **XLA vs CUDA Optimization**
   - PyTorch: Mature CUDA kernels optimized for transformers
   - JAX/XLA: More general-purpose compilation, less optimized for specific patterns

4. **Memory Access Patterns**
   - Chunked attention requires more memory reads/writes
   - PyTorch's Flash Attention minimizes HBM access

## Recommendations

### For Production Use

1. **Always use memory-efficient attention** - Essential for video lengths >10 frames
2. **Enable JIT compilation for 10+ inference steps** - Significant speedup after break-even
3. **Consider PyTorch for latency-critical applications** - 6-8x faster
4. **Use JAX for research/experimentation** - Better flexibility and debugging

### Optimal JAX Configuration

```python
# For maximum compatibility and performance
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
os.environ["JAX_MEMORY_EFFICIENT_ATTENTION"] = "true"

# Use test_jax_generation_jit.py for inference steps >= 10
# Use test_jax_generation.py for quick tests < 10 steps
```

## Future Optimization Opportunities

1. **Native JAX Flash Attention** - Could provide 3-5x speedup
2. **Full JAX Pipeline** - Convert VAE/CLIP to JAX to eliminate transfer overhead
3. **Custom XLA Kernels** - Hand-optimized kernels for critical operations
4. **Model Sharding** - Distribute across multiple GPUs for larger models
5. **Mixed Precision** - More aggressive bfloat16 usage

## Conclusion

The JAX implementation successfully achieves functional parity with PyTorch in terms of memory efficiency, enabling generation of 57+ frame videos on 48GB GPUs through custom memory-efficient attention. However, performance remains the primary challenge, with JAX running 6-8x slower than PyTorch's highly optimized implementation. 

The 32% speedup from XLA JIT compilation helps but doesn't close the gap. For production use cases prioritizing speed, PyTorch remains the better choice. JAX offers advantages for research and experimentation due to its functional programming model and automatic differentiation capabilities.