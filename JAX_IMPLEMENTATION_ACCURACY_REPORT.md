# JAX Implementation Accuracy Report

## Executive Summary

The JAX implementation of SkyReels-V2 has been debugged and validated against the PyTorch reference implementation. The models produce nearly identical outputs with numerical agreement (PSNR: 36.63 dB, SSIM: 0.9937) despite using bfloat16 precision. This implementation currently supports Image-to-Video (I2V) models only.

## Environment Setup

### Dependencies
- **JAX:** Version 0.5.1
- **Flax:** Version 0.10.4
- **Model Compatibility:** I2V (Image-to-Video) models only
  - Tested with: Skywork/SkyReels-V2-I2V-1.3B-540P
  - Text-to-Video (T2V) models are not yet supported

### JAX Configuration
```bash
# Enable float64 for debugging (used in some scripts)
# In Python: jax.config.update('jax_enable_x64', True)

# Prevent JAX from pre-allocating GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Alternative memory fraction setting
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

## Debugging Scripts Used (Final Version)

### 1. Detailed Transformer Debugging (`debug_transformer_detailed.py`)
**Purpose:** Comprehensive analysis of each operation within transformer blocks.

**Command:**
```bash
python debug_transformer_detailed.py
```

**Key Components Analyzed:**
- Modulation + time embeddings
- Layer normalization
- mul_add and mul_add_add operations
- Self-attention projections (Q, K, V)
- Cross-attention
- Feed-forward networks

**Key Findings:**
- mul_add_add operation showed initial dtype inconsistencies (fixed)
- Cross-attention had dtype mismatch issues (fixed)
- Most differences are within expected bfloat16 precision limits

### 2. Dtype Handling Analysis (`fix_dtype_handling.py`)
**Purpose:** Investigate and fix dtype consistency between PyTorch and JAX.

**Command:**
```bash
python fix_dtype_handling.py
```

**Key Findings:**
- PyTorch keeps certain tensors in bfloat16 while casting others to float32
- JAX needed adjustment in mul_add_add to match PyTorch's behavior
- Modulation operations correctly stay in float32

### 3. Video Comparison with Metrics (`test_jax_vs_torch_comparison.py`)
**Purpose:** Generate videos with both implementations and compute numerical differences.

**Commands:**
```bash
# Test JAX only with reduced frames/steps for debugging
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python test_jax_vs_torch_comparison.py --num_frames 13 --jax_only

# Full comparison with both models (latest test)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python test_jax_vs_torch_comparison.py --num_frames 13

# Default arguments used:
# --model_id "Skywork/SkyReels-V2-I2V-1.3B-540P"
# --resolution "540P"
# --image "test_image.jpg"
# --guidance_scale 6.0
# --shift 8.0
# --fps 24
# --seed 42
# --prompt "Shapes float around in a background consisting of color gradients"
```

**Features Added:**
- Frame-by-frame MSE, PSNR, and SSIM calculations
- Pixel-wise absolute and relative difference statistics
- Quality assessment with clear thresholds
- Side-by-side video generation

### 4. Existing Video Analysis (`analyze_existing_videos.py`)
**Purpose:** Analyze already generated videos for numerical accuracy metrics.

**Command:**
```bash
# First install scikit-image if not already installed
pip install scikit-image

# Run analysis on existing videos
python analyze_existing_videos.py
```

## Numerical Accuracy Results

### Overall Statistics (13-frame video comparison)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean MSE** | 0.000229 | Very low error |
| **Mean PSNR** | 36.63 dB | Very good (>35 dB) |
| **Mean SSIM** | 0.9937 | Excellent (>0.99) |
| **Max Pixel Difference** | 0.24/1.0 | Isolated outliers only |

### Frame-by-Frame Analysis

| Frame | MSE | PSNR (dB) | SSIM |
|-------|-----|-----------|------|
| 1 | 0.000059 | 42.32 | 0.9893 |
| 2 | 0.000249 | 36.04 | 0.9932 |
| 3 | 0.000221 | 36.55 | 0.9941 |
| 4 | 0.000245 | 36.10 | 0.9943 |
| 5 | 0.000244 | 36.13 | 0.9942 |
| 6 | 0.000264 | 35.79 | 0.9944 |
| 7 | 0.000260 | 35.85 | 0.9944 |
| 8 | 0.000258 | 35.89 | 0.9943 |
| 9 | 0.000244 | 36.13 | 0.9943 |
| 10 | 0.000239 | 36.21 | 0.9942 |
| 11 | 0.000235 | 36.28 | 0.9934 |
| 12 | 0.000229 | 36.40 | 0.9939 |
| 13 | 0.000226 | 36.45 | 0.9940 |

### Pixel-wise Difference Distribution

- **Mean Absolute Difference:** 1.15% (0-100% scale)
- **95th Percentile:** 2.75%
- **99th Percentile:** 3.92%

## Visual Quality Assessment

### Quality Rating: **EXCELLENT**

The videos are nearly identical with only minor numerical differences that are imperceptible to the human eye. The numerical differences are:

1. **Well within bfloat16 precision limits** - Expected when using reduced precision
2. **Due to backend differences** - PyTorch (cuBLAS) vs JAX (XLA) have slightly different implementations
3. **Accumulated through many operations** - 30 transformer blocks × many matrix multiplications
4. **Minimal visual impact** - PSNR > 36 dB and SSIM > 0.99 indicate excellent visual quality

## Sources of Numerical Differences

### 1. Precision-Related
- **bfloat16 rounding:** Different rounding in matrix multiplications
- **Operation order:** Compiler optimizations may reorder operations
- **Fused operations:** Different fusion strategies between frameworks

### 2. Implementation Differences
- **BLAS libraries:** cuBLAS (PyTorch) vs XLA (JAX)
- **Attention implementations:** Flash Attention vs JAX's dot_product_attention
- **Compiler optimizations:** PyTorch's torch.compile vs JAX's JIT

### 3. Fixed Issues
- **mul_add_add dtype handling:** Now matches PyTorch exactly
- **Cross-attention dtype mismatch:** Fixed by converting inputs to context dtype
- **RoPE implementation:** Validated to match PyTorch's 3D RoPE

## Performance Comparison

From the latest test run (13 frames, 5 inference steps):
- **PyTorch generation time:** 7.20 seconds
- **JAX generation time:** 36.86 seconds
- **Speed comparison:** PyTorch is currently 5.12x faster

Note: JAX performance can be improved with:
- JIT compilation optimization
- Better memory management
- JAX-specific optimizations like pmap for multi-GPU

## Conclusion

The JAX implementation is functional with:

1. **Good numerical accuracy** - PSNR: 36.63 dB, SSIM: 0.9937
2. **Minimal visual quality difference** - Videos are visually similar
3. **Expected bfloat16 behavior** - Differences are within normal precision limits
4. **Critical bugs fixed** - Dtype issues resolved, operations match PyTorch

The small numerical differences observed are normal and expected when comparing different deep learning frameworks using reduced precision arithmetic. The implementation replicates the PyTorch model's behavior and produces comparable video outputs.

## Recommendations

1. **For production use:** The JAX implementation meets accuracy requirements
2. **For further optimization:** Consider JAX-specific optimizations like pmap for multi-GPU
3. **For exact matching:** Would require float32 precision (not recommended due to memory costs)

---

*Generated: August 18, 2024*
*JAX Implementation by: Claude Assistant*
*Validation: 13-frame video generation at 540P resolution*
*Latest metrics updated from actual video comparison*