# SkyReels-V2 JAX Port

This directory contains a JAX implementation of the SkyReels-V2 image-to-video diffusion model using Flax NNX. The implementation ports the main transformer backbone to JAX while keeping the VAE and text encoders in PyTorch for compatibility.

## Overview

The SkyReels-V2 model consists of several components:
- **Transformer Backbone** (WanModel) - Ported to JAX ✅
- **VAE** (WanVAE) - Kept in PyTorch ✅
- **Text Encoder** - Kept in PyTorch ✅
- **Image Encoder** (CLIP) - Kept in PyTorch ✅

## Architecture

### JAX Components

#### 1. **WanModel** (`modules/transformer.py`)
The main transformer backbone supporting both text-to-video (T2V) and image-to-video (I2V) modes.

Key features:
- Multi-head self-attention with RoPE (Rotary Position Embedding)
- Cross-attention for text and image conditioning
- AdaLN (Adaptive Layer Normalization) with time embeddings
- Causal attention support for autoregressive generation

#### 2. **Attention Mechanisms** (`modules/attention.py`)
- Uses JAX's optimized `jax.nn.dot_product_attention`
- Supports causal attention with automatic backend selection
- `WanSelfAttention`: Self-attention with RoPE
- `WanI2VCrossAttention`: Cross-attention for image-to-video
- `WanT2VCrossAttention`: Cross-attention for text-to-video
- `WanRMSNorm` and `WanLayerNorm`: Normalization layers

#### 3. **Utility Functions** (`modules/utils.py`)
- `sinusoidal_embedding_1d`: Time embedding generation
- `rope_params` and `rope_apply`: RoPE implementation
- `mul_add` and `mul_add_add`: Element-wise operations

### Hybrid Pipeline

The `HybridPipeline` class (`utils/pipeline.py`) combines:
- **PyTorch VAE** for encoding/decoding
- **PyTorch Text Encoder** for prompt processing
- **JAX Transformer** for the main diffusion process

## Installation

1. Install JAX dependencies:
```bash
pip install -r jax_implementation/requirements.txt
```

2. Install PyTorch dependencies (from the main repository):
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from jax_implementation.utils import HybridPipeline
from jax_implementation.modules import WanModel
from jax_implementation.utils.weight_converter import load_torch_weights, create_jax_model_from_config

# Load model weights and config
jax_weights, config = load_torch_weights("Skywork/SkyReels-V2-I2V-1.3B-540P")

# Create JAX model
jax_model = create_jax_model_from_config(config)

# Create hybrid pipeline
pipeline = HybridPipeline(
    model_path="/path/to/model",
    jax_model=jax_model,
    device="cuda"
)

# Generate video
video = pipeline.generate(
    prompt="A beautiful landscape with mountains and a lake",
    image=input_image,  # PIL Image
    height=544,
    width=960,
    num_frames=97,
    num_inference_steps=30,
    guidance_scale=6.0
)
```

### Testing

Run the comprehensive testing script:

```bash
python test_jax_port.py \
    --model_id "Skywork/SkyReels-V2-I2V-1.3B-540P" \
    --height 544 \
    --width 960 \
    --num_frames 97 \
    --output_dir ./test_results \
    --num_tests 3
```

This will:
1. Generate videos using both PyTorch and JAX implementations
2. Create side-by-side comparison videos
3. Compute per-layer numerical errors
4. Generate timing comparisons
5. Save detailed results and plots

## Implementation Details

### Weight Conversion

The `weight_converter.py` module handles:
- Downloading PyTorch weights from HuggingFace
- Converting PyTorch tensors to JAX arrays
- Handling data type conversions (bfloat16, float16)
- Transposing weights for JAX conventions

### Key Differences from PyTorch

1. **Attention Implementation**: Uses JAX's optimized `jax.nn.dot_product_attention` with automatic backend selection (XLA, cuDNN Flash Attention)
2. **Memory Layout**: JAX uses different memory layouts for convolutions
3. **Data Types**: Handles bfloat16/float16 conversions explicitly
4. **Gradient Computation**: JAX uses automatic differentiation

### Performance Optimizations

1. **JIT Compilation**: Attention functions are pre-compiled with JIT for optimal performance
2. **Optimized Attention**: Uses JAX's `dot_product_attention` with automatic backend selection (XLA, cuDNN Flash Attention)
3. **Memory Efficiency**: JAX's functional programming model enables better memory management
4. **Parallelization**: JAX's vectorization capabilities can improve throughput

## Testing Results

The testing framework provides:

### 1. **Numerical Accuracy**
- Per-layer maximum absolute errors
- Normalized by mean activation norms
- Comparison plots showing error distributions

### 2. **Performance Metrics**
- Generation time comparisons
- Memory usage analysis
- Speedup measurements

### 3. **Visual Quality**
- Side-by-side video comparisons
- Frame-by-frame analysis
- Quality metrics (PSNR, SSIM)

### 4. **Comprehensive Reports**
- JSON files with detailed results
- Matplotlib plots for error analysis
- Summary statistics

## Limitations and Future Work

### Current Limitations
1. **Weight Mapping**: Complete weight conversion requires careful parameter name mapping
2. **CLIP Integration**: CLIP image encoding needs additional implementation
3. **TeaCache**: Advanced optimization features not yet implemented
4. **Distributed Training**: Multi-GPU support needs additional work

### Future Improvements
1. **Complete Weight Conversion**: Implement full parameter mapping
2. **Performance Optimization**: Add JIT compilation and optimization
3. **Memory Optimization**: Implement gradient checkpointing
4. **Training Support**: Add training loop and optimization
5. **Advanced Features**: Implement TeaCache and other optimizations

## Troubleshooting

### Common Issues

1. **JAX Installation**: Ensure JAX is installed with the correct CUDA version
2. **Memory Issues**: Reduce batch size or use gradient checkpointing
3. **Weight Loading**: Check that model paths are correct
4. **Device Mismatch**: Ensure PyTorch and JAX use compatible devices

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

This implementation follows the same license as the original SkyReels-V2 repository.

## Acknowledgments

- Original SkyReels-V2 implementation by SkyworkAI
- JAX and Flax development teams
- HuggingFace for model hosting
