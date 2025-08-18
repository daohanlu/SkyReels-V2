# SkyReels-V2 PyTorch vs JAX Test Scripts

This directory contains test scripts to compare the PyTorch and JAX implementations of the SkyReels-V2 model.

## Overview

The test scripts are designed to:
1. Generate videos using the original PyTorch implementation
2. Generate videos using the JAX implementation (when available)
3. Create side-by-side comparison videos
4. Provide detailed error reporting and debugging information

## Test Scripts

### 1. `test_jax_loading.py` - JAX Implementation Testing

This script tests the JAX implementation loading and identifies any issues with:
- JAX module imports
- Weight loading from PyTorch models
- JAX model creation
- Weight application
- Forward pass (optional)

**Usage:**
```bash
# Basic JAX loading test
python test_jax_loading.py --model_id "Skywork/SkyReels-V2-I2V-1.3B-540P"

# Test with forward pass
python test_jax_loading.py --model_id "Skywork/SkyReels-V2-I2V-1.3B-540P" --test_forward
```

### 2. `test_simple_torch_vs_jax.py` - Simple Comparison Test

This script provides a simplified comparison with better error handling:
- Tests JAX implementation availability first
- Generates PyTorch video
- Attempts to generate JAX video (if available)
- Creates comparison video
- Provides fallback options if JAX fails

**Usage:**
```bash
# Basic comparison test
python test_simple_torch_vs_jax.py \
  --model_id "Skywork/SkyReels-V2-I2V-1.3B-540P" \
  --resolution 540P \
  --num_frames 97 \
  --guidance_scale 6.0 \
  --shift 8.0 \
  --fps 24 \
  --prompt "A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface." \
  --image test_image.jpg \
  --guidance_scale 5.0 \
  --shift 3.0

# With optimization flags
python test_simple_torch_vs_jax.py \
  --model_id "Skywork/SkyReels-V2-I2V-1.3B-540P" \
  --resolution 540P \
  --num_frames 97 \
  --guidance_scale 6.0 \
  --shift 8.0 \
  --fps 24 \
  --prompt "A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface." \
  --image test_image.jpg \
  --offload \
  --teacache \
  --use_ret_steps \
  --teacache_thresh 0.3 \
  --save_individual

# Test JAX loading only
python test_simple_torch_vs_jax.py \
  --model_id "Skywork/SkyReels-V2-I2V-1.3B-540P" \
  --test_jax_only
```

### 3. `test_torch_vs_jax.py` - Full Comparison Test

This script provides a comprehensive comparison with all features:
- Full PyTorch vs JAX comparison
- Side-by-side video generation
- Individual video saving
- Detailed error reporting

**Usage:**
```bash
# Full comparison test
python test_torch_vs_jax.py \
  --model_id "Skywork/SkyReels-V2-I2V-1.3B-540P" \
  --resolution 540P \
  --num_frames 97 \
  --guidance_scale 6.0 \
  --shift 8.0 \
  --fps 24 \
  --prompt "A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface." \
  --image test_image.jpg \
  --offload \
  --teacache \
  --use_ret_steps \
  --teacache_thresh 0.3 \
  --save_individual
```

## Arguments

### Model Arguments
- `--model_id`: HuggingFace model ID (default: "Skywork/SkyReels-V2-I2V-1.3B-540P")
- `--resolution`: Video resolution ("540P" or "720P", default: "540P")

### Generation Arguments
- `--num_frames`: Number of frames to generate (default: 97)
- `--guidance_scale`: Classifier-free guidance scale (default: 6.0)
- `--shift`: Scheduler shift parameter (default: 8.0)
- `--fps`: Output video FPS (default: 24)
- `--prompt`: Text prompt for video generation
- `--image`: Input image path for image-to-video generation
- `--num_inference_steps`: Number of denoising steps (default: 30)

### Optimization Arguments
- `--offload`: Enable model offloading to CPU
- `--teacache`: Enable TeaCache optimization
- `--teacache_thresh`: TeaCache threshold (default: 0.3)
- `--use_ret_steps`: Use retention steps for faster generation

### Output Arguments
- `--output_dir`: Output directory (default: "test_results")
- `--save_individual`: Save individual PyTorch and JAX videos
- `--seed`: Random seed for reproducible generation
- `--test_jax_only`: Only test JAX implementation loading

## Expected Output

The scripts will generate:

1. **Comparison Video**: Side-by-side video showing PyTorch vs JAX results
2. **Individual Videos**: Separate PyTorch and JAX videos (if `--save_individual` is used)
3. **Console Output**: Detailed progress and error information

### Output Files
- `comparison_<prompt>_<seed>_<timestamp>.mp4`: Side-by-side comparison
- `pytorch_<prompt>_<seed>.mp4`: PyTorch-only video
- `jax_<prompt>_<seed>.mp4`: JAX-only video
- `pytorch_only_<prompt>_<seed>_<timestamp>.mp4`: PyTorch-only (if JAX fails)

## Troubleshooting

### Common Issues

1. **JAX Import Errors**
   - Ensure JAX is properly installed: `pip install jax jaxlib`
   - Check CUDA compatibility for JAX

2. **Weight Loading Errors**
   - Verify model ID is correct
   - Check internet connection for model download
   - Ensure sufficient disk space

3. **Memory Issues**
   - Use `--offload` flag to reduce GPU memory usage
   - Reduce `--num_frames` or `--num_inference_steps`
   - Use lower resolution

4. **Shape Mismatch Errors**
   - Check that input image dimensions match expected resolution
   - Verify model configuration matches weights

### Debugging Steps

1. **Test JAX Loading First**:
   ```bash
   python test_jax_loading.py --model_id "Skywork/SkyReels-V2-I2V-1.3B-540P"
   ```

2. **Test PyTorch Only**:
   ```bash
   python test_simple_torch_vs_jax.py \
     --model_id "Skywork/SkyReels-V2-I2V-1.3B-540P" \
     --prompt "Test prompt"
   ```

3. **Check Dependencies**:
   ```bash
   pip install -r jax_implementation/requirements.txt
   ```

## Performance Notes

- **PyTorch**: Generally more stable and feature-complete
- **JAX**: May provide performance benefits but requires more setup
- **Memory Usage**: JAX implementation may use different memory patterns
- **Speed**: JAX may be faster for inference but slower for setup

## Limitations

- JAX implementation is experimental and may have bugs
- Some advanced features (TeaCache, etc.) may not work in JAX
- Weight conversion may not be 100% accurate
- CLIP integration in JAX pipeline is incomplete

## Contributing

To improve the JAX implementation:

1. Fix issues identified by `test_jax_loading.py`
2. Improve weight conversion in `weight_converter.py`
3. Enhance the hybrid pipeline in `pipeline.py`
4. Add missing features like CLIP integration
5. Optimize performance and memory usage

