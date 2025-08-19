#!/usr/bin/env python3
"""
Debug Conv3d precision issues with actual loaded weights.
"""

import torch
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

sys.path.append('.')
sys.path.append('./jax_implementation')

from skyreels_v2_infer.modules import download_model, get_transformer
from jax_implementation.modules.transformer import WanModel
from jax_implementation.utils.weight_converter import load_torch_weights, apply_weights_to_model


def test_conv3d_precision():
    """Test Conv3d with actual weights and different dtype strategies."""
    print("="*70)
    print("CONV3D PRECISION TEST WITH ACTUAL WEIGHTS")
    print("="*70)
    
    # Load model
    model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P"
    model_path = download_model(model_id)
    
    # Load weights and config
    torch_weights, config = load_torch_weights(model_path)
    
    # Create models
    torch_model = get_transformer(model_path, device="cuda", weight_dtype=torch.bfloat16)
    torch_model.eval()
    
    jax_model = WanModel(
        model_type=config.get('model_type', 'i2v'),
        patch_size=tuple(config.get('patch_size', [1, 2, 2])),
        text_len=config.get('text_len', 512),
        in_dim=config.get('in_dim', 36),
        dim=config.get('dim', 1536),
        ffn_dim=config.get('ffn_dim', 8960),
        freq_dim=config.get('freq_dim', 256),
        text_dim=config.get('text_dim', 4096),
        out_dim=config.get('out_dim', 16),
        num_heads=config.get('num_heads', 12),
        num_layers=config.get('num_layers', 30),
        window_size=tuple(config.get('window_size', [-1, -1])),
        qk_norm=config.get('qk_norm', True),
        cross_attn_norm=config.get('cross_attn_norm', True),
        inject_sample_info=config.get('inject_sample_info', False),
        eps=config.get('eps', 1e-6),
    )
    jax_model = apply_weights_to_model(jax_model, torch_weights)
    
    # Create test input
    batch_size = 1
    num_frames = 2
    height, width = 34, 60
    
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    x_np = np.random.randn(batch_size, 16, num_frames, height, width).astype(np.float32)
    y_np = np.random.randn(batch_size, 20, num_frames, height, width).astype(np.float32)
    concat_np = np.concatenate([x_np, y_np], axis=1)
    
    print(f"\nInput shape: {concat_np.shape}")
    print(f"Input dtype: {concat_np.dtype}")
    print(f"Input range: [{concat_np.min():.6f}, {concat_np.max():.6f}]")
    
    # Test 1: PyTorch with bfloat16
    print("\n" + "-"*50)
    print("TEST 1: PyTorch with bfloat16")
    with torch.no_grad():
        torch_input_bf16 = torch.from_numpy(concat_np).cuda().to(torch.bfloat16)
        torch_output_bf16 = torch_model.patch_embedding(torch_input_bf16)
        torch_output_bf16_np = torch_output_bf16.cpu().float().numpy()
    
    print(f"Output shape: {torch_output_bf16_np.shape}")
    print(f"Output range: [{torch_output_bf16_np.min():.6f}, {torch_output_bf16_np.max():.6f}]")
    print(f"Output mean: {torch_output_bf16_np.mean():.6f}, std: {torch_output_bf16_np.std():.6f}")
    
    # Test 2: PyTorch with float32
    print("\n" + "-"*50)
    print("TEST 2: PyTorch with float32")
    with torch.no_grad():
        # Convert weights to float32 temporarily
        torch_model_f32 = torch_model.patch_embedding.float()
        torch_input_f32 = torch.from_numpy(concat_np).cuda().float()
        torch_output_f32 = torch_model_f32(torch_input_f32)
        torch_output_f32_np = torch_output_f32.cpu().numpy()
        # Convert back
        torch_model.patch_embedding.to(torch.bfloat16)
    
    print(f"Output shape: {torch_output_f32_np.shape}")
    print(f"Output range: [{torch_output_f32_np.min():.6f}, {torch_output_f32_np.max():.6f}]")
    print(f"Output mean: {torch_output_f32_np.mean():.6f}, std: {torch_output_f32_np.std():.6f}")
    
    # Compare PyTorch bf16 vs f32
    diff_torch = np.abs(torch_output_bf16_np - torch_output_f32_np)
    print(f"\nPyTorch bf16 vs f32 difference:")
    print(f"  Max: {diff_torch.max():.8f}, Mean: {diff_torch.mean():.8f}")
    
    # Test 3: JAX with input as float32, weights as loaded
    print("\n" + "-"*50)
    print("TEST 3: JAX with mixed precision (input f32, weights as loaded)")
    jax_input_f32 = jnp.array(concat_np, dtype=jnp.float32)
    jax_output_mixed = jax_model.patch_embedding(jax_input_f32)
    jax_output_mixed_np = np.array(jax_output_mixed).astype(np.float32)
    
    print(f"Output shape: {jax_output_mixed_np.shape}")
    print(f"Output dtype: {jax_output_mixed.dtype}")
    print(f"Output range: [{jax_output_mixed_np.min():.6f}, {jax_output_mixed_np.max():.6f}]")
    print(f"Output mean: {jax_output_mixed_np.mean():.6f}, std: {jax_output_mixed_np.std():.6f}")
    
    # Test 4: JAX with input as bfloat16
    print("\n" + "-"*50)
    print("TEST 4: JAX with bfloat16 input")
    jax_input_bf16 = jnp.array(concat_np, dtype=jnp.bfloat16)
    jax_output_bf16 = jax_model.patch_embedding(jax_input_bf16)
    jax_output_bf16_np = np.array(jax_output_bf16).astype(np.float32)
    
    print(f"Output shape: {jax_output_bf16_np.shape}")
    print(f"Output dtype: {jax_output_bf16.dtype}")
    print(f"Output range: [{jax_output_bf16_np.min():.6f}, {jax_output_bf16_np.max():.6f}]")
    print(f"Output mean: {jax_output_bf16_np.mean():.6f}, std: {jax_output_bf16_np.std():.6f}")
    
    # Comparisons
    print("\n" + "="*50)
    print("COMPARISONS")
    print("="*50)
    
    # PyTorch bf16 vs JAX bf16
    diff = np.abs(torch_output_bf16_np - jax_output_bf16_np)
    rel_diff = diff / (np.abs(torch_output_bf16_np) + 1e-8)
    print(f"\nPyTorch bf16 vs JAX bf16:")
    print(f"  Abs diff - Max: {diff.max():.8f}, Mean: {diff.mean():.8f}")
    print(f"  Rel diff - Max: {rel_diff.max():.8f}, Mean: {rel_diff.mean():.8f}")
    
    # PyTorch f32 vs JAX f32
    diff2 = np.abs(torch_output_f32_np - jax_output_mixed_np)
    rel_diff2 = diff2 / (np.abs(torch_output_f32_np) + 1e-8)
    print(f"\nPyTorch f32 vs JAX mixed precision:")
    print(f"  Abs diff - Max: {diff2.max():.8f}, Mean: {diff2.mean():.8f}")
    print(f"  Rel diff - Max: {rel_diff2.max():.8f}, Mean: {rel_diff2.mean():.8f}")
    
    # Check weight dtypes
    print("\n" + "-"*50)
    print("WEIGHT DTYPES")
    torch_w_dtype = torch_model.patch_embedding.weight.dtype
    torch_b_dtype = torch_model.patch_embedding.bias.dtype
    jax_w_dtype = jax_model.patch_embedding.kernel.value.dtype
    jax_b_dtype = jax_model.patch_embedding.bias.value.dtype
    
    print(f"PyTorch - weight: {torch_w_dtype}, bias: {torch_b_dtype}")
    print(f"JAX - kernel: {jax_w_dtype}, bias: {jax_b_dtype}")
    
    # Check if the issue is accumulation precision
    print("\n" + "-"*50)
    print("ACCUMULATION PRECISION TEST")
    
    # Get a single output channel to analyze
    torch_single = torch_output_bf16_np[0, 0, 0, 0, 0]
    jax_single = jax_output_bf16_np[0, 0, 0, 0, 0]
    print(f"Single output value - PyTorch: {torch_single:.8f}, JAX: {jax_single:.8f}")
    print(f"Difference: {abs(torch_single - jax_single):.8f}")
    
    return torch_output_bf16_np, jax_output_bf16_np


if __name__ == "__main__":
    test_conv3d_precision()