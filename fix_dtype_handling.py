#!/usr/bin/env python3
"""
Fix dtype handling in JAX implementation to match PyTorch exactly.
"""

import os
import sys
import torch
import jax
import jax.numpy as jnp
import numpy as np

# Enable float64 in JAX
jax.config.update('jax_enable_x64', True)

sys.path.append('.')
sys.path.append('./jax_implementation')

from skyreels_v2_infer.modules import download_model, get_transformer
from jax_implementation.modules import WanModel
from jax_implementation.utils.weight_converter import load_torch_weights, apply_weights_to_model


def test_dtype_ops():
    """Test various dtype operations to understand differences."""
    
    print("="*60)
    print("TESTING DTYPE OPERATIONS")
    print("="*60)
    
    # Test data
    np.random.seed(42)
    x_np = np.random.randn(2, 3).astype(np.float32) * 0.1
    y_np = np.random.randn(2, 3).astype(np.float32) * 0.01
    z_np = np.random.randn(2, 3).astype(np.float32) * 0.01
    
    # Convert to bfloat16
    x_torch = torch.from_numpy(x_np).to(torch.bfloat16)
    y_torch = torch.from_numpy(y_np).to(torch.bfloat16)  
    z_torch = torch.from_numpy(z_np).to(torch.bfloat16)
    
    x_jax = jnp.array(x_np, dtype=jnp.bfloat16)
    y_jax = jnp.array(y_np, dtype=jnp.bfloat16)
    z_jax = jnp.array(z_np, dtype=jnp.bfloat16)
    
    print("\n1. Test mul_add_add: x * (1 + y) + z")
    print("-" * 40)
    
    # PyTorch version (as in transformer.py)
    result_torch = x_torch.float() * (1 + y_torch) + z_torch
    print(f"PyTorch: dtype sequence")
    print(f"  x.float(): {x_torch.float().dtype}")
    print(f"  (1 + y): {(1 + y_torch).dtype}")
    print(f"  z: {z_torch.dtype}")
    print(f"  result: {result_torch.dtype}")
    
    # JAX version (current)
    result_jax = x_jax.astype(jnp.float32) * (1 + y_jax) + z_jax
    print(f"\nJAX: dtype sequence")
    print(f"  x.astype(float32): {x_jax.astype(jnp.float32).dtype}")
    print(f"  (1 + y): {(1 + y_jax).dtype}")
    print(f"  z: {z_jax.dtype}")
    print(f"  result: {result_jax.dtype}")
    
    # Compare results
    torch_vals = result_torch.cpu().float().numpy()
    jax_vals = np.array(result_jax).astype(np.float32)
    diff = np.abs(torch_vals - jax_vals)
    
    print(f"\nComparison:")
    print(f"  PyTorch result: {torch_vals}")
    print(f"  JAX result: {jax_vals}")
    print(f"  Max diff: {diff.max():.10f}")
    print(f"  Mean diff: {diff.mean():.10f}")
    
    print("\n2. Test alternative JAX implementations")
    print("-" * 40)
    
    # Alternative 1: Cast all to float32
    result_jax_alt1 = x_jax.astype(jnp.float32) * (1 + y_jax.astype(jnp.float32)) + z_jax.astype(jnp.float32)
    print(f"Alt1 (all float32): max diff = {np.abs(torch_vals - np.array(result_jax_alt1)).max():.10f}")
    
    # Alternative 2: Keep y and z in bfloat16
    result_jax_alt2 = x_jax.astype(jnp.float32) * (1 + y_jax) + z_jax
    print(f"Alt2 (y,z bfloat16): max diff = {np.abs(torch_vals - np.array(result_jax_alt2)).max():.10f}")
    
    # Alternative 3: Cast intermediate result
    temp = (1 + y_jax).astype(jnp.float32)
    result_jax_alt3 = x_jax.astype(jnp.float32) * temp + z_jax.astype(jnp.float32)
    print(f"Alt3 (cast after add): max diff = {np.abs(torch_vals - np.array(result_jax_alt3)).max():.10f}")
    
    print("\n3. Test LayerNorm behavior")
    print("-" * 40)
    
    # Large input to test normalization
    x_large = np.random.randn(10, 100).astype(np.float32)
    x_torch_ln = torch.from_numpy(x_large).to(torch.bfloat16)
    x_jax_ln = jnp.array(x_large, dtype=jnp.bfloat16)
    
    # PyTorch LayerNorm
    ln_torch = torch.nn.LayerNorm(100, eps=1e-6, elementwise_affine=False)
    with torch.no_grad():
        out_torch_ln = ln_torch(x_torch_ln)
    
    # JAX LayerNorm equivalent
    from flax import nnx
    ln_jax = nnx.LayerNorm(100, epsilon=1e-6, use_bias=False, use_scale=False, rngs=nnx.Rngs(0))
    out_jax_ln = ln_jax(x_jax_ln)
    
    torch_ln_vals = out_torch_ln.cpu().float().numpy()
    jax_ln_vals = np.array(out_jax_ln).astype(np.float32)
    ln_diff = np.abs(torch_ln_vals - jax_ln_vals)
    
    print(f"LayerNorm comparison:")
    print(f"  Max diff: {ln_diff.max():.10f}")
    print(f"  Mean diff: {ln_diff.mean():.10f}")
    print(f"  PyTorch output dtype: {out_torch_ln.dtype}")
    print(f"  JAX output dtype: {out_jax_ln.dtype}")


def test_block_modulation():
    """Test modulation operations in transformer blocks."""
    
    print("\n" + "="*60)
    print("TESTING BLOCK MODULATION")
    print("="*60)
    
    # Load models
    model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P"
    model_path = download_model(model_id)
    
    torch_model = get_transformer(model_path, device="cuda", weight_dtype=torch.bfloat16)
    torch_model.eval()
    
    jax_weights, config = load_torch_weights(model_path)
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
    jax_model = apply_weights_to_model(jax_model, jax_weights)
    
    # Test modulation in first block
    torch_block = torch_model.blocks[0]
    jax_block = jax_model.blocks[0]
    
    # Create test inputs
    np.random.seed(42)
    e0_np = np.random.randn(1, 6, config['dim']).astype(np.float32) * 0.01
    
    # PyTorch keeps e0 in float32
    e0_torch = torch.from_numpy(e0_np).cuda().to(torch.float32)
    e0_jax = jnp.array(e0_np, dtype=jnp.float32)
    
    print("\n1. Modulation + e0 operation")
    print("-" * 40)
    
    with torch.no_grad():
        # PyTorch: uses autocast to float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            mod_torch = torch_block.modulation
            e_combined_torch = mod_torch + e0_torch
            print(f"PyTorch modulation dtype: {mod_torch.dtype}")
            print(f"PyTorch e0 dtype: {e0_torch.dtype}")
            print(f"PyTorch combined dtype: {e_combined_torch.dtype}")
    
    # JAX version
    mod_jax = jax_block.modulation.value
    e_combined_jax = mod_jax + e0_jax
    print(f"\nJAX modulation dtype: {mod_jax.dtype}")
    print(f"JAX e0 dtype: {e0_jax.dtype}")
    print(f"JAX combined dtype: {e_combined_jax.dtype}")
    
    # Compare
    torch_vals = e_combined_torch.cpu().float().numpy()
    jax_vals = np.array(e_combined_jax).astype(np.float32)
    diff = np.abs(torch_vals - jax_vals)
    
    print(f"\nComparison:")
    print(f"  Max diff: {diff.max():.10f}")
    print(f"  Mean diff: {diff.mean():.10f}")
    
    # Check if dtypes need adjustment
    if e_combined_jax.dtype != jnp.float32:
        print(f"\n⚠️ JAX combined dtype is {e_combined_jax.dtype}, should be float32")
    
    if mod_jax.dtype != jnp.bfloat16:
        print(f"\n⚠️ JAX modulation dtype is {mod_jax.dtype}, should be bfloat16")


if __name__ == "__main__":
    test_dtype_ops()
    test_block_modulation()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
Based on the tests above, the main dtype handling differences are:

1. PyTorch's mul_add_add keeps y and z in their original dtype (bfloat16)
   while only casting x to float32
   
2. The modulation operations should stay in float32 throughout

3. LayerNorm has minimal differences, likely acceptable

The fix is to ensure mul_add_add matches PyTorch's dtype handling exactly.
""")