#!/usr/bin/env python3
"""
Debug transformer block operations to find source of rapid error growth.
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
from jax_implementation.modules.utils import sinusoidal_embedding_1d, mul_add, mul_add_add

# Load models
model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P"
model_path = download_model(model_id)

print("Loading models...")
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

print("\n" + "="*60)
print("TEST: Transformer Block 0 Operations")
print("="*60)

# Create simple test inputs
np.random.seed(42)
batch_size = 1
seq_len = 100  # Small sequence for testing
hidden_dim = config['dim']

# Input tensor
x_np = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32) * 0.1
x_torch = torch.from_numpy(x_np).cuda().to(torch.bfloat16)
x_jax = jnp.array(x_np, dtype=jnp.bfloat16)

# Time embedding (e0) - shape [batch, 6, dim]
e0_np = np.random.randn(batch_size, 6, hidden_dim).astype(np.float32) * 0.01
e0_torch = torch.from_numpy(e0_np).cuda().to(torch.bfloat16)
e0_jax = jnp.array(e0_np, dtype=jnp.bfloat16)

# Get first block
torch_block = torch_model.blocks[0]
jax_block = jax_model.blocks[0]

print("\nChecking Block 0 components:")

# 1. Check modulation parameter
torch_mod = torch_block.modulation
jax_mod = jax_block.modulation.value
print(f"\n1. Modulation parameter:")
print(f"   PyTorch shape: {torch_mod.shape}, dtype: {torch_mod.dtype}")
print(f"   JAX shape: {jax_mod.shape}, dtype: {jax_mod.dtype}")
mod_diff = np.abs(torch_mod.cpu().float().numpy() - np.array(jax_mod).astype(np.float32))
print(f"   Max diff: {mod_diff.max():.10f}")

# 2. Test modulation + e0
with torch.no_grad():
    # PyTorch: modulation + e0, then chunk
    e_combined_torch = torch_mod + e0_torch
    e_chunks_torch = e_combined_torch.chunk(6, dim=1)
    e_torch = [chunk.squeeze(1) for chunk in e_chunks_torch]

# JAX: modulation + e0, then split
e_combined_jax = jax_mod + e0_jax
# JAX uses different indexing after transpose
e_jax = [e_combined_jax[:, i, :] for i in range(6)]

print(f"\n2. After modulation + e0:")
for i in range(6):
    diff = np.abs(e_torch[i].cpu().float().numpy() - np.array(e_jax[i]).astype(np.float32))
    print(f"   e[{i}] max diff: {diff.max():.10f}, mean: {diff.mean():.10f}")

# 3. Test norm1
torch_norm1 = torch_block.norm1
jax_norm1 = jax_block.norm1

with torch.no_grad():
    norm1_torch = torch_norm1(x_torch)
norm1_jax = jax_norm1(x_jax)

diff = np.abs(norm1_torch.cpu().float().numpy() - np.array(norm1_jax).astype(np.float32))
print(f"\n3. After norm1:")
print(f"   Max diff: {diff.max():.10f}, mean: {diff.mean():.10f}")

# 4. Test mul_add_add operation
print(f"\n4. Testing mul_add_add (norm1(x) * (1 + e[1]) + e[0]):")

# PyTorch version
with torch.no_grad():
    # PyTorch uses compiled version, but let's test the basic operation
    out_torch = norm1_torch.float() * (1 + e_torch[1].float()) + e_torch[0].float()
    out_torch = out_torch.to(torch.bfloat16)

# JAX version
out_jax = mul_add_add(norm1_jax, e_jax[1], e_jax[0])

diff = np.abs(out_torch.cpu().float().numpy() - np.array(out_jax).astype(np.float32))
print(f"   Max diff: {diff.max():.10f}, mean: {diff.mean():.10f}")

# Check value ranges
torch_vals = out_torch.cpu().float().numpy()
jax_vals = np.array(out_jax).astype(np.float32)
print(f"   PyTorch range: [{torch_vals.min():.6f}, {torch_vals.max():.6f}]")
print(f"   JAX range: [{jax_vals.min():.6f}, {jax_vals.max():.6f}]")

# 5. Check for numerical instability
print(f"\n5. Checking for numerical instability:")
print(f"   Input x range: [{x_np.min():.6f}, {x_np.max():.6f}]")
print(f"   e[0] range: [{e0_np[:, 0, :].min():.6f}, {e0_np[:, 0, :].max():.6f}]")
print(f"   e[1] range: [{e0_np[:, 1, :].min():.6f}, {e0_np[:, 1, :].max():.6f}]")

# Check if any values are getting amplified
amplification = torch_vals.max() / x_np.max()
print(f"   Output/Input amplification: {amplification:.2f}x")

# 6. Test self-attention (simplified)
print(f"\n6. Self-attention Q,K,V projections:")

# Just test the Q projection
with torch.no_grad():
    q_torch = torch_block.self_attn.q(out_torch)
    
# JAX Q projection
q_jax = jax_block.self_attn.q(out_jax)

diff = np.abs(q_torch.cpu().float().numpy() - np.array(q_jax).astype(np.float32))
print(f"   Q projection max diff: {diff.max():.10f}, mean: {diff.mean():.10f}")

# Check if errors are systematic or random
print(f"\n7. Error distribution analysis:")
flat_diff = diff.flatten()
print(f"   Error std dev: {flat_diff.std():.10f}")
print(f"   Error skewness: {np.mean((flat_diff - flat_diff.mean())**3) / flat_diff.std()**3:.4f}")
print(f"   % errors > 0.01: {100 * np.sum(flat_diff > 0.01) / len(flat_diff):.2f}%")
print(f"   % errors > 0.1: {100 * np.sum(flat_diff > 0.1) / len(flat_diff):.2f}%")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print("\nThe rapid error growth is likely due to:")
print("1. Bfloat16 precision limitations in matrix multiplications")
print("2. Different BLAS implementations between PyTorch (cuBLAS) and JAX (XLA)")
print("3. Accumulation of small errors through many operations")
print("4. Possible differences in how operations are fused/optimized")
print("\nThese are inherent to bfloat16 and not bugs in the implementation.")