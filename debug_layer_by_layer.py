#!/usr/bin/env python3
"""
Layer-by-layer comparison of JAX vs PyTorch transformer forward pass.
"""

import os
import sys
import torch
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any

# Enable float64 in JAX for better precision
jax.config.update('jax_enable_x64', True)

# Add paths
sys.path.append('.')
sys.path.append('./jax_implementation')

# Import components
from skyreels_v2_infer.modules import download_model, get_transformer
from jax_implementation.modules import WanModel
from jax_implementation.utils.weight_converter import load_torch_weights, apply_weights_to_model
from jax_implementation.modules.utils import sinusoidal_embedding_1d


def create_matching_inputs(batch_size=1, num_frames=2, height=34, width=60, seed=42):
    """Create identical inputs for both models."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create inputs
    in_dim = 16
    text_dim = 4096
    text_len = 512
    
    # Video input
    x_np = np.random.randn(batch_size, in_dim, num_frames, height, width).astype(np.float32)
    x_torch = torch.from_numpy(x_np).cuda()
    x_jax = jnp.array(x_np)
    
    # Conditional input (for i2v)
    y_np = np.random.randn(batch_size, 20, num_frames, height, width).astype(np.float32)
    y_torch = torch.from_numpy(y_np).cuda()
    y_jax = jnp.array(y_np)
    
    # Time step
    t_np = np.array([500], dtype=np.int32)
    t_torch = torch.from_numpy(t_np).cuda()
    t_jax = jnp.array(t_np)
    
    # Text context
    context_np = np.random.randn(batch_size, text_len, text_dim).astype(np.float32)
    context_torch = torch.from_numpy(context_np).cuda()
    context_jax = jnp.array(context_np)
    
    # CLIP features
    clip_np = np.random.randn(batch_size, 257, 1280).astype(np.float32)
    clip_torch = torch.from_numpy(clip_np).cuda()
    clip_jax = jnp.array(clip_np)
    
    return {
        'x': (x_torch, x_jax),
        'y': (y_torch, y_jax),
        't': (t_torch, t_jax),
        'context': (context_torch, context_jax),
        'clip_fea': (clip_torch, clip_jax)
    }


def compare_tensors(name: str, torch_tensor: torch.Tensor, jax_array: jax.Array, rtol=1e-3, atol=1e-4):
    """Compare PyTorch tensor and JAX array."""
    # Convert to numpy
    torch_np = torch_tensor.detach().cpu().float().numpy()
    jax_np = np.array(jax_array).astype(np.float32)
    
    # Calculate statistics
    abs_diff = np.abs(torch_np - jax_np)
    rel_diff = abs_diff / (np.abs(torch_np) + 1e-8)
    
    print(f"\n{name}:")
    print(f"  Shapes: PyTorch {torch_np.shape}, JAX {jax_np.shape}")
    print(f"  PyTorch - Mean: {torch_np.mean():.6f}, Std: {torch_np.std():.6f}, Range: [{torch_np.min():.6f}, {torch_np.max():.6f}]")
    print(f"  JAX     - Mean: {jax_np.mean():.6f}, Std: {jax_np.std():.6f}, Range: [{jax_np.min():.6f}, {jax_np.max():.6f}]")
    print(f"  Abs diff - Mean: {abs_diff.mean():.6f}, Max: {abs_diff.max():.6f}")
    print(f"  Rel diff - Mean: {rel_diff.mean():.6f}, Max: {rel_diff.max():.6f}")
    
    # Check if close
    is_close = np.allclose(torch_np, jax_np, rtol=rtol, atol=atol)
    if is_close:
        print(f"  ✅ Values match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"  ❌ Values differ beyond tolerance")
        # Find worst mismatches
        worst_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
        print(f"  Worst mismatch at {worst_idx}: PyTorch={torch_np[worst_idx]:.6f}, JAX={jax_np[worst_idx]:.6f}")
    
    return is_close, abs_diff.mean(), abs_diff.max()


def trace_forward_pass():
    """Trace forward pass through both models layer by layer."""
    
    print("=" * 80)
    print("LAYER-BY-LAYER FORWARD PASS COMPARISON")
    print("=" * 80)
    
    # Load models
    model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P"
    model_path = download_model(model_id)
    
    # Load PyTorch model
    print("\nLoading PyTorch model...")
    torch_model = get_transformer(model_path, device="cuda", weight_dtype=torch.bfloat16)
    torch_model.eval()
    
    # Load JAX model
    print("Loading JAX model...")
    jax_weights, config = load_torch_weights(model_path)
    
    # Use config values directly - PyTorch i2v config has in_dim=36 (concatenated)
    jax_model = WanModel(
        model_type=config.get('model_type', 'i2v'),
        patch_size=tuple(config.get('patch_size', [1, 2, 2])),
        text_len=config.get('text_len', 512),
        in_dim=config.get('in_dim', 36),  # Use the actual config value
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
    
    # Create matching inputs
    print("\nCreating matching inputs...")
    inputs = create_matching_inputs()
    
    # Unpack inputs
    x_torch, x_jax = inputs['x']
    y_torch, y_jax = inputs['y']
    t_torch, t_jax = inputs['t']
    context_torch, context_jax = inputs['context']
    clip_torch, clip_jax = inputs['clip_fea']
    
    print("\n" + "=" * 60)
    print("STAGE 1: INPUT CONCATENATION")
    print("=" * 60)
    
    # Concatenate x and y - convert to bfloat16 to match model dtype
    with torch.no_grad():
        x_cat_torch = torch.cat([x_torch, y_torch], dim=1).to(torch.bfloat16)
    x_cat_jax = jnp.concatenate([x_jax, y_jax], axis=1).astype(jnp.bfloat16)
    
    compare_tensors("Concatenated input (x + y)", x_cat_torch, x_cat_jax)
    
    print("\n" + "=" * 60)
    print("STAGE 2: PATCH EMBEDDING")
    print("=" * 60)
    
    # Patch embedding - already in bfloat16
    with torch.no_grad():
        x_patch_torch = torch_model.patch_embedding(x_cat_torch)
    x_patch_jax = jax_model.patch_embedding(x_cat_jax)
    
    compare_tensors("After patch embedding", x_patch_torch, x_patch_jax)
    
    # Reshape for sequence
    with torch.no_grad():
        b, c, *grid_sizes_list = x_patch_torch.shape
        grid_sizes_torch = torch.tensor(grid_sizes_list, dtype=torch.long).cuda()
        x_seq_torch = x_patch_torch.reshape(b, c, -1).transpose(1, 2)
    
    grid_sizes_jax = jnp.array(x_patch_jax.shape[2:], dtype=jnp.int32)
    x_seq_jax = x_patch_jax.reshape(x_patch_jax.shape[0], x_patch_jax.shape[1], -1).transpose(0, 2, 1)
    
    compare_tensors("After reshape to sequence", x_seq_torch, x_seq_jax)
    
    print("\n" + "=" * 60)
    print("STAGE 3: TIME EMBEDDINGS")
    print("=" * 60)
    
    # Time embeddings
    with torch.no_grad():
        # PyTorch time embedding - uses sinusoidal embedding
        from skyreels_v2_infer.modules.transformer import sinusoidal_embedding_1d as torch_sinusoidal
        t_sin_torch = torch_sinusoidal(config['freq_dim'], t_torch).to(torch.bfloat16)
        t_sin_jax = sinusoidal_embedding_1d(config['freq_dim'], t_jax.reshape(-1)).astype(jnp.bfloat16)
        compare_tensors("Sinusoidal embedding", t_sin_torch, t_sin_jax)
        
        # time_embedding: Linear -> SiLU -> Linear
        t_emb = torch_model.time_embedding[0](t_sin_torch)  # First linear
        jax_t_emb = jax_model.time_embedding_1(t_sin_jax)
        compare_tensors("After time_embedding_1", t_emb, jax_t_emb)
        
        t_emb = torch_model.time_embedding[1](t_emb)  # SiLU
        jax_t_emb = jax.nn.silu(jax_t_emb)
        compare_tensors("After SiLU activation", t_emb, jax_t_emb)
        
        t_emb = torch_model.time_embedding[2](t_emb)  # Second linear
        jax_t_emb = jax_model.time_embedding_2(jax_t_emb)
        compare_tensors("After time_embedding_2", t_emb, jax_t_emb)
    
    print("\n" + "=" * 60)
    print("STAGE 4: TEXT EMBEDDINGS")
    print("=" * 60)
    
    # Text embeddings
    with torch.no_grad():
        context_torch_bf16 = context_torch.to(torch.bfloat16)
        context_embed_torch = torch_model.text_embedding[0](context_torch_bf16)  # Linear
        context_embed_torch = torch_model.text_embedding[1](context_embed_torch)  # GELU
        context_embed_torch = torch_model.text_embedding[2](context_embed_torch)  # Linear
    
    context_embed_jax = jax_model.text_embedding_1(context_jax)
    context_embed_jax = jax.nn.gelu(context_embed_jax, approximate=True)
    context_embed_jax = jax_model.text_embedding_2(context_embed_jax)
    
    compare_tensors("Text embeddings", context_embed_torch, context_embed_jax)
    
    print("\n" + "=" * 60)
    print("STAGE 5: CLIP EMBEDDINGS (I2V)")
    print("=" * 60)
    
    # CLIP image embeddings
    with torch.no_grad():
        clip_torch_bf16 = clip_torch.to(torch.bfloat16)
        clip_embed_torch = torch_model.img_emb(clip_torch_bf16)
        # Concatenate with text
        full_context_torch = torch.cat([clip_embed_torch, context_embed_torch], dim=1)
    
    clip_embed_jax = jax_model.img_emb(clip_jax)
    full_context_jax = jnp.concatenate([clip_embed_jax, context_embed_jax], axis=1)
    
    compare_tensors("CLIP embeddings", clip_embed_torch, clip_embed_jax)
    compare_tensors("Full context (CLIP + text)", full_context_torch, full_context_jax)
    
    print("\n" + "=" * 60)
    print("STAGE 6: TRANSFORMER BLOCKS")
    print("=" * 60)
    
    # Process through transformer blocks
    x_torch_current = x_seq_torch.clone()
    x_jax_current = x_seq_jax
    
    # Prepare PyTorch time embeddings for blocks
    with torch.no_grad():
        e0_torch = torch.nn.functional.silu(t_emb)
        e0_torch = torch_model.time_projection[1](e0_torch)  # Linear to 6*dim
        e0_torch = e0_torch.unflatten(1, (6, config['dim']))  # [batch, 6, dim] - Keep this shape!
        
        # Ensure freqs is on the same device
        freqs_torch = torch_model.freqs.to(x_torch_current.device)
    
    # Prepare JAX time embeddings for blocks
    e0_jax = jax.nn.silu(jax_t_emb)
    e0_jax = jax_model.time_projection_1(e0_jax).reshape(1, 6, config['dim'])
    
    num_blocks_to_check = min(5, config['num_layers'])  # Check first 5 blocks
    
    for i in range(num_blocks_to_check):
        print(f"\n--- Block {i} ---")
        
        with torch.no_grad():
            # PyTorch block forward
            x_torch_current = torch_model.blocks[i](
                x_torch_current,
                e0_torch,
                grid_sizes_torch,
                freqs_torch,
                full_context_torch,
                None
            )
        
        # JAX block forward
        x_jax_current = jax_model.blocks[i](
            x_jax_current,
            e0_jax,
            grid_sizes_jax,
            jax_model.freqs,
            full_context_jax,
            None
        )
        
        is_close, mean_diff, max_diff = compare_tensors(f"After block {i}", x_torch_current, x_jax_current)
        
        if not is_close and mean_diff > 0.5:  # Only stop if deviation is very large
            print(f"\n⚠️ Very large deviation detected at block {i}!")
            print(f"Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")
            break
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Set precision for better comparison
    torch.set_printoptions(precision=6, sci_mode=False)
    np.set_printoptions(precision=6, suppress=True)
    
    trace_forward_pass()