import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from flax import nnx
from functools import partial
import os

# Enable float64 types for higher precision - this only works on startup!
jax.config.update("jax_enable_x64", True)

# Import memory-efficient attention if enabled
USE_MEMORY_EFFICIENT_ATTENTION = os.environ.get("JAX_MEMORY_EFFICIENT_ATTENTION", "true").lower() == "true"
if USE_MEMORY_EFFICIENT_ATTENTION:
    from .attention_memory_efficient import memory_efficient_attention


@partial(jax.jit, static_argnames=['is_causal'])
def attention_wrapper(query, key, value, is_causal):
    """JIT-compiled wrapper for JAX's dot_product_attention."""
    # JAX's dot_product_attention requires float32, so convert if needed
    orig_dtype = query.dtype
    if orig_dtype == jnp.bfloat16:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)
        value = value.astype(jnp.float32)
    
    result = jax.nn.dot_product_attention(query, key, value, is_causal=is_causal)
    
    # Convert back to original dtype
    if orig_dtype == jnp.bfloat16:
        result = result.astype(jnp.bfloat16)
    
    return result


@jax.jit
def _attention_non_causal(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """JIT-compiled non-causal attention."""
    return attention_wrapper(q, k, v, is_causal=False)


@jax.jit
def _attention_causal(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """JIT-compiled causal attention."""
    return attention_wrapper(q, k, v, is_causal=True)


def attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    deterministic: bool = True,
) -> jax.Array:
    """
    Attention implementation matching PyTorch's flash attention behavior.
    Uses bfloat16 by default like PyTorch's flash attention implementation.
    
    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        causal: Whether to apply causal masking
        dropout_p: Dropout probability (must be 0.0, not supported)
        softmax_scale: Scaling factor for attention scores (must be None, not supported)
        deterministic: Whether to use deterministic dropout (ignored)
        
    Returns:
        Attention output of shape [batch, seq_len, num_heads, head_dim]
    """
    # Store original dtype
    orig_dtype = q.dtype
    
    # Convert to bfloat16 to match PyTorch's flash attention default behavior
    # PyTorch flash attention typically uses bfloat16 for efficiency
    if orig_dtype != jnp.bfloat16:
        q = q.astype(jnp.bfloat16)
        k = k.astype(jnp.bfloat16)
        v = v.astype(jnp.bfloat16)
    
    # Use memory-efficient attention if enabled
    if USE_MEMORY_EFFICIENT_ATTENTION:
        result = memory_efficient_attention(
            q, k, v, causal=causal, dropout_p=dropout_p,
            softmax_scale=softmax_scale, deterministic=deterministic,
            use_flash=False, chunk_size=256  # Use chunked attention with smaller chunks
        )
    else:
        # Otherwise use standard implementation
        # Check unsupported arguments (outside of JIT-compiled function)
        if dropout_p != 0.0:
            raise ValueError("dropout_p must be 0.0, dropout is not supported in this implementation")
        
        if softmax_scale is not None:
            raise ValueError("softmax_scale must be None, custom scaling is not supported in this implementation")
        
        # Use JAX's optimized dot_product_attention
        if causal:
            result = _attention_causal(q, k, v)
        else:
            result = _attention_non_causal(q, k, v)
    
    # Convert back to original dtype if needed
    if result.dtype != orig_dtype:
        result = result.astype(orig_dtype)
    
    return result


class WanRMSNorm(nnx.Module):
    """RMS normalization layer matching PyTorch's fast_rms_norm behavior."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(dim))
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
        
        Matches PyTorch's fast_rms_norm implementation:
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        x = x.type_as(x) * weight
        """
        orig_dtype = x.dtype
        
        # Force float32 calculation to match PyTorch's fast_rms_norm
        x_f32 = x.astype(jnp.float32)
        
        # Compute RMS normalization in float32
        variance = jnp.mean(x_f32**2, axis=-1, keepdims=True)
        x_normalized = x_f32 * jax.lax.rsqrt(variance + self.eps)
        
        # Convert back to original dtype and apply weight scaling
        # This matches PyTorch's "x.type_as(x) * weight" behavior
        x_result = x_normalized.astype(orig_dtype) * self.weight.astype(orig_dtype)
        
        return x_result
