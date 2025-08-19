import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from flax import nnx
from functools import partial
import os

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
    Attention implementation with optional memory-efficient mode.
    
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
    # Use memory-efficient attention if enabled
    if USE_MEMORY_EFFICIENT_ATTENTION:
        return memory_efficient_attention(
            q, k, v, causal=causal, dropout_p=dropout_p,
            softmax_scale=softmax_scale, deterministic=deterministic,
            use_flash=False, chunk_size=256  # Use chunked attention with smaller chunks
        )
    
    # Otherwise use standard implementation
    # Check unsupported arguments (outside of JIT-compiled function)
    if dropout_p != 0.0:
        raise ValueError("dropout_p must be 0.0, dropout is not supported in this implementation")
    
    if softmax_scale is not None:
        raise ValueError("softmax_scale must be None, custom scaling is not supported in this implementation")
    
    # Use JAX's optimized dot_product_attention
    if causal:
        return _attention_causal(q, k, v)
    else:
        return _attention_non_causal(q, k, v)


class WanRMSNorm(nnx.Module):
    """RMS normalization layer."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(dim))
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
        """
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        x = x.astype(orig_dtype) * self.weight
        return x
