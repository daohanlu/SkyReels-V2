import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from flax import nnx
from functools import partial


@partial(jax.jit, static_argnames=['is_causal'])
def attention_wrapper(query, key, value, is_causal):
    """JIT-compiled wrapper for JAX's dot_product_attention."""
    return jax.nn.dot_product_attention(query, key, value, is_causal=is_causal)


@jax.jit
def _attention_non_causal(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """JIT-compiled non-causal attention."""
    return attention_wrapper(q, k, v, is_causal=False)


@jax.jit
def _attention_causal(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    """JIT-compiled causal attention."""
    return attention_wrapper(q, k, v, is_causal=True)


def flash_attention(
    q: jax.Array,
    k: jax.Array, 
    v: jax.Array,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    deterministic: bool = True,
) -> jax.Array:
    """
    Flash attention implementation using JAX's dot_product_attention.
    
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
    Standard attention implementation using JAX's dot_product_attention.
    
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
        x = x.astype(jnp.float32)
        x = x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        x = x.astype(x.dtype) * self.weight
        return x


class WanLayerNorm(nnx.Module):
    """Layer normalization layer."""
    
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nnx.Param(jnp.ones(dim))
            self.bias = nnx.Param(jnp.zeros(dim))
        else:
            self.weight = None
            self.bias = None
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
        """
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            x = x * self.weight + self.bias
            
        return x
