"""Memory-efficient attention implementation for JAX."""

import jax
import jax.numpy as jnp
from typing import Optional
import math
from functools import partial


@partial(jax.jit, static_argnames=['chunk_size', 'causal'])
def chunked_attention(
    q: jax.Array,
    k: jax.Array, 
    v: jax.Array,
    chunk_size: int = 512,
    causal: bool = False
) -> jax.Array:
    """
    Memory-efficient attention using chunking.
    Processes attention in smaller chunks to reduce memory usage.
    
    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        chunk_size: Size of chunks to process
        causal: Whether to apply causal masking
        
    Returns:
        Attention output of shape [batch, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    # Process in chunks to save memory
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    output_chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, seq_len)
        
        q_chunk = q[:, start_idx:end_idx]
        
        # Compute attention scores for this chunk
        # [batch, chunk_len, num_heads, head_dim] @ [batch, seq_len, num_heads, head_dim]^T
        # -> [batch, chunk_len, num_heads, seq_len]
        scores = jnp.einsum('bqhd,bkhd->bqhk', q_chunk, k) * scale
        
        # Apply causal mask if needed
        if causal:
            chunk_len = end_idx - start_idx
            # Create causal mask for this chunk
            row_indices = jnp.arange(start_idx, end_idx)[:, None]
            col_indices = jnp.arange(seq_len)[None, :]
            causal_mask = row_indices >= col_indices
            scores = jnp.where(causal_mask[None, :, None, :], scores, -jnp.inf)
        
        # Softmax
        scores = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        # [batch, chunk_len, num_heads, seq_len] @ [batch, seq_len, num_heads, head_dim]
        # -> [batch, chunk_len, num_heads, head_dim]
        output_chunk = jnp.einsum('bqhk,bkhd->bqhd', scores, v)
        output_chunks.append(output_chunk)
    
    # Concatenate all chunks
    output = jnp.concatenate(output_chunks, axis=1)
    return output


def flash_attention_simple(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool = False
) -> jax.Array:
    """
    Simple flash attention implementation.
    Uses tiling and recomputation to reduce memory usage.
    
    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        causal: Whether to apply causal masking
        
    Returns:
        Attention output of shape [batch, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Use smaller dtype for intermediate computations
    compute_dtype = jnp.float32 if q.dtype == jnp.bfloat16 else q.dtype
    q = q.astype(compute_dtype)
    k = k.astype(compute_dtype)
    v = v.astype(compute_dtype)
    
    scale = 1.0 / jnp.sqrt(head_dim).astype(compute_dtype)
    
    # Block size for tiling (tune based on available memory)
    block_size = min(512, seq_len)
    
    # Initialize output and running statistics
    output = jnp.zeros_like(q)
    row_max = jnp.full((batch_size, seq_len, num_heads, 1), -jnp.inf, dtype=compute_dtype)
    row_sum = jnp.zeros((batch_size, seq_len, num_heads, 1), dtype=compute_dtype)
    
    # Process in blocks
    num_blocks = (seq_len + block_size - 1) // block_size
    
    for block_idx in range(num_blocks):
        kv_start = block_idx * block_size
        kv_end = min((block_idx + 1) * block_size, seq_len)
        
        k_block = k[:, kv_start:kv_end]
        v_block = v[:, kv_start:kv_end]
        
        # Compute attention scores for all queries with this k/v block
        # [batch, seq_len, num_heads, head_dim] @ [batch, block_size, num_heads, head_dim]^T
        scores = jnp.einsum('bqhd,bkhd->bqhk', q, k_block) * scale
        
        # Apply causal mask if needed
        if causal:
            q_indices = jnp.arange(seq_len)[:, None]
            kv_indices = jnp.arange(kv_start, kv_end)[None, :]
            causal_mask = q_indices >= kv_indices
            scores = jnp.where(causal_mask[None, :, None, :], scores, -jnp.inf)
        
        # Online softmax computation
        # Handle empty blocks
        if scores.shape[-1] == 0:
            continue
        block_max = jnp.max(scores, axis=-1, keepdims=True)
        
        # Update running max
        new_max = jnp.maximum(row_max, block_max)
        
        # Compute exp with numerical stability
        exp_scores = jnp.exp(scores - new_max)
        block_sum = jnp.sum(exp_scores, axis=-1, keepdims=True)
        
        # Update output with correction factor
        correction = jnp.exp(row_max - new_max)
        output = output * correction
        row_sum = row_sum * correction
        
        # Add contribution from this block
        # [batch, seq_len, num_heads, block_size] @ [batch, block_size, num_heads, head_dim]
        block_output = jnp.einsum('bqhk,bkhd->bqhd', exp_scores, v_block)
        output = output + block_output
        
        # Update running statistics
        row_max = new_max
        row_sum = row_sum + block_sum
    
    # Final normalization
    output = output / row_sum
    
    # Convert back to original dtype
    if q.dtype == jnp.bfloat16:
        output = output.astype(jnp.bfloat16)
    
    return output


def memory_efficient_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    causal: bool = False,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    deterministic: bool = True,
    use_flash: bool = True,
    chunk_size: int = 512
) -> jax.Array:
    """
    Memory-efficient attention implementation.
    
    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        causal: Whether to apply causal masking
        dropout_p: Dropout probability (must be 0.0)
        softmax_scale: Scaling factor for attention scores (ignored)
        deterministic: Whether to use deterministic dropout (ignored)
        use_flash: Whether to use flash attention algorithm
        chunk_size: Size of chunks for chunked attention
        
    Returns:
        Attention output of shape [batch, seq_len, num_heads, head_dim]
    """
    if dropout_p != 0.0:
        raise ValueError("dropout_p must be 0.0, dropout is not supported")
    
    if use_flash:
        return flash_attention_simple(q, k, v, causal)
    else:
        return chunked_attention(q, k, v, chunk_size, causal)