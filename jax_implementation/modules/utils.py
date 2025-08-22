import math
import jax
import jax.numpy as jnp
from typing import Tuple


def sinusoidal_embedding_1d(dim: int, position: jax.Array) -> jax.Array:
    """
    Create sinusoidal positional embeddings for 1D sequences.
    
    Args:
        dim: Embedding dimension (must be even)
        position: Position indices of shape [seq_len]
        
    Returns:
        Sinusoidal embeddings of shape [seq_len, dim]
    """
    assert dim % 2 == 0, "Dimension must be even"
    half = dim // 2
    
    # Use float64 if available (requires jax.config.update('jax_enable_x64', True))
    # Otherwise falls back to float32
    try:
        position = position.astype(jnp.float64)
        dtype = jnp.float64
    except:
        position = position.astype(jnp.float32)
        dtype = jnp.float32
    
    # Calculate sinusoidal embeddings matching PyTorch's implementation
    freqs = jnp.outer(position, jnp.power(10000.0, -jnp.arange(half, dtype=dtype) / dtype(half)))
    x = jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)], axis=1)
    
    # Return in the computed dtype - will be converted to bfloat16 later as needed
    return x


def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> jax.Array:
    """
    Generate RoPE (Rotary Position Embedding) parameters.
    
    Args:
        max_seq_len: Maximum sequence length
        dim: Embedding dimension (must be even)
        theta: Base frequency for RoPE
        
    Returns:
        RoPE frequency parameters of shape [max_seq_len, dim//2]
    """
    assert dim % 2 == 0, "Dimension must be even"
    
    freqs = jnp.outer(
        jnp.arange(max_seq_len),
        1.0 / jnp.power(theta, jnp.arange(0, dim, 2) / dim)
    )
    
    # Convert to complex numbers
    freqs = jnp.exp(1j * freqs)
    
    return freqs


def rope_apply(x: jax.Array, grid_sizes: jax.Array, freqs: jax.Array) -> jax.Array:
    """
    Applies 3D Rotary Position Embeddings (RoPE) to the input tensor using JAX.

    This function extends 2D RoPE to 3D for video data by factorizing the
    embeddings across the frame, height, and width dimensions.

    Args:
        x (jax.Array): The input tensor (e.g., query or key) with a shape of
                       (batch_size, seq_len, num_heads, head_dim).
        grid_sizes (jax.Array | tuple): A 1D array or tuple containing the
                                        grid dimensions (frames, height, width).
        freqs (jax.Array): A precomputed complex-valued tensor of rotary frequencies
                           with a shape of (max_len, head_dim // 2).

    Returns:
        jax.Array: The tensor with RoPE applied, having the same shape as the input `x`.
    """
    # 1. Get dimensions from the input tensor
    # 'n' corresponds to num_heads, and 'c' is half of the head_dim
    bs, seq_len_from_x, n, head_dim = x.shape
    c = head_dim // 2

    # 2. Split the frequency tensor for each dimension (frame, height, width)
    # The head dimension is partitioned to handle each spatial/temporal dimension separately.
    # Use static indices for JIT compatibility
    split_1 = c - 2 * (c // 3)
    split_2 = c // 3
    freqs_f = freqs[:, :split_1]
    freqs_h = freqs[:, split_1:split_1 + split_2]
    freqs_w = freqs[:, split_1 + split_2:]

    # 3. Unpack grid dimensions and calculate the total sequence length
    f, h, w = grid_sizes
    # Use the actual sequence length from x's shape to avoid tracing issues
    seq_len = seq_len_from_x
    
    # 4. Reshape the input to view pairs of values as complex numbers
    # Shape: (bs, seq_len, n, head_dim) -> (bs, seq_len, n, c, 2)
    x_reshaped = x.astype(jnp.float32).reshape(bs, seq_len, n, c, 2)
    # Shape: (bs, seq_len, n, c, 2) -> (bs, seq_len, n, c) [complex]
    x_complex = jax.lax.complex(x_reshaped[..., 0], x_reshaped[..., 1])

    # 5. Construct the 3D rotary embedding tensor
    # Use take instead of slicing for JIT compatibility
    freqs_f = jnp.take(freqs_f, jnp.arange(f), axis=0, mode='clip')
    freqs_h = jnp.take(freqs_h, jnp.arange(h), axis=0, mode='clip') 
    freqs_w = jnp.take(freqs_w, jnp.arange(w), axis=0, mode='clip')

    # Reshape and broadcast each frequency component to the full (f, h, w, c_part) grid.
    # This creates a unique rotary angle for each token's position.
    freqs_f_grid = jnp.broadcast_to(freqs_f.reshape(f, 1, 1, -1), (f, h, w, freqs_f.shape[-1]))
    freqs_h_grid = jnp.broadcast_to(freqs_h.reshape(1, h, 1, -1), (f, h, w, freqs_h.shape[-1]))
    freqs_w_grid = jnp.broadcast_to(freqs_w.reshape(1, 1, w, -1), (f, h, w, freqs_w.shape[-1]))

    # Concatenate along the last dimension to form the complete embedding
    freqs_3d = jnp.concatenate([freqs_f_grid, freqs_h_grid, freqs_w_grid], axis=-1)
    
    # Reshape to be broadcastable with the input tensor
    freqs_i = freqs_3d.reshape(seq_len, 1, c)

    # 6. Apply rotary embeddings via element-wise complex multiplication
    # JAX's broadcasting rules handle the shape alignment:
    # x_complex shape: (bs, seq_len, n, c)
    # freqs_i shape:        (seq_len, 1, c) -> broadcasted to (1, seq_len, n, c)
    rotated_x_complex = x_complex * freqs_i

    # 7. Convert the complex tensor back to its real representation
    # Shape: (bs, seq_len, n, c) [complex] -> (bs, seq_len, n, c, 2)
    rotated_x_real_parts = jnp.stack([rotated_x_complex.real, rotated_x_complex.imag], axis=-1)
    
    # Flatten the last two dimensions to restore the original head_dim
    # Shape: (bs, seq_len, n, c, 2) -> (bs, seq_len, n, head_dim)
    output = rotated_x_real_parts.reshape(bs, seq_len, n, head_dim)
    
    # Preserve original dtype
    return output.astype(x.dtype)


def rope_apply_static(x: jax.Array, grid_sizes: Tuple[int, int, int], freqs: jax.Array) -> jax.Array:
    """
    Applies 3D Rotary Position Embeddings (RoPE) to the input tensor using JAX.

    This function extends 2D RoPE to 3D for video data by factorizing the
    embeddings across the frame, height, and width dimensions.

    NOTE: `grid_sizes` must be a tuple of Python integers (`(f, h, w)`) to be
    compatible with `jax.jit`. This allows it to be treated as a "static"
    argument, enabling JAX to trace the function correctly.

    Args:
        x (jax.Array): The input tensor (e.g., query or key) with a shape of
                       (batch_size, seq_len, num_heads, head_dim).
        grid_sizes (Tuple[int, int, int]): A tuple containing the static
                                            grid dimensions (frames, height, width).
        freqs (jax.Array): A precomputed complex-valued tensor of rotary frequencies
                           with a shape of (max_len, head_dim // 2).

    Returns:
        jax.Array: The tensor with RoPE applied, having the same shape as the input `x`.
    """
    # 1. Get dimensions from the input tensor
    # 'n' corresponds to num_heads, and 'c' is half of the head_dim
    bs, seq_len_from_x, n, head_dim = x.shape
    c = head_dim // 2

    # 2. Split the frequency tensor for each dimension (frame, height, width)
    # The head dimension is partitioned to handle each spatial/temporal dimension separately.
    split_proportions = jnp.array([c - 2 * (c // 3), c // 3, c // 3])
    split_indices = jnp.cumsum(split_proportions)[:-1]
    freqs_f, freqs_h, freqs_w = jnp.split(freqs, split_indices, axis=1)

    # 3. Unpack grid dimensions and calculate the total sequence length
    f, h, w = grid_sizes
    seq_len = f * h * w
    # Ensure the input tensor's sequence length matches the grid product
    # Inside a jit, this will raise a ConcretizationTypeError if shapes mismatch,
    # which correctly stops execution.
    if seq_len_from_x != seq_len:
        raise ValueError(
            f"Input sequence length {seq_len_from_x} must match the product of grid_sizes {seq_len}."
        )

    # 4. Reshape the input to view pairs of values as complex numbers
    # Shape: (bs, seq_len, n, head_dim) -> (bs, seq_len, n, c, 2)
    x_reshaped = x.astype(jnp.float32).reshape(bs, seq_len, n, c, 2)
    # Shape: (bs, seq_len, n, c, 2) -> (bs, seq_len, n, c) [complex]
    x_complex = jax.lax.complex(x_reshaped[..., 0], x_reshaped[..., 1])

    # 5. Construct the 3D rotary embedding tensor
    # Slice the frequencies to match the actual grid dimensions
    # This now works under jit because f, h, and w are concrete integers.
    freqs_f = freqs_f[:f]
    freqs_h = freqs_h[:h]
    freqs_w = freqs_w[:w]

    # Reshape and broadcast each frequency component to the full (f, h, w, c_part) grid.
    # This creates a unique rotary angle for each token's position.
    freqs_f_grid = jnp.broadcast_to(freqs_f.reshape(f, 1, 1, -1), (f, h, w, freqs_f.shape[-1]))
    freqs_h_grid = jnp.broadcast_to(freqs_h.reshape(1, h, 1, -1), (f, h, w, freqs_h.shape[-1]))
    freqs_w_grid = jnp.broadcast_to(freqs_w.reshape(1, 1, w, -1), (f, h, w, freqs_w.shape[-1]))

    # Concatenate along the last dimension to form the complete embedding
    freqs_3d = jnp.concatenate([freqs_f_grid, freqs_h_grid, freqs_w_grid], axis=-1)

    # Reshape to be broadcastable with the input tensor
    freqs_i = freqs_3d.reshape(seq_len, 1, c)

    # 6. Apply rotary embeddings via element-wise complex multiplication
    # JAX's broadcasting rules handle the shape alignment:
    # x_complex shape: (bs, seq_len, n, c)
    # freqs_i shape:       (seq_len, 1, c) -> broadcasted to (1, seq_len, n, c)
    rotated_x_complex = x_complex * freqs_i

    # 7. Convert the complex tensor back to its real representation
    # Shape: (bs, seq_len, n, c) [complex] -> (bs, seq_len, n, c, 2)
    rotated_x_real_parts = jnp.stack([rotated_x_complex.real, rotated_x_complex.imag], axis=-1)
    
    # Flatten the last two dimensions to restore the original head_dim
    # Shape: (bs, seq_len, n, c, 2) -> (bs, seq_len, n, head_dim)
    output = rotated_x_real_parts.reshape(bs, seq_len, n, head_dim)
    
    # Preserve original dtype
    return output.astype(x.dtype)


def apply_rope_1d(x: jax.Array, freq: jax.Array) -> jax.Array:
    """
    Apply 1D RoPE to input tensor.
    
    Args:
        x: Input tensor of shape [batch, seq_len, num_heads, head_dim]
        freq: Frequency parameters of shape [head_dim]
        
    Returns:
        Tensor with 1D RoPE applied
    """
    # Apply rotary position embedding
    cos_freq = jnp.cos(freq)
    sin_freq = jnp.sin(freq)
    
    # Split input into even and odd indices
    x_even = x[:, :, :, ::2]
    x_odd = x[:, :, :, 1::2]
    
    # Apply rotation
    x_rotated = jnp.concatenate([
        x_even * cos_freq[::2] - x_odd * sin_freq[1::2],
        x_even * sin_freq[::2] + x_odd * cos_freq[1::2]
    ], axis=-1)
    
    return x_rotated


def mul_add(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    """Element-wise multiplication and addition: x + y * z"""
    orig_dtype = x.dtype
    result = x.astype(jnp.float32) + y.astype(jnp.float32) * z.astype(jnp.float32)
    return result.astype(orig_dtype)


def mul_add_add(x: jax.Array, y: jax.Array, z: jax.Array) -> jax.Array:
    """Element-wise operation: x * (1 + y) + z
    
    Matches PyTorch's implementation: x.float() * (1 + y) + z
    In PyTorch, precision promotion rules mean the entire expression is computed in float32
    """
    orig_dtype = x.dtype
    # Convert all to float32 to match PyTorch's precision promotion behavior
    x_f32 = x.astype(jnp.float32)
    y_f32 = y.astype(jnp.float32)
    z_f32 = z.astype(jnp.float32)
    result = x_f32 * (1 + y_f32) + z_f32
    return result.astype(orig_dtype)

