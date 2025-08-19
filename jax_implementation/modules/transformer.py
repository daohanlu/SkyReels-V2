import math
import jax
import jax.numpy as jnp
from jax import lax
from typing import List, Optional, Tuple, Union
from flax import nnx
from .attention import WanRMSNorm, attention
from .utils import sinusoidal_embedding_1d, rope_params, rope_apply, mul_add, mul_add_add


class WanLayerNorm(nnx.LayerNorm):
    """
    A wrapper around flax.nnx.LayerNorm that mimics the `elementwise_affine`
    parameter from the PyTorch nn.LayerNorm API for compatibility.
    """
    def __init__(self, num_features: int, eps: float = 1e-6, elementwise_affine: bool = False, rngs: nnx.Rngs = nnx.Rngs(0)):
        """
        Initializes the LayerNorm wrapper.

        Args:
            num_features: The number of num_features in the input.
            eps: A small float added to variance to avoid dividing by zero.
            elementwise_affine: If True, this module has learnable affine
                parameters (scale and bias). Corresponds to `use_scale` and
                `use_bias` in the parent `nnx.LayerNorm`. Default is False
                according to the PyTorch model at
                https://github.com/SkyworkAI/SkyReels-V2/blob/main/skyreels_v2_infer/modules/transformer.py#L104.
            rngs: The random number generator to use for initialization.
        """
        # The `use_scale` and `use_bias` arguments in nnx.LayerNorm directly
        # correspond to the behavior of `elementwise_affine`.
        super().__init__(
            num_features=num_features,
            epsilon=eps,
            use_bias=elementwise_affine,
            use_scale=elementwise_affine,
            rngs=rngs
        )


class Conv3d(nnx.Module):
    """3D Convolution implementation using JAX's lax.conv_general_dilated."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int], 
                 strides: Tuple[int, int, int] = (1, 1, 1), padding: str = "VALID"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
        # Initialize 3D convolution kernel
        # Kernel shape: [kernel_d, kernel_h, kernel_w, in_channels, out_channels]
        key = jax.random.PRNGKey(42)
        kernel_shape = (*kernel_size, in_channels, out_channels)
        self.kernel = nnx.Param(
            jax.random.normal(key, kernel_shape) * (2.0 / math.sqrt(in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]))
        )
        
        # Initialize bias
        self.bias = nnx.Param(jnp.zeros(out_channels))
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [batch, channels, depth, height, width]
        """
        # Define dimension numbers for 3D convolution
        # Input: [batch, channels, depth, height, width] -> [batch, depth, height, width, channels]
        # Kernel: [depth, height, width, in_channels, out_channels]
        # Output: [batch, depth_out, height_out, width_out, out_channels] -> [batch, out_channels, depth_out, height_out, width_out]
        
        # Transpose input to NHWDC format
        x = x.transpose(0, 2, 3, 4, 1)  # [batch, depth, height, width, channels]
        
        # Define dimension numbers for 3D convolution
        # N=batch, D=depth, H=height, W=width, C=channels, I=input_channels, O=output_channels
        # The input is transposed to [batch, depth, height, width, channels]
        # The kernel is [depth, height, width, in_channels, out_channels]
        dn = ('NDHWC', 'DHWIO', 'NDHWC')
        
        
        # Handle padding
        if self.padding == "SAME":
            padding = "SAME"
        elif self.padding == "VALID":
            padding = "VALID"
        else:
            # Custom padding
            padding = ((0, 0), (0, 0), (0, 0))
        
        # Apply 3D convolution - ensure matching dtypes
        kernel = self.kernel
        if x.dtype != kernel.dtype:
            kernel = kernel.astype(x.dtype)
        
        output = lax.conv_general_dilated(
            lhs=x,                    # Input tensor
            rhs=kernel,               # Kernel tensor
            window_strides=self.strides,  # Strides
            padding=padding,          # Padding
            lhs_dilation=(1, 1, 1),   # Input dilation
            rhs_dilation=(1, 1, 1),   # Kernel dilation
            dimension_numbers=dn      # Dimension numbers
        )
        
        # Add bias - ensure matching dtype
        bias = self.bias
        if output.dtype != bias.dtype:
            bias = bias.astype(output.dtype)
        output = output + bias
        
        # Transpose back to [batch, channels, depth, height, width] format
        output = output.transpose(0, 4, 1, 2, 3)
        
        return output


class WanSelfAttention(nnx.Module):
    """Self-attention layer with RoPE."""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        window_size: Tuple[int, int] = (-1, -1), 
        qk_norm: bool = True, 
        eps: float = 1e-6
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        
        # Linear projections
        self.q = nnx.Linear(dim, dim, rngs=nnx.Rngs(0))
        self.k = nnx.Linear(dim, dim, rngs=nnx.Rngs(0))
        self.v = nnx.Linear(dim, dim, rngs=nnx.Rngs(0))
        self.o = nnx.Linear(dim, dim, rngs=nnx.Rngs(0))
        
        # Normalization layers
        self.norm_q = WanRMSNorm(dim, eps) if qk_norm else nnx.Identity()
        self.norm_k = WanRMSNorm(dim, eps) if qk_norm else nnx.Identity()
    
    def set_ar_attention(self):
        """Set autoregressive attention mode."""
        pass  # Implement if needed
    
    def __call__(
        self, 
        x: jax.Array, 
        grid_sizes: jax.Array, 
        freqs: jax.Array, 
        block_mask: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            grid_sizes: Grid sizes [F, H, W]
            freqs: RoPE frequency parameters
            block_mask: Optional attention mask
        """
        b, n, d = x.shape[0], self.num_heads, self.head_dim
        
        # Compute query, key, value
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        k = self.norm_k(self.k(x)).reshape(b, -1, n, d)
        v = self.v(x).reshape(b, -1, n, d)
        
        # Apply RoPE
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        
        # Compute attention
        x = attention(q, k, v)
        
        # Output
        x = x.reshape(b, -1, self.dim)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    """Cross-attention for image-to-video with image context."""
    
    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int] = (-1, -1), qk_norm: bool = True, eps: float = 1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)
        
        # Additional projections for image context
        self.k_img = nnx.Linear(dim, dim, rngs=nnx.Rngs(0))
        self.v_img = nnx.Linear(dim, dim, rngs=nnx.Rngs(0))
        self.norm_k_img = WanRMSNorm(dim, eps) if qk_norm else nnx.Identity()
    
    def __call__(self, x: jax.Array, context: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            context: Context tensor of shape [batch, context_len, dim]
        """
        # Split context into image and text parts
        context_img = context[:, :257]  # First 257 tokens are image features
        context_text = context[:, 257:]  # Remaining tokens are text features
        
        b, n, d = x.shape[0], self.num_heads, self.head_dim
        
        # Compute query, key, value
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        k = self.norm_k(self.k(context_text)).reshape(b, -1, n, d)
        v = self.v(context_text).reshape(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).reshape(b, -1, n, d)
        v_img = self.v_img(context_img).reshape(b, -1, n, d)
        
        # Compute attention for both image and text
        img_x = attention(q, k_img, v_img)
        x = attention(q, k, v)
        
        # Combine outputs
        x = x.reshape(b, -1, self.dim)
        img_x = img_x.reshape(b, -1, self.dim)
        x = x + img_x
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    """Cross-attention for text-to-video."""
    
    def __call__(self, x: jax.Array, context: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            context: Context tensor of shape [batch, context_len, dim]
        """
        b, n, d = x.shape[0], self.num_heads, self.head_dim
        
        # Compute query, key, value
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        k = self.norm_k(self.k(context)).reshape(b, -1, n, d)
        v = self.v(context).reshape(b, -1, n, d)
        
        # Compute attention
        x = attention(q, k, v)
        
        # Output
        x = x.reshape(b, -1, self.dim)
        x = self.o(x)
        return x


class WanAttentionBlock(nnx.Module):
    """Attention block with self-attention and cross-attention."""
    
    def __init__(
        self,
        cross_attn_type: str,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        
        # Layers
        self.norm1 = WanLayerNorm(dim, eps, rngs=nnx.Rngs(0))
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nnx.Identity()
        
        # Cross-attention based on type
        if cross_attn_type == "t2v_cross_attn":
            self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        elif cross_attn_type == "i2v_cross_attn":
            self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        else:
            raise ValueError(f"Unknown cross attention type: {cross_attn_type}")
        
        self.norm2 = WanLayerNorm(dim, eps, rngs=nnx.Rngs(0))
        self.ffn_1 = nnx.Linear(dim, ffn_dim, rngs=nnx.Rngs(0))
        self.ffn_2 = nnx.Linear(ffn_dim, dim, rngs=nnx.Rngs(0))
        
        # Modulation - use jax.random instead of jnp.random
        key = jax.random.PRNGKey(42)
        self.modulation = nnx.Param(jax.random.normal(key, (1, 6, dim)) * (1/math.sqrt(dim)))
    
    def set_ar_attention(self):
        """Set autoregressive attention mode."""
        self.self_attn.set_ar_attention()
    
    def __call__(
        self,
        x: jax.Array,
        e: jax.Array,
        grid_sizes: jax.Array,
        freqs: jax.Array,
        context: jax.Array,
        block_mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            e: Time embeddings of shape [batch, 6, dim] or [batch, seq_len, 6, dim]
            grid_sizes: Grid sizes [F, H, W]
            freqs: RoPE frequency parameters
            context: Context tensor for cross-attention
            block_mask: Optional attention mask
        """
        # Handle modulation
        if e.ndim == 3: # Shape is [B, 6, dim]
            modulation = self.modulation
            e = modulation + e
            # Split along dim=1 to match PyTorch's chunk(6, dim=1)
            # e has shape [B, 6, dim], split into 6 parts of [B, 1, dim]
            e_split = jnp.split(e, 6, axis=1)
            # Squeeze to get [B, dim] for each part
            e = [jnp.squeeze(ei, axis=1) for ei in e_split]
        elif e.ndim == 4: # Shape is [B, seq_len, 6, dim]
            modulation = self.modulation[:, None, :, :] # Shape becomes [1, 1, 6, dim]
            e = modulation + e
            # Split along dim=2 (the 6 dimension)
            e_split = jnp.split(e, 6, axis=2)
            # Squeeze to get [B, seq_len, dim] for each part
            e = [jnp.squeeze(ei, axis=2) for ei in e_split]
        
        # Self-attention
        out = mul_add_add(self.norm1(x), e[1], e[0])
        y = self.self_attn(out, grid_sizes, freqs, block_mask)
        x = mul_add(x, y, e[2])
        
        # Cross-attention & FFN
        def cross_attn_ffn(x, context, e):
            x = x + self.cross_attn(self.norm3(x), context)
            y = self.ffn_2(nnx.gelu(self.ffn_1(mul_add_add(self.norm2(x), e[4], e[3]))))
            x = mul_add(x, y, e[5])
            return x
        
        x = cross_attn_ffn(x, context, e)
        return x  # Don't force dtype conversion


class Head(nnx.Module):
    """Output projection head."""
    
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        
        # Layers
        out_dim_total = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps, rngs=nnx.Rngs(0))
        self.head = nnx.Linear(dim, out_dim_total, rngs=nnx.Rngs(0))
        
        # Modulation - use jax.random instead of jnp.random
        key = jax.random.PRNGKey(42)
        self.modulation = nnx.Param(jax.random.normal(key, (1, 2, dim)) * (1/math.sqrt(dim)))
    
    def __call__(self, x: jax.Array, e: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            e: Time embeddings of shape [batch, dim] or [batch, seq_len, dim]
        """
        if e.ndim == 2:
            modulation = self.modulation  # [1, 2, dim]
            e = (modulation + e[None, :, :]).reshape(2, -1)
        elif e.ndim == 3:
            modulation = self.modulation[:, :, None, :]  # [1, 2, 1, dim]
            e = (modulation + e[None, :, :, :]).reshape(e.shape[0], 2, -1)
            e = [e[:, i, :] for i in range(2)]
        
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(nnx.Module):
    """
    MLP projection for image embeddings.
    CORRECTED: This class now uses a standard Python list to hold modules,
    which nnx correctly nests to match the PyTorch nn.Sequential structure.
    """
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # This epsilon is the hardcoded default in torch.nn.LayerNorm.
        # The original MLPProj module does NOT receive the main model's
        # `eps` from the config, so we must match the PyTorch default here.
        TORCH_DEFAULT_EPS = 1e-5
        self.proj = [
            WanLayerNorm(in_dim, eps=TORCH_DEFAULT_EPS, elementwise_affine=True, rngs=nnx.Rngs(0)),      # index 0
            nnx.Linear(in_dim, in_dim, rngs=nnx.Rngs(0)),     # index 1
            # nn.GELU is just a function, so it doesn't have parameters and
            # doesn't need a placeholder in the list.
            nnx.Linear(in_dim, out_dim, rngs=nnx.Rngs(0)),    # index 3 in PyTorch
            WanLayerNorm(out_dim, eps=TORCH_DEFAULT_EPS, elementwise_affine=True, rngs=nnx.Rngs(0))      # index 4 in PyTorch
        ]
    
    def __call__(self, image_embeds: jax.Array) -> jax.Array:
        # Apply the layers sequentially, just like nn.Sequential.
        x = self.proj[0](image_embeds)
        x = self.proj[1](x)
        x = jax.nn.gelu(x, approximate=True) # Apply GELU activation.
        x = self.proj[2](x) # Corresponds to index 3 in PyTorch
        x = self.proj[3](x) # Corresponds to index 4 in PyTorch
        return x


class WanModel(nnx.Module):
    """
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """
    
    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 1536,  # 1.3B model default
        ffn_dim: int = 8960,  # 1.3B model default
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 12,  # 1.3B model default
        num_layers: int = 30,  # 1.3B model default
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        inject_sample_info: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        
        assert model_type in ["t2v", "i2v"], f"Unknown model type: {model_type}"
        self.model_type = model_type
        
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = 1
        self.flag_causal_attention = False
        self.block_mask = None
        self.enable_teacache = False
        
        # Embeddings
        # For i2v mode, PyTorch config already includes concatenated channels (36 = 16 + 20)
        # So we use in_dim directly which should already be the total channels
        self.patch_embedding = Conv3d(in_dim, dim, kernel_size=patch_size, strides=patch_size)
        
        # Text embedding layers
        self.text_embedding_1 = nnx.Linear(text_dim, dim, rngs=nnx.Rngs(0))
        self.text_embedding_2 = nnx.Linear(dim, dim, rngs=nnx.Rngs(0))
        
        # Time embedding layers
        self.time_embedding_1 = nnx.Linear(freq_dim, dim, rngs=nnx.Rngs(0))
        self.time_embedding_2 = nnx.Linear(dim, dim, rngs=nnx.Rngs(0))
        self.time_projection_1 = nnx.Linear(dim, dim * 6, rngs=nnx.Rngs(0))
        
        if inject_sample_info:
            self.fps_embedding = nnx.Embedding(2, dim, rngs=nnx.Rngs(0))
            self.fps_projection_1 = nnx.Linear(dim, dim, rngs=nnx.Rngs(0))
            self.fps_projection_2 = nnx.Linear(dim, dim * 6, rngs=nnx.Rngs(0))
        
        # Blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = [
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ]
        
        # Head
        self.head = Head(dim, out_dim, patch_size, eps)
        
        # RoPE frequencies
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = jnp.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], axis=1)
        
        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)
        
        self.inject_sample_info = inject_sample_info
    
    def __call__(
        self, 
        x: jax.Array, 
        t: jax.Array, 
        context: jax.Array, 
        clip_fea: Optional[jax.Array] = None, 
        y: Optional[jax.Array] = None, 
        fps: Optional[int] = None
    ) -> jax.Array:
        """
        Forward pass through the diffusion model.
        
        Args:
            x: Input video tensor of shape [batch, in_dim, F, H, W]
            t: Diffusion timesteps tensor of shape [batch] or [batch, frames]
            context: Text embeddings of shape [batch, text_len, text_dim]
            clip_fea: CLIP image features for i2v mode
            y: Conditional video inputs for i2v mode
            fps: FPS information for sample injection
            
        Returns:
            Denoised video tensor of shape [batch, out_dim, F, H//8, W//8]
        """
        if self.model_type == "i2v":
            assert clip_fea is not None and y is not None
        
        # Concatenate inputs if needed
        if y is not None:
            x = jnp.concatenate([x, y], axis=1)
        
        # Patch embedding
        x = self.patch_embedding(x)
        grid_sizes = jnp.array(x.shape[2:], dtype=jnp.int32)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(0, 2, 1)
        
        # Time embeddings
        if t.ndim == 2:
            b, f = t.shape
            _flag_df = True
        else:
            b = t.shape[0]
            _flag_df = False
        
        e = self.time_embedding_1(
            sinusoidal_embedding_1d(self.freq_dim, t.reshape(-1))
        )  # [batch, dim]
        e = jax.nn.silu(e)
        e = self.time_embedding_2(e)
        e0 = jax.nn.silu(e)
        e0 = self.time_projection_1(e0).reshape(b, 6, self.dim)  # [batch, 6, dim]
        e = e.astype(jnp.float32)
        e0 = e0.astype(jnp.float32)
        
        if self.inject_sample_info and fps is not None:
            fps_tensor = jnp.array(fps, dtype=jnp.int32)
            fps_emb = self.fps_embedding(fps_tensor).astype(jnp.float32)
            fps_proj = self.fps_projection_1(fps_emb)
            fps_proj = jax.nn.silu(fps_proj)
            fps_proj = self.fps_projection_2(fps_proj)
            if _flag_df:
                e0 = e0 + fps_proj.reshape(6, self.dim)[None, :, :].repeat(t.shape[1], axis=0)
            else:
                e0 = e0 + fps_proj.reshape(6, self.dim)
        
        if _flag_df:
            e = e.reshape(b, f, 1, 1, self.dim)
            e0 = e0.reshape(b, f, 1, 1, 6, self.dim)
            e = e.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1).reshape(b, -1, self.dim)
            e0 = e0.repeat(1, 1, grid_sizes[1], grid_sizes[2], 1, 1).reshape(b, -1, 6, self.dim)
            e0 = e0.transpose(1, 2)
        
        # Context processing
        context = self.text_embedding_1(context)
        context = jax.nn.gelu(context, approximate=True)
        context = self.text_embedding_2(context)
        
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # [batch, 257, dim]
            context = jnp.concatenate([context_clip, context], axis=1)
        
        # Forward through blocks
        kwargs = dict(e=e0, grid_sizes=grid_sizes, freqs=self.freqs, context=context, block_mask=self.block_mask)
        
        # Simple forward pass (without teacache for now)
        for block in self.blocks:
            x = block(x, **kwargs)
        
        # Output projection
        x = self.head(x, e)
        
        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        
        # Return in float32 for compatibility with VAE decoder
        if x.dtype == jnp.bfloat16:
            x = x.astype(jnp.float32)
        return x
    
    def unpatchify(self, x: jax.Array, grid_sizes: jax.Array) -> jax.Array:
        """
        Reconstruct video tensors from patch embeddings.
        
        Args:
            x: Patchified features of shape [batch, seq_len, out_dim * prod(patch_size)]
            grid_sizes: Original spatial-temporal grid dimensions [F, H, W]
            
        Returns:
            Reconstructed video tensor of shape [batch, out_dim, F, H//8, W//8]
        """
        c = self.out_dim
        bs = x.shape[0]
        
        # Reshape to spatial dimensions
        x = x.reshape(bs, *grid_sizes, *self.patch_size, c)
        
        # Rearrange dimensions
        x = jnp.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(bs, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        
        return x
    
    def set_ar_attention(self, causal_block_size: int):
        """Set autoregressive attention mode."""
        self.num_frame_per_block = causal_block_size
        self.flag_causal_attention = True
        for block in self.blocks:
            block.set_ar_attention()

