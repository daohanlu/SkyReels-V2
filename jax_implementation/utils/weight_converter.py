import jax
import jax.numpy as jnp
import numpy as np
import os
import json
from huggingface_hub import hf_hub_download
from typing import Dict, Any, Tuple

# Assuming your model definition is in a file named `model.py`
from modules.transformer import WanModel

def load_torch_weights(model_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load PyTorch weights and config from a HuggingFace model repo or a local path.

    Args:
        model_id: HuggingFace model ID (e.g., "Skywork/SkyReels-V2-I2V-1.3B-540P") or a local directory path.

    Returns:
        A tuple containing:
        - A dictionary of the model's state dict with tensor values.
        - A dictionary of the model's configuration.
    """
    print(f"Attempting to load weights and config for model: {model_id}...")

    # Determine if model_id is a local path or a HuggingFace repo ID
    if os.path.isdir(model_id):
        print(f"'{model_id}' is a local directory.")
        model_path = model_id
    else:
        print(f"'{model_id}' is not a local directory, treating as HuggingFace repo ID.")
        try:
            # First, try to download the safetensors file to get the cache path
            weights_path_for_dir = hf_hub_download(repo_id=model_id, filename="model.safetensors")
            model_path = os.path.dirname(weights_path_for_dir) # Get directory from file path
        except Exception as e:
            raise FileNotFoundError(f"Could not download or find model repository for '{model_id}'. Error: {e}")

    config_path = os.path.join(model_path, "config.json")
    weights_path = os.path.join(model_path, "model.safetensors")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file (model.safetensors) not found at: {weights_path}")

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"✅ Loaded config from {config_path}")

    # Load weights using safetensors
    try:
        from safetensors import safe_open
        torch_weights = {}
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                torch_weights[key] = f.get_tensor(key)
        print(f"✅ Loaded {len(torch_weights)} weight tensors from {weights_path}")
    except ImportError:
        raise ImportError("The 'safetensors' library is required. Please install it with 'pip install safetensors'.")

    return torch_weights, config


def apply_weights_to_model(model: WanModel, pt_state_dict: Dict[str, Any]) -> WanModel:
    """
    Converts a PyTorch state dictionary and applies it to a JAX NNX WanModel.

    This function handles:
    - Transposing weights for linear layers.
    - Permuting dimensions for 3D convolution kernels.
    - Splitting combined QKV/KV layers from PyTorch into separate Q, K, V layers in JAX.
    - Renaming parameters (e.g., 'weight' to 'kernel' or 'scale').

    Args:
        model: An instance of the JAX NNX WanModel.
        pt_state_dict: A dictionary mapping parameter names to tensor objects.

    Returns:
        The WanModel instance with the loaded weights.
    """
    print("Starting weight conversion and application...")

    # --- Helper functions for clarity ---
    def conv_linear(pt_key_weight: str, pt_key_bias: str):
        """Converts a PyTorch linear layer's weights and biases."""
        w = jnp.asarray(pt_state_dict[pt_key_weight]).T
        b = jnp.asarray(pt_state_dict[pt_key_bias])
        return w, b

    def conv_param(pt_key: str):
        """Converts a generic PyTorch parameter."""
        return jnp.asarray(pt_state_dict[pt_key])

    # --- Top-level parameters ---
    print("Loading top-level embeddings and projections...")
    pt_w = pt_state_dict['patch_embedding.weight']
    model.patch_embedding.kernel.value = jnp.asarray(pt_w).transpose(2, 3, 4, 1, 0)
    model.patch_embedding.bias.value = conv_param('patch_embedding.bias')

    # Text Embeddings (from nn.Sequential)
    model.text_embedding_1.kernel.value, model.text_embedding_1.bias.value = conv_linear('text_embedding.0.weight', 'text_embedding.0.bias')
    model.text_embedding_2.kernel.value, model.text_embedding_2.bias.value = conv_linear('text_embedding.2.weight', 'text_embedding.2.bias')

    # Time Embeddings & Projections (from nn.Sequential)
    model.time_embedding_1.kernel.value, model.time_embedding_1.bias.value = conv_linear('time_embedding.0.weight', 'time_embedding.0.bias')
    model.time_embedding_2.kernel.value, model.time_embedding_2.bias.value = conv_linear('time_embedding.2.weight', 'time_embedding.2.bias')
    model.time_projection_1.kernel.value, model.time_projection_1.bias.value = conv_linear('time_projection.1.weight', 'time_projection.1.bias')

    # FPS embeddings (if enabled)
    if model.inject_sample_info:
        print("Loading FPS info injection parameters...")
        model.fps_embedding.embedding.value = conv_param('fps_embedding.weight')
        model.fps_projection_1.kernel.value, model.fps_projection_1.bias.value = conv_linear('fps_projection.0.weight', 'fps_projection.0.bias')
        model.fps_projection_2.kernel.value, model.fps_projection_2.bias.value = conv_linear('fps_projection.2.weight', 'fps_projection.2.bias')

    # Image embedding projection (for i2v model)
    if model.model_type == 'i2v':
        print("Loading image embedding projection for i2v model...")
        model.img_emb.proj[0].scale.value = conv_param('img_emb.proj.0.weight')
        model.img_emb.proj[0].bias.value = conv_param('img_emb.proj.0.bias')
        model.img_emb.proj[1].kernel.value, model.img_emb.proj[1].bias.value = conv_linear('img_emb.proj.1.weight', 'img_emb.proj.1.bias')
        model.img_emb.proj[2].kernel.value, model.img_emb.proj[2].bias.value = conv_linear('img_emb.proj.3.weight', 'img_emb.proj.3.bias')
        model.img_emb.proj[3].scale.value = conv_param('img_emb.proj.4.weight')
        model.img_emb.proj[3].bias.value = conv_param('img_emb.proj.4.bias')

    # --- Transformer Blocks ---
    for i in range(model.num_layers):
        block = model.blocks[i]
        prefix = f'blocks.{i}.'

        # Modulation parameters
        block.modulation.value = conv_param(f'{prefix}modulation')

        # Normalization layers
        if model.cross_attn_norm:
            block.norm3.scale.value = conv_param(f'{prefix}norm3.weight')
            block.norm3.bias.value = conv_param(f'{prefix}norm3.bias')

        # Self-Attention
        # CORRECTED: Load q, k, and v layers separately as they are not combined in the PyTorch model.
        block.self_attn.q.kernel.value, block.self_attn.q.bias.value = conv_linear(f'{prefix}self_attn.q.weight', f'{prefix}self_attn.q.bias')
        block.self_attn.k.kernel.value, block.self_attn.k.bias.value = conv_linear(f'{prefix}self_attn.k.weight', f'{prefix}self_attn.k.bias')
        block.self_attn.v.kernel.value, block.self_attn.v.bias.value = conv_linear(f'{prefix}self_attn.v.weight', f'{prefix}self_attn.v.bias')
        block.self_attn.o.kernel.value, block.self_attn.o.bias.value = conv_linear(f'{prefix}self_attn.o.weight', f'{prefix}self_attn.o.bias')
        
        if model.qk_norm:
            # WanRMSNorm uses 'weight', not 'scale'.
            block.self_attn.norm_q.weight.value = conv_param(f'{prefix}self_attn.norm_q.weight')
            block.self_attn.norm_k.weight.value = conv_param(f'{prefix}self_attn.norm_k.weight')

        # Cross-Attention
        block.cross_attn.q.kernel.value, block.cross_attn.q.bias.value = conv_linear(f'{prefix}cross_attn.q.weight', f'{prefix}cross_attn.q.bias')
        block.cross_attn.k.kernel.value, block.cross_attn.k.bias.value = conv_linear(f'{prefix}cross_attn.k.weight', f'{prefix}cross_attn.k.bias')
        block.cross_attn.v.kernel.value, block.cross_attn.v.bias.value = conv_linear(f'{prefix}cross_attn.v.weight', f'{prefix}cross_attn.v.bias')
        block.cross_attn.o.kernel.value, block.cross_attn.o.bias.value = conv_linear(f'{prefix}cross_attn.o.weight', f'{prefix}cross_attn.o.bias')
        
        if model.qk_norm:
            block.cross_attn.norm_q.weight.value = conv_param(f'{prefix}cross_attn.norm_q.weight')
            block.cross_attn.norm_k.weight.value = conv_param(f'{prefix}cross_attn.norm_k.weight')

        if model.model_type == 'i2v':
            block.cross_attn.k_img.kernel.value, block.cross_attn.k_img.bias.value = conv_linear(f'{prefix}cross_attn.k_img.weight', f'{prefix}cross_attn.k_img.bias')
            block.cross_attn.v_img.kernel.value, block.cross_attn.v_img.bias.value = conv_linear(f'{prefix}cross_attn.v_img.weight', f'{prefix}cross_attn.v_img.bias')
            if model.qk_norm:
                block.cross_attn.norm_k_img.weight.value = conv_param(f'{prefix}cross_attn.norm_k_img.weight')

        # FFN (from nn.Sequential)
        block.ffn_1.kernel.value, block.ffn_1.bias.value = conv_linear(f'{prefix}ffn.0.weight', f'{prefix}ffn.0.bias')
        block.ffn_2.kernel.value, block.ffn_2.bias.value = conv_linear(f'{prefix}ffn.2.weight', f'{prefix}ffn.2.bias')

    # --- Head ---
    print("Loading final output head...")
    model.head.modulation.value = conv_param('head.modulation')
    model.head.head.kernel.value, model.head.head.bias.value = conv_linear('head.head.weight', 'head.head.bias')

    print("\nSuccessfully applied all weights to the JAX NNX model.")
    return model

if __name__ == '__main__':
    # --- Example Usage ---
    # This script is intended to be used as a module by other scripts (e.g., a testing or inference script).
    # Below is a basic example of how to use its functions.

    # 1. Define the model ID (Hugging Face repo or local path)
    model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P"  # Replace with your model if needed

    try:
        # 2. Load the PyTorch weights and configuration
        torch_weights, config = load_torch_weights(model_id)

        # 3. Create the JAX model from the loaded configuration
        print("Creating JAX model from config...")
        jax_model = WanModel(
            model_type=config.get('model_type', 'i2v'),
            patch_size=tuple(config.get('patch_size', [1, 2, 2])),
            text_len=config.get('text_len', 512),
            in_dim=config.get('in_dim', 36),
            dim=config.get('dim', 1536),  # 1.3B model uses 1536
            ffn_dim=config.get('ffn_dim', 8960),  # 1.3B model uses 8960
            freq_dim=config.get('freq_dim', 256),
            text_dim=config.get('text_dim', 4096),
            out_dim=config.get('out_dim', 16),
            num_heads=config.get('num_heads', 12),  # 1.3B model uses 12 heads
            num_layers=config.get('num_layers', 30),  # 1.3B model uses 30 layers
            window_size=tuple(config.get('window_size', [-1, -1])),
            qk_norm=config.get('qk_norm', True),
            cross_attn_norm=config.get('cross_attn_norm', True),
            inject_sample_info=config.get('inject_sample_info', False),
            eps=config.get('eps', 1e-6),
        )
        print("JAX model created.")

        # 4. Apply the loaded weights to the JAX model
        jax_model_with_weights = apply_weights_to_model(jax_model, torch_weights)

        print("\n✅ Weight conversion and application complete.")
        print("The model is now ready for inference.")

    except (FileNotFoundError, ImportError, KeyError) as e:
        print(f"\n❌ An error occurred: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
