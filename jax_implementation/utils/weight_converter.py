import os
import torch
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple
from huggingface_hub import hf_hub_download
import json
import sys
sys.path.append('..')
from modules import WanModel


def convert_torch_to_jax_weights(torch_weights: Dict[str, torch.Tensor]) -> Dict[str, jax.Array]:
    """
    Convert PyTorch weights to JAX format.
    
    Args:
        torch_weights: Dictionary of PyTorch tensors
        
    Returns:
        Dictionary of JAX arrays
    """
    jax_weights = {}
    
    for name, tensor in torch_weights.items():
        # Convert to numpy and handle data types
        if tensor.dtype == torch.bfloat16:
            numpy_tensor = tensor.float().numpy()
        elif tensor.dtype == torch.float16:
            numpy_tensor = tensor.float().numpy()
        else:
            numpy_tensor = tensor.numpy()
        
        # Convert to JAX array
        jax_array = jnp.array(numpy_tensor, dtype=jnp.float32)
        
        # Handle transpositions for different layer types
        if 'conv' in name.lower():
            # Conv layers: PyTorch (out, in, ...) -> JAX (in, out, ...)
            if len(jax_array.shape) == 4:  # Conv2d
                jax_array = jax_array.transpose(1, 0, 2, 3)
            elif len(jax_array.shape) == 5:  # Conv3d
                jax_array = jax_array.transpose(1, 0, 2, 3, 4)
        elif 'linear' in name.lower() or 'fc' in name.lower():
            # Linear layers: PyTorch (out, in) -> JAX (in, out)
            if len(jax_array.shape) == 2:
                jax_array = jax_array.transpose(1, 0)
        
        jax_weights[name] = jax_array
    
    return jax_weights


def load_torch_weights(model_id: str) -> Tuple[Dict[str, jax.Array], Dict[str, Any]]:
    """
    Load PyTorch weights from HuggingFace model or local path.
    
    Args:
        model_id: HuggingFace model ID (e.g., "Skywork/SkyReels-V2-I2V-1.3B-540P") or local path
        
    Returns:
        Tuple of (jax_weights, config)
    """
    print(f"Loading weights from {model_id}...")
    
    # Check if model_id is a local path or HuggingFace ID
    if os.path.exists(model_id):
        # Local path
        model_path = model_id
        config_path = os.path.join(model_path, "config.json")
        weights_path = os.path.join(model_path, "model.safetensors")
    else:
        # HuggingFace model ID
        try:
            config_path = hf_hub_download(repo_id=model_id, filename="config.json")
            weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
            model_path = os.path.dirname(config_path)
        except Exception as e:
            print(f"Failed to download from HuggingFace: {e}")
            print("Trying to find local files...")
            # Try to find local files in common locations
            possible_paths = [
                model_id,
                os.path.join(".", model_id),
                os.path.join("./models", model_id),
                os.path.join("./checkpoints", model_id),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    config_path = os.path.join(model_path, "config.json")
                    weights_path = os.path.join(model_path, "model.safetensors")
                    if os.path.exists(config_path) and os.path.exists(weights_path):
                        break
            else:
                raise ValueError(f"Could not find model files for {model_id}")
    
    # Load config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✅ Loaded config from {config_path}")
    
    # Load weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    print(f"Loading weights from {weights_path}...")
    torch_weights = {}
    
    try:
        from safetensors import safe_open
        with safe_open(weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                torch_weights[key] = f.get_tensor(key)
    except ImportError:
        print("Warning: safetensors not available, trying torch.load...")
        torch_weights = torch.load(weights_path, map_location="cpu")
        if isinstance(torch_weights, dict) and "state_dict" in torch_weights:
            torch_weights = torch_weights["state_dict"]
    
    print(f"✅ Loaded {len(torch_weights)} weight tensors")
    
    # Convert to JAX format
    jax_weights = convert_torch_to_jax_weights(torch_weights)
    
    return jax_weights, config


def create_jax_model_from_config(config: Dict[str, Any]) -> WanModel:
    """
    Create JAX model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        JAX WanModel instance
    """
    # Extract model parameters from config
    model_params = {
        'model_type': config.get('model_type', 'i2v'),
        'patch_size': tuple(config.get('patch_size', [1, 2, 2])),
        'text_len': config.get('text_len', 512),
        'in_dim': config.get('in_dim', 16),
        'dim': config.get('dim', 2048),
        'ffn_dim': config.get('ffn_dim', 8192),
        'freq_dim': config.get('freq_dim', 256),
        'text_dim': config.get('text_dim', 4096),
        'out_dim': config.get('out_dim', 16),
        'num_heads': config.get('num_heads', 16),
        'num_layers': config.get('num_layers', 32),
        'window_size': tuple(config.get('window_size', [-1, -1])),
        'qk_norm': config.get('qk_norm', True),
        'cross_attn_norm': config.get('cross_attn_norm', True),
        'inject_sample_info': config.get('inject_sample_info', False),
        'eps': config.get('eps', 1e-6),
    }
    
    return WanModel(**model_params)


def map_pytorch_to_jax_names(pytorch_name: str) -> str:
    """
    Map PyTorch parameter names to JAX parameter names.
    
    Args:
        pytorch_name: PyTorch parameter name
        
    Returns:
        JAX parameter name
    """
    # Define mapping rules
    name_mapping = {
        # Patch embedding
        'patch_embedding.weight': 'patch_embedding.conv2d.kernel',
        'patch_embedding.bias': 'patch_embedding.conv2d.bias',
        
        # Text embedding
        'text_embedding.0.weight': 'text_embedding.0.kernel',
        'text_embedding.0.bias': 'text_embedding.0.bias',
        'text_embedding.2.weight': 'text_embedding.2.kernel',
        'text_embedding.2.bias': 'text_embedding.2.bias',
        
        # Time embedding
        'time_embedding.0.weight': 'time_embedding.0.kernel',
        'time_embedding.0.bias': 'time_embedding.0.bias',
        'time_embedding.2.weight': 'time_embedding.2.kernel',
        'time_embedding.2.bias': 'time_embedding.2.bias',
        
        # Time projection
        'time_projection.1.weight': 'time_projection.1.kernel',
        'time_projection.1.bias': 'time_projection.1.bias',
        
        # Modulation
        'modulation': 'modulation',
    }
    
    # Handle attention blocks
    if 'blocks.' in pytorch_name:
        # Extract block index
        parts = pytorch_name.split('.')
        block_idx = int(parts[1])
        
        # Map attention block parameters
        if 'self_attn' in pytorch_name:
            if 'q.weight' in pytorch_name:
                return f'blocks.{block_idx}.self_attn.q.kernel'
            elif 'q.bias' in pytorch_name:
                return f'blocks.{block_idx}.self_attn.q.bias'
            elif 'k.weight' in pytorch_name:
                return f'blocks.{block_idx}.self_attn.k.kernel'
            elif 'k.bias' in pytorch_name:
                return f'blocks.{block_idx}.self_attn.k.bias'
            elif 'v.weight' in pytorch_name:
                return f'blocks.{block_idx}.self_attn.v.kernel'
            elif 'v.bias' in pytorch_name:
                return f'blocks.{block_idx}.self_attn.v.bias'
            elif 'o.weight' in pytorch_name:
                return f'blocks.{block_idx}.self_attn.o.kernel'
            elif 'o.bias' in pytorch_name:
                return f'blocks.{block_idx}.self_attn.o.bias'
            elif 'norm_q.weight' in pytorch_name:
                return f'blocks.{block_idx}.self_attn.norm_q.weight'
            elif 'norm_k.weight' in pytorch_name:
                return f'blocks.{block_idx}.self_attn.norm_k.weight'
            elif 'modulation' in pytorch_name:
                return f'blocks.{block_idx}.self_attn.modulation'
        
        elif 'cross_attn' in pytorch_name:
            if 'q.weight' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.q.kernel'
            elif 'q.bias' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.q.bias'
            elif 'k.weight' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.k.kernel'
            elif 'k.bias' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.k.bias'
            elif 'v.weight' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.v.kernel'
            elif 'v.bias' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.v.bias'
            elif 'o.weight' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.o.kernel'
            elif 'o.bias' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.o.bias'
            elif 'norm_q.weight' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.norm_q.weight'
            elif 'norm_k.weight' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.norm_k.weight'
            elif 'norm_k_img.weight' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.norm_k_img.weight'
            elif 'k_img.weight' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.k_img.kernel'
            elif 'k_img.bias' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.k_img.bias'
            elif 'v_img.weight' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.v_img.kernel'
            elif 'v_img.bias' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.v_img.bias'
            elif 'modulation' in pytorch_name:
                return f'blocks.{block_idx}.cross_attn.modulation'
        
        elif 'norm1' in pytorch_name:
            if 'weight' in pytorch_name:
                return f'blocks.{block_idx}.norm1.weight'
            elif 'bias' in pytorch_name:
                return f'blocks.{block_idx}.norm1.bias'
        
        elif 'norm2' in pytorch_name:
            if 'weight' in pytorch_name:
                return f'blocks.{block_idx}.norm2.weight'
            elif 'bias' in pytorch_name:
                return f'blocks.{block_idx}.norm2.bias'
        
        elif 'norm3' in pytorch_name:
            if 'weight' in pytorch_name:
                return f'blocks.{block_idx}.norm3.weight'
            elif 'bias' in pytorch_name:
                return f'blocks.{block_idx}.norm3.bias'
        
        elif 'ffn' in pytorch_name:
            if '0.weight' in pytorch_name:
                return f'blocks.{block_idx}.ffn.0.kernel'
            elif '0.bias' in pytorch_name:
                return f'blocks.{block_idx}.ffn.0.bias'
            elif '2.weight' in pytorch_name:
                return f'blocks.{block_idx}.ffn.2.kernel'
            elif '2.bias' in pytorch_name:
                return f'blocks.{block_idx}.ffn.2.bias'
        
        elif 'modulation' in pytorch_name:
            return f'blocks.{block_idx}.modulation'
    
    # Handle head
    elif 'head' in pytorch_name:
        if 'norm.weight' in pytorch_name:
            return 'head.norm.weight'
        elif 'norm.bias' in pytorch_name:
            return 'head.norm.bias'
        elif 'head.weight' in pytorch_name:
            return 'head.head.kernel'
        elif 'head.bias' in pytorch_name:
            return 'head.head.bias'
        elif 'modulation' in pytorch_name:
            return 'head.modulation'
    
    # Handle image embedding
    elif 'img_emb' in pytorch_name:
        if 'proj.0.weight' in pytorch_name:
            return 'img_emb.proj.0.weight'
        elif 'proj.0.bias' in pytorch_name:
            return 'img_emb.proj.0.bias'
        elif 'proj.2.weight' in pytorch_name:
            return 'img_emb.proj.2.kernel'
        elif 'proj.2.bias' in pytorch_name:
            return 'img_emb.proj.2.bias'
        elif 'proj.4.weight' in pytorch_name:
            return 'img_emb.proj.4.weight'
        elif 'proj.4.bias' in pytorch_name:
            return 'img_emb.proj.4.bias'
    
    # Default mapping
    return name_mapping.get(pytorch_name, pytorch_name)


def apply_weights_to_model(model: WanModel, jax_weights: Dict[str, jax.Array]) -> WanModel:
    """
    Apply weights to JAX model.
    
    Args:
        model: JAX model to apply weights to
        jax_weights: Dictionary of JAX weights
        
    Returns:
        Model with applied weights
    """
    print("Applying weights to JAX model...")
    
    # For Flax NNX models, we need to use the state attribute
    try:
        # Get current state
        state = model.state
        new_state = {}
        applied_count = 0
        total_weights = len(jax_weights)
        
        for pytorch_name, jax_weight in jax_weights.items():
            jax_name = map_pytorch_to_jax_names(pytorch_name)
            
            # Try to find the parameter in the current state
            if jax_name in state:
                new_state[jax_name] = jax_weight
                applied_count += 1
            else:
                print(f"Warning: Parameter {jax_name} not found in model (from {pytorch_name})")
        
        print(f"Applied {applied_count}/{total_weights} weights to model")
        
        # Update model state
        model.state = new_state
        
        return model
        
    except Exception as e:
        print(f"Warning: Could not apply weights: {e}")
        print("Continuing without weight application...")
        return model

