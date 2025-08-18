#!/usr/bin/env python3
"""
Simple script to test JAX implementation loading and identify issues.
"""

import argparse
import os
import sys
import torch
import jax
import jax.numpy as jnp

# Add paths
sys.path.append('.')
sys.path.append('./jax_implementation')

# Import PyTorch components
from skyreels_v2_infer.modules import download_model


def test_jax_imports():
    """Test if JAX modules can be imported."""
    print("🔄 Testing JAX imports...")
    
    try:
        from jax_implementation.modules import WanModel
        print("✅ WanModel imported successfully")
        
        from jax_implementation.modules import attention
        print("✅ attention module imported successfully")
        
        from jax_implementation.modules import utils
        print("✅ utils module imported successfully")
        
        from jax_implementation.utils import weight_converter
        print("✅ weight_converter imported successfully")
        
        from jax_implementation.utils import pipeline
        print("✅ pipeline imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weight_loading(model_id: str):
    """Test weight loading from PyTorch model."""
    print(f"🔄 Testing weight loading from {model_id}...")
    
    try:
        from jax_implementation.utils.weight_converter import load_torch_weights
        
        # Load PyTorch weights and config
        torch_weights, config = load_torch_weights(model_id)
        
        print(f"✅ Loaded {len(torch_weights)} weights")
        print(f"✅ Config keys: {list(config.keys())}")
        
        # Print some sample weight names
        sample_weights = list(torch_weights.keys())[:10]
        print(f"✅ Sample weight names: {sample_weights}")
        
        return torch_weights, config
        
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_model_creation(config: dict):
    """Test JAX model creation."""
    print("🔄 Testing JAX model creation...")
    
    try:
        from jax_implementation.modules import WanModel
        
        # Create JAX model with same config
        jax_model = WanModel(
            model_type=config.get('model_type', 'i2v'),
            patch_size=tuple(config.get('patch_size', [1, 2, 2])),
            text_len=config.get('text_len', 512),
            in_dim=config.get('in_dim', 16),
            dim=config.get('dim', 2048),
            ffn_dim=config.get('ffn_dim', 8192),
            freq_dim=config.get('freq_dim', 256),
            text_dim=config.get('text_dim', 4096),
            out_dim=config.get('out_dim', 16),
            num_heads=config.get('num_heads', 16),
            num_layers=config.get('num_layers', 32),
            window_size=tuple(config.get('window_size', [-1, -1])),
            qk_norm=config.get('qk_norm', True),
            cross_attn_norm=config.get('cross_attn_norm', True),
            inject_sample_info=config.get('inject_sample_info', False),
            eps=config.get('eps', 1e-6),
        )
        
        print("✅ JAX model created successfully")
        print(f"✅ Model type: {type(jax_model)}")
        
        return jax_model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_weight_application(jax_model, torch_weights):
    """Test applying weights to JAX model."""
    print("🔄 Testing weight application...")
    
    try:
        from jax_implementation.utils.weight_converter import apply_weights_to_model
        
        # Apply weights to JAX model
        jax_model = apply_weights_to_model(jax_model, torch_weights)
        
        print("✅ Weights applied successfully")
        return jax_model
        
    except Exception as e:
        print(f"❌ Weight application failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_forward_pass(jax_model):
    """Test a simple forward pass through the JAX model."""
    print("🔄 Testing forward pass...")
    
    try:
        # Create dummy inputs
        batch_size = 1
        num_frames = 16
        height = 34  # 544 // 16
        width = 60   # 960 // 16
        channels = 16
        
        # Create dummy latents (16 channels for in_dim)
        latents = jax.random.normal(
            jax.random.PRNGKey(42), 
            (batch_size, 16, num_frames, height, width)  # in_dim = 16
        )
        
        # Create dummy timestep
        timestep = jnp.array([0.5])
        
        # Create dummy context (text embeddings)
        context = jax.random.normal(
            jax.random.PRNGKey(43), 
            (batch_size, 512, 4096)  # text_len=512, text_dim=4096
        )
        
        # Create dummy CLIP features
        clip_fea = jax.random.normal(
            jax.random.PRNGKey(44), 
            (batch_size, 257, 4096)  # CLIP features
        )
        
        # Create dummy conditional video (should be 16 channels for I2V mode)
        y = jax.random.normal(
            jax.random.PRNGKey(45), 
            (batch_size, 16, num_frames, height, width)  # 16 channels for I2V mode
        )
        
        print(f"✅ Input shapes:")
        print(f"   latents: {latents.shape}")
        print(f"   timestep: {timestep.shape}")
        print(f"   context: {context.shape}")
        print(f"   clip_fea: {clip_fea.shape}")
        print(f"   y: {y.shape}")
        
        # Run forward pass
        output = jax_model(latents, timestep, context, clip_fea, y)
        
        print(f"✅ Forward pass successful")
        print(f"✅ Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test JAX implementation loading")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-I2V-1.3B-540P")
    parser.add_argument("--test_forward", action="store_true", help="Test forward pass")
    
    args = parser.parse_args()
    
    print("🚀 Starting JAX implementation tests...")
    print(f"Model ID: {args.model_id}")
    
    # Test 1: JAX imports
    if not test_jax_imports():
        print("❌ JAX imports failed, stopping tests")
        return 1
    
    # Download model
    model_id = download_model(args.model_id)
    print(f"Downloaded model path: {model_id}")
    
    # Test 2: Weight loading
    torch_weights, config = test_weight_loading(model_id)
    if torch_weights is None:
        print("❌ Weight loading failed, stopping tests")
        return 1
    
    # Test 3: Model creation
    jax_model = test_model_creation(config)
    if jax_model is None:
        print("❌ Model creation failed, stopping tests")
        return 1
    
    # Test 4: Weight application
    jax_model = test_weight_application(jax_model, torch_weights)
    if jax_model is None:
        print("❌ Weight application failed, stopping tests")
        return 1
    
    # Test 5: Forward pass (optional)
    if args.test_forward:
        if not test_model_forward_pass(jax_model):
            print("❌ Forward pass failed")
            return 1
    
    print("\n🎉 All tests passed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
