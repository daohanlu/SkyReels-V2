#!/usr/bin/env python3
"""
Debug script to compare JAX and PyTorch Conv3d implementations with bfloat16 inputs.
Tests that both implementations produce allclose outputs.
"""

import torch
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
import math
import os

# Set CUDA visible devices if needed
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Check GPU availability
print("GPU Setup:")
print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  PyTorch GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch GPU count: {torch.cuda.device_count()}")

print(f"  JAX devices: {jax.devices()}")
print(f"  JAX default backend: {jax.default_backend()}")
print()

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def create_torch_conv3d(in_channels, out_channels, kernel_size, stride):
    """Create and initialize a PyTorch Conv3d layer."""
    conv = torch.nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,  # VALID padding
        bias=True
    )
    
    # Initialize with specific values for reproducibility
    torch.nn.init.normal_(conv.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(conv.bias)
    
    return conv


def jax_conv3d(x, kernel, bias, strides):
    """JAX Conv3d implementation matching the one in jax_implementation/modules/transformer.py."""
    # Transpose input from NCDHW to NDHWC format
    x = x.transpose(0, 2, 3, 4, 1)
    
    # Define dimension numbers for 3D convolution
    dn = ('NDHWC', 'DHWIO', 'NDHWC')
    
    # Apply 3D convolution
    output = lax.conv_general_dilated(
        lhs=x,
        rhs=kernel,
        window_strides=strides,
        padding='VALID',
        lhs_dilation=(1, 1, 1),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=dn
    )
    
    # Add bias
    output = output + bias
    
    # Transpose back to NCDHW format
    output = output.transpose(0, 4, 1, 2, 3)
    
    return output


def compare_conv3d_implementations():
    """Compare JAX and PyTorch Conv3d implementations."""
    print("="*70)
    print("Conv3d Implementation Comparison (JAX vs PyTorch)")
    print("="*70)
    
    # Test parameters
    batch_size = 2
    in_channels = 16
    out_channels = 2048
    kernel_size = (1, 2, 2)
    strides = (1, 2, 2)
    input_shape = (batch_size, in_channels, 5, 8, 8)  # (B, C, D, H, W)
    
    print(f"\nTest Configuration:")
    print(f"  Input shape: {input_shape}")
    print(f"  Kernel size: {kernel_size}")
    print(f"  Stride: {strides}")
    print(f"  In channels: {in_channels}")
    print(f"  Out channels: {out_channels}")
    print(f"  Input dtype: bfloat16")
    
    # Create random bfloat16 input
    input_np = np.random.randn(*input_shape).astype(np.float32)
    
    # PyTorch implementation
    print("\n" + "-"*50)
    print("PyTorch Implementation:")
    torch_conv = create_torch_conv3d(in_channels, out_channels, kernel_size, strides)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_input = torch.from_numpy(input_np).to(device).to(torch.bfloat16)
    torch_conv = torch_conv.to(device).to(torch.bfloat16)
    
    with torch.no_grad():
        torch_output = torch_conv(torch_input)
    
    print(f"  Device: {device}")
    
    print(f"  Output shape: {torch_output.shape}")
    print(f"  Output dtype: {torch_output.dtype}")
    print(f"  Output range: [{torch_output.min().item():.6f}, {torch_output.max().item():.6f}]")
    
    # JAX implementation
    print("\n" + "-"*50)
    print("JAX Implementation:")
    
    # Convert PyTorch weights to JAX format
    # PyTorch weight shape: [out_channels, in_channels, D, H, W]
    # JAX weight shape: [D, H, W, in_channels, out_channels]
    torch_weight_np = torch_conv.weight.detach().cpu().float().numpy()
    jax_weight_np = np.transpose(torch_weight_np, (2, 3, 4, 1, 0))
    
    # Convert to JAX arrays with bfloat16
    jax_input = jnp.array(input_np, dtype=jnp.bfloat16)
    jax_kernel = jnp.array(jax_weight_np, dtype=jnp.bfloat16)
    jax_bias = jnp.array(torch_conv.bias.detach().cpu().float().numpy(), dtype=jnp.bfloat16)
    
    # Run JAX conv3d
    jax_output = jax_conv3d(jax_input, jax_kernel, jax_bias, strides)
    
    # JAX automatically uses GPU if available
    print(f"  Device: {jax_input.device}")
    print(f"  Output shape: {jax_output.shape}")
    print(f"  Output dtype: {jax_output.dtype}")
    print(f"  Output range: [{float(jax_output.min()):.6f}, {float(jax_output.max()):.6f}]")
    
    # Compare outputs
    print("\n" + "="*50)
    print("Comparison Results:")
    print("="*50)
    
    # Convert both outputs to numpy for comparison
    torch_output_np = torch_output.detach().cpu().float().numpy()
    jax_output_np = np.array(jax_output).astype(np.float32)
    
    # Calculate differences
    abs_diff = np.abs(torch_output_np - jax_output_np)
    rel_diff = abs_diff / (np.abs(torch_output_np) + 1e-10)
    
    print(f"\nAbsolute difference:")
    print(f"  Max: {abs_diff.max():.8f}")
    print(f"  Mean: {abs_diff.mean():.8f}")
    print(f"  Std: {abs_diff.std():.8f}")
    
    print(f"\nRelative difference:")
    print(f"  Max: {rel_diff.max():.8f}")
    print(f"  Mean: {rel_diff.mean():.8f}")
    print(f"  Std: {rel_diff.std():.8f}")
    
    # Check if outputs are close
    rtol = 1e-2  # Relative tolerance for bfloat16
    atol = 1e-2  # Absolute tolerance for bfloat16
    
    is_close = np.allclose(torch_output_np, jax_output_np, rtol=rtol, atol=atol)
    print(f"\nOutputs allclose (rtol={rtol}, atol={atol}): {is_close}")
    
    if is_close:
        print("\n✅ SUCCESS: JAX and PyTorch Conv3d implementations produce matching outputs!")
    else:
        print("\n❌ FAILURE: Outputs do not match within tolerance.")
        
        # Show some example differences
        print("\nExample differences (first 5 elements):")
        flat_torch = torch_output_np.flatten()
        flat_jax = jax_output_np.flatten()
        for i in range(min(5, len(flat_torch))):
            print(f"  Index {i}: PyTorch={flat_torch[i]:.6f}, JAX={flat_jax[i]:.6f}, Diff={flat_torch[i]-flat_jax[i]:.6f}")
    
    return is_close


def test_different_configurations():
    """Test multiple configurations to ensure robustness."""
    print("\n" + "="*70)
    print("Testing Multiple Configurations")
    print("="*70)
    
    test_configs = [
        # (batch_size, in_channels, out_channels, input_dims, kernel_size, strides)
        (1, 16, 1536, (4, 16, 16), (1, 2, 2), (1, 2, 2)),  # Typical patch embedding
        (2, 32, 64, (8, 8, 8), (2, 2, 2), (2, 2, 2)),      # Uniform kernel and stride
        (1, 8, 16, (3, 6, 6), (1, 3, 3), (1, 1, 1)),       # No stride
        (2, 16, 2048, (17, 32, 32), (1, 2, 2), (1, 2, 2)), # Large model config
    ]
    
    all_passed = True
    
    for i, (bs, in_c, out_c, in_dims, k_size, stride) in enumerate(test_configs, 1):
        print(f"\nTest {i}: BS={bs}, In={in_c}, Out={out_c}, Dims={in_dims}, Kernel={k_size}, Stride={stride}")
        
        # Create input
        input_shape = (bs, in_c, *in_dims)
        input_np = np.random.randn(*input_shape).astype(np.float32) * 0.1
        
        # PyTorch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch_conv = create_torch_conv3d(in_c, out_c, k_size, stride).to(device).to(torch.bfloat16)
        torch_input = torch.from_numpy(input_np).to(device).to(torch.bfloat16)
        with torch.no_grad():
            torch_output = torch_conv(torch_input)
        
        # JAX
        torch_weight_np = torch_conv.weight.detach().cpu().float().numpy()
        jax_weight_np = np.transpose(torch_weight_np, (2, 3, 4, 1, 0))
        jax_input = jnp.array(input_np, dtype=jnp.bfloat16)
        jax_kernel = jnp.array(jax_weight_np, dtype=jnp.bfloat16)
        jax_bias = jnp.array(torch_conv.bias.detach().cpu().float().numpy(), dtype=jnp.bfloat16)
        jax_output = jax_conv3d(jax_input, jax_kernel, jax_bias, stride)
        
        # Compare
        torch_output_np = torch_output.detach().cpu().float().numpy()
        jax_output_np = np.array(jax_output).astype(np.float32)
        
        is_close = np.allclose(torch_output_np, jax_output_np, rtol=1e-2, atol=1e-2)
        status = "✅ PASS" if is_close else "❌ FAIL"
        
        max_diff = np.abs(torch_output_np - jax_output_np).max()
        print(f"  Result: {status} (Max diff: {max_diff:.8f})")
        
        if not is_close:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    
    return all_passed


if __name__ == "__main__":
    print("Starting Conv3d implementation comparison...\n")
    
    # Run main comparison
    main_result = compare_conv3d_implementations()
    
    # Run additional configuration tests
    config_result = test_different_configurations()
    
    print("\n" + "="*70)
    print("Final Summary:")
    if main_result and config_result:
        print("✅ All Conv3d tests passed successfully!")
        print("The JAX and PyTorch implementations produce matching outputs.")
    else:
        print("❌ Some tests failed. Please review the differences above.")
    print("="*70)