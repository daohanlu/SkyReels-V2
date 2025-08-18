#!/usr/bin/env python3
"""
Simple test script to compare PyTorch and JAX implementations of SkyReels-V2.
This script provides better error handling and focuses on core functionality.
"""

import argparse
import gc
import os
import random
import time
import numpy as np
import torch
import jax
import jax.numpy as jnp
import imageio
from PIL import Image
from diffusers.utils import load_image
from typing import Dict, Any, Tuple, Optional, List
import sys
import cv2

# Add paths
sys.path.append('.')
sys.path.append('./jax_implementation')

# Import PyTorch components
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline
from skyreels_v2_infer.pipelines import resizecrop


def set_seed(seed: int):
    """Set random seed for both PyTorch and JAX."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    jax.random.PRNGKey(seed)


def load_image_and_preprocess(image_path: str, height: int, width: int) -> Image.Image:
    """Load and preprocess image for both PyTorch and JAX."""
    if image_path:
        image = load_image(image_path)
        image_width, image_height = image.size
        if image_height > image_width:
            height, width = width, height
        image = resizecrop(image, height, width)
        return image.convert("RGB")
    return None


def generate_pytorch_video(
    model_id: str,
    prompt: str,
    negative_prompt: str,
    image: Optional[Image.Image],
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    shift: float,
    seed: int,
    offload: bool = False,
    teacache: bool = False,
    teacache_thresh: float = 0.3,
    use_ret_steps: bool = False,
) -> List[np.ndarray]:
    """Generate video using PyTorch implementation."""
    print("🔄 Generating video with PyTorch implementation...")
    
    # Set seed
    set_seed(seed)
    
    # Create PyTorch pipeline
    pipe = Image2VideoPipeline(
        model_path=model_id, 
        dit_path=model_id, 
        use_usp=False, 
        offload=offload
    )
    
    # Initialize teacache if requested
    if teacache:
        try:
            pipe.transformer.initialize_teacache(
                enable_teacache=True, 
                num_steps=num_inference_steps, 
                teacache_thresh=teacache_thresh, 
                use_ret_steps=use_ret_steps, 
                ckpt_dir=model_id
            )
        except Exception as e:
            print(f"Warning: Could not initialize teacache: {e}")
    
    # Generate video
    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        video_frames = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
    
    print("✅ PyTorch video generation completed")
    return video_frames


def test_jax_implementation(model_id: str) -> bool:
    """Test if JAX implementation can be loaded and initialized."""
    print("🔄 Testing JAX implementation...")
    
    try:
        # Import JAX components
        from jax_implementation.modules import WanModel
        from jax_implementation.utils.weight_converter import load_torch_weights, apply_weights_to_model
        
        # Load PyTorch weights and config
        jax_weights, config = load_torch_weights(model_id)
        
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
        
        # Apply weights to JAX model
        jax_model = apply_weights_to_model(jax_model, jax_weights)
        
        print(f"✅ JAX model created with {len(jax_weights)} parameters")
        return True
        
    except Exception as e:
        print(f"❌ JAX implementation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_side_by_side_video(
    pytorch_frames: List[np.ndarray],
    output_path: str,
    jax_frames: Optional[List[np.ndarray]] = None,
    fps: int = 24
) -> None:
    """Create a side-by-side comparison video."""
    print("🔄 Creating comparison video...")
    
    if jax_frames is None:
        # Only PyTorch video
        video_frames = pytorch_frames
        labels = ["PyTorch"]
        frame_width = pytorch_frames[0].shape[1]
        combined_width = frame_width
    else:
        # Ensure both videos have the same number of frames
        num_frames = min(len(pytorch_frames), len(jax_frames))
        pytorch_frames = pytorch_frames[:num_frames]
        jax_frames = jax_frames[:num_frames]
        
        # Get frame dimensions
        h, w, c = pytorch_frames[0].shape
        
        # Create side-by-side frames
        video_frames = []
        for i in range(num_frames):
            # Create side-by-side frame
            combined_frame = np.zeros((h, w * 2, c), dtype=np.uint8)
            combined_frame[:, :w] = pytorch_frames[i]
            combined_frame[:, w:] = jax_frames[i]
            
            # Add labels
            cv2.putText(combined_frame, "PyTorch", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_frame, "JAX", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            video_frames.append(combined_frame)
        
        labels = ["PyTorch", "JAX"]
        combined_width = w * 2
    
    # Save video
    imageio.mimwrite(
        output_path, 
        video_frames, 
        fps=fps, 
        quality=8, 
        output_params=["-loglevel", "error"]
    )
    
    print(f"✅ Comparison video saved to {output_path}")
    print(f"📊 Video dimensions: {video_frames[0].shape[0]}x{combined_width}")
    print(f"📊 Number of frames: {len(video_frames)}")
    print(f"📊 Labels: {', '.join(labels)}")


def save_individual_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 24
) -> None:
    """Save individual video."""
    imageio.mimwrite(
        output_path, 
        frames, 
        fps=fps, 
        quality=8, 
        output_params=["-loglevel", "error"]
    )
    print(f"✅ Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple test script for PyTorch vs JAX SkyReels-V2 comparison")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-I2V-1.3B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"], default="540P")
    
    # Generation arguments
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--prompt", type=str, 
                       default="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.")
    parser.add_argument("--image", type=str, default=None)
    
    # Optimization arguments
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument("--teacache_thresh", type=float, default=0.3)
    parser.add_argument("--use_ret_steps", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="test_results")
    parser.add_argument("--save_individual", action="store_true", help="Save individual videos")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test_jax_only", action="store_true", help="Only test JAX implementation loading")
    
    args = parser.parse_args()
    
    # Set resolution
    if args.resolution == "540P":
        height, width = 544, 960
    elif args.resolution == "720P":
        height, width = 720, 1280
    else:
        raise ValueError(f"Invalid resolution: {args.resolution}")
    
    # Set seed
    if args.seed is None:
        args.seed = int(random.randrange(4294967294))
    
    # Download model
    model_id = download_model(args.model_id)
    print(f"Model ID: {model_id}")
    
    # Load and preprocess image
    image = load_image_and_preprocess(args.image, height, width) if args.image else None
    
    # Negative prompt
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test JAX implementation first
    jax_available = test_jax_implementation(model_id)
    
    if args.test_jax_only:
        if jax_available:
            print("✅ JAX implementation test passed!")
            return 0
        else:
            print("❌ JAX implementation test failed!")
            return 1
    
    try:
        # Generate PyTorch video
        pytorch_frames = generate_pytorch_video(
            model_id=model_id,
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            image=image,
            height=height,
            width=width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            shift=args.shift,
            seed=args.seed,
            offload=args.offload,
            teacache=args.teacache,
            teacache_thresh=args.teacache_thresh,
            use_ret_steps=args.use_ret_steps,
        )
        
        # Create output filename
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        
        if jax_available:
            # Try to generate JAX video
            try:
                from jax_implementation.utils.pipeline import HybridPipeline
                from jax_implementation.modules import WanModel
                from jax_implementation.utils.weight_converter import load_torch_weights, apply_weights_to_model
                
                # Create JAX model
                jax_weights, config = load_torch_weights(model_id)
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
                jax_model = apply_weights_to_model(jax_model, jax_weights)
                
                # Create hybrid pipeline
                pipeline = HybridPipeline(
                    model_path=model_id,
                    jax_model=jax_model,
                    device="cuda",
                    weight_dtype=torch.bfloat16,
                )
                
                # Generate JAX video
                print("🔄 Generating video with JAX implementation...")
                video_tensor = pipeline.generate(
                    prompt=args.prompt,
                    image=image,
                    height=height,
                    width=width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=negative_prompt,
                    seed=args.seed,
                )
                
                # Convert to numpy frames
                jax_frames = []
                for i in range(video_tensor.shape[2]):  # Iterate over frames
                    frame = video_tensor[0, :, i, :, :].permute(1, 2, 0)  # HWC format
                    frame = (frame * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
                    jax_frames.append(frame)
                
                print("✅ JAX video generation completed")
                
                # Create side-by-side video
                output_filename = f"comparison_{args.prompt[:50].replace('/', '')}_{args.seed}_{current_time}.mp4"
                output_path = os.path.join(args.output_dir, output_filename)
                create_side_by_side_video(pytorch_frames, jax_frames, output_path, args.fps)
                
                # Save individual videos if requested
                if args.save_individual:
                    pytorch_path = os.path.join(args.output_dir, f"pytorch_{args.prompt[:50].replace('/', '')}_{args.seed}.mp4")
                    jax_path = os.path.join(args.output_dir, f"jax_{args.prompt[:50].replace('/', '')}_{args.seed}.mp4")
                    
                    save_individual_video(pytorch_frames, pytorch_path, args.fps)
                    save_individual_video(jax_frames, jax_path, args.fps)
                
            except Exception as e:
                print(f"❌ JAX video generation failed: {e}")
                import traceback
                traceback.print_exc()
                
                # Create PyTorch-only video
                output_filename = f"pytorch_only_{args.prompt[:50].replace('/', '')}_{args.seed}_{current_time}.mp4"
                output_path = os.path.join(args.output_dir, output_filename)
                create_side_by_side_video(pytorch_frames, None, output_path, args.fps)
                
                # Save individual PyTorch video if requested
                if args.save_individual:
                    pytorch_path = os.path.join(args.output_dir, f"pytorch_{args.prompt[:50].replace('/', '')}_{args.seed}.mp4")
                    save_individual_video(pytorch_frames, pytorch_path, args.fps)
        else:
            # Only PyTorch video
            output_filename = f"pytorch_only_{args.prompt[:50].replace('/', '')}_{args.seed}_{current_time}.mp4"
            output_path = os.path.join(args.output_dir, output_filename)
            create_side_by_side_video(pytorch_frames, None, output_path, args.fps)
            
            # Save individual PyTorch video if requested
            if args.save_individual:
                pytorch_path = os.path.join(args.output_dir, f"pytorch_{args.prompt[:50].replace('/', '')}_{args.seed}.mp4")
                save_individual_video(pytorch_frames, pytorch_path, args.fps)
        
        print(f"\n🎉 Test completed successfully!")
        print(f"🔢 Seed used: {args.seed}")
        print(f"📊 PyTorch frames: {len(pytorch_frames)}")
        if jax_available:
            print(f"📊 JAX frames: {len(jax_frames) if 'jax_frames' in locals() else 'Failed'}")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
