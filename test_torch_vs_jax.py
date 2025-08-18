#!/usr/bin/env python3
"""
Test script to compare PyTorch and JAX implementations of SkyReels-V2.
This script generates videos using both implementations with identical parameters
and creates a side-by-side comparison video.
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

# Import JAX components
from jax_implementation.modules import WanModel
from jax_implementation.utils.weight_converter import load_torch_weights, apply_weights_to_model
from jax_implementation.utils.pipeline import HybridPipeline


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


def create_jax_model_from_torch_config(model_id: str) -> Tuple[WanModel, Dict[str, Any]]:
    """Create JAX model and load weights from PyTorch model."""
    print(f"Loading PyTorch weights from {model_id}...")
    
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
    return jax_model, config


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
        pipe.transformer.initialize_teacache(
            enable_teacache=True, 
            num_steps=num_inference_steps, 
            teacache_thresh=teacache_thresh, 
            use_ret_steps=use_ret_steps, 
            ckpt_dir=model_id
        )
    
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


def generate_jax_video(
    model_id: str,
    jax_model: WanModel,
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
) -> List[np.ndarray]:
    """Generate video using JAX implementation."""
    print("🔄 Generating video with JAX implementation...")
    
    # Set seed
    set_seed(seed)
    
    # Create hybrid pipeline
    pipeline = HybridPipeline(
        model_path=model_id,
        jax_model=jax_model,
        device="cuda",
        weight_dtype=torch.bfloat16,
    )
    
    # Generate video
    video_tensor = pipeline.generate(
        prompt=prompt,
        image=image,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        seed=seed,
    )
    
    # Convert to numpy frames
    video_frames = []
    for i in range(video_tensor.shape[2]):  # Iterate over frames
        frame = video_tensor[0, :, i, :, :].permute(1, 2, 0)  # HWC format
        frame = (frame * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        video_frames.append(frame)
    
    print("✅ JAX video generation completed")
    return video_frames


def create_side_by_side_video(
    pytorch_frames: List[np.ndarray],
    jax_frames: List[np.ndarray],
    output_path: str,
    fps: int = 24
) -> None:
    """Create a side-by-side comparison video."""
    print("🔄 Creating side-by-side comparison video...")
    
    # Ensure both videos have the same number of frames
    num_frames = min(len(pytorch_frames), len(jax_frames))
    pytorch_frames = pytorch_frames[:num_frames]
    jax_frames = jax_frames[:num_frames]
    
    # Get frame dimensions
    h, w, c = pytorch_frames[0].shape
    
    # Create side-by-side frames
    side_by_side_frames = []
    for i in range(num_frames):
        # Create side-by-side frame
        combined_frame = np.zeros((h, w * 2, c), dtype=np.uint8)
        combined_frame[:, :w] = pytorch_frames[i]
        combined_frame[:, w:] = jax_frames[i]
        
        # Add labels
        cv2.putText(combined_frame, "PyTorch", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, "JAX", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        side_by_side_frames.append(combined_frame)
    
    # Save video
    imageio.mimwrite(
        output_path, 
        side_by_side_frames, 
        fps=fps, 
        quality=8, 
        output_params=["-loglevel", "error"]
    )
    
    print(f"✅ Side-by-side video saved to {output_path}")


def save_individual_videos(
    pytorch_frames: List[np.ndarray],
    jax_frames: List[np.ndarray],
    output_dir: str,
    prompt: str,
    seed: int,
    fps: int = 24
) -> None:
    """Save individual PyTorch and JAX videos."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PyTorch video
    pytorch_path = os.path.join(output_dir, f"pytorch_{prompt[:50].replace('/', '')}_{seed}.mp4")
    imageio.mimwrite(
        pytorch_path, 
        pytorch_frames, 
        fps=fps, 
        quality=8, 
        output_params=["-loglevel", "error"]
    )
    print(f"✅ PyTorch video saved to {pytorch_path}")
    
    # Save JAX video
    jax_path = os.path.join(output_dir, f"jax_{prompt[:50].replace('/', '')}_{seed}.mp4")
    imageio.mimwrite(
        jax_path, 
        jax_frames, 
        fps=fps, 
        quality=8, 
        output_params=["-loglevel", "error"]
    )
    print(f"✅ JAX video saved to {jax_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch and JAX implementations of SkyReels-V2")
    
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
    parser.add_argument("--save_individual", action="store_true", help="Save individual videos in addition to side-by-side")
    parser.add_argument("--seed", type=int, default=None)
    
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
    
    try:
        # Create JAX model
        print("🔄 Creating JAX model...")
        jax_model, config = create_jax_model_from_torch_config(model_id)
        
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
        
        # Generate JAX video
        jax_frames = generate_jax_video(
            model_id=model_id,
            jax_model=jax_model,
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
        )
        
        # Create output filename
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        output_filename = f"comparison_{args.prompt[:50].replace('/', '')}_{args.seed}_{current_time}.mp4"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Create side-by-side video
        create_side_by_side_video(pytorch_frames, jax_frames, output_path, args.fps)
        
        # Save individual videos if requested
        if args.save_individual:
            save_individual_videos(
                pytorch_frames, 
                jax_frames, 
                args.output_dir, 
                args.prompt, 
                args.seed, 
                args.fps
            )
        
        print(f"\n🎉 Comparison completed successfully!")
        print(f"📁 Output saved to: {output_path}")
        print(f"🔢 Seed used: {args.seed}")
        print(f"📊 PyTorch frames: {len(pytorch_frames)}")
        print(f"📊 JAX frames: {len(jax_frames)}")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

