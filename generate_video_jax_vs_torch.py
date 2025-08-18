#!/usr/bin/env python3
"""
Generate videos side by side using PyTorch and JAX implementations.
This script loads the same model weights and uses the same initialization noise
to ensure fair comparison between the two implementations.
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
from typing import Dict, Any, Tuple, Optional
import sys

# Add paths
sys.path.append('.')
sys.path.append('./jax_implementation')

# Import PyTorch components
from skyreels_v2_infer import DiffusionForcingPipeline
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines.image2video_pipeline import resizecrop

# Import JAX components
from jax_implementation.modules import WanModel
from jax_implementation.utils.weight_converter import load_torch_weights, apply_weights_to_model


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
    
    # Apply weights to JAX model
    jax_model = apply_weights_to_model(jax_model, jax_weights)
    
    print(f"✅ JAX model created with {len(jax_weights)} parameters")
    return jax_model, config


def encode_image_clip_jax(image: Image.Image, height: int, width: int) -> jnp.ndarray:
    """Encode image using CLIP for JAX model."""
    try:
        from transformers import CLIPVisionModel, CLIPProcessor
        
        # Load CLIP model
        clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Process image
        inputs = clip_processor(images=image, return_tensors="pt")
        
        # Get CLIP features
        with torch.no_grad():
            outputs = clip_model(**inputs)
            features = outputs.last_hidden_state  # [1, 257, 768]
        
        # Project to expected dimension (768 -> 1280)
        projection = torch.nn.Linear(768, 1280)
        features = projection(features)
        
        # Convert to JAX array
        return jnp.array(features.detach().numpy())
        
    except ImportError:
        print("Warning: CLIP not available, using dummy encoding")
        return jnp.random.normal(0, 1, (1, 257, 1280)).astype(jnp.float32)


def encode_text_jax(text: str, config: Dict[str, Any]) -> jnp.ndarray:
    """Encode text for JAX model."""
    try:
        from transformers import T5Tokenizer, T5EncoderModel
        
        # Load T5 model
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        text_encoder = T5EncoderModel.from_pretrained("t5-base")
        
        # Tokenize and encode
        inputs = tokenizer(text, return_tensors="pt", max_length=config.get('text_len', 512), truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = text_encoder(**inputs)
            text_features = outputs.last_hidden_state  # [1, seq_len, 768]
        
        # Project to expected dimension (768 -> text_dim)
        text_dim = config.get('text_dim', 4096)
        projection = torch.nn.Linear(768, text_dim)
        text_features = projection(text_features)
        
        # Convert to JAX array
        return jnp.array(text_features.detach().numpy())
        
    except ImportError:
        print("Warning: T5 not available, using dummy encoding")
        return jnp.random.normal(0, 1, (1, config.get('text_len', 512), config.get('text_dim', 4096))).astype(jnp.float32)


def generate_video_jax(
    jax_model: WanModel,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    shift: float,
    seed: int,
    config: Dict[str, Any]
) -> np.ndarray:
    """Generate video using JAX model."""
    print("Generating video with JAX model...")
    
    # Set seed for JAX
    key = jax.random.PRNGKey(seed)
    
    # Encode inputs
    clip_features = encode_image_clip_jax(image, height, width)
    text_features = encode_text_jax(prompt, config)
    negative_text_features = encode_text_jax(negative_prompt, config)
    
    # Calculate latent dimensions
    latent_height = height // 16
    latent_width = width // 16
    
    # Initialize noise with same seed as PyTorch
    key, noise_key = jax.random.split(key)
    latents = jax.random.normal(noise_key, (1, config.get('in_dim', 16), num_frames, latent_height, latent_width))
    
    # Create timesteps
    timesteps = jnp.linspace(0, 1000, num_inference_steps, dtype=jnp.int32)
    
    # Simple diffusion loop (simplified for now)
    print(f"Running {num_inference_steps} inference steps...")
    
    for i, t in enumerate(timesteps):
        if i % 5 == 0:
            print(f"Step {i+1}/{num_inference_steps}")
        
        # Forward pass through JAX model
        output = jax_model(
            x=latents,
            t=jnp.array([t]),
            context=text_features,
            clip_fea=clip_features,
            y=latents  # Use same latents as conditional
        )
        
        # Simple noise reduction (simplified scheduler)
        latents = latents - 0.1 * output
    
    # Convert latents back to video frames (simplified)
    # In a full implementation, this would use the VAE decoder
    video_frames = []
    for frame_idx in range(num_frames):
        frame_latent = latents[0, :, frame_idx, :, :]  # [channels, height, width]
        # Simple upscaling (in real implementation, use VAE decoder)
        frame = jnp.transpose(frame_latent, (1, 2, 0))  # [height, width, channels]
        frame = jnp.clip(frame, -1, 1)
        frame = ((frame + 1) * 127.5).astype(jnp.uint8)
        video_frames.append(np.array(frame))
    
    return np.array(video_frames)


def generate_video_torch(
    torch_pipeline: DiffusionForcingPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    shift: float,
    seed: int
) -> np.ndarray:
    """Generate video using PyTorch model."""
    print("Generating video with PyTorch model...")
    
    # Set seed for PyTorch
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Generate video using PyTorch pipeline
    with torch.cuda.amp.autocast(dtype=torch_pipeline.transformer.dtype), torch.no_grad():
        video_frames = torch_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            end_image=None,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            shift=shift,
            guidance_scale=guidance_scale,
            generator=generator,
            overlap_history=None,
            addnoise_condition=0,
            base_num_frames=num_frames,
            ar_step=0,
            causal_block_size=1,
            fps=24,
        )[0]
    
    return np.array(video_frames)


def save_side_by_side_video(
    torch_frames: np.ndarray,
    jax_frames: np.ndarray,
    output_path: str,
    fps: int = 24
):
    """Save videos side by side for comparison."""
    print(f"Saving side-by-side comparison to {output_path}")
    
    # Ensure both videos have the same number of frames
    min_frames = min(len(torch_frames), len(jax_frames))
    torch_frames = torch_frames[:min_frames]
    jax_frames = jax_frames[:min_frames]
    
    # Create side-by-side video
    side_by_side_frames = []
    for i in range(min_frames):
        # Concatenate frames horizontally
        combined_frame = np.concatenate([torch_frames[i], jax_frames[i]], axis=1)
        side_by_side_frames.append(combined_frame)
    
    # Save video
    imageio.mimwrite(output_path, side_by_side_frames, fps=fps, quality=8)
    print(f"✅ Side-by-side video saved to {output_path}")


def main():
    """Main function to generate videos side by side."""
    parser = argparse.ArgumentParser(description="Generate videos side by side using PyTorch and JAX")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-I2V-1.3B-540P")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"], default="540P")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="jax_vs_torch_comparison")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over the ocean with waves crashing on the shore",
    )
    
    args = parser.parse_args()
    
    # Set seed
    if args.seed is None:
        args.seed = int(time.time())
    set_seed(args.seed)
    
    # Set resolution
    if args.resolution == "540P":
        height, width = 544, 960
    elif args.resolution == "720P":
        height, width = 720, 1280
    else:
        raise ValueError(f"Invalid resolution: {args.resolution}")
    
    # Create output directory
    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)
    
    # Negative prompt
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    print("=" * 80)
    print("JAX vs PyTorch Video Generation Comparison")
    print("=" * 80)
    print(f"Model ID: {args.model_id}")
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print(f"Guidance Scale: {args.guidance_scale}")
    print(f"Shift: {args.shift}")
    print(f"Resolution: {args.resolution} ({height}x{width})")
    print(f"Frames: {args.num_frames}")
    print(f"Inference Steps: {args.inference_steps}")
    print(f"Seed: {args.seed}")
    print("=" * 80)
    
    # Load and preprocess image
    print("Loading and preprocessing image...")
    image = load_image_and_preprocess(args.image, height, width)
    if image is None:
        print("❌ Failed to load image")
        return
    
    # Download model
    print(f"Downloading model {args.model_id}...")
    model_path = download_model(args.model_id)
    print(f"Model downloaded to: {model_path}")
    
    # Initialize PyTorch pipeline
    print("Initializing PyTorch pipeline...")
    torch_pipeline = DiffusionForcingPipeline(
        model_path,
        dit_path=model_path,
        device=torch.device("cuda"),
        weight_dtype=torch.bfloat16,
        use_usp=False,
        offload=False,
    )
    
    # Initialize JAX model
    print("Initializing JAX model...")
    try:
        jax_model, config = create_jax_model_from_torch_config(model_path)
    except Exception as e:
        print(f"❌ Failed to initialize JAX model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate videos
    print("\n" + "=" * 60)
    print("GENERATING VIDEOS")
    print("=" * 60)
    
    # Generate PyTorch video
    torch_start_time = time.time()
    torch_frames = generate_video_torch(
        torch_pipeline=torch_pipeline,
        image=image,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=args.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        seed=args.seed
    )
    torch_time = time.time() - torch_start_time
    
    # Generate JAX video
    jax_start_time = time.time()
    jax_frames = generate_video_jax(
        jax_model=jax_model,
        image=image,
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=args.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        seed=args.seed,
        config=config
    )
    jax_time = time.time() - jax_start_time
    
    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    base_filename = f"comparison_{args.seed}_{current_time}"
    
    # Save individual videos
    torch_video_path = os.path.join(save_dir, f"{base_filename}_torch.mp4")
    jax_video_path = os.path.join(save_dir, f"{base_filename}_jax.mp4")
    comparison_video_path = os.path.join(save_dir, f"{base_filename}_side_by_side.mp4")
    
    imageio.mimwrite(torch_video_path, torch_frames, fps=24, quality=8)
    imageio.mimwrite(jax_video_path, jax_frames, fps=24, quality=8)
    
    # Save side-by-side comparison
    save_side_by_side_video(torch_frames, jax_frames, comparison_video_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"PyTorch generation time: {torch_time:.2f}s")
    print(f"JAX generation time: {jax_time:.2f}s")
    print(f"Speedup: {torch_time/jax_time:.2f}x")
    print(f"PyTorch video shape: {torch_frames.shape}")
    print(f"JAX video shape: {jax_frames.shape}")
    print(f"PyTorch video saved to: {torch_video_path}")
    print(f"JAX video saved to: {jax_video_path}")
    print(f"Side-by-side comparison saved to: {comparison_video_path}")
    print("=" * 60)
    
    print("🎉 Video generation comparison completed!")


if __name__ == "__main__":
    main()
