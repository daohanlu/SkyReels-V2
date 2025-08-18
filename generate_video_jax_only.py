#!/usr/bin/env python3
"""
JAX-only video generation for testing the JAX implementation.
"""

import argparse
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
sys.path.append('./jax_implementation')

# Import JAX components
from jax_implementation.modules import WanModel


def set_seed(seed: int):
    """Set random seed for JAX."""
    random.seed(seed)
    np.random.seed(seed)
    jax.random.PRNGKey(seed)


def load_image_and_preprocess(image_path: str, height: int, width: int) -> Image.Image:
    """Load and preprocess image."""
    if image_path:
        image = load_image(image_path)
        image_width, image_height = image.size
        if image_height > image_width:
            height, width = width, height
        # Simple resize for now
        image = image.resize((width, height))
        return image.convert("RGB")
    return None


def create_jax_model_simple() -> WanModel:
    """Create a simple JAX model for testing."""
    print("Creating simple JAX model...")
    
    # Create a smaller model for testing
    jax_model = WanModel(
        model_type='i2v',
        patch_size=(1, 2, 2),
        text_len=64,
        in_dim=16,
        dim=256,  # Even smaller dimension
        ffn_dim=1024,  # Smaller FFN
        freq_dim=64,
        text_dim=256,
        out_dim=16,
        num_heads=4,
        num_layers=2,  # Very few layers
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        inject_sample_info=False,
        eps=1e-6,
    )
    
    print(f"✅ JAX model created")
    print(f"   Model type: {jax_model.model_type}")
    print(f"   Hidden dim: {jax_model.dim}")
    print(f"   Num layers: {jax_model.num_layers}")
    print(f"   Num heads: {jax_model.num_heads}")
    
    return jax_model


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


def encode_text_jax(text: str, text_len: int = 64, text_dim: int = 256) -> jnp.ndarray:
    """Encode text for JAX model."""
    try:
        from transformers import T5Tokenizer, T5EncoderModel
        import torch
        
        # Load T5 model
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        text_encoder = T5EncoderModel.from_pretrained("t5-base")
        
        # Tokenize and encode
        inputs = tokenizer(text, return_tensors="pt", max_length=text_len, truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = text_encoder(**inputs)
            text_features = outputs.last_hidden_state  # [1, seq_len, 768]
        
        # Project to expected dimension (768 -> text_dim)
        projection = torch.nn.Linear(768, text_dim)
        text_features = projection(text_features)
        
        # Convert to JAX array
        return jnp.array(text_features.detach().numpy())
        
    except ImportError:
        print("Warning: T5 not available, using dummy encoding")
        return jnp.random.normal(0, 1, (1, text_len, text_dim)).astype(jnp.float32)


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
    seed: int
) -> np.ndarray:
    """Generate video using JAX model."""
    print("Generating video with JAX model...")
    
    # Set seed for JAX
    key = jax.random.PRNGKey(seed)
    
    # Encode inputs
    clip_features = encode_image_clip_jax(image, height, width)
    text_features = encode_text_jax(prompt, jax_model.text_len, jax_model.text_dim)
    negative_text_features = encode_text_jax(negative_prompt, jax_model.text_len, jax_model.text_dim)
    
    # Calculate latent dimensions
    latent_height = height // 16
    latent_width = width // 16
    
    # Initialize noise with same seed as PyTorch
    key, noise_key = jax.random.split(key)
    latents = jax.random.normal(noise_key, (1, jax_model.in_dim, num_frames, latent_height, latent_width))
    
    # Create timesteps
    timesteps = jnp.linspace(0, 1000, num_inference_steps, dtype=jnp.int32)
    
    # Simple diffusion loop (simplified for now)
    print(f"Running {num_inference_steps} inference steps...")
    
    for i, t in enumerate(timesteps):
        if i % 2 == 0:
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
        
        # Ensure we have exactly 3 channels (RGB)
        if frame.shape[2] > 3:
            frame = frame[:, :, :3]  # Take first 3 channels
        elif frame.shape[2] < 3:
            # Pad with zeros if we have fewer than 3 channels
            padding = jnp.zeros((frame.shape[0], frame.shape[1], 3 - frame.shape[2]), dtype=jnp.uint8)
            frame = jnp.concatenate([frame, padding], axis=2)
        
        video_frames.append(np.array(frame))
    
    return np.array(video_frames)


def main():
    """Main function to generate video using JAX only."""
    parser = argparse.ArgumentParser(description="Generate video using JAX only")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--inference_steps", type=int, default=5)
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"], default="540P")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", type=str, default="jax_only")
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
    print("JAX-Only Video Generation")
    print("=" * 80)
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
    
    # Initialize JAX model
    print("Initializing JAX model...")
    try:
        jax_model = create_jax_model_simple()
    except Exception as e:
        print(f"❌ Failed to initialize JAX model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate video
    print("\n" + "=" * 60)
    print("GENERATING VIDEO")
    print("=" * 60)
    
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
        seed=args.seed
    )
    jax_time = time.time() - jax_start_time
    
    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    base_filename = f"jax_video_{args.seed}_{current_time}"
    
    # Save video
    jax_video_path = os.path.join(save_dir, f"{base_filename}.mp4")
    imageio.mimwrite(jax_video_path, jax_frames, fps=24, quality=8)
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"JAX generation time: {jax_time:.2f}s")
    print(f"JAX video shape: {jax_frames.shape}")
    print(f"JAX video saved to: {jax_video_path}")
    print("=" * 60)
    
    print("🎉 JAX video generation completed!")
    print("Note: JAX model uses random weights (no pre-trained weights loaded)")


if __name__ == "__main__":
    main()
