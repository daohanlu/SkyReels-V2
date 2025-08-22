#!/usr/bin/env python3
"""
Test JAX diffusion model implementation with actual generation.
Modified from generate_video.py to use JAX transformer instead of PyTorch.
Includes a JIT-compiled option for the denoising loop.
"""

import argparse
import gc
import os
import sys
import time
from functools import partial

import torch
import jax
import jax.numpy as jnp
import numpy as np
import imageio
from PIL import Image
from diffusers.utils import load_image
from tqdm import tqdm

# Add paths
sys.path.append('.')
sys.path.append('./jax_implementation')

# Import standard components
from skyreels_v2_infer.modules import download_model, get_vae, get_text_encoder, get_image_encoder
from skyreels_v2_infer.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
from skyreels_v2_infer.pipelines import resizecrop

# Import JAX components
from jax_implementation.modules import WanModel
from jax_implementation.utils.weight_converter import load_torch_weights, apply_weights_to_model


class JAXImage2VideoPipeline:
    """
    Modified Image2VideoPipeline that uses a JAX transformer with an optional JIT-compiled denoising step.
    """
    
    def __init__(self, model_path: str, dit_path: str, use_usp: bool = False, offload: bool = False, jit: bool = False):
        """Initialize pipeline with JAX transformer."""
        self.model_path = model_path
        self.dit_path = dit_path
        self.use_usp = use_usp
        self.offload = offload
        self.device = "cuda"
        self.jit = jit
        
        # Load PyTorch components (VAE, text encoder, CLIP)
        load_device = "cpu" if offload else self.device
        weight_dtype = torch.bfloat16
        
        print("📚 Loading PyTorch components...")
        vae_model_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        self.vae = get_vae(vae_model_path, load_device, weight_dtype=torch.float32)
        self.text_encoder = get_text_encoder(model_path, load_device, weight_dtype)
        self.clip = get_image_encoder(model_path, load_device, weight_dtype)
        self.scheduler = FlowUniPCMultistepScheduler()
        
        # Load JAX transformer
        print("🔧 Loading JAX transformer...")
        torch_weights, config = load_torch_weights(dit_path)
        
        self.transformer = WanModel(
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
        
        self.transformer = apply_weights_to_model(self.transformer, torch_weights)
        self.transformer.dtype = jnp.bfloat16 # Use JAX bfloat16
        
        if self.jit:
            print("🚀 Compiling JIT function for denoising step...")
            # Note: 'guidance_scale' must be static for JIT compilation.
            self.predict_step_jit = jax.jit(self._predict_step, static_argnames=('guidance_scale',))
        
        print("✅ JAX transformer loaded and weights applied")

    @staticmethod
    def _to_jax(tensor: torch.Tensor) -> jax.Array:
        """Converts a PyTorch tensor to a JAX array, handling bfloat16."""
        if tensor is None:
            return None
        if tensor.dtype == torch.bfloat16:
            # Convert bfloat16 to float32 numpy, then to jax bfloat16
            return jnp.array(tensor.cpu().to(torch.float32).numpy(), dtype=jnp.bfloat16)
        else:
            return jnp.array(tensor.cpu().numpy())

    def _from_jax(self, array: jax.Array, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Converts a JAX array back to a PyTorch tensor, handling bfloat16."""
        if array is None:
            return None
        if array.dtype == jnp.bfloat16:
            # Convert jax bfloat16 to float32 numpy, then to torch bfloat16
            output_np = np.array(array, dtype=np.float32)
            return torch.from_numpy(output_np).to(self.device).to(dtype)
        else:
            return torch.from_numpy(np.array(array)).to(self.device)

    def _predict_step(self, latent_model_input, t, arg_c, arg_null, guidance_scale):
        """
        JAX-based function for a single denoising prediction step.
        This function is intended to be JIT-compiled.
        """
        # Run JAX transformer for conditional and unconditional
        noise_pred_cond = self.transformer(
            latent_model_input, t, arg_c['context'], arg_c['clip_fea'], arg_c['y']
        )
        noise_pred_uncond = self.transformer(
            latent_model_input, t, arg_null['context'], arg_null['clip_fea'], arg_null['y']
        )
        # Classifier-free guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        return noise_pred

    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: str = "",
        num_frames: int = 97,
        height: int = 544,
        width: int = 960,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        shift: float = 8.0,
        generator: torch.Generator = None,
    ):
        """Generate video (matching Image2VideoPipeline interface)."""
        latent_height, latent_width = height // 8, width // 8
        latent_length = (num_frames - 1) // 4 + 1
        h, w = latent_height * 8, latent_width * 8
        image_resized = image.resize((w, h))
        
        # Encode image with VAE
        img = torch.from_numpy(np.array(image_resized)).permute(2, 0, 1).unsqueeze(0)
        img = img.to(self.device).to(torch.float32)
        if img.max() > 1:
            img = (img.float() / 127.5) - 1.0
        
        # Create video with padding frames for VAE encoding
        F = num_frames  # Total frames
        padding_video = torch.zeros(img.shape[0], 3, F - 1, h, w, device=self.device)
        img = img.unsqueeze(2)  # Add temporal dimension
        img_cond = torch.concat([img, padding_video], dim=2)
        
        self.vae.to(self.device)
        img_cond = self.vae.encode(img_cond)
        if self.offload:
            self.vae.cpu()
            torch.cuda.empty_cache()
        
        # Create mask (1 for first frame, 0 for others)
        mask = torch.ones_like(img_cond)
        mask[:, :, 1:] = 0
        
        # Combine mask and image conditioning
        y = torch.cat([mask[:, :4], img_cond], dim=1)
        
        # Encode image with CLIP
        self.clip.to(self.device)
        clip_context = self.clip.encode_video(img)
        if self.offload:
            self.clip.cpu()
            torch.cuda.empty_cache()
        
        # Encode text prompts
        self.text_encoder.to(self.device)
        context = self.text_encoder.encode(prompt).to(self.device)
        context_null = self.text_encoder.encode(negative_prompt).to(self.device)
        if self.offload:
            self.text_encoder.cpu()
            torch.cuda.empty_cache()
        
        # Initialize latents
        latent = torch.randn(
            16, latent_length, latent_height, latent_width, 
            dtype=torch.float32, generator=generator, device=self.device
        )
        
        # Denoising loop
        with torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
            self.scheduler.set_timesteps(num_inference_steps, device=self.device, shift=shift)
            timesteps = self.scheduler.timesteps

            if self.jit:
                # --- JIT-COMPILED DENOISING LOOP ---
                print("⚡ Running JIT-compiled denoising loop...")
                arg_c = {"context": context, "clip_fea": clip_context, "y": y}
                arg_null = {"context": context_null, "clip_fea": clip_context, "y": y}
                
                # Convert all conditioning tensors to JAX arrays ONCE before the loop
                arg_c_jax = jax.tree_util.tree_map(self._to_jax, arg_c)
                arg_null_jax = jax.tree_util.tree_map(self._to_jax, arg_null)
                
                for _, t in enumerate(tqdm(timesteps, desc="Denoising (JIT)")):
                    latent_model_input = torch.stack([latent]).to(self.device)
                    
                    # Convert only the changing latent tensor inside the loop
                    latent_model_input_jax = self._to_jax(latent_model_input)
                    t_jax = jnp.array([t.item() if hasattr(t, 'item') else t])

                    # Call the single compiled JAX function
                    noise_pred_jax = self.predict_step_jit(
                        latent_model_input_jax, t_jax, arg_c_jax, arg_null_jax, guidance_scale=guidance_scale
                    )

                    # Convert result back to PyTorch
                    noise_pred = self._from_jax(noise_pred_jax, dtype=torch.bfloat16)
                    
                    if noise_pred.dim() == 5 and noise_pred.shape[0] == 1:
                        noise_pred = noise_pred.squeeze(0)
                    
                    latent = self.scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)

            else:
                # --- STANDARD (EAGER) DENOISING LOOP ---
                print("🐌 Running standard (eager) denoising loop...")
                arg_c = {"context": context, "clip_fea": clip_context, "y": y}
                arg_null = {"context": context_null, "clip_fea": clip_context, "y": y}
                
                for _, t in enumerate(tqdm(timesteps, desc="Denoising (Eager)")):
                    latent_model_input = torch.stack([latent]).to(self.device)
                    
                    # Manual conversion and execution for conditional pass
                    noise_pred_cond = self._from_jax(
                        self.transformer(self._to_jax(latent_model_input), jnp.array([t.item()]), 
                                         self._to_jax(arg_c['context']), self._to_jax(arg_c['clip_fea']), self._to_jax(arg_c['y']))
                    )
                    
                    # Manual conversion and execution for unconditional pass
                    noise_pred_uncond = self._from_jax(
                        self.transformer(self._to_jax(latent_model_input), jnp.array([t.item()]), 
                                         self._to_jax(arg_null['context']), self._to_jax(arg_null['clip_fea']), self._to_jax(arg_null['y']))
                    )
                    
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    if noise_pred.dim() == 5 and noise_pred.shape[0] == 1:
                        noise_pred = noise_pred.squeeze(0)

                    latent = self.scheduler.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False)[0].squeeze(0)

            # --- VAE Decoding ---
            self.vae.to(self.device)
            videos = self.vae.decode(latent)
            videos = (videos / 2 + 0.5).clamp(0, 1)
            videos = [video.permute(1, 2, 3, 0) * 255 for video in videos]
            videos = [video.cpu().numpy().astype(np.uint8) for video in videos]
        
        return videos


def main():
    parser = argparse.ArgumentParser(description="Test JAX generation")
    parser.add_argument("--outdir", type=str, default="jax_test")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-I2V-1.3B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"], default="540P")
    parser.add_argument("--num_frames", type=int, default=57)
    parser.add_argument("--image", type=str, default="test_image.jpg")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jit", action="store_true", help="Enable JIT compilation for the denoising loop.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Shapes float around in a background consisting of color gradients",
    )
    
    args = parser.parse_args()
    
    args.model_id = download_model(args.model_id)
    print("model_id:", args.model_id)
    
    height, width = (544, 960) if args.resolution == "540P" else (720, 1280)
    
    image = load_image(args.image).convert("RGB")
    image_width, image_height = image.size
    if image_height > image_width:
        height, width = width, height
    image = resizecrop(image, height, width)
    
    # Create JAX pipeline
    print("init JAX img2video pipeline")
    pipe = JAXImage2VideoPipeline(
        model_path=args.model_id, 
        dit_path=args.model_id, 
        use_usp=False, 
        offload=args.offload,
        jit=args.jit
    )
    
    # Set up generation parameters
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality"
    
    kwargs = {
        "prompt": args.prompt,
        "negative_prompt": negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "height": height,
        "width": width,
        "image": image.convert("RGB"),
    }
    
    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"infer kwargs: {kwargs}")
    
    # Generate video
    start_time = time.time()
    video_frames = pipe(**kwargs)[0]
    end_time = time.time()
    
    print(f"⏱️ Generation time: {end_time - start_time:.2f} seconds")
    
    # Save video
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_out_file = f"jax_{'jit' if args.jit else 'eager'}_{args.prompt[:40].replace(' ','_')}_{args.seed}_{current_time}.mp4"
    output_path = os.path.join(save_dir, video_out_file)
    imageio.mimwrite(output_path, video_frames, fps=args.fps, quality=8, output_params=["-loglevel", "error"])
    
    print(f"✅ Video saved to: {output_path}")
    
    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()