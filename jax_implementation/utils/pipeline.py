import torch
import jax
import jax.numpy as jnp
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from diffusers.video_processor import VideoProcessor

from skyreels_v2_infer.modules import get_vae, get_text_encoder
from skyreels_v2_infer.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..modules import WanModel


class HybridPipeline:
    """
    Hybrid pipeline that uses PyTorch VAE + JAX transformer.
    """
    
    def __init__(
        self,
        model_path: str,
        jax_model: WanModel,
        device: str = "cuda",
        weight_dtype=torch.bfloat16,
    ):
        """
        Initialize the hybrid pipeline.
        
        Args:
            model_path: Path to the PyTorch model (for VAE and text encoder)
            jax_model: JAX transformer model
            device: Device to run PyTorch components on
            weight_dtype: Weight data type for PyTorch components
        """
        self.jax_model = jax_model
        self.device = device
        
        # Load PyTorch components
        vae_model_path = f"{model_path}/Wan2.1_VAE.pth"
        self.vae = get_vae(vae_model_path, device, weight_dtype=torch.float32)
        self.text_encoder = get_text_encoder(model_path, device, weight_dtype)
        self.video_processor = VideoProcessor(vae_scale_factor=16)
        
        # Scheduler
        self.scheduler = FlowUniPCMultistepScheduler()
        
        # JAX device
        self.jax_device = jax.devices()[0] if jax.devices() else None
    
    def encode_image(
        self, 
        image: Image.Image, 
        height: int, 
        width: int, 
        num_frames: int
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Encode image using PyTorch VAE.
        
        Args:
            image: Input image
            height: Target height
            width: Target width
            num_frames: Number of frames
            
        Returns:
            Tuple of (prefix_video, prefix_video_latent_length)
        """
        # Resize image
        image = image.resize((width, height))
        prefix_video = np.array(image).transpose(2, 0, 1)
        prefix_video = torch.tensor(prefix_video).unsqueeze(1)
        
        if prefix_video.dtype == torch.uint8:
            prefix_video = (prefix_video.float() / (255.0 / 2.0)) - 1.0
        
        prefix_video = prefix_video.to(self.device)
        prefix_video = [self.vae.encode(prefix_video.unsqueeze(0))[0]]
        
        # Handle causal block size alignment
        causal_block_size = self.jax_model.num_frame_per_block
        if prefix_video[0].shape[1] % causal_block_size != 0:
            truncate_len = prefix_video[0].shape[1] % causal_block_size
            print("The length of prefix video is truncated for the causal block size alignment.")
            prefix_video[0] = prefix_video[0][:, :prefix_video[0].shape[1] - truncate_len]
        
        prefix_video_latent_length = prefix_video[0].shape[1]
        return prefix_video, prefix_video_latent_length
    
    def encode_prompt(
        self, 
        prompt: str, 
        negative_prompt: str = ""
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts using PyTorch text encoder.
        
        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            
        Returns:
            Tuple of (prompt_embeds, negative_prompt_embeds)
        """
        prompt_embeds = self.text_encoder(prompt)
        negative_prompt_embeds = self.text_encoder(negative_prompt)
        
        return prompt_embeds, negative_prompt_embeds
    
    def denoise_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        guidance_scale: float = 6.0,
        clip_fea: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform one denoising step using JAX transformer.
        
        Args:
            latents: Current latents
            timestep: Current timestep
            prompt_embeds: Prompt embeddings
            negative_prompt_embeds: Negative prompt embeddings
            guidance_scale: Guidance scale for classifier-free guidance
            clip_fea: CLIP features for i2v mode
            y: Conditional video inputs for i2v mode
            
        Returns:
            Denoised latents
        """
        # Convert to JAX arrays
        latents_jax = jnp.array(latents.cpu().numpy())
        timestep_jax = jnp.array(timestep.cpu().numpy())
        prompt_embeds_jax = jnp.array(prompt_embeds.cpu().numpy())
        negative_prompt_embeds_jax = jnp.array(negative_prompt_embeds.cpu().numpy())
        
        if clip_fea is not None:
            clip_fea_jax = jnp.array(clip_fea.cpu().numpy())
        else:
            clip_fea_jax = None
            
        if y is not None:
            y_jax = jnp.array(y.cpu().numpy())
        else:
            y_jax = None
        
        # Run JAX model
        def model_fn(latents, timestep, context, clip_fea, y):
            return self.jax_model(latents, timestep, context, clip_fea, y)
        
        # Apply classifier-free guidance
        if guidance_scale > 1.0:
            # Unconditional prediction
            noise_pred_uncond = model_fn(
                latents_jax, timestep_jax, negative_prompt_embeds_jax, clip_fea_jax, y_jax
            )
            
            # Conditional prediction
            noise_pred_cond = model_fn(
                latents_jax, timestep_jax, prompt_embeds_jax, clip_fea_jax, y_jax
            )
            
            # Combine predictions
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = model_fn(
                latents_jax, timestep_jax, prompt_embeds_jax, clip_fea_jax, y_jax
            )
        
        # Convert back to PyTorch
        noise_pred = torch.from_numpy(np.array(noise_pred)).to(latents.device)
        
        return noise_pred
    
    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        height: int = 544,
        width: int = 960,
        num_frames: int = 97,
        num_inference_steps: int = 30,
        guidance_scale: float = 6.0,
        negative_prompt: str = "",
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate video using the hybrid pipeline.
        
        Args:
            prompt: Text prompt
            image: Input image for i2v mode
            height: Video height
            width: Video width
            num_frames: Number of frames
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            negative_prompt: Negative prompt
            seed: Random seed
            
        Returns:
            Generated video tensor
        """
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Encode prompts
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt)
        
        # Initialize latents
        latents = torch.randn(
            (1, 16, num_frames, height // 16, width // 16),
            device=self.device,
            dtype=torch.float32
        )
        
        # Encode image if provided
        clip_fea = None
        y = None
        if image is not None:
            prefix_video, prefix_video_latent_length = self.encode_image(
                image, height, width, num_frames
            )
            y = prefix_video[0]
            
            # For now, we'll skip CLIP encoding as it requires additional setup
            # clip_fea = self.encode_clip_features(image)
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Denoising loop
        for i, timestep in enumerate(timesteps):
            # Predict noise
            noise_pred = self.denoise_step(
                latents,
                timestep,
                prompt_embeds,
                negative_prompt_embeds,
                guidance_scale,
                clip_fea,
                y,
            )
            
            # Update latents
            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
        
        # Decode latents
        video = self.vae.decode(latents)
        
        return video

