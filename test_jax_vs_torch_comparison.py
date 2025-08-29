#!/usr/bin/env python3
"""
Compare JAX and PyTorch implementations by generating videos sequentially.
Runs each model separately to avoid memory issues, then creates a side-by-side comparison.
"""

import argparse
import gc
import os
import sys
import time
import torch
import jax
import jax.numpy as jnp
import numpy as np
import imageio
from PIL import Image
from diffusers.utils import load_image
from tqdm import tqdm
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob

# Add paths
sys.path.append('.')
sys.path.append('./jax_implementation')

# Import components
from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline
from skyreels_v2_infer.pipelines import resizecrop


def discover_test_cases(batch_folder):
    """Discover image/prompt pairs in a folder."""
    print(f"\n🔍 Discovering test cases in: {batch_folder}")
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    test_cases = []
    
    # Find all text files (prompts)
    prompt_files = glob.glob(os.path.join(batch_folder, "*.txt"))
    
    for prompt_file in sorted(prompt_files):
        base_name = os.path.splitext(prompt_file)[0]
        
        # Look for corresponding image file
        image_file = None
        for ext in image_extensions:
            candidate = base_name + ext
            if os.path.exists(candidate):
                image_file = candidate
                break
        
        if image_file:
            # Read prompt
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            test_case = {
                'name': os.path.basename(base_name),
                'image_path': image_file,
                'prompt': prompt,
                'prompt_file': prompt_file
            }
            test_cases.append(test_case)
            print(f"  ✅ Found: {test_case['name']}")
        else:
            print(f"  ⚠️  No image found for prompt: {os.path.basename(prompt_file)}")
    
    print(f"\n📊 Total test cases found: {len(test_cases)}")
    return test_cases


def run_torch_generation(args, test_case=None):
    """Run PyTorch generation."""
    print("\n" + "="*60)
    print("🔥 Running PyTorch Generation")
    print("="*60)
    
    # Load image
    if test_case:
        image = load_image(test_case['image_path']).convert("RGB")
        prompt = test_case['prompt']
        print(f"Processing: {test_case['name']}")
        print(f"Prompt: {prompt}")
    else:
        image = load_image(args.image).convert("RGB")
        prompt = args.prompt
    if args.resolution == "540P":
        height = 544
        width = 960
    else:
        height = 720
        width = 1280
    
    image_width, image_height = image.size
    if image_height > image_width:
        height, width = width, height
    image = resizecrop(image, height, width)
    
    # Create pipeline
    print("Loading PyTorch pipeline...")
    pipe = Image2VideoPipeline(
        model_path=args.model_path, 
        dit_path=args.model_path,
        use_usp=False,
        offload=args.offload
    )
    
    # Generate
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality"
    
    print(f"Generating {args.num_frames} frames...")
    start_time = time.time()
    
    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        video_frames = pipe(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            num_frames=args.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            shift=args.shift,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            height=height,
            width=width,
        )[0]
    
    torch_time = time.time() - start_time
    print(f"PyTorch generation time: {torch_time:.2f} seconds")
    
    # Clean up
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    
    return video_frames, torch_time


def run_jax_generation(args, test_case=None):
    """Run JAX generation with custom pipeline."""
    print("\n" + "="*60)
    print("🚀 Running JAX Generation (Optimized)")
    print("="*60)
    
    # Set memory optimization
    import os
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    
    # Import JAX components here to avoid early memory allocation
    from jax_implementation.modules import WanModel
    from jax_implementation.utils.weight_converter import load_torch_weights, apply_weights_to_model
    from test_jax_generation_optimized import JAXImage2VideoPipelineOptimized as JAXImage2VideoPipeline
    
    # Load image
    if test_case:
        image = load_image(test_case['image_path']).convert("RGB")
        prompt = test_case['prompt']
        print(f"Processing: {test_case['name']}")
        print(f"Prompt: {prompt}")
    else:
        image = load_image(args.image).convert("RGB")
        prompt = args.prompt
    if args.resolution == "540P":
        height = 544
        width = 960
    else:
        height = 720
        width = 1280
    
    image_width, image_height = image.size
    if image_height > image_width:
        height, width = width, height
    image = resizecrop(image, height, width)
    
    # Create pipeline
    print("Loading JAX pipeline...")
    pipe = JAXImage2VideoPipeline(
        model_path=args.model_path,
        dit_path=args.model_path,
        use_usp=False,
        offload=args.offload
    )
    
    # Generate
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality"
    
    print(f"Generating {args.num_frames} frames...")
    start_time = time.time()
    
    video_frames = pipe(
        prompt=prompt,
        image=image,
        negative_prompt=negative_prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        height=height,
        width=width,
    )[0]
    
    jax_time = time.time() - start_time
    print(f"JAX generation time: {jax_time:.2f} seconds")
    
    # Clean up
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    jax.clear_caches()
    
    return video_frames, jax_time


def compute_video_metrics(torch_frames, jax_frames):
    """Compute numerical error statistics between PyTorch and JAX videos."""
    print("\n" + "="*60)
    print("📊 Computing Numerical Error Statistics")
    print("="*60)
    
    # Ensure both have same number of frames
    min_frames = min(len(torch_frames), len(jax_frames))
    torch_frames = torch_frames[:min_frames]
    jax_frames = jax_frames[:min_frames]
    
    # Storage for metrics
    mse_values = []
    psnr_values = []
    ssim_values = []
    abs_diff_values = []
    rel_diff_values = []
    
    print(f"\nAnalyzing {min_frames} frames...")
    
    for i in range(min_frames):
        # Convert to float32 for accurate comparison
        torch_frame = torch_frames[i].astype(np.float32) / 255.0
        jax_frame = jax_frames[i].astype(np.float32) / 255.0
        
        # Compute MSE
        mse = np.mean((torch_frame - jax_frame) ** 2)
        mse_values.append(mse)
        
        # Compute PSNR (higher is better, 30+ is good)
        psnr_val = psnr(torch_frame, jax_frame, data_range=1.0)
        psnr_values.append(psnr_val)
        
        # Compute SSIM (structural similarity, 1.0 is perfect)
        # Use multichannel for RGB images
        ssim_val = ssim(torch_frame, jax_frame, data_range=1.0, channel_axis=2)
        ssim_values.append(ssim_val)
        
        # Compute absolute and relative differences
        abs_diff = np.abs(torch_frame - jax_frame)
        abs_diff_values.append(abs_diff)
        
        # Relative difference (avoid division by zero)
        rel_diff = abs_diff / (np.abs(torch_frame) + 1e-8)
        rel_diff_values.append(rel_diff)
    
    # Aggregate statistics
    print("\n📈 Frame-by-Frame Statistics:")
    print("-" * 40)
    
    for i in range(min_frames):
        print(f"Frame {i+1:2d}: MSE={mse_values[i]:.6f}, PSNR={psnr_values[i]:.2f}dB, SSIM={ssim_values[i]:.4f}")
    
    print("\n📊 Overall Statistics:")
    print("-" * 40)
    
    # MSE statistics
    print(f"MSE (Mean Squared Error):")
    print(f"  Mean: {np.mean(mse_values):.6f}")
    print(f"  Std:  {np.std(mse_values):.6f}")
    print(f"  Min:  {np.min(mse_values):.6f}")
    print(f"  Max:  {np.max(mse_values):.6f}")
    
    # PSNR statistics
    print(f"\nPSNR (Peak Signal-to-Noise Ratio) [dB]:")
    print(f"  Mean: {np.mean(psnr_values):.2f}")
    print(f"  Std:  {np.std(psnr_values):.2f}")
    print(f"  Min:  {np.min(psnr_values):.2f}")
    print(f"  Max:  {np.max(psnr_values):.2f}")
    print(f"  (Higher is better, >30dB is good, >40dB is excellent)")
    
    # SSIM statistics
    print(f"\nSSIM (Structural Similarity Index):")
    print(f"  Mean: {np.mean(ssim_values):.4f}")
    print(f"  Std:  {np.std(ssim_values):.4f}")
    print(f"  Min:  {np.min(ssim_values):.4f}")
    print(f"  Max:  {np.max(ssim_values):.4f}")
    print(f"  (1.0 is perfect, >0.95 is excellent, >0.90 is good)")
    
    # Pixel-wise statistics
    all_abs_diffs = np.concatenate([d.flatten() for d in abs_diff_values])
    all_rel_diffs = np.concatenate([d.flatten() for d in rel_diff_values])
    
    print(f"\nPixel-wise Absolute Difference (0-1 scale):")
    print(f"  Mean: {np.mean(all_abs_diffs):.6f}")
    print(f"  Std:  {np.std(all_abs_diffs):.6f}")
    print(f"  Max:  {np.max(all_abs_diffs):.6f}")
    print(f"  95th percentile: {np.percentile(all_abs_diffs, 95):.6f}")
    print(f"  99th percentile: {np.percentile(all_abs_diffs, 99):.6f}")
    
    print(f"\nPixel-wise Relative Difference (%):")
    print(f"  Mean: {np.mean(all_rel_diffs)*100:.2f}%")
    print(f"  Std:  {np.std(all_rel_diffs)*100:.2f}%")
    print(f"  Max:  {np.max(all_rel_diffs)*100:.2f}%")
    print(f"  95th percentile: {np.percentile(all_rel_diffs, 95)*100:.2f}%")
    print(f"  99th percentile: {np.percentile(all_rel_diffs, 99)*100:.2f}%")
    
    # Quality assessment
    print("\n🏆 Quality Assessment:")
    print("-" * 40)
    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)
    
    if mean_psnr > 40 and mean_ssim > 0.95:
        print("✅ EXCELLENT: Videos are nearly identical")
    elif mean_psnr > 35 and mean_ssim > 0.90:
        print("✅ VERY GOOD: Minor differences, visually very similar")
    elif mean_psnr > 30 and mean_ssim > 0.85:
        print("⚠️ GOOD: Noticeable but acceptable differences")
    elif mean_psnr > 25 and mean_ssim > 0.80:
        print("⚠️ FAIR: Significant differences visible")
    else:
        print("❌ POOR: Major differences between implementations")
    
    return {
        'mse': mse_values,
        'psnr': psnr_values,
        'ssim': ssim_values,
        'mean_mse': np.mean(mse_values),
        'mean_psnr': np.mean(psnr_values),
        'mean_ssim': np.mean(ssim_values)
    }


def create_side_by_side_video(torch_frames, jax_frames, output_path, fps=24):
    """Create a side-by-side comparison video."""
    print("\n" + "="*60)
    print("🎬 Creating Side-by-Side Comparison Video")
    print("="*60)
    
    # Ensure both have same number of frames
    min_frames = min(len(torch_frames), len(jax_frames))
    torch_frames = torch_frames[:min_frames]
    jax_frames = jax_frames[:min_frames]
    
    # Get dimensions
    h, w = torch_frames[0].shape[:2]
    
    # Create side-by-side frames
    combined_frames = []
    for i, (torch_frame, jax_frame) in enumerate(zip(torch_frames, jax_frames)):
        # Add labels
        torch_labeled = cv2.putText(
            torch_frame.copy(),
            "PyTorch",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        jax_labeled = cv2.putText(
            jax_frame.copy(),
            "JAX",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Combine horizontally
        combined = np.concatenate([torch_labeled, jax_labeled], axis=1)
        combined_frames.append(combined)
    
    # Save video
    imageio.mimwrite(output_path, combined_frames, fps=fps, quality=8)
    print(f"✅ Saved comparison video to: {output_path}")
    
    return combined_frames


def main():
    parser = argparse.ArgumentParser(description="Compare JAX vs PyTorch generation")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-I2V-1.3B-540P")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"], default="540P")
    parser.add_argument("--num_frames", type=int, default=13)
    parser.add_argument("--image", type=str, default="test_image.jpg")
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=5)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, 
                       default="Shapes float around in a background consisting of color gradients")
    parser.add_argument("--torch_only", action="store_true", help="Only run PyTorch")
    parser.add_argument("--jax_only", action="store_true", help="Only run JAX")
    
    # Batch processing arguments
    parser.add_argument("--batch_folder", type=str, help="Folder containing image/prompt pairs for batch processing")
    parser.add_argument("--batch_output", type=str, default="result/batch_comparison", 
                       help="Output folder for batch comparison results")
    
    args = parser.parse_args()
    
    # Download model
    args.model_path = download_model(args.model_id)
    print(f"Model path: {args.model_path}")
    
    # Handle batch processing
    if args.batch_folder:
        # Batch processing mode
        test_cases = discover_test_cases(args.batch_folder)
        if not test_cases:
            print("❌ No test cases found in batch folder!")
            return
        
        # Create batch output directory
        os.makedirs(args.batch_output, exist_ok=True)
        
        print(f"\n🚀 Starting batch processing of {len(test_cases)} test cases...")
        
        all_results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*60}")
            print(f"Processing {i+1}/{len(test_cases)}: {test_case['name']}")
            print(f"{'='*60}")
            
            torch_frames = None
            jax_frames = None
            torch_time = 0
            jax_time = 0
            
            # Create case-specific output directory
            case_output_dir = os.path.join(args.batch_output, test_case['name'])
            os.makedirs(case_output_dir, exist_ok=True)
            
            # Run PyTorch generation
            if not args.jax_only:
                try:
                    torch_frames, torch_time = run_torch_generation(args, test_case)
                    # Save PyTorch video
                    torch_path = os.path.join(case_output_dir, "pytorch_output.mp4")
                    imageio.mimwrite(torch_path, torch_frames, fps=args.fps, quality=8)
                    print(f"Saved PyTorch video to: {torch_path}")
                except Exception as e:
                    print(f"❌ PyTorch generation failed for {test_case['name']}: {e}")
                    if args.torch_only:
                        continue
            
            # Run JAX generation
            if not args.torch_only:
                try:
                    # Set JAX memory allocation
                    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
                    jax_frames, jax_time = run_jax_generation(args, test_case)
                    # Save JAX video
                    jax_path = os.path.join(case_output_dir, "jax_output.mp4")
                    imageio.mimwrite(jax_path, jax_frames, fps=args.fps, quality=8)
                    print(f"Saved JAX video to: {jax_path}")
                except Exception as e:
                    print(f"❌ JAX generation failed for {test_case['name']}: {e}")
                    import traceback
                    traceback.print_exc()
                    if args.jax_only:
                        continue
            
            # Create comparison if both succeeded
            if torch_frames is not None and jax_frames is not None:
                # Compute numerical metrics
                metrics = compute_video_metrics(torch_frames, jax_frames)
                
                # Create side-by-side video
                comparison_path = os.path.join(case_output_dir, "side_by_side_comparison.mp4")
                create_side_by_side_video(torch_frames, jax_frames, comparison_path, args.fps)
                
                # Store results
                result = {
                    'name': test_case['name'],
                    'prompt': test_case['prompt'],
                    'torch_time': torch_time,
                    'jax_time': jax_time,
                    'metrics': metrics
                }
                all_results.append(result)
        
        # Print batch summary
        print(f"\n{'='*60}")
        print("🎯 BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        
        if all_results:
            total_torch_time = sum(r['torch_time'] for r in all_results)
            total_jax_time = sum(r['jax_time'] for r in all_results)
            avg_psnr = np.mean([r['metrics']['mean_psnr'] for r in all_results])
            avg_ssim = np.mean([r['metrics']['mean_ssim'] for r in all_results])
            avg_mse = np.mean([r['metrics']['mean_mse'] for r in all_results])
            
            print(f"Processed: {len(all_results)}/{len(test_cases)} test cases")
            print(f"Total PyTorch time: {total_torch_time:.2f}s")
            print(f"Total JAX time: {total_jax_time:.2f}s")
            if total_jax_time > 0:
                speedup = total_torch_time / total_jax_time
                print(f"Overall speedup: {speedup:.2f}x {'JAX' if speedup > 1 else 'PyTorch'}")
            
            print(f"\nAverage Quality Metrics:")
            print(f"  PSNR: {avg_psnr:.2f} dB")
            print(f"  SSIM: {avg_ssim:.4f}")
            print(f"  MSE:  {avg_mse:.6f}")
            
            # Save summary to file
            summary_path = os.path.join(args.batch_output, "batch_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("BATCH PROCESSING SUMMARY\n")
                f.write("="*60 + "\n\n")
                for result in all_results:
                    f.write(f"Test Case: {result['name']}\n")
                    f.write(f"Prompt: {result['prompt']}\n")
                    f.write(f"PyTorch Time: {result['torch_time']:.2f}s\n")
                    f.write(f"JAX Time: {result['jax_time']:.2f}s\n")
                    f.write(f"PSNR: {result['metrics']['mean_psnr']:.2f} dB\n")
                    f.write(f"SSIM: {result['metrics']['mean_ssim']:.4f}\n")
                    f.write(f"MSE: {result['metrics']['mean_mse']:.6f}\n")
                    f.write("-" * 40 + "\n")
                f.write(f"\nOverall Statistics:\n")
                f.write(f"Total PyTorch time: {total_torch_time:.2f}s\n")
                f.write(f"Total JAX time: {total_jax_time:.2f}s\n")
                f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
                f.write(f"Average SSIM: {avg_ssim:.4f}\n")
                f.write(f"Average MSE: {avg_mse:.6f}\n")
            
            print(f"\nSummary saved to: {summary_path}")
        
        print("\n✅ Batch processing complete!")
        return
    
    # Single case processing mode (original behavior)
    # Create output directory
    os.makedirs("result/comparison", exist_ok=True)
    
    torch_frames = None
    jax_frames = None
    torch_time = 0
    jax_time = 0
    
    # Run PyTorch generation
    if not args.jax_only:
        try:
            torch_frames, torch_time = run_torch_generation(args)
            # Save PyTorch video
            torch_path = "result/comparison/pytorch_output.mp4"
            imageio.mimwrite(torch_path, torch_frames, fps=args.fps, quality=8)
            print(f"Saved PyTorch video to: {torch_path}")
        except Exception as e:
            print(f"❌ PyTorch generation failed: {e}")
            if args.torch_only:
                return
    
    # Run JAX generation
    if not args.torch_only:
        try:
            # Set JAX memory allocation
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            jax_frames, jax_time = run_jax_generation(args)
            # Save JAX video
            jax_path = "result/comparison/jax_output.mp4"
            imageio.mimwrite(jax_path, jax_frames, fps=args.fps, quality=8)
            print(f"Saved JAX video to: {jax_path}")
        except Exception as e:
            print(f"❌ JAX generation failed: {e}")
            import traceback
            traceback.print_exc()
            if args.jax_only:
                return
    
    # Create comparison if both succeeded
    if torch_frames is not None and jax_frames is not None:
        # Compute numerical metrics
        metrics = compute_video_metrics(torch_frames, jax_frames)
        
        # Create side-by-side video
        comparison_path = "result/comparison/side_by_side_comparison.mp4"
        create_side_by_side_video(torch_frames, jax_frames, comparison_path, args.fps)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Performance Summary")
        print("="*60)
        print(f"PyTorch time: {torch_time:.2f}s")
        print(f"JAX time: {jax_time:.2f}s")
        if jax_time > 0:
            speedup = torch_time / jax_time
            if speedup > 1:
                print(f"JAX is {speedup:.2f}x faster")
            else:
                print(f"PyTorch is {1/speedup:.2f}x faster")
        
        print("\n" + "="*60)
        print("🎯 Final Quality Metrics")
        print("="*60)
        print(f"Mean PSNR: {metrics['mean_psnr']:.2f} dB")
        print(f"Mean SSIM: {metrics['mean_ssim']:.4f}")
        print(f"Mean MSE:  {metrics['mean_mse']:.6f}")
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()