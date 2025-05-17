#!/usr/bin/env python
# InstructPix2Pix Optimization Project
# Main script for running experiments with different optimization techniques
# and assessing output quality.

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from datasets import load_from_disk
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
import torch.nn.functional as F
# Corrected import for scikit-image SSIM
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Set the random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create output directories
os.makedirs("results", exist_ok=True)
os.makedirs("results/images", exist_ok=True) # Original directory for general images if any
os.makedirs("results/benchmarks", exist_ok=True)
os.makedirs("results/optimized_images", exist_ok=True) # Stores sample outputs from each optimization
os.makedirs("results/quality_comparison_images", exist_ok=True) # Stores input/output/reference for quality check
os.makedirs("logs", exist_ok=True)

# Configuration for the experiments
CONFIG = {
    "model_id": "timbrooks/instruct-pix2pix",
    "batch_size": 1, # Currently, inference is done one by one. Batching could be another optimization.
    "num_inference_steps": 20,
    "image_guidance_scale": 1.5,
    "guidance_scale": 7.5,
    "num_samples_to_benchmark": 5,  # Number of samples from dataset to use for benchmarking each optimization
    "num_quality_samples_to_save": 3, # Number of image sets (input, generated, reference) to save for visual quality check
    "reference_image_key": "edited_image", # UPDATED: Key in your dataset for the ground-truth edited image
    "optimizations": [
        "baseline",
        "fp16",
        "xformers",
        # "sliced_attention", # enable_attention_slicing(slice_size="max") - often for lower VRAM
        "attention_slicing", # enable_attention_slicing() - let diffusers decide chunk size
        "torch_compile",
        # "cudagraph", # CUDA graphs can be tricky with dynamic aspects of diffusion models
        "vae_slicing",
        "memory_efficient_attention", # Diffusers' AttnProcessor2_0
        "scaled_dot_product_attention" # PyTorch 2.0 SDPA
    ]
}

def load_model(optimization="baseline"):
    """
    Load the InstructPix2Pix model with specified optimizations.
    """
    print(f"Loading model with optimization: {optimization}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        print("WARNING: CUDA is not available. Using CPU instead, which will be extremely slow.")
    
    # Start with default configuration
    torch_dtype = torch.float32
    use_xformers = False
    enable_attention_slicing_mode = None # To store 'max', True, or None
    enable_vae_slicing = False
    enable_torch_compile = False
    # enable_cudagraph = False # CUDAGraph applied differently
    use_memory_efficient_attention = False
    use_sdpa = False
    
    # Apply optimizations based on the specified configuration
    if optimization == "fp16":
        torch_dtype = torch.float16
    
    # Load the model with the appropriate configuration
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        CONFIG["model_id"],
        torch_dtype=torch_dtype,
        safety_checker=None,
        feature_extractor=None
    )
    
    pipe = pipe.to(device)
    
    if optimization == "xformers" and device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            use_xformers = True
            print("xFormers optimization enabled")
        except ImportError:
            print("xFormers not installed or compatible, skipping this optimization")
        except Exception as e:
            print(f"Could not enable xFormers: {e}")

    # Note: 'sliced_attention' was in original config, often means slice_size="auto" or "max"
    # The original code had "sliced_attention" map to slice_size="max"
    # And "attention_slicing" map to simple .enable_attention_slicing()
    if optimization == "sliced_attention": # Kept for consistency if user had specific reason for "max"
        pipe.enable_attention_slicing(slice_size="max")
        enable_attention_slicing_mode = "max"
        print("Attention slicing enabled with slice_size='max'")
    
    if optimization == "attention_slicing":
        pipe.enable_attention_slicing() # Let diffusers decide best slice size or use default
        enable_attention_slicing_mode = True
        print("Attention slicing enabled (auto slice size)")
    
    if optimization == "vae_slicing":
        pipe.enable_vae_slicing()
        enable_vae_slicing = True
        print("VAE slicing enabled")
    
    if optimization == "memory_efficient_attention" and device == "cuda": # This is essentially Diffusers' AttnProcessor2_0
        pipe.unet.set_attn_processor(AttnProcessor2_0())
        use_memory_efficient_attention = True
        print("Memory efficient attention (AttnProcessor2_0) enabled")
    
    if optimization == "scaled_dot_product_attention" and device == "cuda":
        # This is for PyTorch 2.0 Scaled Dot Product Attention
        # It usually requires torch >= 2.0
        # Diffusers will attempt to use it if conditions are met when setting default_attn_processor
        try:
            # The most straightforward way to enable PyTorch 2.0 SDPA is to just use the default
            # if the environment supports it. Diffusers handles this internally.
            # For explicit use, one might need to set specific processors if not default.
            # Forcing default can be a way to test if it's picked up.
            pipe.unet.set_default_attn_processor()
            # We can't easily confirm here if SDPA is *actually* used without deeper checks,
            # but this is the recommended way to let Diffusers try to use it.
            use_sdpa = True # Assume it's enabled if torch version is appropriate
            print("Attempting to use PyTorch scaled dot product attention (if available and PyTorch >= 2.0)")
        except Exception as e:
            print(f"Could not set default_attn_processor for SDPA: {e}")

    if optimization == "torch_compile" and device == "cuda":
        if hasattr(torch, "compile"):
            try:
                print("Applying torch.compile to UNet (mode='reduce-overhead', fullgraph=True)...")
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
                enable_torch_compile = True
                print("Torch compile optimization enabled for UNet.")
            except Exception as e:
                print(f"Failed to apply torch.compile: {e}")
        else:
            print("torch.compile not available (requires PyTorch 2.0+). Skipping.")
            
    # CUDAGraph is handled differently, during the inference call, so not set up here.
    
    optimization_status = {
        "optimization": optimization,
        "torch_dtype": str(torch_dtype), # store as string for easier CSV/JSON
        "use_xformers": use_xformers,
        "enable_attention_slicing": enable_attention_slicing_mode,
        "enable_vae_slicing": enable_vae_slicing,
        "enable_torch_compile": enable_torch_compile,
        "use_memory_efficient_attention": use_memory_efficient_attention,
        "use_sdpa": use_sdpa, # Indicates intent to use SDPA
        "device": device
    }
    
    return pipe, optimization_status

def calculate_quality_metrics(generated_pil, reference_pil):
    """
    Calculates PSNR and SSIM between two PIL Images.
    """
    if generated_pil is None or reference_pil is None:
        return None, None

    try:
        # Convert PIL Images to NumPy arrays
        # Ensure reference is RGB
        if reference_pil.mode != "RGB":
            reference_pil = reference_pil.convert("RGB")
        # Ensure generated is RGB
        if generated_pil.mode != "RGB":
            generated_pil = generated_pil.convert("RGB")

        reference_np = np.array(reference_pil)
        generated_np = np.array(generated_pil)

        # Resize generated image to match reference image dimensions if they differ
        if generated_np.shape != reference_np.shape:
            print(f"Warning: Generated image shape {generated_np.shape} differs from reference {reference_np.shape}. Resizing generated image.")
            generated_pil_resized = generated_pil.resize(reference_pil.size, Image.Resampling.LANCZOS)
            generated_np = np.array(generated_pil_resized)

        # Calculate PSNR
        # data_range is max possible pixel value (255 for uint8 images)
        psnr_score = peak_signal_noise_ratio(reference_np, generated_np, data_range=255)
        
        # Calculate SSIM
        # For RGB images, set multichannel=True. data_range is 255.
        # The win_size for SSIM must be odd and smaller than image dimensions.
        # Default win_size is 7. If images are smaller, this might fail.
        min_dim = min(reference_np.shape[0], reference_np.shape[1])
        # Ensure win_size is odd and <= min_dim. Also, scikit-image's default gaussian_weights used internally
        # for SSIM might require win_size to be at least 3 (for a sigma of 1.5, it usually needs a few pixels).
        # Let's ensure win_size is at least 3 and odd.
        win_size = max(3, min(7, min_dim)) 
        if win_size > min_dim : # If min_dim itself is < 3
            win_size = min_dim
        
        if win_size % 2 == 0: # Ensure win_size is odd
            win_size -= 1 
            
        if win_size < 3 : 
             print(f"Warning: Image dimension ({min_dim}) too small for SSIM window size {win_size}. Skipping SSIM.")
             ssim_score = None
        else:
            # Corrected function call to structural_similarity
            ssim_score = structural_similarity(reference_np, generated_np, multichannel=True, data_range=255, win_size=win_size, channel_axis=2)
            
        return psnr_score, ssim_score
    except Exception as e:
        print(f"Error calculating quality metrics: {e}")
        return None, None

def run_inference(pipe, prompt, image, optimization_status_dict):
    """
    Run inference with the model and measure performance.
    `image` should be a PIL Image.
    """
    device = optimization_status_dict["device"]
    # optimization_name = optimization_status_dict["optimization"] # For CUDAGraph if used

    # Ensure image is PIL Image and in RGB
    if not isinstance(image, Image.Image):
        # This case should ideally not happen if dataset provides PIL images
        print(f"Warning: Input image is not a PIL Image type: {type(image)}. Trying to convert.")
        try:
            image = Image.fromarray(np.array(image)) # Basic attempt if it's array-like
        except Exception as e:
            print(f"Could not convert input to PIL Image: {e}")
            return None, 0, 0


    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Warm-up run (important for accurate benchmarking, especially with torch.compile or CUDA graphs)
    # print("Performing warm-up run...")
    # Using fewer steps for warm-up to make it faster.
    _ = pipe(prompt, image=image, num_inference_steps=3, output_type="pil")
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Actual timed run
    # print(f"Running timed inference for {optimization_name}...")
    if device == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        output = pipe(
            prompt,
            image=image,
            num_inference_steps=CONFIG["num_inference_steps"],
            image_guidance_scale=CONFIG["image_guidance_scale"],
            guidance_scale=CONFIG["guidance_scale"],
            output_type="pil" # Ensure PIL image output
        )
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    
    inference_time = end_time - start_time
    
    # Get memory usage
    memory_allocated = 0
    if device == "cuda":
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
        torch.cuda.reset_peak_memory_stats() # Reset for next measurement
    
    # print(f"Inference completed in {inference_time:.4f} seconds")
    # if device == "cuda":
    #     print(f"Max GPU memory allocated during this run: {memory_allocated:.4f} GB")
        
    return output, inference_time, memory_allocated


def benchmark_optimizations(dataset, num_samples_for_benchmark=10):
    """
    Benchmark different optimization techniques and assess image quality.
    """
    results = []
    
    # Limit to a subset for benchmarking
    if "train" not in dataset or len(dataset["train"]) == 0 :
        print("Error: Dataset 'train' split is missing or empty. Cannot run benchmarks.")
        return pd.DataFrame()
        
    actual_num_samples = min(num_samples_for_benchmark, len(dataset["train"]))
    if actual_num_samples == 0 :
        print("Warning: No samples to benchmark after filtering.")
        return pd.DataFrame()
    benchmark_dataset = dataset["train"].shuffle(seed=42).select(range(actual_num_samples))
    
    print(f"Benchmarking on {actual_num_samples} samples for each optimization.")

    for optimization in CONFIG["optimizations"]:
        print(f"\n{'='*40}\nBenchmarking: {optimization}\n{'='*40}")
        pipe = None # Ensure pipe is reset
        try:
            pipe, optimization_status = load_model(optimization)
            
            inference_times = []
            memory_usages = []
            psnr_scores = []
            ssim_scores = []
            
            for i, example in enumerate(tqdm(benchmark_dataset, desc=f"Running {optimization}")):
                input_image_pil = example["input_image"]
                edit_prompt = example["edit_prompt"]
                reference_image_pil = example.get(CONFIG["reference_image_key"])

                if not isinstance(input_image_pil, Image.Image):
                    print(f"Warning: Sample {i} 'input_image' is not a PIL image. Skipping sample.")
                    continue
                if reference_image_pil is None:
                    print(f"Warning: Sample {i} is missing reference image ('{CONFIG['reference_image_key']}'). Skipping quality metrics for this sample.")
                elif not isinstance(reference_image_pil, Image.Image):
                    print(f"Warning: Sample {i} '{CONFIG['reference_image_key']}' is not a PIL image. Skipping quality metrics for this sample.")
                    reference_image_pil = None # Ensure it's None if not valid

                # Run inference (this is the main timed part)
                output_obj, current_inference_time, mem_alloc = run_inference(pipe, edit_prompt, input_image_pil, optimization_status)
                
                if output_obj is None or not output_obj.images:
                    print(f"Warning: Inference failed or produced no image for sample {i} with {optimization}. Skipping.")
                    continue

                generated_image_pil = output_obj.images[0]
                
                inference_times.append(current_inference_time)
                if optimization_status["device"] == "cuda": # Only track GPU memory
                    memory_usages.append(mem_alloc)
                
                # Calculate quality metrics if reference is available and valid
                if reference_image_pil: # It would have been set to None above if invalid
                    current_psnr, current_ssim = calculate_quality_metrics(generated_image_pil, reference_image_pil)
                    if current_psnr is not None:
                        psnr_scores.append(current_psnr)
                    if current_ssim is not None:
                        ssim_scores.append(current_ssim)
                
                # Save some sample images for visual comparison
                if i < CONFIG["num_quality_samples_to_save"]:
                    input_image_pil.save(f"results/quality_comparison_images/{optimization}_sample_{i}_0_input.png")
                    generated_image_pil.save(f"results/quality_comparison_images/{optimization}_sample_{i}_1_generated.png")
                    if reference_image_pil:
                        # Ensure reference is RGB before saving, if it's PIL
                        if reference_image_pil.mode != "RGB":
                            reference_image_pil.convert("RGB").save(f"results/quality_comparison_images/{optimization}_sample_{i}_2_reference.png")
                        else:
                            reference_image_pil.save(f"results/quality_comparison_images/{optimization}_sample_{i}_2_reference.png")
                    # Also save to the general optimized_images folder
                    generated_image_pil.save(f"results/optimized_images/{optimization}_sample_{i}_generated.png")

            # Calculate statistics
            avg_time = np.mean(inference_times) if inference_times else None
            median_time = np.median(inference_times) if inference_times else None
            std_time = np.std(inference_times) if inference_times else None
            avg_memory = np.mean(memory_usages) if memory_usages else None # Avg GPU memory
            
            avg_psnr = np.mean(psnr_scores) if psnr_scores else None
            avg_ssim = np.mean(ssim_scores) if ssim_scores else None

            result_data = {
                "optimization": optimization,
                "avg_inference_time": avg_time,
                "median_inference_time": median_time,
                "std_inference_time": std_time,
                "avg_memory_usage_gb": avg_memory,
                "avg_psnr": avg_psnr,
                "avg_ssim": avg_ssim,
                "num_samples_metrics_calculated_for": len(psnr_scores) # or len(ssim_scores) assuming they are paired
            }
            result_data.update(optimization_status) # Add details like dtype, xformers status etc.
            results.append(result_data)
            
            print(f"Avg inference time: {avg_time:.4f}s" if avg_time is not None else "Avg inference time: N/A")
            if avg_memory is not None: print(f"Avg GPU memory usage: {avg_memory:.4f} GB")
            if avg_psnr is not None: print(f"Avg PSNR: {avg_psnr:.2f} (from {len(psnr_scores)} samples)")
            if avg_ssim is not None: print(f"Avg SSIM: {avg_ssim:.4f} (from {len(ssim_scores)} samples)")
            
        except Exception as e:
            print(f"Error benchmarking {optimization}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            results.append({
                "optimization": optimization, "error": str(e),
                "avg_inference_time": None, "median_inference_time": None, "std_inference_time": None,
                "avg_memory_usage_gb": None, "avg_psnr": None, "avg_ssim": None,
                "num_samples_metrics_calculated_for": 0
            })
        finally:
            # Clear GPU memory
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats() # Reset global peak for next optimization
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df.to_csv("results/benchmarks/optimization_results_with_quality.csv", index=False)
        plot_benchmark_results(results_df)
    else:
        print("No results generated, skipping CSV saving and plotting.")
    
    return results_df

def plot_benchmark_results(results_df):
    """
    Create visualizations of the benchmark results, including quality metrics.
    """
    valid_results = results_df.dropna(subset=["avg_inference_time"]) # Base filtering on time
    
    if len(valid_results) == 0:
        print("No valid results with inference times to plot.")
        # Check for quality metric results even if time is missing for some reason
        quality_results = results_df.dropna(subset=["avg_psnr", "avg_ssim"], how='all')
        if len(quality_results) == 0:
            print("No valid quality metric results to plot either.")
            return
    else: # Plot performance metrics if valid_results exist
        plt.figure(figsize=(14, 7))
        plt.bar(valid_results["optimization"], valid_results["avg_inference_time"], yerr=valid_results["std_inference_time"], capsize=4)
        plt.ylabel("Inference Time (seconds)")
        plt.xlabel("Optimization Method")
        plt.title("Average Inference Time by Optimization Method")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("results/benchmarks/inference_times.png")
        plt.close()

        # Plot memory usage (only if there are memory numbers)
        mem_results = valid_results.dropna(subset=["avg_memory_usage_gb"])
        if not mem_results.empty:
            plt.figure(figsize=(14, 7))
            plt.bar(mem_results["optimization"], mem_results["avg_memory_usage_gb"])
            plt.ylabel("Avg Peak Memory Usage (GB)")
            plt.xlabel("Optimization Method")
            plt.title("Average Peak GPU Memory Usage by Optimization Method")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig("results/benchmarks/memory_usage.png")
            plt.close()
        else:
            print("No memory usage data to plot (e.g. running on CPU or all CUDA runs failed).")


        if "baseline" in valid_results["optimization"].values:
            baseline_row = valid_results[valid_results["optimization"] == "baseline"]
            if not baseline_row.empty and pd.notna(baseline_row["avg_inference_time"].iloc[0]):
                baseline_time = baseline_row["avg_inference_time"].iloc[0]
                if baseline_time > 0: # Ensure baseline_time is positive to avoid division by zero or meaningless speedup
                    valid_results_for_speedup = valid_results.copy() # Avoid SettingWithCopyWarning
                    # Calculate speedup only for rows with valid avg_inference_time > 0
                    valid_results_for_speedup["speedup"] = valid_results_for_speedup["avg_inference_time"].apply(
                        lambda x: baseline_time / x if pd.notna(x) and x > 0 else np.nan
                    )
                    valid_results_for_speedup.dropna(subset=["speedup"], inplace=True) # Remove rows where speedup couldn't be calculated
                    
                    if not valid_results_for_speedup.empty:
                        plt.figure(figsize=(14, 7))
                        plt.bar(valid_results_for_speedup["optimization"], valid_results_for_speedup["speedup"])
                        plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.7, label="Baseline")
                        plt.ylabel("Speedup (relative to baseline)")
                        plt.xlabel("Optimization Method")
                        plt.title("Speedup by Optimization Method (Higher is Better)")
                        plt.xticks(rotation=45, ha="right")
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig("results/benchmarks/speedup.png")
                        plt.close()
                    else:
                        print("Not enough valid data to plot speedup after filtering.")
                else:
                    print("Baseline time is not positive, cannot calculate speedup.")
            else:
                print("Baseline performance data missing or invalid, cannot plot speedup.")
        else:
            print("Baseline optimization not found in results, cannot plot speedup.")

    # Plot Quality Metrics (PSNR and SSIM) - use original df to include all that have metrics
    psnr_results = results_df.dropna(subset=["avg_psnr"])
    if not psnr_results.empty:
        plt.figure(figsize=(14, 7))
        plt.bar(psnr_results["optimization"], psnr_results["avg_psnr"])
        plt.ylabel("Average PSNR (Higher is Better)")
        plt.xlabel("Optimization Method")
        plt.title("Average PSNR by Optimization Method")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("results/benchmarks/avg_psnr.png")
        plt.close()
    else:
        print("No PSNR data to plot.")

    ssim_results = results_df.dropna(subset=["avg_ssim"])
    if not ssim_results.empty:
        plt.figure(figsize=(14, 7))
        plt.bar(ssim_results["optimization"], ssim_results["avg_ssim"])
        plt.ylabel("Average SSIM (Higher is Better, Max 1.0)")
        plt.xlabel("Optimization Method")
        plt.title("Average SSIM by Optimization Method")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("results/benchmarks/avg_ssim.png")
        plt.close()
    else:
        print("No SSIM data to plot.")
        
    print("Benchmark plots saved to results/benchmarks/")


def generate_report(results_df):
    """
    Generate a Markdown report of the benchmark results, including quality metrics.
    """
    report = f"# InstructPix2Pix Optimization & Quality Report\n\n"
    report += f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    if results_df.empty or results_df["avg_inference_time"].isnull().all():
        report += "No valid benchmark results (for inference time) were obtained. Check logs.\n"
    else:
        # Hardware information
        report += "## Hardware & Setup\n\n"
        if torch.cuda.is_available():
            report += f"- GPU: {torch.cuda.get_device_name(0)}\n"
            report += f"- CUDA Version: {torch.version.cuda}\n"
        else:
            report += "- Running on CPU (No GPU detected)\n"
        report += f"- PyTorch Version: {torch.__version__}\n"
        # To get diffusers version, you'd typically `import diffusers` and use `diffusers.__version__`
        try:
            import diffusers
            report += f"- Diffusers Version: {diffusers.__version__}\n"
        except ImportError:
            report += f"- Diffusers Version: (Not detected, import diffusers to check)\n"

        report += f"- Num Samples per Optimization: {CONFIG['num_samples_to_benchmark']}\n"
        report += f"- Num Inference Steps: {CONFIG['num_inference_steps']}\n\n"

        # Performance and Quality Summary Table
        report += "## Performance & Quality Summary\n\n"
        report += "| Optimization | Avg Time (s) | Speedup | Memory (GB) | Avg PSNR | Avg SSIM | Samples for Metrics |\n"
        report += "|--------------|--------------|---------|-------------|----------|----------|---------------------|\n"
        
        baseline_time_val = None
        baseline_row = results_df[results_df["optimization"] == "baseline"]
        if not baseline_row.empty and pd.notna(baseline_row["avg_inference_time"].iloc[0]):
            baseline_time_val = baseline_row["avg_inference_time"].iloc[0]
            if baseline_time_val <=0: baseline_time_val = None # Ensure positive baseline time for speedup calc

        # Sort by average inference time for the table display
        sorted_df = results_df.sort_values(by="avg_inference_time", na_position='last')

        for _, row in sorted_df.iterrows():
            opt = row["optimization"]
            time_s = f"{row['avg_inference_time']:.2f}" if pd.notna(row['avg_inference_time']) else "N/A"
            
            speedup_str = "N/A"
            if baseline_time_val is not None and pd.notna(row['avg_inference_time']) and row['avg_inference_time'] > 0:
                speedup = baseline_time_val / row['avg_inference_time']
                speedup_str = f"{speedup:.2f}x"
            elif opt == "baseline" and pd.notna(row['avg_inference_time']) and row['avg_inference_time'] > 0 : # Baseline is 1.00x if valid
                 speedup_str = "1.00x"


            mem_gb = f"{row['avg_memory_usage_gb']:.2f}" if pd.notna(row['avg_memory_usage_gb']) else "N/A (CPU/Error)"
            psnr_val = f"{row['avg_psnr']:.2f}" if pd.notna(row['avg_psnr']) else "N/A"
            ssim_val = f"{row['avg_ssim']:.3f}" if pd.notna(row['avg_ssim']) else "N/A"
            num_metric_samples = f"{int(row.get('num_samples_metrics_calculated_for', 0))}" # Ensure int for display

            report += f"| {opt} | {time_s} | {speedup_str} | {mem_gb} | {psnr_val} | {ssim_val} | {num_metric_samples} |\n"
        report += "\n"

        # Best performing optimization based on time
        valid_time_results = results_df.dropna(subset=["avg_inference_time"])
        if not valid_time_results.empty:
            best_opt_by_time_row = valid_time_results.loc[valid_time_results["avg_inference_time"].idxmin()]
            best_opt_name_time = best_opt_by_time_row['optimization']
            best_time_val = best_opt_by_time_row['avg_inference_time']

            report += f"### Fastest Optimization: **{best_opt_name_time}**\n"
            report += f"- Average inference time: {best_time_val:.3f} seconds.\n"
            if baseline_time_val is not None and best_opt_name_time != "baseline" and best_time_val > 0:
                speedup = baseline_time_val / best_time_val
                report += f"- Speedup over baseline: {speedup:.2f}x\n"
            report += "\n"
        
        # Best quality optimization based on SSIM (or PSNR if SSIM is not available)
        best_quality_metric_name = "avg_ssim" if "avg_ssim" in results_df.columns and results_df["avg_ssim"].notna().any() else "avg_psnr"
        valid_quality_results = results_df.dropna(subset=[best_quality_metric_name])
        if not valid_quality_results.empty:
            best_opt_by_quality_row = valid_quality_results.loc[valid_quality_results[best_quality_metric_name].idxmax()]
            best_opt_name_quality = best_opt_by_quality_row['optimization']
            best_quality_val = best_opt_by_quality_row[best_quality_metric_name]
            report += f"### Best Quality Optimization (by {best_quality_metric_name}): **{best_opt_name_quality}**\n"
            report += f"- {best_quality_metric_name}: {best_quality_val:.3f}\n\n"


    report += "## Detailed Optimization Analysis\n\n"
    for optimization in CONFIG["optimizations"]: # Iterate in defined order
        rows = results_df[results_df["optimization"] == optimization]
        if rows.empty:
            report += f"### {optimization}\n\nNo results recorded.\n\n"
            continue
        
        row = rows.iloc[0]
        report += f"### {optimization}\n\n"
        if pd.notna(row.get("error")):
            report += f"**Error encountered**: {row['error']}\n\n"
            continue
            
        time_s = f"{row['avg_inference_time']:.3f}" if pd.notna(row['avg_inference_time']) else "N/A"
        median_s = f"{row['median_inference_time']:.3f}" if pd.notna(row['median_inference_time']) else "N/A"
        std_s = f"{row['std_inference_time']:.3f}" if pd.notna(row['std_inference_time']) else "N/A"
        mem_gb = f"{row['avg_memory_usage_gb']:.2f}" if pd.notna(row['avg_memory_usage_gb']) else "N/A"
        psnr_val = f"{row['avg_psnr']:.2f}" if pd.notna(row['avg_psnr']) else "N/A"
        ssim_val = f"{row['avg_ssim']:.3f}" if pd.notna(row['avg_ssim']) else "N/A"
        num_metric_samples_detail = int(row.get('num_samples_metrics_calculated_for',0))
        
        report += f"- Average inference time: {time_s} seconds\n"
        report += f"- Median inference time: {median_s} seconds\n"
        report += f"- Std dev inference time: {std_s} seconds\n"
        report += f"- Average GPU memory: {mem_gb} GB\n"
        report += f"- Average PSNR: {psnr_val} (from {num_metric_samples_detail} samples)\n"
        report += f"- Average SSIM: {ssim_val} (from {num_metric_samples_detail} samples)\n"
        
        # Add notes about the optimization from original code
        if optimization == "baseline": report += "_Baseline model with no explicit optimizations beyond default Diffusers setup._\n"
        elif optimization == "fp16": report += "_Half-precision (FP16) computation._\n"
        elif optimization == "xformers": report += "_xFormers memory-efficient attention (if available)._\n"
        elif optimization == "sliced_attention": report += "_Attention computation sliced (slice_size='max')._\n"
        elif optimization == "attention_slicing": report += "_Attention computation sliced (auto slice size)._\n"
        elif optimization == "torch_compile": report += "_PyTorch 2.0+ `torch.compile` on UNet (mode='reduce-overhead', fullgraph=True)._\n"
        # elif optimization == "cudagraph": report += "_CUDA Graphs (experimental, applied to UNet)._\n" # If re-enabled
        elif optimization == "vae_slicing": report += "_VAE computation sliced to reduce memory._\n"
        elif optimization == "memory_efficient_attention": report += "_Diffusers AttnProcessor2_0 for memory efficient attention._\n"
        elif optimization == "scaled_dot_product_attention": report += "_Attempted PyTorch 2.0+ native Scaled Dot Product Attention._\n"
        report += "\n"

    report += "## Benchmark Plots\n\n"
    # Assuming report is in results/benchmarks/, relative paths are fine
    if Path("results/benchmarks/inference_times.png").exists():
        report += "### Inference Time Comparison\n![Inference Times](inference_times.png)\n\n"
    if Path("results/benchmarks/memory_usage.png").exists():
        report += "### Memory Usage Comparison\n![Memory Usage](memory_usage.png)\n\n"
    if Path("results/benchmarks/speedup.png").exists():
        report += "### Speedup Comparison\n![Speedup](speedup.png)\n\n"
    if Path("results/benchmarks/avg_psnr.png").exists():
        report += "### Average PSNR Comparison\n![Average PSNR](avg_psnr.png)\n\n"
    if Path("results/benchmarks/avg_ssim.png").exists():
        report += "### Average SSIM Comparison\n![Average SSIM](avg_ssim.png)\n\n"
    
    report += "## Conclusion & Recommendations\n\n"
    report += "This report summarizes performance and quality metrics for various optimizations. "
    report += "The 'best' optimization depends on the specific trade-offs desired between speed, memory usage, and output quality.\n"
    report += "Refer to the summary tables and detailed analysis for specific values.\n"
    report += "Recommendations may include:\n"
    report += "- For maximum speed on compatible hardware: `torch_compile` (if PyTorch 2+) and `fp16` are often strong candidates.\n"
    report += "- For memory savings: `fp16`, `attention_slicing`, `vae_slicing`, `memory_efficient_attention` (or `xformers`).\n"
    report += "- Evaluate quality metrics (PSNR/SSIM) alongside visual inspection of generated images, as some optimizations might slightly alter output.\n\n"
    report += f"**Note on Quality Metrics**: PSNR/SSIM calculated against reference images from key '{CONFIG['reference_image_key']}'. "
    report += "Ensure these references are appropriate for the desired output. Higher is generally better.\n"

    with open("results/benchmarks/benchmark_report.md", "w") as f:
        f.write(report)
    print("Report generated and saved to results/benchmarks/benchmark_report.md")


def main():
    """
    Main function to run the benchmark and generate the report.
    """
    print(f"Starting InstructPix2Pix optimization and quality benchmark")
    print(f"PyTorch version: {torch.__version__}")
    try:
        import diffusers
        print(f"Diffusers version: {diffusers.__version__}")
    except ImportError:
        print("Diffusers version: (Not installed or not in PYTHONPATH)")
        
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print(f"Number of samples per optimization: {CONFIG['num_samples_to_benchmark']}")
    print(f"Reference image key from dataset: '{CONFIG['reference_image_key']}'")

    # Load the dataset
    print("Loading dataset...")
    try:
        # Ensure your dataset is saved at this path or change the path
        dataset = load_from_disk("saved_dataset") 
        if "train" not in dataset:
            raise ValueError("Dataset loaded but does not contain a 'train' split.")
        if len(dataset['train']) == 0:
             print("Warning: 'train' split of the dataset is empty.")
        else:
            # Check for required keys in the first sample as a quick validation
            first_sample = dataset['train'][0]
            required_keys_for_op = ["input_image", "edit_prompt"]
            all_required_keys = required_keys_for_op + [CONFIG["reference_image_key"]]
            
            for key in all_required_keys:
                if key not in first_sample:
                    warning_msg = f"Warning: Key '{key}' not found in the first sample of the dataset. "
                    if key == CONFIG["reference_image_key"]:
                        warning_msg += "Quality metrics (PSNR/SSIM) will not be calculated for samples missing this key."
                    else: # input_image or edit_prompt
                        warning_msg += "This key is essential for running inference. Samples missing it will be skipped."
                    print(warning_msg)
            
            if "input_image" in first_sample and not isinstance(first_sample["input_image"], Image.Image):
                 print("Warning: 'input_image' in the dataset is not a PIL Image. Conversion will be attempted, but errors may occur if conversion fails.")
            if CONFIG["reference_image_key"] in first_sample and not isinstance(first_sample[CONFIG["reference_image_key"]], Image.Image):
                 print(f"Warning: '{CONFIG['reference_image_key']}' in the dataset is not a PIL Image. Quality metrics may fail for such samples if conversion is not possible.")


        print(f"Dataset loaded with {len(dataset['train'])} examples in 'train' split.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at 'saved_dataset'. Please ensure your dataset is correctly saved or update the path.")
        print("To save a Hugging Face dataset: `your_dataset_object.save_to_disk('saved_dataset')`")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run benchmarks
    results_df = benchmark_optimizations(dataset, num_samples_for_benchmark=CONFIG["num_samples_to_benchmark"])
    
    # Generate the report
    if not results_df.empty:
        generate_report(results_df)
    else:
        print("No results were generated from benchmarking. Report generation skipped.")
        
    print("Benchmark completed!")

if __name__ == "__main__":
    main()
