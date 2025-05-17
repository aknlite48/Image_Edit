# Image Editor with InstructPix2Pix

This repository contains a web-based image editing application that uses the InstructPix2Pix model to perform AI-powered image transformations based on text prompts. The project includes an extensive optimization study (final course project) that benchmarks various performance optimization techniques for the InstructPix2Pix model on high-performance computing environments.

## Project Structure

- `app.py`: Main web application for image editing
- `optimization/`: Contains the final course project code for performance optimization
  - `run_optim.py`: Main optimization benchmarking script
  - `run_optim.slurm`: SLURM script for running optimizations on NYU HPC Greene

## Optimization Project Features

The optimization project (`optimization/run_optim.py`) implements and benchmarks multiple optimization techniques:

1. **Model Optimizations**:
   - FP16 (half-precision) inference
   - xFormers memory-efficient attention
   - Attention slicing (auto and max modes)
   - VAE slicing
   - PyTorch 2.0 compilation
   - Memory-efficient attention (AttnProcessor2_0)
   - Scaled dot product attention (SDPA)

2. **Performance Metrics**:
   - Inference time per image
   - Memory usage
   - Quality metrics (PSNR, SSIM)
   - Comprehensive benchmarking reports

3. **Output Analysis**:
   - Automated quality comparison
   - Performance vs. quality trade-off analysis
   - Visual quality assessment
   - Detailed benchmarking reports

## Running the Optimization Project

### NYU HPC Greene Setup

1. Connect to Greene and request an interactive session:
```bash
ssh <your-netid>@greene.hpc.nyu.edu
ssh burst
srun --account=csci_ga_3033_077-2025sp --partition=interactive --time=04:00:00 --pty /bin/bash
```

2. Load required modules:
```bash
module load python/intel/3.8.6
module load cuda/11.7.0
```

3. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the optimization script:
```bash
cd optimization
python run_optim.py
```

Or submit as a SLURM job:
```bash
sbatch run_optim.slurm
```

## Web Application Features

- Web-based interface for easy image editing
- Support for text-guided image transformations
- Optional masking for selective editing
- Real-time progress tracking
- GPU acceleration support (CUDA for GCP, MPS for Apple Silicon)
- Interactive preview of the editing process

## Configuration

The application uses the following key parameters that can be adjusted:

- `image_guidance_scale`: Controls how closely the output follows the input image (default: 1.5)
- `text_guidance_scale`: Controls how closely the output follows the text prompt (default: 7.5)
- `num_steps`: Number of diffusion steps (default: 20)

## Technical Details

- Built with Flask for the backend
- Uses StableDiffusionInstructPix2Pix for image editing
- Supports multiple GPU backends:
  - CUDA for GCP and NYU HPC Greene
  - MPS for Apple Silicon devices
- Implements asynchronous processing with progress tracking
- Uses DDIM scheduler for improved quality
- Optimized for high-performance computing environments

## Dependencies

- torch >= 2.0.0
- diffusers >= 0.21.0
- transformers >= 4.30.0
- safetensors >= 0.3.0
- accelerate >= 0.20.0
- flask >= 2.0.0
- pillow >= 9.3.0
- opencv-python >= 4.7.0
- numpy >= 1.22.0
- xformers (optional, for memory optimization)
- scikit-image (for quality metrics)

## Notes

- The first run will download the InstructPix2Pix model, which may take some time depending on your internet connection
- For optimal performance, it's recommended to use NYU HPC Greene with GPU support
- The application automatically uses the appropriate GPU acceleration based on the environment:
  - CUDA on GCP and NYU HPC Greene
  - MPS on Apple Silicon devices
- When running on NYU HPC Greene, make sure to request appropriate GPU resources
- The optimization project generates comprehensive reports in the `results/` directory
