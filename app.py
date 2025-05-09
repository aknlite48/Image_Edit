import os
if 'darwin' in os.uname().sysname.lower():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import io
import base64
import torch
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from diffusers import StableDiffusionInstructPix2PixPipeline, DDIMScheduler
import cv2
import time
import random
from threading import Thread, Lock

# Set environment variable to handle operations not supported in MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

app = Flask(__name__)

# Global variables
model = None
ongoing_processes = {}
process_locks = {}

def load_model():
    """Load the InstructPix2Pix model"""
    global model
    if model is not None:
        return model
    
    # Check if MPS is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA (GPU)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    
    # Load model
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        safety_checker=None
    )
    
    # Enable memory optimizations
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    
    # Use DDIM scheduler for better quality
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    model = pipe
    return model

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

# Create a class to handle diffusion callbacks
class DiffusionProcessCallback:
    def __init__(self, process_id):
        self.process_id = process_id
    
    def __call__(self, pipe, step, timestep, callback_kwargs):
        """Callback method for diffusion process
        
        This follows the diffusers callback_on_step_end signature:
        (pipe, step, timestep, callback_kwargs) -> callback_kwargs
        """
        # Get latents from callback kwargs
        latents = callback_kwargs["latents"]
        
        # Only process every 3 steps to avoid overwhelming the browser
        if step % 3 == 0 or step == 1:
            with torch.no_grad():
                # Decode latents to image
                latents_processed = 1 / 0.18215 * latents
                image = pipe.vae.decode(latents_processed).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                
                # Convert to PIL Image
                image_pil = pipe.numpy_to_pil(image)[0]
                
                # Convert to base64
                buffer = io.BytesIO()
                image_pil.save(buffer, format="JPEG", quality=80)
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Store the intermediate result in the process data
                with process_locks[self.process_id]:
                    ongoing_processes[self.process_id]["current_step"] = step
                    ongoing_processes[self.process_id]["steps_image"] = f"data:image/jpeg;base64,{base64_image}"
                    
                    # Calculate progress percentage
                    total_steps = ongoing_processes[self.process_id]["total_steps"]
                    progress = round((step / total_steps) * 100)
                    ongoing_processes[self.process_id]["progress"] = progress
        
        # Must return the callback_kwargs
        return callback_kwargs

@app.route('/start_process', methods=['POST'])
def start_process():
    """Start the image processing with progress tracking"""
    # Load model if not already loaded
    pipe = load_model()
    
    # Generate unique process ID
    process_id = str(random.randint(10000, 99999))
    
    # Get data from request
    data = request.get_json()
    image_data = data.get('image')
    mask_data = data.get('mask')
    prompt = data.get('prompt')
    image_guidance_scale = float(data.get('image_guidance_scale', 1.5))
    text_guidance_scale = float(data.get('text_guidance_scale', 7.5))
    num_steps = int(data.get('num_steps', 20))
    
    # Validate parameters
    if not image_data or not prompt:
        return jsonify({'error': 'Missing image or prompt'}), 400
    
    # Initialize process data and lock
    ongoing_processes[process_id] = {
        "status": "initializing",
        "progress": 0,
        "current_step": 0,
        "total_steps": num_steps,
        "result": None,
        "steps_image": None,
        "message": "Starting process...",
    }
    process_locks[process_id] = Lock()
    
    # Start processing thread
    thread = Thread(
        target=process_image_thread,
        args=(process_id, image_data, mask_data, prompt, image_guidance_scale, text_guidance_scale, num_steps)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({"process_id": process_id})

def process_image_thread(process_id, image_data, mask_data, prompt, image_guidance_scale, text_guidance_scale, num_steps):
    """Process the image in a separate thread"""
    pipe = load_model()
    
    try:
        # Update status
        with process_locks[process_id]:
            ongoing_processes[process_id]["status"] = "processing"
            ongoing_processes[process_id]["message"] = "Processing image..."
        
        # Convert base64 to image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Process mask if provided
        mask = None
        if mask_data:
            mask_bytes = base64.b64decode(mask_data.split(',')[1])
            mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        
        # Create a callback instance
        callback = DiffusionProcessCallback(process_id)
        
        # Run InstructPix2Pix with callback
        edited_image = pipe(
            prompt,
            image=image,
            num_inference_steps=num_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=text_guidance_scale,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]
        
        # If we have a mask, do selective editing
        if mask:
            # Convert to numpy arrays for compositing
            img_np = np.array(image)
            edited_np = np.array(edited_image)
            
            # IMPORTANT: Make sure both images have the same dimensions before applying the mask
            # Resize edited image to match original if needed
            if img_np.shape != edited_np.shape:
                print(f"Resizing edited image from {edited_np.shape} to {img_np.shape}")
                edited_image = edited_image.resize((image.width, image.height), Image.LANCZOS)
                edited_np = np.array(edited_image)
            
            # Process the mask
            mask_np = np.array(mask)
            
            # Resize mask to exactly match the original image dimensions
            if mask_np.shape[:2] != img_np.shape[:2]:
                print(f"Resizing mask from {mask_np.shape} to {img_np.shape[:2]}")
                mask = mask.resize((image.width, image.height), Image.LANCZOS)
                mask_np = np.array(mask)
            
            # Normalize mask values to [0, 1]
            mask_np = mask_np / 255.0
            
            # Expand mask dimensions for broadcasting
            if len(mask_np.shape) == 2:
                mask_np = np.expand_dims(mask_np, axis=2)
                mask_np = np.repeat(mask_np, 3, axis=2)
            
            # Double-check dimensions before compositing
            if img_np.shape != edited_np.shape:
                print(f"Error: Image shapes still don't match after resizing: {img_np.shape} vs {edited_np.shape}")
                with process_locks[process_id]:
                    ongoing_processes[process_id]["status"] = "error"
                    ongoing_processes[process_id]["message"] = f"Error: Image dimension mismatch"
                return
            
            if img_np.shape != mask_np.shape:
                print(f"Error: Mask shape doesn't match image: {img_np.shape} vs {mask_np.shape}")
                # Try one more resizing approach
                try:
                    mask_np = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]))
                    print(f"Resized mask with cv2 to {mask_np.shape}")
                except Exception as e:
                    print(f"Failed to resize mask: {e}")
                    with process_locks[process_id]:
                        ongoing_processes[process_id]["status"] = "error"
                        ongoing_processes[process_id]["message"] = f"Error: Mask and image dimension mismatch"
                    return
            
            # Composite: edited * mask + original * (1 - mask)
            try:
                result_np = edited_np * mask_np + img_np * (1 - mask_np)
                result_image = Image.fromarray(result_np.astype(np.uint8))
            except Exception as e:
                print(f"Error during compositing: {e}")
                # Fall back to just using the edited image without masking
                result_image = edited_image
        else:
            result_image = edited_image
        
        # Convert result to base64
        buffer = io.BytesIO()
        result_image.save(buffer, format="PNG")
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Update process data with result
        with process_locks[process_id]:
            ongoing_processes[process_id]["status"] = "completed"
            ongoing_processes[process_id]["progress"] = 100
            ongoing_processes[process_id]["result"] = f"data:image/png;base64,{result_base64}"
            ongoing_processes[process_id]["message"] = f"Editing completed with prompt: {prompt}"
        
        # Keep the result for 5 minutes, then clean up
        def cleanup_later():
            time.sleep(300)  # 5 minutes
            if process_id in ongoing_processes:
                del ongoing_processes[process_id]
            if process_id in process_locks:
                del process_locks[process_id]
        
        cleanup_thread = Thread(target=cleanup_later)
        cleanup_thread.daemon = True
        cleanup_thread.start()
            
    except Exception as e:
        print(f"Error processing image: {e}")
        with process_locks[process_id]:
            ongoing_processes[process_id]["status"] = "error"
            ongoing_processes[process_id]["message"] = f"Error: {str(e)}"

@app.route('/check_progress/<process_id>', methods=['GET'])
def check_progress(process_id):
    """Check the progress of a running process"""
    if process_id not in ongoing_processes:
        return jsonify({"error": "Process not found"}), 404
    
    with process_locks[process_id]:
        return jsonify(ongoing_processes[process_id])

if __name__ == '__main__':
    # Preload model in the main thread
    print("Loading InstructPix2Pix model...")
    load_model()
    print("Model loaded successfully!")
    
    # Run the app
    app.run(debug=False, host='0.0.0.0', port=8000, threaded=True)
