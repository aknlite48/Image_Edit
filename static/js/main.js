// Main JavaScript for InstructPix2Pix Web Interface with Simplified UI

document.addEventListener('DOMContentLoaded', () => {
  // Canvas elements
  const inputCanvas = document.getElementById('input-canvas');
  const maskCanvas = document.getElementById('mask-canvas');
  const resultCanvas = document.getElementById('result-canvas');
  
  // Get canvas contexts
  const inputCtx = inputCanvas.getContext('2d');
  const maskCtx = maskCanvas.getContext('2d');
  const resultCtx = resultCanvas.getContext('2d');
  
  // UI controls
  const imageUpload = document.getElementById('image-upload');
  const brushSizeSlider = document.getElementById('brush-size');
  const brushSizeValue = document.getElementById('brush-size-value');
  const imgGuidanceSlider = document.getElementById('image-guidance');
  const imgGuidanceValue = document.getElementById('image-guidance-value');
  const textGuidanceSlider = document.getElementById('text-guidance');
  const textGuidanceValue = document.getElementById('text-guidance-value');
  const stepsSlider = document.getElementById('steps');
  const stepsValue = document.getElementById('steps-value');
  const promptInput = document.getElementById('prompt-input');
  const processBtn = document.getElementById('process-image');
  const clearMaskBtn = document.getElementById('clear-mask');
  const undoBtn = document.getElementById('undo-last');
  const downloadBtn = document.getElementById('download-result');
  const statusMessage = document.getElementById('status-message');
  const loadingOverlay = document.getElementById('loading-overlay');
  const loadingMessage = document.getElementById('loading-message');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const progressPercentage = document.getElementById('progress-percentage');
  const resultPlaceholder = document.getElementById('result-placeholder');
  const advancedHeader = document.getElementById('advanced-header');
  const advancedContent = document.getElementById('advanced-content');
  
  // Drawing state
  let originalImage = null;
  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;
  let brushSize = parseInt(brushSizeSlider.value);
  let undoStack = [];
  let currentProcessId = null;
  let progressCheckInterval = null;
  
  // Initialize UI
  initializeUI();
  
  // Set up event listeners
  setupEventListeners();
  
  // Set up responsive canvas sizing
  setupCanvasSizing();
  
  function initializeUI() {
    // Set initial values for sliders
    brushSizeValue.textContent = brushSizeSlider.value;
    imgGuidanceValue.textContent = imgGuidanceSlider.value;
    textGuidanceValue.textContent = textGuidanceSlider.value;
    stepsValue.textContent = stepsSlider.value;
    
    // Initialize advanced settings accordion
    advancedHeader.addEventListener('click', () => {
      advancedHeader.classList.toggle('active');
      advancedContent.classList.toggle('open');
    });
    
    // Initialize button states
    updateButtonStates();
  }
  
  function setupEventListeners() {
    // File upload handling
    imageUpload.addEventListener('change', handleImageUpload);
    
    // Drawing events
    inputCanvas.addEventListener('mousedown', startDrawing);
    inputCanvas.addEventListener('mousemove', draw);
    inputCanvas.addEventListener('mouseup', stopDrawing);
    inputCanvas.addEventListener('mouseout', stopDrawing);
    
    // Touch support
    inputCanvas.addEventListener('touchstart', handleTouchStart);
    inputCanvas.addEventListener('touchmove', handleTouchMove);
    inputCanvas.addEventListener('touchend', stopDrawing);
    
    // UI controls events
    brushSizeSlider.addEventListener('input', updateBrushSize);
    imgGuidanceSlider.addEventListener('input', () => {
      imgGuidanceValue.textContent = imgGuidanceSlider.value;
    });
    textGuidanceSlider.addEventListener('input', () => {
      textGuidanceValue.textContent = textGuidanceSlider.value;
    });
    stepsSlider.addEventListener('input', () => {
      stepsValue.textContent = stepsSlider.value;
    });
    
    // Button actions
    clearMaskBtn.addEventListener('click', clearMask);
    undoBtn.addEventListener('click', undoLastStroke);
    processBtn.addEventListener('click', processImage);
    downloadBtn.addEventListener('click', downloadResult);
    
    // Prompt input change to update button state
    promptInput.addEventListener('input', updateButtonStates);
    
    // Handle drag and drop for image upload
    const fileUploadLabel = document.querySelector('.file-upload-label');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      fileUploadLabel.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
      fileUploadLabel.addEventListener(eventName, () => {
        fileUploadLabel.classList.add('highlight');
      }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
      fileUploadLabel.addEventListener(eventName, () => {
        fileUploadLabel.classList.remove('highlight');
      }, false);
    });
    
    fileUploadLabel.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      
      if (files.length) {
        imageUpload.files = files;
        handleImageUpload({ target: { files } });
      }
    }, false);
  }
  
  function setupCanvasSizing() {
    // Initial sizing
    resizeCanvases();
    
    // Resize on window resize
    window.addEventListener('resize', resizeCanvases);
    
    function resizeCanvases() {
      // Get the current width of the canvas containers
      const containers = document.querySelectorAll('.canvas-content');
      if (!containers.length) return;
      
      const containerWidth = containers[0].clientWidth;
      const containerHeight = containers[0].clientHeight;
      
      if (containerWidth <= 0 || containerHeight <= 0) return;
      
      // Set all canvases to the same size
      [inputCanvas, maskCanvas, resultCanvas].forEach(canvas => {
        canvas.width = containerWidth;
        canvas.height = containerHeight;
      });
      
      // Redraw canvases if needed
      if (originalImage) {
        drawImageOnCanvas(originalImage, inputCanvas);
        redrawMask();
      }
    }
  }
  
  function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    // Check file type
    if (!file.type.match('image.*')) {
      setStatus('Please select an image file', 'error');
      return;
    }
    
    const reader = new FileReader();
    reader.onload = function(event) {
      const img = new Image();
      img.onload = function() {
        originalImage = img;
        drawImageOnCanvas(img, inputCanvas);
        clearMask();
        updateButtonStates();
        setStatus('Image loaded. Draw on it to create a mask for selective editing');
      };
      img.onerror = function() {
        setStatus('Failed to load image. Please try another file', 'error');
      };
      img.src = event.target.result;
    };
    
    reader.onerror = function() {
      setStatus('Failed to read file. Please try again', 'error');
    };
    
    reader.readAsDataURL(file);
  }
  
  function drawImageOnCanvas(img, canvas) {
    const ctx = canvas.getContext('2d');
    
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate scaling to fit image while maintaining aspect ratio
    const canvasRatio = canvas.width / canvas.height;
    const imgRatio = img.width / img.height;
    
    let drawWidth, drawHeight, offsetX, offsetY;
    
    if (canvasRatio > imgRatio) {
      // Canvas is wider than the image (relative to height)
      drawHeight = canvas.height;
      drawWidth = img.width * (canvas.height / img.height);
      offsetX = (canvas.width - drawWidth) / 2;
      offsetY = 0;
    } else {
      // Canvas is taller than the image (relative to width)
      drawWidth = canvas.width;
      drawHeight = img.height * (canvas.width / img.width);
      offsetX = 0;
      offsetY = (canvas.height - drawHeight) / 2;
    }
    
    // Draw the image
    ctx.drawImage(img, offsetX, offsetY, drawWidth, drawHeight);
    
    // Store the image dimensions for coordinate mapping
    canvas.imageData = {
      offsetX,
      offsetY,
      drawWidth,
      drawHeight,
      originalWidth: img.width,
      originalHeight: img.height
    };
  }
  
  function startDrawing(e) {
    if (!originalImage) return;
    
    e.preventDefault();
    isDrawing = true;
    
    // Save current state for undo
    saveMaskState();
    
    // Get the position
    const { x, y } = getPointerPos(inputCanvas, e);
    lastX = x;
    lastY = y;
    
    // Draw a dot at start position
    drawMaskStroke(x, y, x, y);
  }
  
  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    
    // Get current position
    const { x, y } = getPointerPos(inputCanvas, e);
    
    // Draw line from last position to current position
    drawMaskStroke(lastX, lastY, x, y);
    
    // Update last position
    lastX = x;
    lastY = y;
  }
  
  function stopDrawing() {
    isDrawing = false;
  }
  
  function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousedown', {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    inputCanvas.dispatchEvent(mouseEvent);
  }
  
  function handleTouchMove(e) {
    e.preventDefault();
    if (!isDrawing) return;
    
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent('mousemove', {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    inputCanvas.dispatchEvent(mouseEvent);
  }
  
  function getPointerPos(canvas, e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }
  
  function drawMaskStroke(x1, y1, x2, y2) {
    // Draw on the mask canvas
    maskCtx.strokeStyle = '#ffffff'; // White mask
    maskCtx.lineWidth = brushSize;
    maskCtx.lineCap = 'round';
    maskCtx.lineJoin = 'round';
    
    maskCtx.beginPath();
    maskCtx.moveTo(x1, y1);
    maskCtx.lineTo(x2, y2);
    maskCtx.stroke();
    
    // Draw visual representation on the input canvas
    redrawMask();
  }
  
  function redrawMask() {
    if (!originalImage) return;
    
    // Redraw the original image
    drawImageOnCanvas(originalImage, inputCanvas);
    
    // Draw the mask as a red overlay on input canvas
    inputCtx.globalCompositeOperation = 'source-over';
    inputCtx.fillStyle = 'rgba(255, 0, 0, 0.5)';
    
    // Get mask data
    const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const maskData = maskImageData.data;
    
    // Create a temporary canvas for the overlay
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = maskCanvas.width;
    tempCanvas.height = maskCanvas.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Create an RGBA overlay image
    const overlayImageData = tempCtx.createImageData(maskCanvas.width, maskCanvas.height);
    const overlayData = overlayImageData.data;
    
    for (let i = 0; i < maskData.length; i += 4) {
      if (maskData[i + 3] > 0) {
        overlayData[i] = 255; // R
        overlayData[i + 1] = 0; // G
        overlayData[i + 2] = 0; // B
        overlayData[i + 3] = 128; // A (semi-transparent)
      }
    }
    
    tempCtx.putImageData(overlayImageData, 0, 0);
    inputCtx.drawImage(tempCanvas, 0, 0);
    
    // Update button states
    updateButtonStates();
  }
  
  function updateBrushSize() {
    brushSize = parseInt(brushSizeSlider.value);
    brushSizeValue.textContent = brushSize;
  }
  
  function clearMask() {
    // Save current state for undo
    saveMaskState();
    
    // Clear the mask canvas
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    
    // Redraw the input canvas without the mask
    if (originalImage) {
      drawImageOnCanvas(originalImage, inputCanvas);
    }
    
    setStatus('Mask cleared. Draw a new mask on the image');
    updateButtonStates();
  }
  
  function saveMaskState() {
    const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    undoStack.push(maskImageData);
    updateButtonStates();
  }
  
  function undoLastStroke() {
    if (undoStack.length === 0) return;
    
    const prevState = undoStack.pop();
    maskCtx.putImageData(prevState, 0, 0);
    
    // Redraw the input canvas with the updated mask
    redrawMask();
    
    setStatus('Undid last drawing stroke');
    updateButtonStates();
  }
  
  function updateButtonStates() {
    const hasImage = originalImage !== null;
    const hasMask = hasMaskContent();
    const hasPrompt = promptInput.value.trim() !== '';
    const isProcessing = currentProcessId !== null;
    const hasResult = resultCanvas.hasDrawnImage;
    
    // Enable/disable buttons based on state
    clearMaskBtn.disabled = !hasImage || !hasMask;
    undoBtn.disabled = !hasImage || undoStack.length === 0;
    processBtn.disabled = !hasImage || !hasPrompt || isProcessing;
    downloadBtn.disabled = !hasResult;
    
    // Update visual states
    if (undoBtn.disabled) {
      undoBtn.classList.add('disabled');
    } else {
      undoBtn.classList.remove('disabled');
    }
    
    if (clearMaskBtn.disabled) {
      clearMaskBtn.classList.add('disabled');
    } else {
      clearMaskBtn.classList.remove('disabled');
    }
    
    if (processBtn.disabled) {
      processBtn.classList.add('disabled');
    } else {
      processBtn.classList.remove('disabled');
    }
    
    if (downloadBtn.disabled) {
      downloadBtn.classList.add('disabled');
    } else {
      downloadBtn.classList.remove('disabled');
    }
  }
  
  function hasMaskContent() {
    if (!maskCanvas) return false;
    
    const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    const maskData = maskImageData.data;
    
    for (let i = 3; i < maskData.length; i += 4) {
      if (maskData[i] > 0) {
        return true;
      }
    }
    
    return false;
  }
  
  function processImage() {
    const prompt = promptInput.value.trim();
    
    if (!originalImage) {
      setStatus('Please upload an image first', 'error');
      return;
    }
    
    if (!prompt) {
      setStatus('Please enter an edit instruction', 'error');
      return;
    }
    
    // Disable process button and show loading
    processBtn.disabled = true;
    showLoading(true, 'Processing image...');
    
    // Clear progress check interval if it exists
    if (progressCheckInterval) {
      clearInterval(progressCheckInterval);
      progressCheckInterval = null;
    }
    
    // Reset progress visualization
    resetProgress();
    
    // Get the full-sized image and mask
    const imageBase64 = getFullSizeImageBase64(originalImage);
    const maskBase64 = getFullSizeMaskBase64();
    
    // Get parameters
    const imageGuidance = parseFloat(imgGuidanceSlider.value);
    const textGuidance = parseFloat(textGuidanceSlider.value);
    const steps = parseInt(stepsSlider.value);
    
    // Prepare data for API
    const requestData = {
      image: imageBase64,
      mask: hasMaskContent() ? maskBase64 : null,
      prompt: prompt,
      image_guidance_scale: imageGuidance,
      text_guidance_scale: textGuidance,
      num_steps: steps
    };
    
    // Send request to start the process
    fetch('/start_process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        setStatus(`Error: ${data.error}`, 'error');
        showLoading(false);
        return;
      }
      
      // Store the process ID
      currentProcessId = data.process_id;
      
      // Start checking progress regularly
      progressCheckInterval = setInterval(() => {
        checkProgress(currentProcessId);
      }, 1000);
      
      // Show status
      setStatus('Processing image, please wait...', 'info');
    })
    .catch(error => {
      console.error('Error:', error);
      setStatus(`Error: ${error.message}`, 'error');
      showLoading(false);
      processBtn.disabled = false;
    });
  }
  
  function checkProgress(processId) {
    if (!processId) return;
    
    fetch(`/check_progress/${processId}`)
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          clearInterval(progressCheckInterval);
          progressCheckInterval = null;
          currentProcessId = null;
          setStatus(`Error: ${data.error}`, 'error');
          showLoading(false);
          processBtn.disabled = false;
          return;
        }
        
        // Update progress bar
        updateProgressBar(data.progress);
        
        // Update loading message
        showLoading(true, data.message);
        
        // Check if process is completed
        if (data.status === 'completed') {
          // Clear interval and reset process ID
          clearInterval(progressCheckInterval);
          progressCheckInterval = null;
          currentProcessId = null;
          
          // Hide loading overlay
          showLoading(false);
          
          // Update button states
          processBtn.disabled = false;
          
          // Display final result
          if (data.result) {
            // Hide placeholder
            resultPlaceholder.style.display = 'none';
            
            // Display result
            displayImage(data.result, resultCanvas);
            resultCanvas.hasDrawnImage = true;
            
            // Set success status
            setStatus(data.message, 'success');
            
            // Enable download button
            downloadBtn.disabled = false;
            downloadBtn.classList.remove('disabled');
          }
        } else if (data.status === 'error') {
          // Clear interval and reset process ID
          clearInterval(progressCheckInterval);
          progressCheckInterval = null;
          currentProcessId = null;
          
          // Hide loading overlay
          showLoading(false);
          
          // Update button states
          processBtn.disabled = false;
          
          // Set error status
          setStatus(data.message, 'error');
        }
      })
      .catch(error => {
        console.error('Error checking progress:', error);
      });
  }
  
  function updateProgressBar(progress) {
    progressBarFill.style.width = `${progress}%`;
    progressPercentage.textContent = `${progress}%`;
  }
  
  function resetProgress() {
    // Reset progress bar
    updateProgressBar(0);
    
    // Show placeholder
    resultPlaceholder.style.display = 'flex';
    
    // Clear result canvas
    resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
    
    // Reset result flag
    resultCanvas.hasDrawnImage = false;
  }
  
  function getFullSizeImageBase64(image) {
    // Create a canvas with the original image dimensions
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = image.width;
    tempCanvas.height = image.height;
    
    // Draw the image at its original size
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(image, 0, 0);
    
    // Return as base64
    return tempCanvas.toDataURL('image/png');
  }
  
  function getFullSizeMaskBase64() {
    if (!originalImage || !maskCanvas) return null;
    
    // Create a canvas with the original image dimensions
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = originalImage.width;
    tempCanvas.height = originalImage.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Get the scaled mask from the mask canvas
    const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    
    // Scale the mask to match the original image dimensions
    const imageData = inputCanvas.imageData;
    if (!imageData) return null;
    
    // Draw mask at original image size and account for canvas positioning
    tempCtx.fillStyle = 'black';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
    
    tempCtx.save();
    
    // Calculate scaling factors
    const scaleX = originalImage.width / imageData.drawWidth;
    const scaleY = originalImage.height / imageData.drawHeight;
    
    // Calculate offset adjustment
    const offsetAdjustX = imageData.offsetX * scaleX;
    const offsetAdjustY = imageData.offsetY * scaleY;
    
    // Set white fill for the mask
    tempCtx.fillStyle = 'white';
    
    // Draw each pixel of the mask at the correct scaled position
    for (let y = 0; y < maskCanvas.height; y++) {
      for (let x = 0; x < maskCanvas.width; x++) {
        // Get pixel alpha value from mask
        const pixelIndex = (y * maskCanvas.width + x) * 4 + 3;
        const alpha = maskImageData.data[pixelIndex];
        
        if (alpha > 0) {
          // Calculate original image coordinates
          const canvasX = (x - imageData.offsetX) / imageData.drawWidth * originalImage.width;
          const canvasY = (y - imageData.offsetY) / imageData.drawHeight * originalImage.height;
          
          // Draw a point at the scaled position if it's a valid position
          if (canvasX >= 0 && canvasX < originalImage.width && 
              canvasY >= 0 && canvasY < originalImage.height) {
            tempCtx.fillRect(Math.floor(canvasX), Math.floor(canvasY), 1, 1);
          }
        }
      }
    }
    
    tempCtx.restore();
    
    // Return as base64
    return tempCanvas.toDataURL('image/png');
  }
  
  function displayImage(src, canvas) {
    const img = new Image();
    img.onload = function() {
      drawImageOnCanvas(img, canvas);
    };
    img.src = src;
  }
  
  function downloadResult() {
    if (!resultCanvas.hasDrawnImage) return;
    
    // Create a temporary link element
    const link = document.createElement('a');
    
    // Set link properties
    link.download = `instructpix2pix-${Date.now()}.png`;
    link.href = resultCanvas.toDataURL('image/png');
    
    // Append to document, click, and remove
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
  
  function setStatus(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.className = 'status-message';
    
    if (type) {
      statusMessage.classList.add(type);
    }
  }
  
  function showLoading(show, message = 'Processing...') {
    if (show) {
      loadingMessage.textContent = message;
      loadingOverlay.classList.add('visible');
    } else {
      loadingOverlay.classList.remove('visible');
    }
  }
});

