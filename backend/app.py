
import os
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torchaudio
import soundfile as sf
from werkzeug.utils import secure_filename
from scipy import stats

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
SAMPLE_RATE = 16000

# Enable CORS for all routes
CORS(app)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Constants for watermarking
NUM_WATERMARKS = 4
NUM_AUDIOS_PER_METHOD = 5

# Try to import AudioSeal for actual watermarking
try:
    from audioseal import AudioSeal
    # Initialize models
    generator = AudioSeal.load_generator("audioseal_wm_16bits")
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    AUDIOSEAL_AVAILABLE = True
    print("AudioSeal imported successfully!")
except ImportError:
    print("Warning: AudioSeal not available. Using placeholder functions.")
    AUDIOSEAL_AVAILABLE = False


def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_audio_for_flask(audio_path, target_sr=SAMPLE_RATE):
    """
    Load, convert to mono, resample, and normalize audio for processing
    
    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate (default: 16000)
        
    Returns:
        Processed audio tensor or None if processing fails
    """
    try:
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Normalize audio (peak normalization)
        max_amplitude = torch.max(torch.abs(waveform))
        if max_amplitude > 0:  # Avoid division by zero for silent audio
            waveform = waveform / max_amplitude
        
        # Ensure shape is (1, T) for processing (we'll handle adding another dimension if needed)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, T)
            
        return waveform, sample_rate
        
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None, None


def save_audio(audio_tensor, filename, sample_rate=SAMPLE_RATE):
    """
    Save processed audio tensor to file
    
    Args:
        audio_tensor: Audio tensor to save
        filename: Output filename
        sample_rate: Sample rate for the output file
        
    Returns:
        Path to the saved file or None if saving fails
    """
    try:
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Ensure tensor is in the format (C, T) for torchaudio.save
        if audio_tensor.dim() > 2:
            audio_tensor = audio_tensor.squeeze(0)  # Remove batch dimension if present
        
        torchaudio.save(output_path, audio_tensor, sample_rate)
        return output_path
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None


# Implementation of preprocessing from notebook
def preprocess_audio_from_tensor(audio, sr):
    """Convert audio to standard format: (1, 1, T) @ 16kHz"""
    # Convert numpy arrays to tensor if needed
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()

    # Ensure 3D shape: (batch=1, channels=1, time)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)  # (1, C, T)

    # Convert to mono if needed
    if audio.shape[1] > 1:
        audio = audio.mean(dim=1, keepdim=True)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)

    return audio


# Watermarking Methods Implementation
def embed_sfa(audio, sr, message_bits=None, alpha=0.3, step=0):
    """
    Sequential Fixed Alpha watermarking method
    
    Args:
        audio: Audio tensor
        sr: Sample rate
        message_bits: Binary message string (not used in actual implementation)
        alpha: Watermark strength (default: 0.3)
        
    Returns:
        Tuple of (watermarked_audio_tensor, results_dict)
    """
    if not AUDIOSEAL_AVAILABLE:
        return placeholder_embed_watermark(audio, message_bits)
    
    try:
        # Convert string message to binary tensor if provided
        if message_bits:
            bits = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            detector.message = bits
        
        # Actually embed the watermark
        watermarked = generator(audio, sample_rate=sr, alpha=alpha)
        
        # Calculate SNR
        noise = watermarked - audio
        snr = 10 * torch.log10(audio.pow(2).mean() / noise.pow(2).mean()).item()
        
        # Run detection to confirm
        prob, detected = detector.detect_watermark(watermarked, sr)
        
        # Calculate BER
        ber = 0
        if message_bits:
            ber = ((bits != detected.round()).float().mean()).item()
        
        # Create results dictionary
        results = {
            "status": "success",
            "action": "embed",
            "method": "sfa",
            "message_embedded": message_bits,
            "snr_db": round(float(snr), 2),
            "detection_probability": round(float(prob.item() if hasattr(prob, 'item') else prob), 3),
            "ber": round(float(ber), 3),
            "info": "Watermark embedded using Sequential Fixed Alpha method"
        }
        
        return watermarked, results
        
    except Exception as e:
        print(f"Error in SFA embedding: {e}")
        return placeholder_embed_watermark(audio, message_bits)


def embed_sda(audio, sr, message_bits=None, base_alpha=0.5, step=0):
    """
    Sequential Decaying Alpha watermarking method
    
    Args:
        audio: Audio tensor
        sr: Sample rate
        message_bits: Binary message string (not used in actual implementation)
        base_alpha: Base watermark strength (default: 0.5)
        step: Watermark step (used to calculate decaying alpha)
        
    Returns:
        Tuple of (watermarked_audio_tensor, results_dict)
    """
    if not AUDIOSEAL_AVAILABLE:
        return placeholder_embed_watermark(audio, message_bits)
    
    try:
        # Convert string message to binary tensor if provided
        if message_bits:
            bits = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            detector.message = bits
        
        # Calculate decaying alpha based on step
        alpha = base_alpha / (step + 1)  # +1 to avoid division by zero
        
        # Actually embed the watermark
        watermarked = generator(audio, sample_rate=sr, alpha=alpha)
        
        # Calculate SNR
        noise = watermarked - audio
        snr = 10 * torch.log10(audio.pow(2).mean() / noise.pow(2).mean()).item()
        
        # Run detection to confirm
        prob, detected = detector.detect_watermark(watermarked, sr)
        
        # Calculate BER
        ber = 0
        if message_bits:
            ber = ((bits != detected.round()).float().mean()).item()
        
        # Create results dictionary
        results = {
            "status": "success",
            "action": "embed",
            "method": "sda",
            "message_embedded": message_bits,
            "snr_db": round(float(snr), 2),
            "detection_probability": round(float(prob.item() if hasattr(prob, 'item') else prob), 3),
            "ber": round(float(ber), 3),
            "info": "Watermark embedded using Sequential Decaying Alpha method"
        }
        
        return watermarked, results
        
    except Exception as e:
        print(f"Error in SDA embedding: {e}")
        return placeholder_embed_watermark(audio, message_bits)


def embed_pfb(audio, sr, message_bits=None, alpha=0.5, step=0):
    """
    Parallel Frequency Bands watermarking method
    
    Args:
        audio: Audio tensor
        sr: Sample rate
        message_bits: Binary message string (not used in actual implementation)
        alpha: Watermark strength (default: 0.5)
        step: Watermark step (used to determine the frequency band)
        
    Returns:
        Tuple of (watermarked_audio_tensor, results_dict)
    """
    if not AUDIOSEAL_AVAILABLE:
        return placeholder_embed_watermark(audio, message_bits)
    
    try:
        # Convert string message to binary tensor if provided
        if message_bits:
            bits = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            detector.message = bits
        
        # Transform to frequency domain
        fft = torch.fft.fft(audio)
        bands = torch.chunk(fft, 4, dim=-1)
        
        # Select band based on step (cycle through bands)
        band_idx = step % 4
        band = bands[band_idx]
        
        # Watermark real part of selected band and keep imaginary part
        wm_band = generator(band.real, sample_rate=sr, alpha=alpha) + 1j * band.imag
        
        # Replace the selected band
        watermarked_bands = list(bands)
        watermarked_bands[band_idx] = wm_band
        
        # Transform back to time domain
        watermarked = torch.fft.ifft(torch.cat(watermarked_bands, dim=-1)).real
        
        # Calculate SNR
        noise = watermarked - audio
        snr = 10 * torch.log10(audio.pow(2).mean() / noise.pow(2).mean()).item()
        
        # Run detection to confirm
        prob, detected = detector.detect_watermark(watermarked, sr)
        
        # Calculate BER
        ber = 0
        if message_bits:
            ber = ((bits != detected.round()).float().mean()).item()
        
        # Create results dictionary
        results = {
            "status": "success",
            "action": "embed",
            "method": "pfb",
            "message_embedded": message_bits,
            "snr_db": round(float(snr), 2),
            "detection_probability": round(float(prob.item() if hasattr(prob, 'item') else prob), 3),
            "ber": round(float(ber), 3),
            "info": "Watermark embedded using Parallel Frequency Bands method"
        }
        
        return watermarked, results
        
    except Exception as e:
        print(f"Error in PFB embedding: {e}")
        return placeholder_embed_watermark(audio, message_bits)


def embed_pca(audio, sr, message_bits=None, alpha=0.4, n_components=32, step=0):
    """
    PCA-based watermarking method
    
    Args:
        audio: Audio tensor
        sr: Sample rate
        message_bits: Binary message string
        alpha: Watermark strength (default: 0.4)
        n_components: Number of PCA components to use (default: 32)
        step: Watermark step
        
    Returns:
        Tuple of (watermarked_audio_tensor, results_dict)
    """
    if not AUDIOSEAL_AVAILABLE:
        return placeholder_embed_watermark(audio, message_bits)
    
    try:
        # Convert string message to binary tensor if provided
        if message_bits:
            bits = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            detector.message = bits
        
        # Convert audio to frequency domain
        fft = torch.fft.fft(audio)
        mag = torch.abs(fft)
        phase = torch.angle(fft)
        
        # Use magnitude for PCA-based embedding (simplified PCA implementation)
        mag_flat = mag.view(-1, mag.size(-1))
        mag_mean = torch.mean(mag_flat, dim=1, keepdim=True)
        mag_centered = mag_flat - mag_mean
        
        # Compute covariance matrix (simplified)
        cov = torch.matmul(mag_centered, mag_centered.transpose(-1, -2))
        
        # Compute eigenvalues and eigenvectors (using torch.linalg.eigh)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Sort eigenvectors by decreasing eigenvalues
        _, indices = torch.sort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, indices]
        
        # Select principal components
        n_components = min(n_components, eigenvectors.size(1))
        principal_components = eigenvectors[:, :n_components]
        
        # Project data onto principal components
        projected = torch.matmul(mag_centered, principal_components)
        
        # Embed watermark in the principal components
        projected_wm = generator(projected.unsqueeze(0), sample_rate=sr, alpha=alpha).squeeze(0)
        
        # Reconstruct magnitude
        mag_reconstructed = torch.matmul(projected_wm, principal_components.transpose(-1, -2)) + mag_mean
        mag_reconstructed = mag_reconstructed.view(mag.shape)
        
        # Combine with original phase to get watermarked FFT
        watermarked_fft = mag_reconstructed * torch.exp(1j * phase)
        
        # Convert back to time domain
        watermarked = torch.fft.ifft(watermarked_fft).real
        
        # Calculate SNR
        noise = watermarked - audio
        snr = 10 * torch.log10(audio.pow(2).mean() / noise.pow(2).mean()).item()
        
        # Run detection to confirm
        prob, detected = detector.detect_watermark(watermarked, sr)
        
        # Calculate BER
        ber = 0
        if message_bits:
            ber = ((bits != detected.round()).float().mean()).item()
        
        # Create results dictionary
        results = {
            "status": "success",
            "action": "embed",
            "method": "pca",
            "message_embedded": message_bits,
            "snr_db": round(float(snr), 2),
            "detection_probability": round(float(prob.item() if hasattr(prob, 'item') else prob), 3),
            "ber": round(float(ber), 3),
            "n_components": n_components,
            "info": "Watermark embedded using PCA-based method"
        }
        
        return watermarked, results
        
    except Exception as e:
        print(f"Error in PCA embedding: {e}")
        return placeholder_embed_watermark(audio, message_bits)


def detect_watermark(audio, method, message_bits_to_check, sr=SAMPLE_RATE):
    """
    Detect watermark in audio using specified method
    
    Args:
        audio: Audio tensor
        method: Watermarking method (sfa, sda, pfb, pca, placeholder)
        message_bits_to_check: Binary message string to check against
        sr: Sample rate
        
    Returns:
        Dictionary with detection results
    """
    if not AUDIOSEAL_AVAILABLE:
        return placeholder_detect_watermark(audio, message_bits_to_check)
    
    try:
        # Convert string message to binary tensor
        bits = torch.tensor([[int(bit) for bit in message_bits_to_check]], dtype=torch.float32)
        detector.message = bits
        
        # Detect watermark
        prob, detected = detector.detect_watermark(audio, sr)
        
        # Calculate BER (bit error rate)
        ber = (bits != detected.round()).float().mean().item()
        
        # Create results dictionary
        return {
            "status": "success",
            "action": "detect",
            "method": method,
            "message_checked": message_bits_to_check,
            "detection_probability": round(float(prob.item() if hasattr(prob, 'item') else prob), 3),
            "is_detected": bool(prob > 0.5),  # Threshold at 50% confidence
            "ber": round(float(ber), 3),
            "info": f"Watermark detection performed using {method} method"
        }
        
    except Exception as e:
        print(f"Error in watermark detection: {e}")
        return placeholder_detect_watermark(audio, message_bits_to_check)


def placeholder_embed_watermark(audio_tensor, message_bits_str):
    """
    Placeholder function for watermark embedding
    
    Args:
        audio_tensor: Processed audio tensor
        message_bits_str: Binary message to embed (e.g., "1010101010101010")
        
    Returns:
        Tuple of (watermarked_audio_tensor, results_dict)
    """
    print(f"Embedding watermark message using placeholder: {message_bits_str}")
    
    # Convert to numpy for manipulation
    audio_np = audio_tensor.numpy()
    
    # Generate very subtle noise (simulate watermarking)
    noise_factor = 0.001  # Very low amplitude
    noise = np.random.normal(0, noise_factor, audio_np.shape)
    
    # Add noise to audio (simulating watermark embedding)
    watermarked_audio_np = audio_np + noise
    
    # Calculate mock SNR
    signal_power = np.mean(audio_np ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100.0
    
    # Convert back to torch tensor
    watermarked_audio_tensor = torch.from_numpy(watermarked_audio_np).float()
    
    # Create results dictionary
    results = {
        "status": "success",
        "action": "embed",
        "method": "placeholder",
        "message_embedded": message_bits_str,
        "snr_db": round(float(snr), 2),
        "detection_probability": 0.85,  # Mock probability
        "ber": 0.25,  # Mock BER
        "info": "Watermark embedded (placeholder implementation)"
    }
    
    return watermarked_audio_tensor, results


def placeholder_detect_watermark(audio_tensor, message_bits_to_check_str):
    """
    Placeholder function for watermark detection
    
    Args:
        audio_tensor: Processed audio tensor
        message_bits_to_check_str: Binary message to check against
        
    Returns:
        Dictionary with detection results
    """
    print(f"Detecting watermark using placeholder, checking for message: {message_bits_to_check_str}")
    
    # Generate mock detection results
    # In a real implementation, this would analyze the audio
    detection_probability = 0.85 + np.random.uniform(-0.15, 0.15)  # Random variation for demo
    is_detected = detection_probability > 0.7  # Mock threshold
    
    # Calculate a mock BER (bit error rate)
    # In a real implementation, this would compare detected bits with expected bits
    ber = np.random.uniform(0.0, 0.3) if is_detected else np.random.uniform(0.3, 0.5)
    
    # Create results dictionary
    results = {
        "status": "success",
        "action": "detect",
        "method": "placeholder",
        "message_checked": message_bits_to_check_str,
        "detection_probability": round(float(detection_probability), 3),
        "is_detected": bool(is_detected),
        "ber": round(float(ber), 3),
        "info": "Watermark detection performed (placeholder implementation)"
    }
    
    return results


# Flask API routes
@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Process uploaded audio file based on action (embed/detect) and method
    """
    try:
        # Check if file part exists in request
        if 'audio_file' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No audio file provided"
            }), 400
            
        file = request.files['audio_file']
        
        # Check if file was actually selected
        if file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected"
            }), 400
            
        # Get parameters from request
        action = request.form.get('action', '')
        method = request.form.get('method', '')
        message = request.form.get('message', '')
        
        # Get optional parameters
        pca_components = int(request.form.get('pca_components', 32))
        watermark_count = int(request.form.get('watermark_count', 1))
        
        # Validate action
        if action not in ['embed', 'detect']:
            return jsonify({
                "status": "error",
                "message": "Invalid action. Must be 'embed' or 'detect'"
            }), 400
            
        # Validate method
        if method not in ['placeholder', 'sfa', 'sda', 'pfb', 'pca']:
            return jsonify({
                "status": "error",
                "message": f"Unsupported watermarking method: {method}"
            }), 400
        
        # Validate message
        if not message or not all(bit in '01' for bit in message):
            return jsonify({
                "status": "error",
                "message": "Invalid message. Must be a binary string (e.g., '1010101010101010')"
            }), 400
            
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "status": "error",
                "message": f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        print(f"Saved file: {file_path}")
        
        # Preprocess audio
        audio_tensor, sr = preprocess_audio_for_flask(file_path)
        if audio_tensor is None:
            return jsonify({
                "status": "error",
                "message": "Failed to process audio file"
            }), 500
            
        # Process based on action
        if action == 'embed':
            # Initialize result metrics list to collect all steps
            all_results = []
            current_audio = audio_tensor
            
            # Apply multiple watermarks if requested
            for step in range(watermark_count):
                # Select watermarking method
                if method == 'sfa':
                    current_audio, results = embed_sfa(current_audio, sr, message, step=step)
                elif method == 'sda':
                    current_audio, results = embed_sda(current_audio, sr, message, step=step)
                elif method == 'pfb':
                    current_audio, results = embed_pfb(current_audio, sr, message, step=step)
                elif method == 'pca':
                    current_audio, results = embed_pca(current_audio, sr, message, 
                                                      n_components=pca_components, step=step)
                else:  # placeholder
                    current_audio, results = placeholder_embed_watermark(current_audio, message)
                
                # Add step information to results
                results["step"] = step + 1
                all_results.append(results)
            
            # Save watermarked audio
            output_filename = f"watermarked_{method}_{unique_filename}"
            output_path = save_audio(current_audio, output_filename, sr)
            if output_path is None:
                return jsonify({
                    "status": "error",
                    "message": "Failed to save watermarked audio"
                }), 500
            
            # Create final response
            response = {
                "status": "success",
                "action": "embed",
                "method": method,
                "message_embedded": message,
                "watermark_count": watermark_count,
                "processed_audio_url": f"/uploads/{output_filename}",
                "results": all_results
            }
            
            return jsonify(response), 200
                
        elif action == 'detect':
            # Use the appropriate detection function
            results = detect_watermark(audio_tensor, method, message, sr)
            return jsonify(results), 200
                
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve processed audio files
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health')
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "message": "Audio Watermark Lab API is running",
        "audioseal_available": AUDIOSEAL_AVAILABLE
    }), 200


@app.route('/methods')
def get_available_methods():
    """
    Return information about available watermarking methods
    """
    methods = {
        "placeholder": {
            "name": "Placeholder",
            "description": "A simple placeholder implementation that adds subtle random noise to simulate watermarking",
            "available": True
        },
        "sfa": {
            "name": "Sequential Fixed Alpha (SFA)",
            "description": "Embeds watermark with fixed strength parameter alpha",
            "available": AUDIOSEAL_AVAILABLE
        },
        "sda": {
            "name": "Sequential Decaying Alpha (SDA)",
            "description": "Embeds watermark with decaying strength parameter alpha",
            "available": AUDIOSEAL_AVAILABLE
        },
        "pfb": {
            "name": "Parallel Frequency Bands (PFB)",
            "description": "Embeds watermark in parallel across different frequency bands",
            "available": AUDIOSEAL_AVAILABLE
        },
        "pca": {
            "name": "Principal Component Analysis (PCA)",
            "description": "Dynamic band selection using PCA for optimal watermark embedding",
            "available": AUDIOSEAL_AVAILABLE
        }
    }
    
    return jsonify({
        "status": "success",
        "audioseal_available": AUDIOSEAL_AVAILABLE,
        "methods": methods
    }), 200


@app.route('/')
def index():
    """
    Root endpoint with API information
    """
    return jsonify({
        "name": "Audio Watermark Lab API",
        "version": "0.2.0",
        "endpoints": {
            "/process_audio": "POST - Process audio file (embed/detect watermark)",
            "/uploads/<filename>": "GET - Access processed audio files",
            "/health": "GET - API health check",
            "/methods": "GET - Get information about available watermarking methods"
        }
    }), 200


if __name__ == '__main__':
    print("Starting Audio Watermark Lab API...")
    print(f"Uploads directory: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    print(f"AudioSeal available: {AUDIOSEAL_AVAILABLE}")
    app.run(debug=True, host='0.0.0.0', port=5000)

