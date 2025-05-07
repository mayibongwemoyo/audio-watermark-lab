
import os
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torchaudio
import soundfile as sf
from werkzeug.utils import secure_filename

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
        
        # Ensure shape is (1, 1, T) for processing
        if waveform.dim() == 2:  # (1, T)
            waveform = waveform.unsqueeze(1)  # (1, 1, T)
            
        return waveform
        
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None


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
        # Squeeze tensor to (C, T) format for torchaudio.save
        if audio_tensor.dim() == 3:  # (B, C, T)
            audio_tensor = audio_tensor.squeeze(0)  # Remove batch dimension
            
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        torchaudio.save(output_path, audio_tensor, sample_rate)
        return output_path
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None


def placeholder_embed_watermark(audio_tensor, message_bits_str):
    """
    Placeholder function for watermark embedding
    
    Args:
        audio_tensor: Processed audio tensor
        message_bits_str: Binary message to embed (e.g., "1010101010101010")
        
    Returns:
        Tuple of (watermarked_audio_tensor, results_dict)
    """
    print(f"Embedding watermark message: {message_bits_str}")
    
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
        "message_embedded": message_bits_str,
        "snr_db": round(float(snr), 2),
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
    print(f"Detecting watermark, checking for message: {message_bits_to_check_str}")
    
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
        "message_checked": message_bits_to_check_str,
        "detection_probability": round(float(detection_probability), 3),
        "is_detected": bool(is_detected),
        "ber": round(float(ber), 3),
        "info": "Watermark detection performed (placeholder implementation)"
    }
    
    return results


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
        
        # Validate action
        if action not in ['embed', 'detect']:
            return jsonify({
                "status": "error",
                "message": "Invalid action. Must be 'embed' or 'detect'"
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
        audio_tensor = preprocess_audio_for_flask(file_path)
        if audio_tensor is None:
            return jsonify({
                "status": "error",
                "message": "Failed to process audio file"
            }), 500
            
        # Process based on action
        if action == 'embed':
            # Currently only supporting placeholder method
            if method == 'placeholder':
                watermarked_audio, results = placeholder_embed_watermark(audio_tensor, message)
                
                # Save watermarked audio
                output_filename = f"watermarked_{unique_filename}"
                save_audio(watermarked_audio, output_filename)
                
                # Add URL to processed file in response
                results["processed_audio_url"] = f"/uploads/{output_filename}"
                
                return jsonify(results), 200
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Unsupported watermarking method: {method}"
                }), 400
                
        elif action == 'detect':
            # Currently only supporting placeholder method
            if method == 'placeholder':
                results = placeholder_detect_watermark(audio_tensor, message)
                return jsonify(results), 200
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Unsupported watermarking method: {method}"
                }), 400
                
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
        "message": "Audio Watermark Lab API is running"
    }), 200


@app.route('/')
def index():
    """
    Root endpoint with API information
    """
    return jsonify({
        "name": "Audio Watermark Lab API",
        "version": "0.1.0",
        "endpoints": {
            "/process_audio": "POST - Process audio file (embed/detect watermark)",
            "/uploads/<filename>": "GET - Access processed audio files",
            "/health": "GET - API health check"
        }
    }), 200


if __name__ == '__main__':
    print("Starting Audio Watermark Lab API...")
    print(f"Uploads directory: {os.path.abspath(app.config['UPLOAD_FOLDER'])}")
    app.run(debug=True, host='0.0.0.0', port=5000)
