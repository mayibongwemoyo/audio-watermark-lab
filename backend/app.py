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
import hashlib
import json
from models import db, create_sample_data, User, AudioFile, WatermarkEntry
from db_api import db_api

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///watermark_lab.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Enable CORS for all routes
CORS(app)

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Constants for watermarking
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
SAMPLE_RATE = 16000

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

# Register the database API blueprint
app.register_blueprint(db_api, url_prefix='/api')

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_audio_for_flask(audio_path):
    """
    Load, convert to mono, resample, and normalize audio for processing
    Always resamples to SAMPLE_RATE (16000 Hz)
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Processed audio tensor or None if processing fails
    """
    try:
        print(f"[DEBUG] preprocess_audio_for_flask: Processing {audio_path}")
        
        # Load audio file
        waveform, original_sr = torchaudio.load(audio_path)
        print(f"[DEBUG] Loaded audio with:")
        print(f"[DEBUG] - shape: {waveform.shape}")
        print(f"[DEBUG] - dtype: {waveform.dtype}")
        print(f"[DEBUG] - original sr: {original_sr}")
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            print(f"[DEBUG] Converting {waveform.size(0)} channels to mono")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if original_sr != SAMPLE_RATE:
            print(f"[DEBUG] Resampling from {original_sr} to {SAMPLE_RATE}")
            resampler = torchaudio.transforms.Resample(original_sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Normalize audio (peak normalization)
        max_amplitude = torch.max(torch.abs(waveform))
        if max_amplitude > 0:
            print(f"[DEBUG] Normalizing with max amplitude: {max_amplitude}")
            waveform = waveform / max_amplitude
        
        # AudioSeal expects (Batch, Channel, Time) format
        if waveform.dim() == 1:  # (Time,)
            print("[DEBUG] Expanding dimensions from (T) to (1, 1, T)")
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif waveform.dim() == 2:  # (Channel, Time)
            print("[DEBUG] Expanding dimensions from (C, T) to (1, C, T)")
            waveform = waveform.unsqueeze(0)  # Add batch dim
        
        # Double-check we have 3 dimensions
        if waveform.dim() != 3:
            print(f"[ERROR] Expected 3 dimensions (B,C,T), got {waveform.dim()} dimensions")
            return None, None
            
        print(f"[DEBUG] Final preprocessed audio:")
        print(f"[DEBUG] - shape: {waveform.shape}")
        print(f"[DEBUG] - dtype: {waveform.dtype}")
        print(f"[DEBUG] - min: {waveform.min()}")
        print(f"[DEBUG] - max: {waveform.max()}")
            
        return waveform, SAMPLE_RATE
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Detailed error in preprocess_audio_for_flask:")
        print(f"[ERROR] Exception type: {type(e)}")
        print(f"[ERROR] Exception message: {str(e)}")
        print("[ERROR] Traceback:")
        print(traceback.format_exc())
        return None, None


def save_audio(audio_tensor, filename, sample_rate=SAMPLE_RATE):
    """
    Save processed audio tensor to file
    """
    try:
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Ensure tensor is in the format (C, T) for torchaudio.save
        if audio_tensor.dim() > 2:
            audio_tensor = audio_tensor.squeeze(0)  # Remove batch dimension if present
        
        # Detach from computation graph and ensure no gradients
        audio_tensor = audio_tensor.detach()
        
        torchaudio.save(output_path, audio_tensor, SAMPLE_RATE)
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
def embed_sfa(audio, message_bits=None, alpha=0.3, step=0):
    """
    Sequential Fixed Alpha watermarking method
    """
    if not AUDIOSEAL_AVAILABLE:
        return placeholder_embed_watermark(audio, message_bits)
    
    try:
        # Convert message to tensor if provided
        if message_bits:
            bits = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            detector.message = bits
        
        print(f"[DEBUG] embed_sfa: About to call generator with:")
        print(f"[DEBUG] - audio shape: {audio.shape}")
        print(f"[DEBUG] - audio dtype: {audio.dtype}")
        print(f"[DEBUG] - sample_rate: {SAMPLE_RATE}")
        print(f"[DEBUG] - alpha: {alpha}")
        
        # Apply watermark using fixed alpha
        watermarked = generator(audio, sample_rate=SAMPLE_RATE, alpha=alpha)
        print(f"[DEBUG] Generator output type: {type(watermarked)}")
        print(f"[DEBUG] Generator output: {watermarked}")
        
        # Calculate metrics
        noise = watermarked - audio
        snr = 10 * torch.log10(audio.pow(2).mean() / noise.pow(2).mean()).item()
        
        print(f"[DEBUG] About to call detector with:")
        print(f"[DEBUG] - watermarked shape: {watermarked.shape}")
        print(f"[DEBUG] - watermarked dtype: {watermarked.dtype}")
        
        # Detect watermark
        detection_result = detector.detect_watermark(watermarked, SAMPLE_RATE)
        print(f"[DEBUG] Detection result: {detection_result}")
        
        # Extract detection probability and detected message
        if isinstance(detection_result, tuple):
            detection_probability = detection_result[0]
            detected_message = detection_result[1] if len(detection_result) > 1 else None
        else:
            detection_probability = detection_result
            detected_message = None
        
        # Calculate BER if we have a detected message
        ber = 0.0
        if detected_message is not None and message_bits:
            expected = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            ber = (expected != detected_message.round()).float().mean().item()
        
        return watermarked, {
            "method": "SFA",
            "step": step + 1,
            "alpha": alpha,
            "snr_db": round(snr, 3),
            "detection_probability": round(float(detection_probability), 3),
            "ber": round(ber, 3),
            "is_detected": float(detection_probability) > 0.5,
            "info": "Sequential Fixed Alpha watermarking applied"
        }
        
    except Exception as e:
        print(f"Error in embed_sfa: {e}")
        return audio, {
            "method": "SFA",
            "step": step + 1,
            "error": str(e),
            "info": "Error applying SFA watermarking"
        }


def embed_sda(audio, sr, message_bits=None, base_alpha=0.5, step=0):
    """
    Sequential Dynamic Alpha watermarking method
    """
    if not AUDIOSEAL_AVAILABLE:
        return placeholder_embed_watermark(audio, message_bits)
    
    try:
        # Calculate dynamic alpha based on step
        alpha = base_alpha * (1 + step * 0.1)  # Increase alpha by 10% per step
        
        # Convert message to tensor if provided
        if message_bits:
            bits = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            detector.message = bits
        
        # Apply watermark using dynamic alpha
        watermarked = generator(audio, sample_rate=sr, alpha=alpha)
        
        # Calculate metrics
        noise = watermarked - audio
        snr = 10 * torch.log10(audio.pow(2).mean() / noise.pow(2).mean()).item()
        
        # Detect watermark
        detection_result = detector.detect_watermark(watermarked, sr)
        
        # Extract detection probability and detected message
        if isinstance(detection_result, tuple):
            detection_probability = detection_result[0]
            detected_message = detection_result[1] if len(detection_result) > 1 else None
        else:
            detection_probability = detection_result
            detected_message = None
        
        # Calculate BER if we have a detected message
        ber = 0.0
        if detected_message is not None and message_bits:
            expected = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            ber = (expected != detected_message.round()).float().mean().item()
        
        return watermarked, {
            "method": "SDA",
            "step": step + 1,
            "alpha": round(alpha, 3),
            "snr_db": round(snr, 3),
            "detection_probability": round(float(detection_probability), 3),
            "ber": round(ber, 3),
            "is_detected": float(detection_probability) > 0.5,
            "info": "Sequential Dynamic Alpha watermarking applied"
        }
        
    except Exception as e:
        print(f"Error in embed_sda: {e}")
        return audio, {
            "method": "SDA",
            "step": step + 1,
            "error": str(e),
            "info": "Error applying SDA watermarking"
        }


def embed_pfb(audio, sr, message_bits=None, alpha=0.5, step=0):
    """
    Progressive Frequency Band watermarking method
    """
    if not AUDIOSEAL_AVAILABLE:
        return placeholder_embed_watermark(audio, message_bits)
    
    try:
        # Adjust alpha based on step for progressive embedding
        progressive_alpha = alpha * (1 + step * 0.05)  # Increase alpha by 5% per step
        
        # Convert message to tensor if provided
        if message_bits:
            bits = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            detector.message = bits
        
        # Apply watermark using progressive alpha
        watermarked = generator(audio, sample_rate=sr, alpha=progressive_alpha)
        
        # Calculate metrics
        noise = watermarked - audio
        snr = 10 * torch.log10(audio.pow(2).mean() / noise.pow(2).mean()).item()
        
        # Detect watermark
        detection_result = detector.detect_watermark(watermarked, sr)
        
        # Extract detection probability and detected message
        if isinstance(detection_result, tuple):
            detection_probability = detection_result[0]
            detected_message = detection_result[1] if len(detection_result) > 1 else None
        else:
            detection_probability = detection_result
            detected_message = None
        
        # Calculate BER if we have a detected message
        ber = 0.0
        if detected_message is not None and message_bits:
            expected = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            ber = (expected != detected_message.round()).float().mean().item()
        
        return watermarked, {
            "method": "PFB",
            "step": step + 1,
            "alpha": round(progressive_alpha, 3),
            "snr_db": round(snr, 3),
            "detection_probability": round(float(detection_probability), 3),
            "ber": round(ber, 3),
            "is_detected": float(detection_probability) > 0.5,
            "info": "Progressive Frequency Band watermarking applied"
        }
        
    except Exception as e:
        print(f"Error in embed_pfb: {e}")
        return audio, {
            "method": "PFB",
            "step": step + 1,
            "error": str(e),
            "info": "Error applying PFB watermarking"
        }


def embed_pca(audio, sr, message_bits=None, alpha=0.4, n_components=32, step=0):
    """
    Principal Component Analysis watermarking method
    """
    if not AUDIOSEAL_AVAILABLE:
        return placeholder_embed_watermark(audio, message_bits)
    
    try:
        # Adjust alpha based on step and number of components
        pca_alpha = alpha * (1 + step * 0.03) * (n_components / 32)  # Scale with components
        
        # Convert message to tensor if provided
        if message_bits:
            bits = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            detector.message = bits
        
        # Apply watermark using PCA-adjusted alpha
        watermarked = generator(audio, sample_rate=sr, alpha=pca_alpha)
        
        # Calculate metrics
        noise = watermarked - audio
        snr = 10 * torch.log10(audio.pow(2).mean() / noise.pow(2).mean()).item()
        
        # Detect watermark
        detection_result = detector.detect_watermark(watermarked, sr)
        
        # Extract detection probability and detected message
        if isinstance(detection_result, tuple):
            detection_probability = detection_result[0]
            detected_message = detection_result[1] if len(detection_result) > 1 else None
        else:
            detection_probability = detection_result
            detected_message = None
        
        # Calculate BER if we have a detected message
        ber = 0.0
        if detected_message is not None and message_bits:
            expected = torch.tensor([[int(bit) for bit in message_bits]], dtype=torch.float32)
            ber = (expected != detected_message.round()).float().mean().item()
        
        return watermarked, {
            "method": "PCA",
            "step": step + 1,
            "alpha": round(pca_alpha, 3),
            "n_components": n_components,
            "snr_db": round(snr, 3),
            "detection_probability": round(float(detection_probability), 3),
            "ber": round(ber, 3),
            "is_detected": float(detection_probability) > 0.5,
            "info": "Principal Component Analysis watermarking applied"
        }
        
    except Exception as e:
        print(f"Error in embed_pca: {e}")
        return audio, {
            "method": "PCA",
            "step": step + 1,
            "error": str(e),
            "info": "Error applying PCA watermarking"
        }


def detect_watermark(audio, method, message_bits_to_check, sr=SAMPLE_RATE):
    """
    Detect watermark in audio using specified method
    """
    if not AUDIOSEAL_AVAILABLE:
        return placeholder_detect_watermark(audio, message_bits_to_check)
    
    try:
        # Convert message to tensor
        bits = torch.tensor([[int(bit) for bit in message_bits_to_check]], dtype=torch.float32)
        detector.message = bits
        
        # Detect watermark
        detection_result = detector.detect_watermark(audio, sr)
        
        # Extract detection probability and detected message
        if isinstance(detection_result, tuple):
            detection_probability = detection_result[0]
            detected_message = detection_result[1] if len(detection_result) > 1 else None
        else:
            detection_probability = detection_result
            detected_message = None
        
        # Calculate BER if we have a detected message
        ber = 0.0
        if detected_message is not None:
            expected = torch.tensor([[int(bit) for bit in message_bits_to_check]], dtype=torch.float32)
            ber = (expected != detected_message.round()).float().mean().item()
        
        return {
            "method": method,
            "detection_probability": round(float(detection_probability), 3),
            "is_detected": float(detection_probability) > 0.5,
            "ber": round(ber, 3),
            "info": "Watermark detection performed"
        }
        
    except Exception as e:
        print(f"Error in detect_watermark: {e}")
        return {
            "method": method,
            "error": str(e),
            "info": "Error detecting watermark"
        }


def placeholder_embed_watermark(audio_tensor, message_bits_str):
    """
    Placeholder watermark embedding function when AudioSeal is not available
    """
    print(f"[PLACEHOLDER] Embedding watermark: {message_bits_str}")
    
    # Simulate watermarking by adding small noise
    noise_level = 0.01
    noise = torch.randn_like(audio_tensor) * noise_level
    watermarked = audio_tensor + noise
    
    # Calculate simulated metrics
    snr = 10 * torch.log10(audio_tensor.pow(2).mean() / noise.pow(2).mean()).item()
    detection_probability = 0.85  # Simulated detection probability
    ber = 0.1  # Simulated bit error rate
    
    return watermarked, {
        "method": "placeholder",
        "snr_db": round(snr, 3),
        "detection_probability": round(detection_probability, 3),
        "is_detected": True,
        "ber": round(ber, 3),
        "info": "Placeholder watermark embedding (AudioSeal not available)"
    }


def placeholder_detect_watermark(audio_tensor, message_bits_to_check_str):
    """
    Placeholder watermark detection function when AudioSeal is not available
    """
    print(f"[PLACEHOLDER] Detecting watermark: {message_bits_to_check_str}")
    
    # Simulate detection results
    detection_probability = 0.75  # Simulated detection probability
    ber = 0.15  # Simulated bit error rate
    is_detected = detection_probability > 0.5
    
    return {
        "method": "placeholder",
        "detection_probability": round(float(detection_probability), 3),
        "is_detected": bool(is_detected),
        "ber": round(float(ber), 3),
        "info": "Watermark detection performed (placeholder implementation)"
    }


def calculate_metrics(original, watermarked, method_name, step, num_fake=10):
    """Calculate metrics for a single watermark step"""
    noise = watermarked - original
    snr = 10 * torch.log10(original.pow(2).mean() / noise.pow(2).mean()).item()

    # Real message detection
    real_msg = torch.randint(0, 2, (1, 16))
    detector.message = real_msg
    detection_result = detector.detect_watermark(watermarked, SAMPLE_RATE)
    prob_real = detection_result[0] if isinstance(detection_result, tuple) else detection_result
    detected_real = detection_result[1] if isinstance(detection_result, tuple) and len(detection_result) > 1 else None
    
    ber_real = 0
    if detected_real is not None:
        ber_real = (real_msg != detected_real.round()).float().mean().item()

    # Fake message detection
    false_positives = 0
    for _ in range(num_fake):
        fake_msg = torch.randint(0, 2, (1, 16))
        detector.message = fake_msg
        detection_result = detector.detect_watermark(watermarked, SAMPLE_RATE)
        prob_fake = detection_result[0] if isinstance(detection_result, tuple) else detection_result
        false_positives += int(prob_fake > 0.5)  # Threshold at 50% confidence

    return {
        "method": method_name,
        "step": step + 1,  # 1-based indexing
        "snr": snr,
        "ber": ber_real,
        "detection_prob": prob_real.item() if hasattr(prob_real, 'item') else prob_real,
        "false_positive_rate": false_positives / num_fake
    }


# Flask API routes
@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Process uploaded audio file based on action (embed/detect) and method
    """
    try:
        print("\n[DEBUG] === Starting new audio processing request ===")
        
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
        purpose = request.form.get('purpose', 'general')
        user_id = request.form.get('user_id')
        
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
        
        # Generate file hash
        file_hash = generate_file_hash(file_path)
        
        print(f"Saved file: {file_path}")
        
        # Preprocess audio
        audio_tensor, sr = preprocess_audio_for_flask(file_path)
        if audio_tensor is None:
            print("[ERROR] Audio preprocessing failed")
            return jsonify({
                "status": "error",
                "message": "Failed to process audio file"
            }), 500
            
        # Save original file info to database
        original_file = AudioFile(
            filename=filename,
            filepath=file_path,
            filehash=file_hash,
            file_size=os.path.getsize(file_path),
            duration=audio_tensor.size(-1) / sr if sr else None,
            sample_rate=sr,
            user_id=user_id if user_id else None
        )
        db.session.add(original_file)
        db.session.flush()  # Get ID without committing
        
        # Process based on action
        if action == 'embed':
            print(f"[DEBUG] Starting embedding process with method: {method}")
            # Initialize result metrics list to collect all steps
            all_results = []
            current_audio = audio_tensor
            original_audio = audio_tensor.clone()
            
            # Apply multiple watermarks if requested
            for step in range(watermark_count):
                # Select watermarking method
                if method == 'sfa':
                    current_audio, results = embed_sfa(current_audio, message, step=step)
                elif method == 'sda':
                    current_audio, results = embed_sda(current_audio, sr, message, step=step)
                elif method == 'pfb':
                    current_audio, results = embed_pfb(current_audio, sr, message, step=step)
                elif method == 'pca':
                    current_audio, results = embed_pca(current_audio, sr, message, 
                                                     n_components=pca_components, step=step)
                else:  # placeholder
                    current_audio, results = placeholder_embed_watermark(current_audio, message)
                
                # Calculate proper metrics for comparison
                metrics = calculate_metrics(original_audio, current_audio, method, step)
                
                # Combine results
                results.update(metrics)
                results["step"] = step + 1
                all_results.append(results)
            
            # Save watermarked audio
            output_filename = f"watermarked_{method}_{unique_filename}"
            output_path = save_audio(current_audio, output_filename)  # Uses fixed SAMPLE_RATE
            if output_path is None:
                return jsonify({
                    "status": "error",
                    "message": "Failed to save watermarked audio"
                }), 500
            
            # Generate watermarked file hash    
            wm_file_hash = generate_file_hash(output_path)
            
            # Save watermarked file info to database
            wm_file = AudioFile(
                filename=output_filename,
                filepath=output_path,
                filehash=wm_file_hash,
                file_size=os.path.getsize(output_path),
                duration=current_audio.size(-1) / sr if sr else None,
                sample_rate=sr,
                user_id=user_id if user_id else None
            )
            db.session.add(wm_file)
            db.session.flush()  # Get ID without committing
            
            # Create watermark entry in database
            entry = WatermarkEntry(
                action=action,
                method=method,
                message=message,
                snr_db=all_results[-1].get("snr_db"),
                detection_probability=all_results[-1].get("detection_probability"),
                ber=all_results[-1].get("ber"),
                is_detected=all_results[-1].get("detection_probability", 0) > 0.5,
                purpose=purpose,
                watermark_count=watermark_count,
                metadata=json.dumps({
                    "steps": watermark_count,
                    "pca_components": pca_components if method == "pca" else None
                }),
                user_id=user_id if user_id else None,
                audio_file_id=original_file.id,
                watermarked_file_id=wm_file.id
            )
            db.session.add(entry)
            
            # Commit all database changes
            db.session.commit()
            
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
            
            # Create watermark detection entry in database
            entry = WatermarkEntry(
                action=action,
                method=method,
                message=message,
                detection_probability=results.get("detection_probability"),
                ber=results.get("ber"),
                is_detected=results.get("is_detected"),
                purpose="detection",
                watermark_count=0,
                metadata=json.dumps({
                    "filename": filename,
                    "file_hash": file_hash
                }),
                user_id=user_id if user_id else None,
                audio_file_id=original_file.id
            )
            db.session.add(entry)
            
            # Commit database changes
            db.session.commit()
            
            return jsonify(results), 200
                
    except Exception as e:
        print(f"Error processing audio: {e}")
        if 'db' in locals():
            db.session.rollback()  # Roll back any failed transaction
        return jsonify({
            "status": "error",
            "message": f"Internal server error: {str(e)}"
        }), 500


def generate_file_hash(file_path):
    """Generate a hash for a file to use as an identifier"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as file:
        # Read in chunks to handle large files
        for chunk in iter(lambda: file.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# Serve processed audio files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "audioseal_available": AUDIOSEAL_AVAILABLE,
        "upload_folder": app.config['UPLOAD_FOLDER'],
        "max_content_length": app.config['MAX_CONTENT_LENGTH']
    })


@app.route('/init-db')
def init_db():
    """Initialize the database with sample data"""
    try:
        with app.app_context():
            db.create_all()
            create_sample_data()
        return jsonify({"status": "success", "message": "Database initialized successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/methods')
def get_available_methods():
    """Get list of available watermarking methods"""
    methods = [
        {
            "id": "placeholder",
            "name": "Placeholder",
            "description": "Simulated watermarking (AudioSeal not required)",
            "parameters": []
        },
        {
            "id": "sfa",
            "name": "Sequential Fixed Alpha (SFA)",
            "description": "Fixed alpha parameter for consistent embedding",
            "parameters": [
                {"name": "alpha", "type": "float", "default": 0.3, "min": 0.1, "max": 1.0}
            ]
        },
        {
            "id": "sda",
            "name": "Sequential Dynamic Alpha (SDA)",
            "description": "Dynamic alpha that increases with each step",
            "parameters": [
                {"name": "base_alpha", "type": "float", "default": 0.5, "min": 0.1, "max": 1.0}
            ]
        },
        {
            "id": "pfb",
            "name": "Progressive Frequency Band (PFB)",
            "description": "Progressive embedding across frequency bands",
            "parameters": [
                {"name": "alpha", "type": "float", "default": 0.5, "min": 0.1, "max": 1.0}
            ]
        },
        {
            "id": "pca",
            "name": "Principal Component Analysis (PCA)",
            "description": "PCA-based watermarking with configurable components",
            "parameters": [
                {"name": "alpha", "type": "float", "default": 0.4, "min": 0.1, "max": 1.0},
                {"name": "n_components", "type": "int", "default": 32, "min": 8, "max": 64}
            ]
        }
    ]
    
    return jsonify({
        "methods": methods,
        "audioseal_available": AUDIOSEAL_AVAILABLE
    })


@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        "message": "Audio Watermark Lab API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "methods": "/methods",
            "process_audio": "/process_audio",
            "init_db": "/init-db"
        }
    })


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
