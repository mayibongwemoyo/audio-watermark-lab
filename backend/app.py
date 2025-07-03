import os
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torchaudio
import soundfile as sf
import research_methods
import random
import uuid
from werkzeug.utils import secure_filename
from scipy import stats
import hashlib
import json
from models import db, create_sample_data, User, AudioFile, WatermarkEntry
from db_api import db_api
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max upload size
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///watermark_lab.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db.init_app(app)

# Configure CORS more explicitly
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8081", "http://127.0.0.1:8081"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Constants for watermarking
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}
SAMPLE_RATE = 16000 # Align with model's expected sample rate

# --- Helper Functions ---
def add_cors_headers(response):
    """Add CORS headers to response"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

def embed_sfa_watermark(audio_tensor, message, alpha=0.1):
    """Sequential Fixed Alpha watermarking"""
    try:
        # Convert message to binary
        if isinstance(message, str):
            message_bits = [int(bit) for bit in message]
        else:
            message_bits = message
            
        # Apply watermark by modulating amplitude
        watermarked = audio_tensor.clone()
        for i, bit in enumerate(message_bits):
            if i >= watermarked.shape[-1]:
                break
            if bit == 1:
                watermarked[..., i] *= (1 + alpha)
            else:
                watermarked[..., i] *= (1 - alpha)
                
        return watermarked, {"method": "SFA", "alpha": alpha}
    except Exception as e:
        print(f"Error in SFA embedding: {e}")
        return audio_tensor, {"method": "SFA", "error": str(e)}

def embed_sda_watermark(audio_tensor, message, alpha_max=0.2, decay_rate=0.95):
    """Sequential Decaying Alpha watermarking"""
    try:
        # Convert message to binary
        if isinstance(message, str):
            message_bits = [int(bit) for bit in message]
        else:
            message_bits = message
            
        # Apply watermark with decaying alpha
        watermarked = audio_tensor.clone()
        for i, bit in enumerate(message_bits):
            if i >= watermarked.shape[-1]:
                break
            alpha = alpha_max * (decay_rate ** i)
            if bit == 1:
                watermarked[..., i] *= (1 + alpha)
            else:
                watermarked[..., i] *= (1 - alpha)
                
        return watermarked, {"method": "SDA", "alpha_max": alpha_max, "decay_rate": decay_rate}
    except Exception as e:
        print(f"Error in SDA embedding: {e}")
        return audio_tensor, {"method": "SDA", "error": str(e)}

def embed_pfb_watermark(audio_tensor, message, alpha=0.15):
    """Parallel Frequency Bands watermarking"""
    try:
        # Convert message to binary
        if isinstance(message, str):
            message_bits = [int(bit) for bit in message]
        else:
            message_bits = message
            
        # Apply FFT
        fft = torch.fft.fft(audio_tensor)
        
        # Split into frequency bands
        num_bands = 4
        band_size = fft.shape[-1] // num_bands
        bands = []
        for i in range(num_bands):
            start_idx = i * band_size
            end_idx = start_idx + band_size if i < num_bands - 1 else fft.shape[-1]
            bands.append(fft[..., start_idx:end_idx])
        
        # Apply watermark to each band
        for i, band in enumerate(bands):
            if i < len(message_bits):
                bit = message_bits[i]
                if bit == 1:
                    bands[i] *= (1 + alpha)
                else:
                    bands[i] *= (1 - alpha)
        
        # Reconstruct
        watermarked_fft = torch.cat(bands, dim=-1)
        watermarked = torch.fft.ifft(watermarked_fft).real
        
        return watermarked, {"method": "PFB", "alpha": alpha}
    except Exception as e:
        print(f"Error in PFB embedding: {e}")
        return audio_tensor, {"method": "PFB", "error": str(e)}

def detect_watermark(audio_tensor, method, original_tensor=None):
    """Generic watermark detection for basic methods"""
    try:
        if method == "SFA":
            # Simple detection based on amplitude variations
            amplitude_variations = torch.std(audio_tensor, dim=-1)
            detection_probability = torch.sigmoid(amplitude_variations.mean()).item()
            return {
                "is_detected": detection_probability > 0.5,
                "detection_probability": detection_probability,
                "ber": 0.1,  # Placeholder
                "method": "SFA"
            }
        elif method == "SDA":
            # Similar to SFA but with different threshold
            amplitude_variations = torch.std(audio_tensor, dim=-1)
            detection_probability = torch.sigmoid(amplitude_variations.mean() * 1.2).item()
            return {
                "is_detected": detection_probability > 0.5,
                "detection_probability": detection_probability,
                "ber": 0.12,  # Placeholder
                "method": "SDA"
            }
        elif method == "PFB":
            # Frequency domain detection
            fft = torch.fft.fft(audio_tensor)
            freq_variations = torch.std(torch.abs(fft), dim=-1)
            detection_probability = torch.sigmoid(freq_variations.mean()).item()
            return {
                "is_detected": detection_probability > 0.5,
                "detection_probability": detection_probability,
                "ber": 0.08,  # Placeholder
                "method": "PFB"
            }
        else:
            return {
                "is_detected": False,
                "detection_probability": 0.0,
                "ber": 1.0,
                "method": method
            }
    except Exception as e:
        print(f"Error in watermark detection: {e}")
        return {
            "is_detected": False,
            "detection_probability": 0.0,
            "ber": 1.0,
            "method": method,
            "error": str(e)
        }

def calculate_snr(original, processed, epsilon=1e-9):
    """Calculate SNR in dB"""
    try:
        noise = processed - original
        signal_power = torch.mean(original**2)
        noise_power = torch.mean(noise**2)
        snr = 10 * torch.log10(signal_power / (noise_power + epsilon))
        return snr.item()
    except Exception as e:
        print(f"Error calculating SNR: {e}")
        return 0.0

# --- Import PFB/PCA Prime watermarking and load models once ---
try:
    from pca_prime_watermarking import (
        get_models, embed_single_incremental_pca_prime_watermark_step,
        embed_full_multi_pca_prime_watermark,
        detect_pca_prime_watermark, extract_payload_with_original, NUM_FREQ_BANDS, N_BITS
    )
    pfb_generator_instance, pfb_detector_instance = get_models()
    PCA_PRIME_AVAILABLE = True
    print("PCA Prime (PFB Y-Prime) watermarking module loaded and models initialized.")
except ImportError as e:
    print(f"Warning: pca_prime_watermarking module or models not available: {e}")
    print("Ensure pca_prime_watermarking.py is in the same directory and its MODEL_DIR contains 'pfb_parallel_4wm_exp_y_prime_best.pth'.")
    PCA_PRIME_AVAILABLE = False
    def embed_single_incremental_pca_prime_watermark_step(*args, **kwargs):
        print("ERROR: PCA Prime incremental embedding not available.")
        return torch.zeros(1, 1, SAMPLE_RATE), {"status": "error", "message": "PCA Prime incremental not available"}
    def detect_pca_prime_watermark(*args, **kwargs):
        print("ERROR: PCA Prime detection not available.")
        return {"status": "error", "message": "PCA Prime not available"}
    def detect_pca_prime_payload_only(*args, **kwargs):
        print("ERROR: PCA Prime payload detection not available.")
        return {"status": "error", "message": "PCA Prime payload detection not available"}
    NUM_FREQ_BANDS = 4 # Default for validation even if not available
    N_BITS = 8 # Default for validation even if not available


# --- AudioSeal specific initializations (if still needed for other methods) ---
try:
    from audioseal import AudioSeal
    audioseal_generator = AudioSeal.load_generator("audioseal_wm_16bits")
    audioseal_detector = AudioSeal.load_detector("audioseal_detector_16bits")
    AUDIOSEAL_AVAILABLE = True
    print("AudioSeal imported successfully!")
except ImportError:
    print("Warning: AudioSeal not available. SFA/SDA methods will not function.")
    AUDIOSEAL_AVAILABLE = False
    audioseal_generator = None
    audioseal_detector = None


# Register the database API blueprint
app.register_blueprint(db_api, url_prefix='/api')

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_audio_for_flask(audio_path):
    """
    Load, convert to mono, resample, and normalize audio for processing.
    Returns tensor in (1, C, T) format, suitable for model input.
    """
    try:
        print(f"[DEBUG] preprocess_audio_for_flask: Processing {audio_path}")
        
        waveform, original_sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True) # (1, Time)
            
        # Resample
        if original_sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(original_sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # Normalize audio (peak normalization to [-1, 1])
        max_amplitude = torch.max(torch.abs(waveform))
        if max_amplitude > 0:
            waveform = waveform / max_amplitude
        
        # Ensure (1, 1, T) for model input
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
        elif waveform.dim() == 2: # Already (C, T), assuming C=1 after mono conversion
            waveform = waveform.unsqueeze(0) # (1, C, T)
        
        return waveform, SAMPLE_RATE
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Detailed error in preprocess_audio_for_flask: {str(e)}")
        print(traceback.format_exc())
        return None, None


def save_audio(audio_tensor, filename, sample_rate=SAMPLE_RATE):
    """
    Save processed audio tensor to file. Expects (1, C, T) or (C, T) tensor.
    """
    try:
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure tensor is in the format (C, T) for torchaudio.save and detached
        audio_tensor_to_save = audio_tensor.detach().cpu()
        if audio_tensor_to_save.dim() == 3: # (Batch, Channel, Time)
            audio_tensor_to_save = audio_tensor_to_save.squeeze(0) # -> (Channel, Time)
        
        torchaudio.save(output_path, audio_tensor_to_save, SAMPLE_RATE)
        return output_path
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None


@app.route('/pre_embed_detect', methods=['POST'])
def pre_embed_detect():
    """
    Analyzes an uploaded audio file before embedding.
    - Detects existing watermarks.
    - Determines the original file ID.
    - Determines the next available watermark index.
    """
    print("[DEBUG] === Starting Pre-Embed Detection ===")
    if 'audio_file' not in request.files or request.files['audio_file'].filename == '':
        return jsonify({"status": "error", "message": "No audio file provided"}), 400

    file = request.files['audio_file']
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "message": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)
    file_hash = generate_file_hash(file_path)
    
    uploaded_audio_tensor, sr = preprocess_audio_for_flask(file_path)
    if uploaded_audio_tensor is None:
        return jsonify({"status": "error", "message": "Failed to process audio file"}), 500
        
    original_audio_file_id = None
    original_clean_audio_tensor = None
    is_self_comparison = False
    
    # Check if this file is ALREADY in our database
    wm_file_db_entry = AudioFile.query.filter_by(filehash=file_hash).first()

    if wm_file_db_entry:
        print(f"[DEBUG] Uploaded audio found in DB (ID: {wm_file_db_entry.id}). Tracing origin...")
        wm_entry = WatermarkEntry.query.filter_by(watermarked_file_id=wm_file_db_entry.id).order_by(WatermarkEntry.watermark_count.desc()).first()
        if wm_entry:
            original_file_id = wm_entry.audio_file_id
            original_file_db_entry = AudioFile.query.get(original_file_id)
            if original_file_db_entry and os.path.exists(original_file_db_entry.filepath):
                print(f"[DEBUG] Original audio found (ID: {original_file_id}). Loading for detection.")
                original_clean_audio_tensor, _ = preprocess_audio_for_flask(original_file_db_entry.filepath)
                original_audio_file_id = original_file_id
                # Check if the found original is actually the same file that was uploaded
                if original_file_db_entry.id == wm_file_db_entry.id:
                    is_self_comparison = True
    
    if original_clean_audio_tensor is None:
        print("[DEBUG] No valid watermark history found. Treating uploaded file as the original.")
        original_clean_audio_tensor = uploaded_audio_tensor.clone()
        is_self_comparison = True # It will be compared against itself
        
        # If we haven't found the file in the DB, add it now as a new original.
        if not wm_file_db_entry:
            new_original_file_db = AudioFile(
                filename=filename, filepath=file_path, filehash=file_hash,
                file_size=os.path.getsize(file_path), duration=uploaded_audio_tensor.size(-1) / sr,
                sample_rate=sr, user_id=request.form.get('user_id')
            )
            db.session.add(new_original_file_db)
            db.session.commit()
            original_audio_file_id = new_original_file_db.id
            wm_file_db_entry = new_original_file_db # This is now the reference entry
        else:
            original_audio_file_id = wm_file_db_entry.id

    # --- Perform Detection WITH SANITY CHECK ---
    if is_self_comparison:
        print("[DEBUG] Self-comparison detected. Bypassing model, returning all-zero payload.")
        all_zero_band = [0] * N_BITS
        detect_results = {
            "payload": '0' * (N_BITS * NUM_FREQ_BANDS),
            "per_band": [all_zero_band] * NUM_FREQ_BANDS
        }
    else:
        print("[DEBUG] Comparing two different files. Calling detector model.")
        detect_results = extract_payload_with_original(original_clean_audio_tensor, uploaded_audio_tensor)
    
    # --- Determine Next Available Watermark Slot ---
    next_wm_idx = 0
    detected_payloads = detect_results.get("per_band", [])
    for i, band_payload in enumerate(detected_payloads):
        if ''.join(map(str, band_payload)) == '0' * N_BITS:
            next_wm_idx = i
            break
        if i == len(detected_payloads) - 1:
            next_wm_idx = len(detected_payloads)

    print(f"[DEBUG] Detection complete. Next available watermark index: {next_wm_idx}")

    return jsonify({
        "status": "success",
        "detected_per_band": detected_payloads,
        "original_audio_file_id": original_audio_file_id,
        "current_audio_file_id": wm_file_db_entry.id,
        "current_audio_file_path": wm_file_db_entry.filepath,
        "next_wm_idx": next_wm_idx
    })

# Flask API routes
@app.route('/process_audio', methods=['POST'])
def process_audio():
    """
    Process uploaded audio file based on action (embed/detect) and method
    """
    try:
        print("\n[DEBUG] === Starting new audio processing request ===")
        
        if 'audio_file' not in request.files or request.files['audio_file'].filename == '':
            return jsonify({"status": "error", "message": "No audio file provided or selected"}), 400
        
        file = request.files['audio_file']
        
        action = request.form.get('action', '')
        method = request.form.get('method', '')
        message = request.form.get('message', '') # Message for embedding or detection check
        purpose = request.form.get('purpose', 'general')
        user_id = request.form.get('user_id')
        
        # New parameters for incremental embedding
        original_audio_file_id = request.form.get('original_audio_file_id') # ID of the initial clean audio
        current_wm_idx = int(request.form.get('current_wm_idx', 0)) # Which watermark (0-3) to embed next
        
        # Message length validation for the single 8-bit message per step
        if not message or not all(bit in '01' for bit in message) or len(message) != N_BITS:
            return jsonify({
                "status": "error",
                "message": f"Invalid message. Must be a binary string of {N_BITS} bits for this step (e.g., '10101010')."
            }), 400
            
        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        file_hash = generate_file_hash(file_path)
        print(f"Saved original file: {file_path}")
        
        current_audio_tensor, sr = preprocess_audio_for_flask(file_path) # This is the audio from *previous* step (or original)
        if current_audio_tensor is None:
            print("[ERROR] Audio preprocessing failed")
            return jsonify({"status": "error", "message": "Failed to process audio file"}), 500
        
        # Store uploaded file info in DB (this could be original or a previously watermarked file)
        uploaded_file_db = AudioFile(
            filename=filename, filepath=file_path, filehash=file_hash,
            file_size=os.path.getsize(file_path), duration=current_audio_tensor.size(-1) / sr,
            sample_rate=sr, user_id=user_id
        )
        db.session.add(uploaded_file_db)
        db.session.flush()
        
        # --- Retrieve Original Audio for Detector & Metrics (Crucial for Incremental) ---
        original_clean_audio_tensor = None
        if original_audio_file_id: # If this is not the very first embed step
            original_file_db_entry = AudioFile.query.get(original_audio_file_id)
            if original_file_db_entry and os.path.exists(original_file_db_entry.filepath):
                original_clean_audio_tensor, _ = preprocess_audio_for_flask(original_file_db_entry.filepath)
                print(f"[DEBUG] Retrieved original clean audio (ID: {original_audio_file_id}) from DB.")
            else:
                print(f"[WARNING] Original clean audio (ID: {original_audio_file_id}) not found for incremental step. BER/SNR may be inaccurate.")
                # If original clean audio is not found, use current_audio_tensor as reference for metrics,
                # which would skew SNR/MSE but allow processing.
                original_clean_audio_tensor = current_audio_tensor.clone() 
        else: # This is the very first embed step (current_wm_idx == 0)
            original_clean_audio_tensor = current_audio_tensor.clone() # The current upload IS the original for this chain
            original_audio_file_id = uploaded_file_db.id # Record its ID as the official original
            print(f"[DEBUG] First embed step. Set original_audio_file_id to {original_audio_file_id}")


        if action == 'embed':
            print(f"[DEBUG] ONE-SHOT EMBED: Preparing to embed watermark at index {current_wm_idx}")
            
            # The frontend sends the full 32-bit string of all desired watermarks
            full_cumulative_message = request.form.get('full_cumulative_message', '0'*(N_BITS*NUM_FREQ_BANDS))

            print(f"[DEBUG] Calling one-shot embedder with full message: {full_cumulative_message}")

            # Call the CORRECT, EXISTING function from your file
            watermarked_audio_tensor, embed_results = embed_full_multi_pca_prime_watermark(
                original_clean_audio_tensor,
                full_cumulative_message
            )

            # The rest of the logic saves the file and returns the response.
            # This part should be mostly the same.
            output_filename_base = secure_filename(file.filename).rsplit('.', 1)[0]
            output_filename = f"{output_filename_base}_wm_{current_wm_idx + 1}.wav"
            output_path = save_audio(watermarked_audio_tensor, output_filename)
            
            wm_file_hash = generate_file_hash(output_path)
            wm_file_size = os.path.getsize(output_path)
            
            wm_file_db = AudioFile(
                filename=output_filename, filepath=output_path, filehash=wm_file_hash,
                file_size=wm_file_size, duration=watermarked_audio_tensor.size(-1) / sr,
                sample_rate=sr, user_id=user_id
            )
            db.session.add(wm_file_db)
            db.session.flush()

            # Run a quick detection on the newly created file for metrics
            detect_results = detect_pca_prime_watermark(original_clean_audio_tensor, watermarked_audio_tensor, full_cumulative_message)
            
            entry = WatermarkEntry(
                action=action, method=method, message=message, 
                purpose=purpose, watermark_count=current_wm_idx + 1, user_id=user_id,
                snr_db=embed_results.get("snr_db"), mse=embed_results.get("mse"),
                detection_probability=detect_results.get("detection_probability"),
                ber=detect_results.get("ber"), is_detected=detect_results.get("is_detected"),
                audio_file_id=original_audio_file_id, 
                watermarked_file_id=wm_file_db.id,
                meta_data=json.dumps({ "full_cumulative_message": full_cumulative_message })
            )
            db.session.add(entry)
            db.session.commit()
            
            response = {
                "status": "success", "action": "embed", "method": method,
                "message_embedded_this_step": message,
                "original_audio_file_id": original_audio_file_id,
                "current_wm_idx": current_wm_idx + 1,
                "processed_audio_url": f"http://localhost:5000/uploads/{output_filename}",
                "results": {
                    "snr_db": embed_results["snr_db"], "mse": embed_results["mse"],
                    "detection_probability": detect_results["detection_probability"],
                    "ber": detect_results["ber"], "is_detected": detect_results["is_detected"],
                    "ber_per_band": detect_results.get("ber_per_band", [])
                }
            }
            return jsonify(response), 200        
        # --- DETECTION LOGIC (Application mode - payload extraction only) ---
        elif action == 'detect':
            
            print(f"[DEBUG] Starting CORRECTED detection process")

            wm_file_db_entry = AudioFile.query.filter_by(filehash=file_hash).first()
            
            if not wm_file_db_entry:
                return jsonify({"status": "error", "message": "This audio file is not recognized or was not processed by this system. Cannot trace its origin."}), 404
            
            wm_entry = WatermarkEntry.query.filter_by(watermarked_file_id=wm_file_db_entry.id).order_by(WatermarkEntry.watermark_count.desc()).first()

            if not wm_entry:
                # Fallback: if no watermark entry, maybe it's the original file itself?
                # In this case, we compare it against itself, expecting no payload.
                original_file_db_entry = wm_file_db_entry
                print("[WARNING] No watermark entry found. Assuming this is an original file and comparing against itself.")
            else:
                # 3. From that entry, get the ID of the original clean audio file.
                original_file_id = wm_entry.audio_file_id
                original_file_db_entry = AudioFile.query.get(original_file_id)

            if not original_file_db_entry or not os.path.exists(original_file_db_entry.filepath):
                return jsonify({"status": "error", "message": f"The original source audio (ID: {original_file_id}) is missing from the server."}), 500

            # 4. Load the TENSOR of the original clean audio.
            print(f"[DEBUG] Found and loading original clean audio: {original_file_db_entry.filepath}")
            original_clean_audio_tensor, _ = preprocess_audio_for_flask(original_file_db_entry.filepath)

            # 5. Call the NEW, CORRECT detection function with BOTH tensors.
            #    'current_audio_tensor' is the watermarked one the user uploaded.
            detect_results = extract_payload_with_original(original_clean_audio_tensor, current_audio_tensor)
            
            
            # Debug prints to see what was detected
            print(f"[DEBUG] Detection completed!")
            print(f"[DEBUG] Detected payload: {detect_results.get('payload', '')}")
            print(f"[DEBUG] Detected per band: {detect_results.get('per_band', [])}")
            print(f"[DEBUG] Full detection results: {detect_results}")
            
            entry = WatermarkEntry(
                action=action, method=method, message="payload_extraction", purpose="detection",
                is_detected=bool(detect_results.get("payload", "").strip('0')), # Detected if payload is not all zeros
                watermark_count=0, user_id=user_id,
                audio_file_id=original_file_db_entry.id, # Link to the original
                watermarked_file_id=wm_file_db_entry.id, # Link to the file that was checked
                meta_data=json.dumps({
                    "filename": filename, "file_hash": file_hash,
                    "detected_payload": detect_results.get("payload", ""),
                    "detected_per_band": detect_results.get("per_band", []),
                    "mode": "application_comparison"
                })
            )
            db.session.add(entry)
            db.session.commit()
            
            response_data = {
                "status": "success",
                "action": "detect",
                "method": method,
                "detected_payload": detect_results.get("payload", ""),
                "detected_per_band": detect_results.get("per_band", []),
                "info": "PCA Prime payload detection completed (reference-based)."
            }
            
            print(f"[DEBUG] Sending response to frontend: {response_data}")
            return jsonify(response_data), 200
            
    except Exception as e:
        import traceback
        print(f"[ERROR] Error processing audio in /process_audio: {e}")
        print(traceback.format_exc())
        if 'db' in locals():
            db.session.rollback()
        return jsonify({
            "status": "error", "message": f"Internal server error: {str(e)}"
        }), 500


def generate_file_hash(file_path):
    """Generate a hash for a file to use as an identifier"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# Serve processed audio files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and save to database"""
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({'status': 'error', 'message': 'No file provided or selected'}), 400
        
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'message': f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        file_hash = generate_file_hash(filepath)
        file_size = os.path.getsize(filepath)
        
        try:
            waveform, sample_rate = torchaudio.load(filepath)
            duration = waveform.size(-1) / sample_rate if waveform.size(-1) > 0 else 0
        except Exception as e:
            print(f"Warning: Could not get audio info for {filepath}: {e}")
            duration = None
            sample_rate = None
        
        audio_file = AudioFile(
            filename=filename, filepath=filepath, filehash=file_hash,
            file_size=file_size, duration=duration, sample_rate=sample_rate, user_id=None
        )
        db.session.add(audio_file)
        db.session.commit()
        
        return jsonify({
            'status': 'success', 'file_id': audio_file.id, 'filename': filename,
            'filepath': filepath, 'file_size': file_size, 'duration': duration,
            'sample_rate': sample_rate
        }), 200
        
    except Exception as e:
        print(f"Error uploading file: {e}")
        return jsonify({
            'status': 'error', 'message': f'Upload failed: {str(e)}'
        }), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    response = jsonify({
        'status': 'healthy',
        'pca_prime_available': PCA_PRIME_AVAILABLE, # Only check PCA_PRIME for simplicity
        'timestamp': datetime.utcnow().isoformat()
    })
    return add_cors_headers(response)


@app.route('/health', methods=['OPTIONS'])
def health_options():
    """Handle OPTIONS request for health endpoint"""
    response = jsonify({})
    return add_cors_headers(response)


@app.route('/init-db')
def init_db_route(): # Renamed to avoid conflict with init_app's db.init_app
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
    """Get available watermarking methods and their availability status"""
    methods = {
        'pca_prime': {
            'name': 'PCA Prime (Multi-Band NN)',
            'description': 'Refined Parallel Frequency Band watermarking with learned distinctiveness (Experiment Y-Prime)',
            'available': PCA_PRIME_AVAILABLE
        },
        'sfa': {
            'name': 'Sequential Fixed Alpha (SFA)',
            'description': 'Basic sequential watermarking with fixed modulation strength',
            'available': True
        },
        'sda': {
            'name': 'Sequential Decaying Alpha (SDA)',
            'description': 'Sequential watermarking with decaying modulation strength',
            'available': True
        },
        'pfb': {
            'name': 'Parallel Frequency Bands (PFB)',
            'description': 'Basic parallel frequency band watermarking',
            'available': True
        }
    }
    
    response = jsonify({
        'status': 'success',
        'methods': methods,
        'pca_prime_available': PCA_PRIME_AVAILABLE
    })
    return add_cors_headers(response)


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
            "init_db": "/init-db",
            "uploads": "/uploads/<filename>"
        }
    })

@app.route('/run_research_experiment', methods=['POST'])
def run_research_experiment():
    print("[DEBUG] === Starting Dedicated Research Experiment ===")
    try:
        if 'audio_file' not in request.files:
            return jsonify({"status": "error", "message": "No audio file provided"}), 400

        file = request.files['audio_file']
        method = request.form.get('method', 'pca_prime')
        watermark_count = int(request.form.get('watermarkCount', 1))

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        original_audio, sr = preprocess_audio_for_flask(file_path)

        if original_audio is None:
            return jsonify({"status": "error", "message": "Failed to process audio"}), 500

        # --- This logic now correctly mirrors your Colab notebook ---
        if method == 'pca_prime':
            print(f"[DEBUG] Running PCA_PRIME (one-shot) experiment")
            message_to_embed = ''.join([str(random.randint(0, 1)) for _ in range(32)])
            watermarked_audio, embed_results = research_methods.embed_full_multi_pca_prime_watermark(original_audio, message_to_embed)
            detect_results = research_methods.detect_pca_prime_watermark(original_audio, watermarked_audio, message_to_embed)
            
            # Combine embed and detect results for consistent structure
            results = {
                "snr_db": embed_results.get("snr_db", 0),
                "ber": detect_results.get("ber", 1.0),
                "detection_probability": detect_results.get("detection_probability", 0),
                "false_positive_rate": 0.0  # PCA Prime doesn't calculate this
            }
        
        else:
            print(f"[DEBUG] Running incremental experiment for {method.upper()}")
            watermarked_audio = original_audio.clone()
            embed_func = getattr(research_methods, f"embed_{method}_step")
            
            for step in range(watermark_count):
                watermarked_audio = embed_func(watermarked_audio, step)
            
            results = research_methods.calculate_generic_metrics(original_audio, watermarked_audio)
        
        # --- Save the watermarked audio and create the response ---
        output_filename = f"research_{method}_{uuid.uuid4().hex}.wav"
        save_audio(watermarked_audio, output_filename)

        # The frontend expects a 'results' object inside the main response
        final_response = {
            "status": "success",
            "processed_audio_url": f"http://localhost:5000/uploads/{output_filename}",
            "results": results
        }
        return add_cors_headers(jsonify(final_response)), 200

    except Exception as e:
        import traceback
        print(f"[ERROR] Research experiment failed: {e}")
        print(traceback.format_exc())
        return jsonify({"status": "error", "message": f"Internal server error during research: {e}"}), 500

@app.route('/research_process', methods=['POST'])
def research_process_audio():
    """
    Research mode endpoint for processing audio with multiple watermarking methods
    """
    try:
        print("\n[DEBUG] === Starting Research Mode Processing ===")
        
        if 'audio_file' not in request.files or request.files['audio_file'].filename == '':
            return jsonify({"status": "error", "message": "No audio file provided"}), 400
        
        file = request.files['audio_file']
        method = request.form.get('method', '')
        watermark_count = int(request.form.get('watermark_count', 1))
        
        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process audio
        audio_tensor, sr = preprocess_audio_for_flask(file_path)
        if audio_tensor is None:
            return jsonify({"status": "error", "message": "Failed to process audio file"}), 500
        
        # Generate random message
        message_length = 32
        message = ''.join([str(np.random.randint(0, 2)) for _ in range(message_length)])
        
        # Apply watermarking based on method
        watermarked_tensor = None
        embed_info = {}
        
        if method == 'pca_prime' and PCA_PRIME_AVAILABLE:
            # Use PCA Prime method
            watermarked_tensor, output_filename, embed_results = embed_single_incremental_pca_prime_watermark_step(
                audio_tensor, audio_tensor, message[:8], 0
            )
            if embed_results.get("status") == "error":
                return jsonify(embed_results), 500
            embed_info = embed_results
        elif method == 'sfa':
            watermarked_tensor, embed_info = embed_sfa_watermark(audio_tensor, message)
        elif method == 'sda':
            watermarked_tensor, embed_info = embed_sda_watermark(audio_tensor, message)
        elif method == 'pfb':
            watermarked_tensor, embed_info = embed_pfb_watermark(audio_tensor, message)
        else:
            return jsonify({"status": "error", "message": f"Method '{method}' not supported"}), 400
        
        # Save watermarked audio
        output_filename = f"research_{method}_{uuid.uuid4()}.wav"
        output_path = save_audio(watermarked_tensor, output_filename, sr)
        
        if not output_path:
            return jsonify({"status": "error", "message": "Failed to save watermarked audio"}), 500
        
        # Calculate metrics
        snr_db = calculate_snr(audio_tensor, watermarked_tensor)
        mse = torch.mean((audio_tensor - watermarked_tensor) ** 2).item()
        
        # Detect watermark
        detect_results = detect_watermark(watermarked_tensor, method, audio_tensor)
        
        response = {
            "status": "success",
            "action": "embed",
            "method": method,
            "message_embedded_this_step": message,
            "processed_audio_url": f"http://localhost:5000/uploads/{output_filename}",
            "results": {
                "snr_db": snr_db,
                "mse": mse,
                "detection_probability": detect_results.get("detection_probability", 0.0),
                "ber": detect_results.get("ber", 0.0),
                "is_detected": detect_results.get("is_detected", False),
                "ber_per_band": [detect_results.get("ber", 0.0)]  # Single value for basic methods
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Error in research processing: {e}")
        print(traceback.format_exc())
        return jsonify({
            "status": "error", "message": f"Research processing failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)