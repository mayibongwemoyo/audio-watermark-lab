# Remove the circular import
# from app import save_audio
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as torch_F # For padding
import random
import math
import os
import numpy as np
import uuid
from scipy import stats

# --- Constants for PCA-Prime (Aligned with Experiment Y-Prime) ---
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = N_FFT // 4
NUM_FREQ_BANDS = 4 
N_BITS = 8 
SEGMENT_DURATION_S = 1
NUM_SAMPLES_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION_S
NUM_TIME_FRAMES_SEGMENT = (NUM_SAMPLES_SEGMENT // HOP_LENGTH) + 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DETECTOR_FC_HIDDEN_SIZE = 192 * 2 
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')
GENERATOR_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "pfb_parallel_4wm_exp_y_prime_best.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Helper Functions ---
def calculate_snr(original, processed, epsilon=1e-9):
    original = original.to(DEVICE)
    processed = processed.to(DEVICE)
    noise = processed - original
    signal_power_per_item = torch.mean(original**2, dim=list(range(1, original.dim())))
    noise_power_per_item = torch.mean(noise**2, dim=list(range(1, noise.dim())))
    snr_vals_per_item = 10 * torch.log10(signal_power_per_item / (noise_power_per_item + epsilon))
    snr_vals_per_item[noise_power_per_item <= epsilon] = 100.0
    return snr_vals_per_item.mean().item()

def calculate_mse(original, processed):
    original = original.to(DEVICE)
    processed = processed.to(DEVICE)
    # FIX: Ensure both tensors have 2 dimensions (C, T) before passing to MSELoss
    # Squeeze batch dim (if it's 1) and ensure 2D.
    original_shaped = original.squeeze(0) if original.dim() == 3 else original
    processed_shaped = processed.squeeze(0) if processed.dim() == 3 else processed
    
    # Ensure they both have a channel dimension for consistency (e.g., (T,) becomes (1, T))
    if original_shaped.dim() == 1:
        original_shaped = original_shaped.unsqueeze(0)
    if processed_shaped.dim() == 1:
        processed_shaped = processed_shaped.unsqueeze(0)

    return nn.MSELoss()(original_shaped, processed_shaped).item()

stft_transform = T.Spectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH, power=None, return_complex=True, center=True).to(DEVICE)
istft_transform = T.InverseSpectrogram(n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)

# --- Model Class Definitions (Exactly from Experiment Y-Prime Colab) ---
class PFBGenerator(nn.Module):
    def __init__(self, n_fft=N_FFT, num_freq_bands=NUM_FREQ_BANDS, n_bits=N_BITS,
                 message_embedding_dim=16, initial_modulation_strength=0.3):
        super().__init__()
        self.n_fft = n_fft
        self.num_freq_bands = num_freq_bands
        self.n_bits = n_bits
        self.message_embedding_dim = message_embedding_dim
        num_freq_bins_total = self.n_fft // 2 + 1
        self.nominal_bins_per_band = num_freq_bins_total // self.num_freq_bands
        self.message_embedding = nn.Embedding(2, self.message_embedding_dim)
        initial_strength_param = math.log(initial_modulation_strength / (1 - initial_modulation_strength))
        self.learnable_modulation_strengths_raw = nn.Parameter(
            torch.full((self.num_freq_bands,), initial_strength_param, device=DEVICE)
        )
        self.modification_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.n_bits * self.message_embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.nominal_bins_per_band),
                nn.Tanh()
            ) for _ in range(self.num_freq_bands)
        ])

    def forward(self, audio_segment_batch, message_batches: list):
        if len(message_batches) != self.num_freq_bands:
            raise ValueError(f"Expected {self.num_freq_bands} message batches, got {len(message_batches)}")
        if audio_segment_batch.dim() == 2:
            audio_segment_batch = audio_segment_batch.unsqueeze(1)
        audio_segment_batch_device = audio_segment_batch.to(DEVICE)
        spectrogram_complex = stft_transform(audio_segment_batch_device.squeeze(1))
        magnitude = spectrogram_complex.abs()
        angle = spectrogram_complex.angle()
        modified_band_magnitudes = []
        num_freq_bins_total_runtime = magnitude.shape[1]
        modulation_strengths = torch.sigmoid(self.learnable_modulation_strengths_raw)

        for i in range(self.num_freq_bands):
            message_batch = message_batches[i]
            embedded_message_flat = self.message_embedding(message_batch.long()).view(message_batch.size(0), -1)
            projected_deltas_raw = self.modification_projectors[i](embedded_message_flat)
            start_bin = i * self.nominal_bins_per_band
            if i == self.num_freq_bands - 1:
                end_bin = num_freq_bins_total_runtime
            else:
                end_bin = start_bin + self.nominal_bins_per_band
            current_band_actual_size = end_bin - start_bin
            num_deltas_to_apply = min(projected_deltas_raw.shape[1], current_band_actual_size)
            full_deltas_for_band_slice = torch.zeros(message_batch.size(0), current_band_actual_size, device=DEVICE)
            full_deltas_for_band_slice[:, :num_deltas_to_apply] = projected_deltas_raw[:, :num_deltas_to_apply]

            max_clamp_val = 1.0 + modulation_strengths[i].item() # .item() converts scalar tensor to Python float
            scaling_factors = (1.0 + full_deltas_for_band_slice * modulation_strengths[i]).unsqueeze(-1)
            scaling_factors = torch.clamp(scaling_factors, min=0.01, max=max_clamp_val) 

            original_band_slice = magnitude[:, start_bin:end_bin, :]
            modified_slice = original_band_slice * scaling_factors
            modified_band_magnitudes.append(modified_slice)

        modified_magnitude = torch.cat(modified_band_magnitudes, dim=1)
        watermarked_spectrogram_complex = torch.polar(modified_magnitude, angle)
        watermarked_audio_segment = istft_transform(watermarked_spectrogram_complex, length=audio_segment_batch_device.shape[-1])
        if watermarked_audio_segment.dim() == 2:
            watermarked_audio_segment = watermarked_audio_segment.unsqueeze(1)
        return watermarked_audio_segment

class PFBDetector(nn.Module):
    def __init__(self, n_fft=N_FFT, num_freq_bands=NUM_FREQ_BANDS, n_bits=N_BITS,
                 num_time_frames=NUM_TIME_FRAMES_SEGMENT,
                 fc_hidden_size=DETECTOR_FC_HIDDEN_SIZE):
        super().__init__()
        self.n_fft = n_fft
        self.num_freq_bands = num_freq_bands
        self.n_bits = n_bits
        self.num_time_frames = num_time_frames
        self.fc_hidden_size = fc_hidden_size
        num_freq_bins_total = self.n_fft // 2 + 1
        self.freq_bins_per_band_input_shape = num_freq_bins_total // self.num_freq_bands

        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 16, (3, 5), stride=(1, 2), padding=(1, 2)),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 1, self.freq_bins_per_band_input_shape, self.num_time_frames)
            dummy_out = self.conv_net(dummy_input)
            self.conv_output_size = dummy_out.view(1, -1).size(1)

        self.fc_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.conv_output_size, self.fc_hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.fc_hidden_size, self.n_bits)
            ) for _ in range(self.num_freq_bands)
        ])

    def _process_band(self, magnitude_diff_slice, band_idx):
        current_band_freq_size_actual = magnitude_diff_slice.shape[1]
        if current_band_freq_size_actual < self.freq_bins_per_band_input_shape:
            processed_band_slice = torch_F.pad(magnitude_diff_slice, (0, 0, 0, self.freq_bins_per_band_input_shape - current_band_freq_size_actual))
        elif current_band_freq_size_actual > self.freq_bins_per_band_input_shape:
            processed_band_slice = magnitude_diff_slice[:, :self.freq_bins_per_band_input_shape, :]
        else:
            processed_band_slice = magnitude_diff_slice
        current_band_time_size_actual = processed_band_slice.shape[2]
        if current_band_time_size_actual < self.num_time_frames:
            processed_band_slice = torch_F.pad(processed_band_slice, (0, self.num_time_frames - current_band_time_size_actual, 0, 0))
        elif current_band_time_size_actual > self.num_time_frames:
            processed_band_slice = processed_band_slice[:, :, :self.num_time_frames]
        conv_input = processed_band_slice.unsqueeze(1)
        conv_output = self.conv_net(conv_input)
        conv_output_flat = conv_output.view(conv_output.size(0), -1)
        band_logits = self.fc_nets[band_idx](conv_output_flat)
        return band_logits

    def forward(self, original_audio_segment_batch, watermarked_audio_segment_batch):
        if original_audio_segment_batch.dim() == 2:
            original_audio_segment_batch = original_audio_segment_batch.unsqueeze(0)
        if watermarked_audio_segment_batch.dim() == 2:
            watermarked_audio_segment_batch = watermarked_audio_segment_batch.unsqueeze(0)

        original_audio_device = original_audio_segment_batch.to(DEVICE)
        watermarked_audio_device = watermarked_audio_segment_batch.to(DEVICE)

        spec_complex_orig = stft_transform(original_audio_device.squeeze(1))
        spec_complex_wm = stft_transform(watermarked_audio_device.squeeze(1))

        magnitude_orig = spec_complex_orig.abs()
        magnitude_wm = spec_complex_wm.abs()

        magnitude_diff = magnitude_wm - magnitude_orig

        num_freq_bins_total_runtime = magnitude_diff.shape[1]
        bins_per_band_runtime_nominal = num_freq_bins_total_runtime // self.num_freq_bands

        all_logits = []
        for i in range(self.num_freq_bands):
            start_bin = i * bins_per_band_runtime_nominal
            if i == self.num_freq_bands - 1:
                end_bin = num_freq_bins_total_runtime
            else:
                end_bin = start_bin + bins_per_band_runtime_nominal

            target_band_diff_slice = magnitude_diff[:, start_bin:end_bin, :]
            band_logits = self._process_band(target_band_diff_slice, i)
            all_logits.append(band_logits)

        return all_logits

# --- Model Loading and Core Functions ---
_generator_instance = None
_detector_instance = None

def get_models():
    """
    Loads and returns the trained PFBGenerator and PFBDetector instances.
    Ensures models are loaded only once.
    """
    global _generator_instance, _detector_instance
    if _generator_instance is None or _detector_instance is None:
        try:
            print(f"[PCA_PRIME_WATERMARKING] Loading models from {GENERATOR_CHECKPOINT_PATH}")
            current_generator = PFBGenerator().to(DEVICE)
            current_detector = PFBDetector().to(DEVICE)

            checkpoint = torch.load(GENERATOR_CHECKPOINT_PATH, map_location=DEVICE)
            current_generator.load_state_dict(checkpoint['generator_state_dict'])
            current_detector.load_state_dict(checkpoint['detector_state_dict'])

            current_generator.eval() # Set to eval mode
            current_detector.eval() # Set to eval mode

            _generator_instance = current_generator
            _detector_instance = current_detector
            print("[PCA_PRIME_WATERMARKING] Models loaded successfully.")
        except Exception as e:
            print(f"[PCA_PRIME_WATERMARKING] ERROR: Failed to load trained models: {e}")
            print("[PCA_PRIME_WATERMARKING] Ensure the model path is correct and the file exists.")
            print("[PCA_PRIME_WATERMARKING] Instantiating new models (will be untrained).")
            # Fallback to untrained models if loading fails
            _generator_instance = PFBGenerator().to(DEVICE)
            _detector_instance = PFBDetector().to(DEVICE)
            _generator_instance.eval()
            _detector_instance.eval()
    return _generator_instance, _detector_instance

# --- NEW: Function to embed a *single* watermark incrementally onto current audio ---
def embed_single_incremental_pca_prime_watermark_step(
    original_full_audio_tensor: torch.Tensor, 
    current_audio_tensor_from_prev_step: torch.Tensor,   
    message_for_this_band_str: str,       
    target_band_idx: int                  
) -> tuple[torch.Tensor, str, dict]:
    generator, _ = get_models()
    generator.eval()

    if not (0 <= target_band_idx < NUM_FREQ_BANDS):
        raise ValueError(f"target_band_idx must be between 0 and {NUM_FREQ_BANDS - 1}, got {target_band_idx}")
    if len(message_for_this_band_str) != N_BITS:
        raise ValueError(f"Message for single band must be {N_BITS} bits, got {len(message_for_this_band_str)}")

    # Ensure inputs are (1, C, T)
    if original_full_audio_tensor.dim() == 2: original_full_audio_tensor = original_full_audio_tensor.unsqueeze(0)
    if current_audio_tensor_from_prev_step.dim() == 2: current_audio_tensor_from_prev_step = current_audio_tensor_from_prev_step.unsqueeze(0)

    total_samples = original_full_audio_tensor.shape[-1]
    num_chunks = math.ceil(total_samples / NUM_SAMPLES_SEGMENT)
    
    newly_watermarked_chunks_list = []

    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * NUM_SAMPLES_SEGMENT
            end_sample = min((chunk_idx + 1) * NUM_SAMPLES_SEGMENT, total_samples)
            
            original_chunk = original_full_audio_tensor[:, :, start_sample:end_sample]
            current_chunk_from_prev_step = current_audio_tensor_from_prev_step[:, :, start_sample:end_sample]
            
            # Pad chunks if necessary
            if original_chunk.shape[-1] < NUM_SAMPLES_SEGMENT:
                padding_needed = NUM_SAMPLES_SEGMENT - original_chunk.shape[-1]
                original_chunk = torch_F.pad(original_chunk, (0, padding_needed))
                current_chunk_from_prev_step = torch_F.pad(current_chunk_from_prev_step, (0, padding_needed))
            
            # Prepare messages for the generator to embed ONLY the target_band_idx.
            messages_for_generator = []
            for i in range(NUM_FREQ_BANDS):
                if i == target_band_idx:
                    messages_for_generator.append(torch.tensor([[int(b) for b in message_for_this_band_str]], dtype=torch.float32, device=DEVICE))
                else:
                    messages_for_generator.append(torch.zeros(1, N_BITS, dtype=torch.float32, device=DEVICE))
            
            watermarked_only_this_band_from_original = generator(original_chunk, messages_for_generator)
            
            delta_introduced = watermarked_only_this_band_from_original - original_chunk
            
            new_watermarked_chunk = current_chunk_from_prev_step + delta_introduced
            
            newly_watermarked_chunks_list.append(new_watermarked_chunk.squeeze(0))

    newly_watermarked_full_audio_padded = torch.cat(newly_watermarked_chunks_list, dim=-1)
    
    # Trim padding and ensure valid range for output
    newly_watermarked_full_audio_trimmed = newly_watermarked_full_audio_padded[:, :total_samples]
    
    # --- CRITICAL FIXES FOR AUDIO OUTPUT ---
    # 1. Clamp values to ensure they are strictly within [-1, 1]
    newly_watermarked_full_audio_trimmed = torch.clamp(newly_watermarked_full_audio_trimmed, min=-1.0, max=1.0)
    
    # 2. Convert to float32 explicitly, if it's not already, as this is standard for audio samples.
    newly_watermarked_full_audio_trimmed = newly_watermarked_full_audio_trimmed.to(torch.float32)

    # 3. Ensure the tensor has exactly 2 dimensions (channels, samples) for torchaudio.save.
    #    It should be (1, T) after original_audio_tensor.unsqueeze(0) and .squeeze(0) in loop.
    if newly_watermarked_full_audio_trimmed.dim() == 3:
        newly_watermarked_full_audio_trimmed = newly_watermarked_full_audio_trimmed.squeeze(0)
    elif newly_watermarked_full_audio_trimmed.dim() == 1:
        newly_watermarked_full_audio_trimmed = newly_watermarked_full_audio_trimmed.unsqueeze(0)
    # This ensures it's (1, T) for mono audio saving.
    # --- END CRITICAL FIXES ---

    # Check for NaNs/Infs (debugging aid)
    if torch.isnan(newly_watermarked_full_audio_trimmed).any() or torch.isinf(newly_watermarked_full_audio_trimmed).any():
        print("[PCA_PRIME_WATERMARKING] WARNING: NaN or Inf found in newly watermarked audio tensor after incremental embed!")
        newly_watermarked_full_audio_trimmed = torch.nan_to_num(newly_watermarked_full_audio_trimmed, nan=0.0, posinf=1.0, neginf=-1.0)
        # Re-clamp after nan_to_num as it might put values slightly outside [-1,1]
        newly_watermarked_full_audio_trimmed = torch.clamp(newly_watermarked_full_audio_trimmed, min=-1.0, max=1.0)

    print("Saving audio: shape", newly_watermarked_full_audio_trimmed.shape, "dtype", newly_watermarked_full_audio_trimmed.shape)
    print("Sample rate:", SAMPLE_RATE)
    print("Min/max:", newly_watermarked_full_audio_trimmed.min().item(), newly_watermarked_full_audio_trimmed.max().item())
    print("Num samples:", newly_watermarked_full_audio_trimmed.shape[-1])

    output_filename = f"watermarked_incremental_{uuid.uuid4().hex}.wav"
    # Replace save_audio with local torchaudio.save to avoid circular import
    output_path = os.path.join(os.path.dirname(__file__), 'uploads', output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Ensure tensor is in the format (C, T) for torchaudio.save
    audio_tensor_to_save = newly_watermarked_full_audio_trimmed.detach().cpu()
    if audio_tensor_to_save.dim() == 3: # (Batch, Channel, Time)
        audio_tensor_to_save = audio_tensor_to_save.squeeze(0) # -> (Channel, Time)
    
    torchaudio.save(output_path, audio_tensor_to_save, SAMPLE_RATE)
    print("Saved watermarked audio to:", output_path)
    print("File size after save:", os.path.getsize(output_path))

    original_full_audio_trimmed_for_metrics = original_full_audio_tensor[:, :total_samples]
    snr_db = calculate_snr(original_full_audio_trimmed_for_metrics, newly_watermarked_full_audio_trimmed)
    mse = calculate_mse(original_full_audio_trimmed_for_metrics, newly_watermarked_full_audio_trimmed)
    
    return newly_watermarked_full_audio_trimmed, output_filename, {
        "method": "PCA_PRIME_INCREMENTAL",
        "message_embedded_this_step": message_for_this_band_str,
        "band_idx": target_band_idx,
        "snr_db": round(snr_db, 3),
        "mse": round(mse, 6),
        "info": f"PCA Prime incremental watermark for band {target_band_idx} applied."
    }

# --- Full Multi-Watermark Embed (Renamed from embed_pca_prime_watermark) ---
# This function is for embedding all 4 watermarks at once (used by research script)
# It is NOT used by the app's 'embed' action anymore.
def embed_full_multi_pca_prime_watermark(original_full_audio_tensor: torch.Tensor, message_bits_str: str) -> tuple[torch.Tensor, dict]:
    generator, _ = get_models()
    generator.eval()

    if original_full_audio_tensor.dim() == 2: original_full_audio_tensor = original_full_audio_tensor.unsqueeze(0)

    if len(message_bits_str) != N_BITS * NUM_FREQ_BANDS:
        raise ValueError(f"Message length mismatch for full embed. Expected {N_BITS * NUM_FREQ_BANDS} bits, got {len(message_bits_str)}.")
    
    messages_for_all_bands_tensor = []
    for i in range(NUM_FREQ_BANDS):
        band_message_str = message_bits_str[i * N_BITS : (i + 1) * N_BITS]
        messages_for_all_bands_tensor.append(torch.tensor([[int(b) for b in band_message_str]], dtype=torch.float32, device=DEVICE))

    total_samples = original_full_audio_tensor.shape[-1]
    num_chunks = math.ceil(total_samples / NUM_SAMPLES_SEGMENT)
    
    watermarked_chunks_list = []
    original_chunks_padded_list = []
    
    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * NUM_SAMPLES_SEGMENT
            end_sample = min((chunk_idx + 1) * NUM_SAMPLES_SEGMENT, total_samples)
            original_chunk = original_full_audio_tensor[:, :, start_sample:end_sample]
            if original_chunk.shape[-1] < NUM_SAMPLES_SEGMENT:
                padding_needed = NUM_SAMPLES_SEGMENT - original_chunk.shape[-1]
                original_chunk = torch_F.pad(original_chunk, (0, padding_needed))
            
            watermarked_chunk = generator(original_chunk, messages_for_all_bands_tensor)
            
            watermarked_chunks_list.append(watermarked_chunk.squeeze(0))
            original_chunks_padded_list.append(original_chunk.squeeze(0))

    watermarked_full_audio_padded = torch.cat(watermarked_chunks_list, dim=-1)
    original_full_audio_padded = torch.cat(original_chunks_padded_list, dim=-1)

    watermarked_full_audio_trimmed = watermarked_full_audio_padded[:, :total_samples]
    original_full_audio_trimmed = original_full_audio_padded[:, :total_samples]

    watermarked_full_audio_trimmed = torch.clamp(watermarked_full_audio_trimmed, min=-1.0, max=1.0).float()
    
    if torch.isnan(watermarked_full_audio_trimmed).any() or torch.isinf(watermarked_full_audio_trimmed).any():
        print("[PCA_PRIME_WATERMARKING] WARNING: NaN or Inf found in full watermarked audio tensor!")
        watermarked_full_audio_trimmed = torch.nan_to_num(watermarked_full_audio_trimmed, nan=0.0, posinf=1.0, neginf=-1.0)

    snr_db = calculate_snr(original_full_audio_trimmed, watermarked_full_audio_trimmed)
    mse = calculate_mse(original_full_audio_trimmed, watermarked_full_audio_trimmed)
    
    return watermarked_full_audio_trimmed, {
        "method": "PCA_PRIME_FULL",
        "message_embedded": message_bits_str,
        "snr_db": round(snr_db, 3),
        "mse": round(mse, 6),
        "info": "PCA Prime full multi-watermark applied."
    }


def detect_pca_prime_watermark(original_audio_tensor: torch.Tensor | None, watermarked_audio_tensor: torch.Tensor, message_bits_to_check_str: str) -> dict: # Updated type hint
    """
    Detects watermarks using the trained PFBDetector and calculates full metrics.
    original_audio_tensor can be None if the original is not available for BER calculation.
    """
    _, detector = get_models()
    detector.eval()

    if watermarked_audio_tensor.dim() == 2:
        watermarked_audio_tensor = watermarked_audio_tensor.unsqueeze(0)

    original_audio_tensor_processed = original_audio_tensor
    if original_audio_tensor_processed is None:
        original_audio_tensor_processed = torch.zeros_like(watermarked_audio_tensor, device=DEVICE)
        print("[PCA_PRIME_WATERMARKING] Warning: Original audio not provided for detection. BER will be 0.5 (random).")
    elif original_audio_tensor_processed.dim() == 2:
        original_audio_tensor_processed = original_audio_tensor_processed.unsqueeze(0)


    if len(message_bits_to_check_str) != N_BITS * NUM_FREQ_BANDS:
        print(f"[PCA_PRIME_WATERMARKING] Warning: Message_bits_to_check length ({len(message_bits_to_check_str)}) does not match expected {N_BITS * NUM_FREQ_BANDS} bits for {NUM_FREQ_BANDS} bands of {N_BITS} bits each. Adjusting.")
        message_bits_to_check_str = message_bits_to_check_str.ljust(N_BITS * NUM_FREQ_BANDS, '0')[:N_BITS * NUM_FREQ_BANDS]

    expected_messages_per_band = []
    for i in range(NUM_FREQ_BANDS):
        band_message_str = message_bits_to_check_str[i * N_BITS : (i + 1) * N_BITS]
        expected_messages_per_band.append(torch.tensor([[int(b) for b in band_message_str]], dtype=torch.float32, device=DEVICE))

    total_samples = watermarked_audio_tensor.shape[-1]
    num_chunks = math.ceil(total_samples / NUM_SAMPLES_SEGMENT)

    total_ber_per_band_raw = [0] * NUM_FREQ_BANDS
    total_bits_processed_overall = 0
    
    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * NUM_SAMPLES_SEGMENT
            end_sample = min((chunk_idx + 1) * NUM_SAMPLES_SEGMENT, total_samples)
            
            original_chunk = original_audio_tensor_processed[:, :, start_sample:end_sample]
            watermarked_chunk = watermarked_audio_tensor[:, :, start_sample:end_sample]

            if original_chunk.shape[-1] < NUM_SAMPLES_SEGMENT:
                padding_needed = NUM_SAMPLES_SEGMENT - original_chunk.shape[-1]
                original_chunk = torch_F.pad(original_chunk, (0, padding_needed))
                watermarked_chunk = torch_F.pad(watermarked_chunk, (0, padding_needed))
            
            logits_list = detector(original_chunk, watermarked_chunk)

            for i in range(NUM_FREQ_BANDS):
                preds = (torch.sigmoid(logits_list[i]) > 0.5).int()
                ber_item = (expected_messages_per_band[i].int() != preds).float().sum().item()
                total_ber_per_band_raw[i] += ber_item
            total_bits_processed_overall += N_BITS * NUM_FREQ_BANDS 

    avg_ber_per_band = [b_raw / (num_chunks * N_BITS) for b_raw in total_ber_per_band_raw]
    avg_ber_overall = sum(avg_ber_per_band) / NUM_FREQ_BANDS

    is_detected = bool(avg_ber_overall < 0.1)

    return {
        "method": "PCA_PRIME",
        "message_checked": message_bits_to_check_str,
        "detection_probability": float(1.0 - avg_ber_overall),
        "is_detected": is_detected,
        "ber": round(avg_ber_overall, 4),
        "ber_per_band": [round(b, 4) for b in avg_ber_per_band],
        "info": "PCA Prime watermark detection performed."
    }


def extract_payload_with_original(original_audio_tensor: torch.Tensor, watermarked_audio_tensor: torch.Tensor) -> dict:
    """
    Application-mode payload extraction that correctly uses the original audio for comparison,
    as required by the PFBDetector model's design.
    """
    print("[DEBUG] Starting payload extraction via comparison...")
    _, detector = get_models()
    detector.eval()

    if original_audio_tensor.dim() == 2:
        original_audio_tensor = original_audio_tensor.unsqueeze(0)
    if watermarked_audio_tensor.dim() == 2:
        watermarked_audio_tensor = watermarked_audio_tensor.unsqueeze(0)

    # Ensure tensors have the same length for chunking
    min_len = min(original_audio_tensor.shape[-1], watermarked_audio_tensor.shape[-1])
    original_audio_tensor = original_audio_tensor[..., :min_len]
    watermarked_audio_tensor = watermarked_audio_tensor[..., :min_len]
    
    total_samples = min_len
    num_chunks = math.ceil(total_samples / NUM_SAMPLES_SEGMENT)
    print(f"[DEBUG] Total samples: {total_samples}, Num chunks: {num_chunks}")

    # Use a simple majority vote for the final payload across all chunks
    predicted_bits_per_band_chunks = [[] for _ in range(NUM_FREQ_BANDS)]

    with torch.no_grad():
        for chunk_idx in range(num_chunks):
            start_sample = chunk_idx * NUM_SAMPLES_SEGMENT
            end_sample = min((chunk_idx + 1) * NUM_SAMPLES_SEGMENT, total_samples)
            
            original_chunk = original_audio_tensor[:, :, start_sample:end_sample]
            watermarked_chunk = watermarked_audio_tensor[:, :, start_sample:end_sample]

            # Pad the last chunk if it's smaller than the required segment size
            if watermarked_chunk.shape[-1] < NUM_SAMPLES_SEGMENT:
                padding_needed = NUM_SAMPLES_SEGMENT - watermarked_chunk.shape[-1]
                original_chunk = torch_F.pad(original_chunk, (0, padding_needed))
                watermarked_chunk = torch_F.pad(watermarked_chunk, (0, padding_needed))

            # Call the detector with the CORRECT inputs
            logits_list = detector(original_chunk, watermarked_chunk)
            
            for i in range(NUM_FREQ_BANDS):
                preds = (torch.sigmoid(logits_list[i]) > 0.5).int().cpu().numpy().tolist()
                predicted_bits_per_band_chunks[i].append(preds[0])

    # Perform majority voting for each bit across all chunks
    final_bits_per_band = []
    for band_chunks in predicted_bits_per_band_chunks:
        if not band_chunks:
            final_bits_per_band.append([0] * N_BITS) # Default to zeros if no chunks
            continue
        
        bits_over_chunks = np.array(band_chunks).T
        majority_voted_bits = stats.mode(bits_over_chunks, axis=1, keepdims=False)[0].tolist()
        final_bits_per_band.append(majority_voted_bits)

    print(f"[DEBUG] Final bits per band after majority vote: {final_bits_per_band}")
    
    payload = ''.join(str(bit) for band in final_bits_per_band for bit in band)
    print(f"[DEBUG] Final payload: {payload}")
    
    return {
        "payload": payload,
        "per_band": final_bits_per_band
    }

def train_pca_prime_model():
    print("Training function not implemented in module. This module expects pre-trained models.")
    pass