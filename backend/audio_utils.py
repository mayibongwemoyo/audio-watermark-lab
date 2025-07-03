# File: backend/audio_utils.py
import os
import torch
import torchaudio

def save_audio(audio_tensor, filename, upload_folder, sample_rate=16000):
    """
    Saves a processed audio tensor to a file in the specified upload folder.
    """
    try:
        # Ensure the upload folder exists
        os.makedirs(upload_folder, exist_ok=True)
        
        output_path = os.path.join(upload_folder, filename)
        
        # Ensure tensor is in the format (C, T) for torchaudio.save and detached
        audio_tensor_to_save = audio_tensor.detach().cpu()
        if audio_tensor_to_save.dim() == 3:
            audio_tensor_to_save = audio_tensor_to_save.squeeze(0)
        
        torchaudio.save(output_path, audio_tensor_to_save, sample_rate)
        return output_path
    except Exception as e:
        print(f"Error saving audio: {e}")
        return None