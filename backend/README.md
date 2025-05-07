
# Audio Watermark Lab - Backend API

This is the backend API for the Audio Watermark Lab application, providing endpoints for audio watermarking operations.

## Setup and Installation

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

Start the Flask server:
```bash
python app.py
```

The API will be available at http://localhost:5000

## API Endpoints

### Process Audio (`/process_audio`)

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameters:
  - `audio_file`: Audio file (WAV, MP3, FLAC, OGG)
  - `action`: Operation to perform (`embed` or `detect`)
  - `method`: Watermarking method (`placeholder` only for now)
  - `message`: Binary message string (e.g., "1010101010101010")

**Response:**
- For `embed` action:
  ```json
  {
    "status": "success",
    "action": "embed",
    "message_embedded": "1010101010101010",
    "snr_db": 65.78,
    "info": "Watermark embedded (placeholder implementation)",
    "processed_audio_url": "/uploads/watermarked_filename.wav"
  }
  ```

- For `detect` action:
  ```json
  {
    "status": "success",
    "action": "detect",
    "message_checked": "1010101010101010",
    "detection_probability": 0.85,
    "is_detected": true,
    "ber": 0.125,
    "info": "Watermark detection performed (placeholder implementation)"
  }
  ```

### Access Processed Files (`/uploads/<filename>`)

**Request:**
- Method: GET
- URL: `/uploads/watermarked_filename.wav`

**Response:**
- The requested audio file

### Health Check (`/health`)

**Request:**
- Method: GET
- URL: `/health`

**Response:**
```json
{
  "status": "healthy",
  "message": "Audio Watermark Lab API is running"
}
```

## Future Integration

This backend implementation currently uses placeholder functions for watermarking operations. In the future, these placeholders will be replaced with actual watermarking algorithms:
- PFB Phase
- SDA
- AudioSeal Base
- WavMark
- And others

Each method will be selectable via the `method` parameter in the `/process_audio` endpoint.
