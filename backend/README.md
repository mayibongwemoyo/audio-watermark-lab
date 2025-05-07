
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
  - `method`: Watermarking method (`placeholder`, `sfa`, `sda`, `pfb`)
  - `message`: Binary message string (e.g., "1010101010101010")

**Response:**
- For `embed` action:
  ```json
  {
    "status": "success",
    "action": "embed",
    "method": "sfa",
    "message_embedded": "1010101010101010",
    "snr_db": 65.78,
    "detection_probability": 0.85,
    "info": "Watermark embedded using Sequential Fixed Alpha method",
    "processed_audio_url": "/uploads/watermarked_filename.wav"
  }
  ```

- For `detect` action:
  ```json
  {
    "status": "success",
    "action": "detect",
    "method": "sfa",
    "message_checked": "1010101010101010",
    "detection_probability": 0.85,
    "is_detected": true,
    "ber": 0.125,
    "info": "Watermark detection performed using sfa method"
  }
  ```

### Access Processed Files (`/uploads/<filename>`)

**Request:**
- Method: GET
- URL: `/uploads/watermarked_filename.wav`

**Response:**
- The requested audio file

### Available Methods (`/methods`)

**Request:**
- Method: GET
- URL: `/methods`

**Response:**
```json
{
  "status": "success",
  "audioseal_available": true,
  "methods": {
    "placeholder": {
      "name": "Placeholder",
      "description": "A simple placeholder implementation that adds subtle random noise to simulate watermarking",
      "available": true
    },
    "sfa": {
      "name": "Sequential Fixed Alpha (SFA)",
      "description": "Embeds watermark with fixed strength parameter alpha",
      "available": true
    },
    "sda": {
      "name": "Sequential Decaying Alpha (SDA)",
      "description": "Embeds watermark with decaying strength parameter alpha",
      "available": true
    },
    "pfb": {
      "name": "Parallel Frequency Bands (PFB)",
      "description": "Embeds watermark in parallel across different frequency bands",
      "available": true
    }
  }
}
```

### Health Check (`/health`)

**Request:**
- Method: GET
- URL: `/health`

**Response:**
```json
{
  "status": "healthy",
  "message": "Audio Watermark Lab API is running",
  "audioseal_available": true
}
```

## Watermarking Methods

The backend implements several watermarking algorithms:

1. **Placeholder** - A simple placeholder implementation that adds subtle random noise to simulate watermarking.

2. **Sequential Fixed Alpha (SFA)** - Embeds watermark with fixed strength parameter alpha. This method maintains consistent watermark strength through each application.

3. **Sequential Decaying Alpha (SDA)** - Embeds watermark with decaying strength parameter alpha. Each watermark application uses a progressively smaller alpha value, resulting in less audible distortion over multiple applications.

4. **Parallel Frequency Bands (PFB)** - Embeds watermark by splitting audio into frequency bands and processing each band separately. This results in higher SNR by concentrating watermark energy into specific frequency bands.

## AudioSeal Integration

The backend attempts to use the AudioSeal library for actual watermarking operations. If AudioSeal is not available, it falls back to placeholder implementations. The `/health` and `/methods` endpoints indicate whether AudioSeal is available in the current environment.

## Future Integration

This backend will be expanded to include additional watermarking algorithms:
- WavMark
- Additional AudioSeal variants
- Custom methods

Each method is selectable via the `method` parameter in the `/process_audio` endpoint.
