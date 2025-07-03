# Audio Watermark Lab

Asummary of the research project can be found at the end f this manual, and the documentation itself is located in /R214568M Dissertation Final.pdf 
A comprehensive audio watermarking application with research and application modes, featuring PCA-based watermarking algorithms.

## Features

- **Research Mode**: Experiment with different watermarking methods (SFA, SDA, PFB, PCA, PCA Prime)
- **Application Mode**: Production-ready PCA-based watermarking with authentication
- **Real-time Analysis**: Compare watermarking methods and analyze results
- **Batch Processing**: Process multiple audio files efficiently
- **File Management**: Organize and track watermarked files

## Quick Start

### Option 1: Using the Startup Script (Windows)
1. Double-click `start.bat` to automatically start both servers
2. Wait for both servers to fully load
3. Open http://localhost:8080 in your browser

### Option 2: Manual Startup

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```
The backend will be available at http://localhost:5000

#### Frontend Setup
```bash
npm install
npm run dev
```
The frontend will be available at http://localhost:8080

## Troubleshooting

### Connection Issues
If you see "Failed to fetch" or "ERR_INTERNET_DISCONNECTED" errors:

1. **Ensure both servers are running**:
   - Backend: Check http://localhost:5000/health
   - Frontend: Check http://localhost:8080

2. **Check firewall settings**: Ensure ports 5000 and 8080 are not blocked

3. **Restart servers**: Sometimes a restart resolves connection issues

4. **Clear browser cache**: Hard refresh (Ctrl+F5) or clear browser cache

### PCA Watermarking Issues
If PCA embedding fails with tensor size errors:
- The system automatically handles message bit length (pads/truncates to 16 bits)
- Ensure AudioSeal is properly installed in the backend

### Demo Mode
When the backend is not available, the application automatically switches to demo mode:
- Watermarking operations are simulated
- Results show expected performance metrics
- No actual audio processing occurs

## Architecture

- **Frontend**: React + TypeScript + Vite (Port 8080)
- **Backend**: Flask + Python (Port 5000)
- **Database**: SQLite (for file tracking and results)
- **Watermarking**: AudioSeal + Custom PCA implementation

## API Endpoints

- `GET /health` - Backend health check
- `GET /methods` - Available watermarking methods
- `POST /process_audio` - Embed or detect watermarks
- `GET /uploads/<filename>` - Download processed audio files

## Development

### Backend Development
```bash
cd backend
python app.py
```

### Frontend Development
```bash
npm run dev
```

### Building for Production
```bash
npm run build
```

## Project Overview

The Audio Watermark Lab provides tools for studying and evaluating various audio watermarking techniques, including:

- Sequential Fixed Alpha (SFA)
- Sequential Decaying Alpha (SDA)
- Parallel Frequency Bands (PFB)
- Principal Component Analysis (PCA)
- Comparison with external methods (Blockchain, Mobile Cloud)

## Project Structure

The project consists of two parts:

1. Frontend: React/TypeScript application with Tailwind CSS and Shadcn UI
2. Backend: Flask API for audio processing and watermarking

## License

This project is developed under the University of Zimbabwe, Faculty of Engineering - published under the MIT license.

## Research Summary

The research proposes a novel deep learning-based audio watermarking system to address the increasing need for digital audio traceability in a world where content is constantly recycled and manipulated. The system uses a generator-detector neural network model based on a Parallel Frequency Band (PFB) paradigm, incorporating multiple distinct watermarks concurrently into the audio's frequency domain. A key innovation is the use of learned basis vectors, inspired by Principal Component Analysis (PCA), to establish optimal and separable embedding spaces for each watermark.

The dissertation outlines the methodology, including the algorithm overview for the generator and detector, training pipeline, dataset (Vox Populi), and evaluation metrics such as Bit Error Rate (BER), Signal-to-Noise Ratio (SNR), Mean Squared Error (MSE), and Perceptual Evaluation of Speech Quality (PESQ). The system aims to achieve imperceptibility, robustness, and capacity, even under adversarial distortions like Additive White Gaussian Noise and low-pass filtering.

The feasibility study section covers technical, economic, social, and operational aspects, concluding that the project is feasible due to the availability of open-source software and minimal development costs. The significance and motivation highlight the potential for enhanced copyright protection, content authentication, and contributions to audio cryptography and deep learning research.

The research also details the system analysis and design of the "Audio Watermark Lab" application, which operationalizes the developed ML models. This application features dual-mode operation for researchers and application users, handling audio file uploads, watermark embedding and detection, performance metrics display, and user account management. The system architecture is client-server, with a React frontend and a Python Flask backend.



