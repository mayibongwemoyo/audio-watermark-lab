# Audio Watermark Lab

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

This project is developed by the University of Zimbabwe, Faculty of Engineering.

