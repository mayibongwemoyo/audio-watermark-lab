the
# Audio Watermark Lab

A comprehensive application for audio watermarking research, featuring multiple watermarking techniques and comparative analysis.

## Project Overview

The Audio Watermark Lab provides tools for studying and evaluating various audio watermarking techniques, including:

- Sequential Fixed Alpha (SFA)
- Sequential Decaying Alpha (SDA)
- Parallel Frequency Bands (PFB)
- Principal Component Analysis (PCA)
- Comparison with external methods (Blockchain, Mobile Cloud)

## Features

- Upload and record audio files
- Apply multiple watermarking methods
- Configure watermark parameters (bits, components, etc.)
- View performance metrics (SNR, BER, Detection Probability)
- Compare methods with interactive visualizations
- Analyze robustness against common attacks

## Project Structure

The project consists of two parts:

1. Frontend: React/TypeScript application with Tailwind CSS and Shadcn UI
2. Backend: Flask API for audio processing and watermarking

## Setup and Running

### Frontend

1. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

2. Run the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

The frontend will be available at http://localhost:5173

### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Start the Flask server:
   ```bash
   python app.py
   ```

The backend API will be available at http://localhost:5000

## Development

The frontend and backend can be developed independently. The frontend will function in "demo mode" if the backend is not available.

## License

This project is developed by the University of Zimbabwe, Faculty of Engineering.

