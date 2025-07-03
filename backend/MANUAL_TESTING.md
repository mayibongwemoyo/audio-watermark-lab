# Manual Testing Guide for Audio Watermark Lab

## Prerequisites
- Backend server running on `http://localhost:5000`
- Frontend running on `http://localhost:3000` (or your configured port)
- AudioSeal models downloaded and available

## 1. Backend API Testing

### Health Check
```bash
curl http://localhost:5000/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "audioseal_available": true
}
```

### Available Methods
```bash
curl http://localhost:5000/methods
```
**Expected Response:**
```json
{
  "methods": ["sfa", "sda", "pfb", "pca"]
}
```

### File Upload Test
```bash
curl -X POST -F "file=@test_audio.wav" http://localhost:5000/upload
```
**Expected Response:**
```json
{
  "status": "success",
  "file_id": 1,
  "filename": "test_audio.wav"
}
```

### Watermark Embedding Test
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "file_id": 1,
    "action": "embed",
    "method": "sfa",
    "message": "10101010",
    "purpose": "test"
  }' \
  http://localhost:5000/process_audio
```

### Watermark Detection Test
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "file_id": 2,
    "action": "detect",
    "method": "sfa",
    "message": "10101010"
  }' \
  http://localhost:5000/process_audio
```

## 2. Database API Testing

### Get All Users
```bash
curl http://localhost:5000/api/users
```

### Get All Watermarks
```bash
curl http://localhost:5000/api/watermarks
```

### Get Watermarks with Filters
```bash
curl "http://localhost:5000/api/watermarks?method=sfa&action=embed"
```

### Create Watermark Entry
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "action": "embed",
    "method": "sfa",
    "message": "10101010",
    "snr_db": 65.5,
    "detection_probability": 0.85,
    "ber": 0.1,
    "is_detected": true,
    "purpose": "test",
    "watermark_count": 1,
    "meta_data": {"test": "data"}
  }' \
  http://localhost:5000/api/watermarks
```

## 3. Frontend Testing

### 1. Homepage
- Navigate to `http://localhost:3000`
- Verify all sections load properly
- Check navigation works

### 2. File Upload
- Go to Upload section
- Try uploading different audio formats (WAV, MP3, FLAC)
- Verify file validation works
- Check error messages for invalid files

### 3. Watermark Embedding
- Upload an audio file
- Select "Embed" action
- Choose different methods (SFA, SDA, PFB, PCA)
- Enter a binary message (e.g., "10101010")
- Set purpose and watermark count
- Click "Process Audio"
- Verify watermarked file is generated and downloadable

### 4. Watermark Detection
- Upload a watermarked file (or use one from embedding)
- Select "Detect" action
- Choose the same method used for embedding
- Enter the original message
- Click "Process Audio"
- Verify detection results (probability, BER, etc.)

### 5. Ledger/History
- Navigate to the Ledger page
- Verify all watermark entries are displayed
- Test filtering by method, action, purpose
- Test search functionality
- Verify file associations work

### 6. Dashboard
- Check statistics and analytics
- Verify method comparison charts
- Test batch processing if available

## 4. Error Testing

### Invalid File Upload
- Try uploading non-audio files
- Try uploading corrupted audio files
- Try uploading files larger than 16MB

### Invalid API Requests
- Send requests without required fields
- Use invalid method names
- Send malformed JSON

### Database Errors
- Test with database connection issues
- Verify proper error handling

## 5. Performance Testing

### File Size Limits
- Test with small files (< 1MB)
- Test with medium files (1-10MB)
- Test with large files (10-16MB)

### Concurrent Requests
- Test multiple simultaneous uploads
- Test multiple simultaneous processing requests

### Memory Usage
- Monitor memory usage during processing
- Test with long audio files

## 6. Audio Quality Testing

### Before/After Comparison
- Compare original and watermarked audio
- Check for audible artifacts
- Verify audio quality metrics

### Different Audio Types
- Test with speech audio
- Test with music audio
- Test with different sample rates
- Test with different bit depths

## 7. Security Testing

### File Upload Security
- Test for path traversal attacks
- Test for malicious file uploads
- Verify proper file type validation

### API Security
- Test for SQL injection (if applicable)
- Test for XSS attacks
- Verify proper input validation

## Test Data

### Sample Audio Files
Create test audio files with these characteristics:
- **Speech**: 10-30 seconds, 16kHz, WAV format
- **Music**: 30-60 seconds, 44.1kHz, MP3 format
- **Mixed**: 60+ seconds, 48kHz, FLAC format

### Test Messages
Use these binary messages for testing:
- `"10101010"` (8 bits)
- `"1100110011001100"` (16 bits)
- `"1111000011110000"` (16 bits)

## Reporting Issues

When reporting issues, include:
1. **Environment**: OS, Python version, browser
2. **Steps to reproduce**: Detailed step-by-step
3. **Expected vs Actual behavior**
4. **Error messages**: Full error logs
5. **Test data**: Audio files and parameters used
6. **Screenshots**: If applicable

## Success Criteria

A successful test run should demonstrate:
- ✅ All API endpoints respond correctly
- ✅ File upload and processing work
- ✅ Watermark embedding and detection function
- ✅ Database operations work properly
- ✅ Frontend UI is responsive and functional
- ✅ Error handling works as expected
- ✅ Audio quality is maintained
- ✅ Performance is acceptable 