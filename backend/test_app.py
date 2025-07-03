import unittest
import json
import tempfile
import os
import numpy as np
from app import app, db
from models import User, AudioFile, WatermarkEntry
import io

class AudioWatermarkTestCase(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test"""
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        app.config['WTF_CSRF_ENABLED'] = False
        
        self.app = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        
        # Create database tables
        db.create_all()
        
        # Create test upload directory
        self.test_upload_dir = tempfile.mkdtemp()
        app.config['UPLOAD_FOLDER'] = self.test_upload_dir
    
    def tearDown(self):
        """Clean up after each test"""
        db.session.remove()
        db.drop_all()
        self.app_context.pop()
        
        # Clean up test upload directory
        import shutil
        shutil.rmtree(self.test_upload_dir, ignore_errors=True)
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.app.get('/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('audioseal_available', data)
    
    def test_methods_endpoint(self):
        """Test the methods endpoint"""
        response = self.app.get('/methods')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('methods', data)
        self.assertIsInstance(data['methods'], list)
    
    def test_upload_audio_file(self):
        """Test audio file upload"""
        # Create a simple test audio file (sine wave)
        sample_rate = 16000
        duration = 1.0  # 1 second
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save as WAV file
        import soundfile as sf
        test_file_path = os.path.join(self.test_upload_dir, 'test_audio.wav')
        sf.write(test_file_path, audio_data, sample_rate)
        
        # Upload file
        with open(test_file_path, 'rb') as f:
            response = self.app.post('/upload', 
                                   data={'file': (f, 'test_audio.wav')},
                                   content_type='multipart/form-data')
        
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertIn('file_id', data)
    
    def test_watermark_embedding(self):
        """Test watermark embedding functionality"""
        # First upload a test file
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)
        
        import soundfile as sf
        test_file_path = os.path.join(self.test_upload_dir, 'test_embed.wav')
        sf.write(test_file_path, audio_data, sample_rate)
        
        # Upload file
        with open(test_file_path, 'rb') as f:
            upload_response = self.app.post('/upload', 
                                          data={'file': (f, 'test_embed.wav')},
                                          content_type='multipart/form-data')
        
        upload_data = json.loads(upload_response.data)
        file_id = upload_data['file_id']
        
        # Get pre-embed analysis first
        with open(test_file_path, 'rb') as f:
            pre_response = self.app.post('/pre_embed_detect',
                                       data={'file': (f, 'test_embed.wav')},
                                       content_type='multipart/form-data')
        
        pre_data = json.loads(pre_response.data)
        original_audio_file_id = pre_data['original_audio_file_id']
        next_wm_idx = pre_data['next_wm_idx']
        
        # Test watermark embedding - send as form data with new parameters
        with open(test_file_path, 'rb') as f:
            response = self.app.post('/process_audio',
                                   data={
                                       'audio_file': (f, 'test_embed.wav'),
                                       'action': 'embed',
                                       'method': 'sfa',
                                       'message': '10101010',  # 8-bit message
                                       'full_cumulative_message': '10101010' + '0' * 24,  # 32-bit cumulative
                                       'original_audio_file_id': str(original_audio_file_id),
                                       'current_wm_idx': str(next_wm_idx),
                                       'purpose': 'test'
                                   },
                                   content_type='multipart/form-data')
        
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['action'], 'embed')
        self.assertIn('processed_audio_url', data)
    
    def test_watermark_detection(self):
        """Test watermark detection functionality"""
        # First embed a watermark
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)
        
        import soundfile as sf
        test_file_path = os.path.join(self.test_upload_dir, 'test_detect.wav')
        sf.write(test_file_path, audio_data, sample_rate)
        
        # Upload file
        with open(test_file_path, 'rb') as f:
            upload_response = self.app.post('/upload', 
                                          data={'file': (f, 'test_detect.wav')},
                                          content_type='multipart/form-data')
        
        upload_data = json.loads(upload_response.data)
        file_id = upload_data['file_id']
        
        # Get pre-embed analysis first
        with open(test_file_path, 'rb') as f:
            pre_response = self.app.post('/pre_embed_detect',
                                       data={'file': (f, 'test_detect.wav')},
                                       content_type='multipart/form-data')
        
        pre_data = json.loads(pre_response.data)
        original_audio_file_id = pre_data['original_audio_file_id']
        next_wm_idx = pre_data['next_wm_idx']
        
        # Embed watermark first - send as form data with new parameters
        with open(test_file_path, 'rb') as f:
            embed_response = self.app.post('/process_audio',
                                         data={
                                             'audio_file': (f, 'test_detect.wav'),
                                             'action': 'embed',
                                             'method': 'sfa',
                                             'message': '10101010',  # 8-bit message
                                             'full_cumulative_message': '10101010' + '0' * 24,  # 32-bit cumulative
                                             'original_audio_file_id': str(original_audio_file_id),
                                             'current_wm_idx': str(next_wm_idx),
                                             'purpose': 'test'
                                         },
                                         content_type='multipart/form-data')
        
        embed_result = json.loads(embed_response.data)
        watermarked_file_id = embed_result.get('watermarked_file_id', file_id)
        
        # Now detect the watermark - send as form data
        with open(test_file_path, 'rb') as f:
            response = self.app.post('/process_audio',
                                   data={
                                       'audio_file': (f, 'test_detect.wav'),
                                       'action': 'detect',
                                       'method': 'sfa',
                                       'message': '10101010'  # 8-bit message
                                   },
                                   content_type='multipart/form-data')
        
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('detection_probability', data)
        self.assertIn('is_detected', data)
    
    def test_database_api_endpoints(self):
        """Test database API endpoints"""
        # Test getting users
        response = self.app.get('/api/users')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        
        # Test getting watermarks
        response = self.app.get('/api/watermarks')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        
        # Test getting audio files
        response = self.app.get('/api/audio_files')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
    
    def test_invalid_file_upload(self):
        """Test uploading invalid file types"""
        # Create a text file (invalid)
        test_file_path = os.path.join(self.test_upload_dir, 'test.txt')
        with open(test_file_path, 'w') as f:
            f.write('This is not an audio file')
        
        with open(test_file_path, 'rb') as f:
            response = self.app.post('/upload', 
                                   data={'file': (f, 'test.txt')},
                                   content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 400)
    
    def test_missing_file_upload(self):
        """Test upload without file"""
        response = self.app.post('/upload')
        self.assertEqual(response.status_code, 400)
    
    def test_invalid_process_audio_request(self):
        """Test invalid process_audio requests"""
        # Test without required fields
        response = self.app.post('/process_audio',
                               data=json.dumps({}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Test with invalid method
        response = self.app.post('/process_audio',
                               data=json.dumps({
                                   'file_id': 1,
                                   'action': 'embed',
                                   'method': 'invalid_method',
                                   'message': '10101010'
                               }),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_pre_embed_detect_endpoint(self):
        """Test the pre-embed detection endpoint"""
        # Create a test audio file
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)
        
        import soundfile as sf
        test_file_path = os.path.join(self.test_upload_dir, 'test_pre_detect.wav')
        sf.write(test_file_path, audio_data, sample_rate)
        
        # Test pre-embed detection
        with open(test_file_path, 'rb') as f:
            response = self.app.post('/pre_embed_detect',
                                   data={'file': (f, 'test_pre_detect.wav')},
                                   content_type='multipart/form-data')
        
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertIn('next_wm_idx', data)
        self.assertIn('original_audio_file_id', data)
        self.assertIn('detected_per_band', data)
        self.assertIsInstance(data['next_wm_idx'], int)
        self.assertIsInstance(data['original_audio_file_id'], int)
        self.assertIsInstance(data['detected_per_band'], list)

    def test_incremental_watermark_embedding(self):
        """Test incremental watermark embedding with new parameters"""
        # Create a test audio file
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)
        
        import soundfile as sf
        test_file_path = os.path.join(self.test_upload_dir, 'test_incremental.wav')
        sf.write(test_file_path, audio_data, sample_rate)
        
        # First, get the pre-embed analysis
        with open(test_file_path, 'rb') as f:
            pre_response = self.app.post('/pre_embed_detect',
                                       data={'file': (f, 'test_incremental.wav')},
                                       content_type='multipart/form-data')
        
        pre_data = json.loads(pre_response.data)
        original_audio_file_id = pre_data['original_audio_file_id']
        next_wm_idx = pre_data['next_wm_idx']
        
        # Test watermark embedding with new parameters
        with open(test_file_path, 'rb') as f:
            response = self.app.post('/process_audio',
                                   data={
                                       'audio_file': (f, 'test_incremental.wav'),
                                       'action': 'embed',
                                       'method': 'pca',
                                       'message': '10101010',  # 8-bit message
                                       'full_cumulative_message': '10101010' + '0' * 24,  # 32-bit cumulative
                                       'original_audio_file_id': str(original_audio_file_id),
                                       'current_wm_idx': str(next_wm_idx),
                                       'purpose': 'test'
                                   },
                                   content_type='multipart/form-data')
        
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['action'], 'embed')
        self.assertIn('processed_audio_url', data)
        self.assertIn('results', data)
        self.assertIn('snr_db', data['results'])
        self.assertIn('mse', data['results'])
        self.assertIn('ber', data['results'])

if __name__ == '__main__':
    unittest.main() 