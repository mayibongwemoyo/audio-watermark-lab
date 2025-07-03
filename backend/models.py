from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True)
    role = db.Column(db.String(50), default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    audio_files = db.relationship('AudioFile', backref='user', lazy=True)
    watermark_entries = db.relationship('WatermarkEntry', backref='user', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class AudioFile(db.Model):
    __tablename__ = 'audio_files'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False, unique=True)
    filehash = db.Column(db.String(64))  # Hash of the file for integrity/identification
    file_size = db.Column(db.Integer)    # Size in bytes
    duration = db.Column(db.Float)       # Duration in seconds
    sample_rate = db.Column(db.Integer)  # Audio sample rate
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    # Relationships
    watermark_entries = db.relationship('WatermarkEntry', 
                                         foreign_keys='WatermarkEntry.audio_file_id',
                                         backref='audio_file', 
                                         lazy=True)
    watermarked_entries = db.relationship('WatermarkEntry',
                                          foreign_keys='WatermarkEntry.watermarked_file_id',
                                          backref='watermarked_file',
                                          lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'filepath': self.filepath,
            'filehash': self.filehash,
            'file_size': self.file_size,
            'duration': self.duration,
            'sample_rate': self.sample_rate,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'user_id': self.user_id
        }


class WatermarkEntry(db.Model):
    __tablename__ = 'watermark_entries'
    
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(20), nullable=False)   # embed or detect
    method = db.Column(db.String(20), nullable=False)   # sfa, sda, pfb, pca, etc.
    message = db.Column(db.String(100))                 # Binary message
    
    # Metrics for embedding
    snr_db = db.Column(db.Float)                        # Signal-to-noise ratio
    mse = db.Column(db.Float)                           # ADD THIS LINE: Mean Squared Error
    detection_probability = db.Column(db.Float)         # Detection probability
    ber = db.Column(db.Float)                           # Bit error rate
    is_detected = db.Column(db.Boolean)                 # Whether watermark was detected
    
    # Contextual data
    purpose = db.Column(db.String(50))                  # Purpose of watermarking
    watermark_count = db.Column(db.Integer, default=1)  # Number of watermarks applied
    meta_data = db.Column(db.Text)                      # JSON metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    audio_file_id = db.Column(db.Integer, db.ForeignKey('audio_files.id'), nullable=True)
    watermarked_file_id = db.Column(db.Integer, db.ForeignKey('audio_files.id'), nullable=True)
    
    def to_dict(self):
        meta_data_dict = {}
        if self.meta_data:
            try:
                meta_data_dict = json.loads(self.meta_data)
            except json.JSONDecodeError:
                meta_data_dict = {"raw_meta_data": self.meta_data}

        return {
            'id': self.id,
            'action': self.action,
            'method': self.method,
            'message': self.message,
            'snr_db': self.snr_db,
            'mse': self.mse, # Include MSE in dict conversion
            'detection_probability': self.detection_probability,
            'ber': self.ber,
            'is_detected': self.is_detected,
            'purpose': self.purpose,
            'watermark_count': self.watermark_count,
            'meta_data': meta_data_dict,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'user_id': self.user_id,
            'audio_file_id': self.audio_file_id,
            'watermarked_file_id': self.watermarked_file_id
        }

# Functions to initialize the database with sample data
def create_sample_data():
    """Create sample users and entries if the database is empty"""
    # Only create sample data if tables are empty
    if User.query.count() == 0:
        # Create sample users
        users = [
            User(id=1, username="Alex Johnson", email="alex@example.com", role="marketer"),
            User(id=2, username="David Kim", email="david@example.com", role="producer"),
            User(id=3, username="Emma Jackson", email="emma@example.com", role="voice_actor"),
            User(id=5, username="Michael Chen", email="michael@example.com", role="producer"),
            User(id=7, username="Sophie Garcia", email="sophie@example.com", role="editor")
        ]
        db.session.add_all(users)
        
        # Create sample audio files (Simplified for brevity, assuming paths might not exist locally)
        # These are just placeholders to link to WatermarkEntry
        audio_files = [
            AudioFile(
                filename="voice_sample_01.wav", filepath="dummy_path_1.wav", filehash="ae129f8d7c298a759f",
                file_size=1000000, duration=30.0, sample_rate=16000, user_id=3
            ),
            AudioFile(
                filename="track_mixdown_v2.wav", filepath="dummy_path_2.wav", filehash="cb852e4f1d38b612a3",
                file_size=2000000, duration=60.0, sample_rate=16000, user_id=5
            ),
        ]
        db.session.add_all(audio_files)
        
        # Create sample watermark entries
        two_days_ago = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        two_days_ago = two_days_ago.replace(day=two_days_ago.day - 2)
        
        entries = [
            WatermarkEntry(
                action="embed", method="pca_prime", message="10101010101010101010101010101010",
                snr_db=32.5, mse=0.0004, # ADDED MSE HERE
                detection_probability=0.99, ber=0.005, is_detected=True,
                purpose="training", watermark_count=4,
                meta_data=json.dumps({"device": "studio-a", "source": "microphone"}),
                created_at=two_days_ago.replace(hour=10), user_id=3, audio_file_id=1, watermarked_file_id=None # Assuming no specific watermarked_file for sample data
            ),
            WatermarkEntry(
                action="detect", method="pca_prime", message="10101010101010101010101010101010",
                snr_db=None, mse=None, # For detection entries, SNR/MSE are not always applicable
                detection_probability=0.95, ber=0.01, is_detected=True,
                purpose="detection", watermark_count=0,
                meta_data=json.dumps({"result_info": "Detected on modified file"}),
                created_at=two_days_ago.replace(hour=11), user_id=1, audio_file_id=2, watermarked_file_id=None
            ),
        ]
        db.session.add_all(entries)
        
        # Commit all sample data
        db.session.commit()
        print("Sample data created.")
    else:
        print("Sample data already exists.")