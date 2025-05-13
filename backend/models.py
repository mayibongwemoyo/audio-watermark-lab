
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
    watermark_entries = db.relationship('WatermarkEntry', backref='audio_file', lazy=True)

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
    action = db.Column(db.String(20), nullable=False)  # embed or detect
    method = db.Column(db.String(20), nullable=False)  # sfa, sda, pfb, pca, etc.
    message = db.Column(db.String(100))                # Binary message
    snr_db = db.Column(db.Float)                       # Signal-to-noise ratio
    detection_probability = db.Column(db.Float)        # Detection probability
    ber = db.Column(db.Float)                          # Bit error rate
    is_detected = db.Column(db.Boolean)                # Whether watermark was detected
    purpose = db.Column(db.String(50))                 # Purpose of watermarking
    watermark_count = db.Column(db.Integer, default=1) # Number of watermarks applied
    metadata = db.Column(db.Text)                      # JSON metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    audio_file_id = db.Column(db.Integer, db.ForeignKey('audio_files.id'), nullable=True)
    watermarked_file_id = db.Column(db.Integer, db.ForeignKey('audio_files.id'), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'action': self.action,
            'method': self.method,
            'message': self.message,
            'snr_db': self.snr_db,
            'detection_probability': self.detection_probability,
            'ber': self.ber,
            'is_detected': self.is_detected,
            'purpose': self.purpose,
            'watermark_count': self.watermark_count,
            'metadata': json.loads(self.metadata) if self.metadata else {},
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
        
        # Create sample audio files
        audio_files = [
            AudioFile(
                filename="voice_sample_01.wav",
                filepath="/uploads/voice_sample_01.wav",
                filehash="ae129f8d7c298a759f",
                file_size=1024000,
                duration=45.5,
                sample_rate=44100,
                user_id=3
            ),
            AudioFile(
                filename="track_mixdown_v2.wav",
                filepath="/uploads/track_mixdown_v2.wav",
                filehash="cb852e4f1d38b612a3",
                file_size=3072000,
                duration=120.2,
                sample_rate=48000,
                user_id=5
            ),
            AudioFile(
                filename="final_mix_with_fx.wav",
                filepath="/uploads/final_mix_with_fx.wav",
                filehash="f2e901d45c781b324d",
                file_size=4096000,
                duration=180.7,
                sample_rate=48000,
                user_id=7
            ),
            AudioFile(
                filename="podcast_intro.mp3",
                filepath="/uploads/podcast_intro.mp3",
                filehash="ba92c13e487f20d6e5",
                file_size=512000,
                duration=15.3,
                sample_rate=44100,
                user_id=2
            ),
            AudioFile(
                filename="ad_campaign_audio.wav",
                filepath="/uploads/ad_campaign_audio.wav",
                filehash="d31fc9a8526b478209",
                file_size=2048000,
                duration=60.0,
                sample_rate=44100,
                user_id=1
            )
        ]
        db.session.add_all(audio_files)
        
        # Create sample watermark entries
        two_days_ago = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        two_days_ago = two_days_ago.replace(day=two_days_ago.day - 2)
        
        entries = [
            WatermarkEntry(
                action="embed",
                method="sfa",
                message="01100101",
                snr_db=65.78,
                detection_probability=0.85,
                ber=0.125,
                is_detected=True,
                purpose="training",
                watermark_count=1,
                metadata=json.dumps({"device": "studio-a", "source": "microphone"}),
                created_at=two_days_ago.replace(hour=10),
                user_id=3,
                audio_file_id=1
            ),
            WatermarkEntry(
                action="embed",
                method="sda",
                message="10101010",
                snr_db=58.32,
                detection_probability=0.92,
                ber=0.0625,
                is_detected=True,
                purpose="internal",
                watermark_count=2,
                metadata=json.dumps({"device": "mixing-desk", "project": "album-release"}),
                created_at=two_days_ago.replace(hour=14),
                user_id=5,
                audio_file_id=2
            ),
            WatermarkEntry(
                action="embed",
                method="pfb",
                message="11000101",
                snr_db=72.15,
                detection_probability=0.78,
                ber=0.1875,
                is_detected=True,
                purpose="remix",
                watermark_count=3,
                metadata=json.dumps({"software": "protools", "effects": ["reverb", "eq"]}),
                created_at=two_days_ago.replace(hour=18),
                user_id=7,
                audio_file_id=3
            ),
            WatermarkEntry(
                action="embed",
                method="pca",
                message="00111001",
                snr_db=68.94,
                detection_probability=0.88,
                ber=0.125,
                is_detected=True,
                purpose="distribution",
                watermark_count=1,
                metadata=json.dumps({"platform": "spotify", "release": "podcast-ep5"}),
                created_at=two_days_ago.replace(day=two_days_ago.day+1, hour=9),
                user_id=2,
                audio_file_id=4
            ),
            WatermarkEntry(
                action="embed",
                method="sfa",
                message="10010110",
                snr_db=61.47,
                detection_probability=0.81,
                ber=0.25,
                is_detected=True,
                purpose="commercial",
                watermark_count=2,
                metadata=json.dumps({"campaign": "summer-sale", "client": "retail-corp"}),
                created_at=datetime.utcnow().replace(hour=11),
                user_id=1,
                audio_file_id=5
            )
        ]
        db.session.add_all(entries)
        
        # Commit all sample data
        db.session.commit()
        print("Sample data created successfully.")
