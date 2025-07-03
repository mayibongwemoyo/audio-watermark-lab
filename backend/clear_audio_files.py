# backend/clear_audio_files.py

from app import db
from models import AudioFile

if __name__ == "__main__":
    num_deleted = db.session.query(AudioFile).delete()
    db.session.commit()
    print(f"Deleted {num_deleted} records from audio_files table.")

