
from flask import Blueprint, request, jsonify
from models import db, User, AudioFile, WatermarkEntry
import json
from werkzeug.exceptions import NotFound, BadRequest

# Create Blueprint
db_api = Blueprint('db_api', __name__)

# User endpoints
@db_api.route('/users', methods=['GET'])
def get_users():
    """Get all users"""
    users = User.query.all()
    return jsonify({
        'status': 'success',
        'users': [user.to_dict() for user in users]
    })

@db_api.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get a specific user"""
    user = User.query.get_or_404(user_id)
    return jsonify({
        'status': 'success',
        'user': user.to_dict()
    })

# Audio file endpoints
@db_api.route('/audio_files', methods=['GET'])
def get_audio_files():
    """Get all audio files"""
    files = AudioFile.query.all()
    return jsonify({
        'status': 'success',
        'audio_files': [file.to_dict() for file in files]
    })

@db_api.route('/audio_files/<int:file_id>', methods=['GET'])
def get_audio_file(file_id):
    """Get a specific audio file"""
    file = AudioFile.query.get_or_404(file_id)
    return jsonify({
        'status': 'success',
        'audio_file': file.to_dict()
    })

# Watermark ledger endpoints
@db_api.route('/watermarks', methods=['GET'])
def get_watermarks():
    """Get all watermark entries with optional filtering"""
    
    # Get query parameters for filtering
    user_id = request.args.get('user_id', type=int)
    file_id = request.args.get('file_id', type=int)
    method = request.args.get('method')
    action = request.args.get('action')
    purpose = request.args.get('purpose')
    
    # Start with base query
    query = WatermarkEntry.query
    
    # Apply filters if provided
    if user_id:
        query = query.filter_by(user_id=user_id)
    if file_id:
        query = query.filter_by(audio_file_id=file_id)
    if method:
        query = query.filter_by(method=method)
    if action:
        query = query.filter_by(action=action)
    if purpose:
        query = query.filter_by(purpose=purpose)
    
    # Execute query and return results
    entries = query.order_by(WatermarkEntry.created_at.desc()).all()
    
    return jsonify({
        'status': 'success',
        'watermarks': [entry.to_dict() for entry in entries]
    })

@db_api.route('/watermarks/<int:entry_id>', methods=['GET'])
def get_watermark(entry_id):
    """Get a specific watermark entry"""
    entry = WatermarkEntry.query.get_or_404(entry_id)
    return jsonify({
        'status': 'success',
        'watermark': entry.to_dict()
    })

# Create new watermark entry
@db_api.route('/watermarks', methods=['POST'])
def create_watermark():
    """Create a new watermark entry"""
    data = request.json
    
    if not data:
        raise BadRequest("No data provided")
    
    # Convert JSON string to dict if needed
    metadata = data.get('metadata')
    if isinstance(metadata, dict):
        metadata_json = json.dumps(metadata)
    else:
        metadata_json = metadata
        
    # Create new watermark entry
    new_entry = WatermarkEntry(
        action=data.get('action'),
        method=data.get('method'),
        message=data.get('message'),
        snr_db=data.get('snr_db'),
        detection_probability=data.get('detection_probability'),
        ber=data.get('ber'),
        is_detected=data.get('is_detected'),
        purpose=data.get('purpose'),
        watermark_count=data.get('watermark_count', 1),
        metadata=metadata_json,
        user_id=data.get('user_id'),
        audio_file_id=data.get('audio_file_id'),
        watermarked_file_id=data.get('watermarked_file_id')
    )
    
    db.session.add(new_entry)
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'message': 'Watermark entry created successfully',
        'watermark': new_entry.to_dict()
    }), 201

# Get statistics and analytics
@db_api.route('/stats/methods', methods=['GET'])
def get_method_statistics():
    """Get statistics on watermark methods"""
    # Count usage of each method
    result = db.session.query(
        WatermarkEntry.method, 
        db.func.count(WatermarkEntry.id).label('count'),
        db.func.avg(WatermarkEntry.snr_db).label('avg_snr'),
        db.func.avg(WatermarkEntry.detection_probability).label('avg_detection'),
        db.func.avg(WatermarkEntry.ber).label('avg_ber')
    ).group_by(WatermarkEntry.method).all()
    
    stats = [
        {
            'method': r.method,
            'count': r.count,
            'avg_snr': float(r.avg_snr) if r.avg_snr is not None else None,
            'avg_detection': float(r.avg_detection) if r.avg_detection is not None else None,
            'avg_ber': float(r.avg_ber) if r.avg_ber is not None else None,
        }
        for r in result
    ]
    
    return jsonify({
        'status': 'success',
        'statistics': stats
    })

# Error handlers
@db_api.errorhandler(NotFound)
def handle_not_found(e):
    return jsonify({
        'status': 'error',
        'message': str(e)
    }), 404

@db_api.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({
        'status': 'error',
        'message': str(e)
    }), 400

@db_api.errorhandler(Exception)
def handle_exception(e):
    return jsonify({
        'status': 'error',
        'message': str(e)
    }), 500
