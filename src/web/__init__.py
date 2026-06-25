"""
Web Interface Module
Flask-based web interface for Face ID System
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
import base64
from datetime import datetime
from werkzeug.utils import secure_filename
import logging
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .auth import require_api_key, rate_limit

class FaceIDWebInterface:
    """Web interface for Face ID System"""
    
    def __init__(self, face_id_system, host='0.0.0.0', port=5000):
        """
        Initialize web interface
        
        Args:
            face_id_system: FaceIDSystem instance
            host: Host address
            port: Port number
        """
        self.face_id_system = face_id_system
        self.host = host
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        self.app.secret_key = os.environ.get('FACEID_SECRET_KEY', secrets.token_hex(32))
        
        # Enable CORS
        CORS(self.app)
        
        # Configure upload settings
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        self.app.config['UPLOAD_FOLDER'] = 'data/uploads'
        
        # Allowed file extensions
        self.ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        
        # Ensure upload directory exists
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Setup WebSockets
        self._setup_websockets()
        
        # Setup routes
        self._setup_routes()
    
    def allowed_file(self, filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main page"""
            return render_template('index.html')
        
        @self.app.route('/register')
        def register_page():
            """Registration page"""
            return render_template('register.html')
        
        @self.app.route('/video_register')
        def video_register():
            """Video registration page"""
            return render_template('video_register.html')
        
        @self.app.route('/recognize')
        def recognize_page():
            """Recognition page"""
            return render_template('recognize.html')
        
        @self.app.route('/dashboard')
        def dashboard():
            """Dashboard page"""
            stats = self.face_id_system.get_system_stats()
            return render_template('dashboard.html', stats=stats)
        
        @self.app.route('/debug_registration.html')
        def debug_registration():
            """Debug registration page"""
            try:
                with open('debug_registration.html', 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                return "Debug page not found. Run: python create_debug_page.py"
        
        @self.app.route('/api/register', methods=['POST'])
        @require_api_key
        @rate_limit(limit=5, period=60)
        def api_register():
            """API endpoint for person registration"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                if not self.allowed_file(file.filename):
                    return jsonify({'error': 'Invalid file type'}), 400
                
                # Get person name
                person_name = request.form.get('person_name', '').strip()
                if not person_name:
                    return jsonify({'error': 'Person name is required'}), 400
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Register person
                success = self.face_id_system.register_person(filepath, person_name)
                
                if success:
                    return jsonify({
                        'success': True,
                        'message': f'Successfully registered {person_name}',
                        'person_name': person_name
                    })
                else:
                    # Provide more specific error message
                    return jsonify({
                        'error': 'Registration failed. Please ensure the image contains a clear face and try again.'
                    }), 400
                
            except Exception as e:
                logger.error(f"Registration API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/video_register', methods=['POST'])
        @require_api_key
        @rate_limit(limit=2, period=60)
        def api_video_register():
            """API endpoint for video-based person registration"""
            try:
                # Get person name
                person_name = request.form.get('person_name', '').strip()
                if not person_name:
                    return jsonify({'error': 'Person name is required'}), 400
                
                # Check if video file is provided
                if 'video' not in request.files:
                    return jsonify({'error': 'No video file provided'}), 400
                
                video_file = request.files['video']
                if video_file.filename == '':
                    return jsonify({'error': 'No video file selected'}), 400
                
                # Save video file temporarily
                filename = secure_filename(video_file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                video_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                video_file.save(video_path)
                
                # Process video registration using the video registration system
                from video_registration import VideoRegistrationSystem
                
                # Initialize video registration system
                video_reg_system = VideoRegistrationSystem(self.face_id_system)
                
                logger.info(f"Processing video file: {video_path}")
                logger.info(f"Video file size: {os.path.getsize(video_path)} bytes")
                
                # Process the video file
                success = video_reg_system.process_video_file(video_path, person_name)
                
                logger.info(f"Video processing result: {success}")
                
                # Clean up temporary video file
                try:
                    os.remove(video_path)
                    logger.info(f"Cleaned up temporary video file: {video_path}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary video file: {e}")
                
                if success:
                    return jsonify({
                        'success': True,
                        'message': f'Successfully registered {person_name} with video (multiple angles captured)',
                        'person_name': person_name,
                        'method': 'video'
                    })
                else:
                    return jsonify({
                        'error': 'Video registration failed. Please ensure the video contains clear faces and try again.'
                    }), 400
                
            except Exception as e:
                logger.error(f"Video registration API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recognize', methods=['POST'])
        @rate_limit(limit=20, period=60)
        def api_recognize():
            """API endpoint for face recognition"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                if not self.allowed_file(file.filename):
                    return jsonify({'error': 'Invalid file type'}), 400
                
                # Read image
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({'error': 'Invalid image'}), 400
                
                # Recognize face
                person_name, confidence, face_info = self.face_id_system.recognize_face(image)
                
                # Prepare response
                face_bbox = None
                if face_info and 'bbox' in face_info:
                    bbox = face_info['bbox']
                    # Convert numpy int32 to Python int for JSON serialization
                    face_bbox = [int(x) for x in bbox] if bbox else None
                
                is_live = face_info.get('is_live', True) if face_info else True
                liveness_score = face_info.get('liveness_score', 1.0) if face_info else 1.0
                
                result = {
                    'person_name': person_name,
                    'confidence': float(confidence),
                    'face_detected': face_info is not None,
                    'face_bbox': face_bbox,
                    'is_live': is_live,
                    'liveness_score': float(liveness_score)
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Recognition API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recognize_base64', methods=['POST'])
        @rate_limit(limit=20, period=60)
        def api_recognize_base64():
            """API endpoint for face recognition with base64 image"""
            try:
                data = request.get_json()
                if not data or 'image' not in data:
                    return jsonify({'error': 'No image data provided'}), 400
                
                # Decode base64 image
                image_data = data['image']
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return jsonify({'error': 'Invalid image data'}), 400
                
                # Recognize face
                person_name, confidence, face_info = self.face_id_system.recognize_face(image)
                
                # Prepare response
                face_bbox = None
                if face_info and 'bbox' in face_info:
                    bbox = face_info['bbox']
                    # Convert numpy int32 to Python int for JSON serialization
                    face_bbox = [int(x) for x in bbox] if bbox else None
                
                is_live = face_info.get('is_live', True) if face_info else True
                liveness_score = face_info.get('liveness_score', 1.0) if face_info else 1.0
                
                result = {
                    'person_name': person_name,
                    'confidence': float(confidence),
                    'face_detected': face_info is not None,
                    'face_bbox': face_bbox,
                    'is_live': is_live,
                    'liveness_score': float(liveness_score)
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Base64 recognition API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recognize_batch', methods=['POST'])
        @rate_limit(limit=10, period=60)
        def api_recognize_batch():
            """API endpoint for batch face recognition"""
            try:
                results = []
                
                # Check for base64 JSON payload first
                if request.is_json:
                    data = request.get_json()
                    images_data = data.get('images', [])
                    if not images_data:
                        return jsonify({'error': 'No images list provided in JSON'}), 400
                        
                    for i, image_data in enumerate(images_data):
                        try:
                            if image_data.startswith('data:image'):
                                image_data = image_data.split(',')[1]
                            image_bytes = base64.b64decode(image_data)
                            nparr = np.frombuffer(image_bytes, np.uint8)
                            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if image is None:
                                results.append({'index': i, 'error': 'Invalid image data'})
                                continue
                                
                            person_name, confidence, face_info = self.face_id_system.recognize_face(image)
                            
                            face_bbox = None
                            if face_info and 'bbox' in face_info:
                                bbox = face_info['bbox']
                                face_bbox = [int(x) for x in bbox] if bbox else None
                                
                            is_live = face_info.get('is_live', True) if face_info else True
                            liveness_score = face_info.get('liveness_score', 1.0) if face_info else 1.0
                            
                            results.append({
                                'index': i,
                                'person_name': person_name,
                                'confidence': float(confidence),
                                'face_detected': face_info is not None,
                                'face_bbox': face_bbox,
                                'is_live': is_live,
                                'liveness_score': float(liveness_score)
                            })
                        except Exception as inner_e:
                            results.append({'index': i, 'error': str(inner_e)})
                            
                # Otherwise, check for multipart files
                else:
                    if not request.files:
                        return jsonify({'error': 'No files provided'}), 400
                        
                    files = request.files.getlist('file')
                    if not files or (len(files) == 1 and files[0].filename == ''):
                        files = [f for f_list in request.files.lists() for f in f_list[1]]
                        
                    if not files or len(files) == 0:
                        return jsonify({'error': 'No files selected'}), 400
                        
                    for i, file in enumerate(files):
                        try:
                            if not self.allowed_file(file.filename):
                                results.append({'index': i, 'filename': file.filename, 'error': 'Invalid file type'})
                                continue
                                
                            file_bytes = file.read()
                            nparr = np.frombuffer(file_bytes, np.uint8)
                            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if image is None:
                                results.append({'index': i, 'filename': file.filename, 'error': 'Invalid image'})
                                continue
                                
                            person_name, confidence, face_info = self.face_id_system.recognize_face(image)
                            
                            face_bbox = None
                            if face_info and 'bbox' in face_info:
                                bbox = face_info['bbox']
                                face_bbox = [int(x) for x in bbox] if bbox else None
                                
                            is_live = face_info.get('is_live', True) if face_info else True
                            liveness_score = face_info.get('liveness_score', 1.0) if face_info else 1.0
                            
                            results.append({
                                'index': i,
                                'filename': file.filename,
                                'person_name': person_name,
                                'confidence': float(confidence),
                                'face_detected': face_info is not None,
                                'face_bbox': face_bbox,
                                'is_live': is_live,
                                'liveness_score': float(liveness_score)
                            })
                        except Exception as inner_e:
                            results.append({'index': i, 'filename': file.filename, 'error': str(inner_e)})
                            
                return jsonify({'results': results})
                
            except Exception as e:
                logger.error(f"Batch recognition API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stats')
        def api_stats():
            """API endpoint for system statistics"""
            try:
                stats = self.face_id_system.get_system_stats()
                return jsonify(stats)
            except Exception as e:
                logger.error(f"Stats API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/persons')
        def api_persons():
            """API endpoint for getting all registered persons"""
            try:
                persons = self.face_id_system.database.get_all_persons()
                return jsonify({'persons': persons})
            except Exception as e:
                logger.error(f"Persons API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/person/<int:person_id>')
        def api_person_details(person_id):
            """API endpoint for person details"""
            try:
                person = self.face_id_system.database.get_person_by_id(person_id)
                if not person:
                    return jsonify({'error': 'Person not found'}), 404
                
                images = self.face_id_system.database.get_person_images(person_id)
                person['images'] = images
                
                return jsonify(person)
            except Exception as e:
                logger.error(f"Person details API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/delete_person/<int:person_id>', methods=['DELETE'])
        @require_api_key
        def api_delete_person(person_id):
            """API endpoint for deleting a person"""
            try:
                person = self.face_id_system.database.get_person_by_id(person_id)
                if not person:
                    return jsonify({'error': 'Person not found'}), 404
                
                person_name = person['name']
                
                # Use comprehensive deletion method
                success = self.face_id_system.delete_person_comprehensive(person_id)
                
                if success:
                    return jsonify({
                        'success': True,
                        'message': f'Successfully deleted {person_name} and all associated data'
                    })
                else:
                    return jsonify({'error': 'Failed to delete person'}), 500
                
            except Exception as e:
                logger.error(f"Delete person API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/camera/start', methods=['POST'])
        def api_camera_start():
            """API endpoint for starting camera recognition"""
            try:
                camera_index = request.json.get('camera_index', 0) if request.is_json else 0
                
                success = self.face_id_system.start_camera_recognition(
                    camera_index=camera_index,
                    display_window=False  # Web interface doesn't need display window
                )
                
                if success:
                    return jsonify({'success': True, 'message': 'Camera recognition started'})
                else:
                    return jsonify({'error': 'Failed to start camera recognition'}), 500
                
            except Exception as e:
                logger.error(f"Camera start API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/camera/stop', methods=['POST'])
        def api_camera_stop():
            """API endpoint for stopping camera recognition"""
            try:
                self.face_id_system.stop_camera_recognition()
                return jsonify({'success': True, 'message': 'Camera recognition stopped'})
                
            except Exception as e:
                logger.error(f"Camera stop API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/export', methods=['POST'])
        def api_export():
            """API endpoint for exporting system data"""
            try:
                export_path = request.json.get('export_path', 'data/export') if request.is_json else 'data/export'
                self.face_id_system.export_system_data(export_path)
                
                return jsonify({'success': True, 'message': 'System data exported successfully'})
                
            except Exception as e:
                logger.error(f"Export API error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_websockets(self):
        """Setup WebSockets for real-time video stream processing"""
        try:
            from flask_socketio import SocketIO, emit
            self.socketio = SocketIO(self.app, cors_allowed_origins="*")
            self.socketio_available = True
            logger.info("Initializing WebSocket routes on /live_stream...")
            
            @self.socketio.on('connect')
            def handle_connect():
                logger.info("WebSocket client connected to live stream")
                emit('status', {'connected': True, 'msg': 'Connected to FaceID Live Stream'})
                
            @self.socketio.on('frame')
            def handle_frame(data):
                """Handle base64-encoded frame from client"""
                try:
                    if not data or 'image' not in data:
                        emit('error', {'error': 'No image data provided'})
                        return
                    
                    image_data = data['image']
                    if image_data.startswith('data:image'):
                        image_data = image_data.split(',')[1]
                    
                    image_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        emit('error', {'error': 'Failed to decode image frame'})
                        return
                    
                    # Run recognition on all faces in the frame
                    results = self.face_id_system.recognize_faces(image)
                    
                    # Format results to be JSON serializable
                    formatted_results = []
                    for r in results:
                        formatted_results.append({
                            'person_name': r.get('person_name'),
                            'confidence': float(r.get('confidence', 0.0)),
                            'bbox': r.get('bbox'),
                            'recognition_method': r.get('recognition_method'),
                            'is_live': r.get('is_live', True),
                            'liveness_score': float(r.get('liveness_score', 1.0))
                        })
                    
                    emit('response', {'results': formatted_results})
                    
                except Exception as e:
                    logger.error(f"WebSocket frame processing error: {e}")
                    emit('error', {'error': str(e)})
                    
        except ImportError:
            self.socketio_available = False
            self.socketio = None
            logger.warning("flask_socketio is not installed. WebSockets functionality will be bypassed. Run 'pip install flask-socketio' to enable.")

    def run(self, debug=False):
        """Run the web interface"""
        try:
            logger.info(f"Starting Face ID Web Interface on {self.host}:{self.port}")
            if hasattr(self, 'socketio_available') and self.socketio_available and self.socketio:
                self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)
            else:
                self.app.run(host=self.host, port=self.port, debug=debug)
        except Exception as e:
            logger.error(f"Web interface failed to start: {e}")

def create_web_interface(face_id_system, host='0.0.0.0', port=5000):
    """Create and return web interface instance"""
    return FaceIDWebInterface(face_id_system, host, port)

if __name__ == "__main__":
    # Test web interface
    print("Web Interface Module Test")
    
    # This would normally be initialized with a FaceIDSystem
    # For testing, we'll just show the structure
    print("Web interface module initialized successfully")
    print("To run the web interface, use:")
    print("web_interface = create_web_interface(face_id_system)")
    print("web_interface.run()")
