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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.app.secret_key = 'face_id_secret_key_2024'
        
        # Enable CORS
        CORS(self.app)
        
        # Configure upload settings
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        self.app.config['UPLOAD_FOLDER'] = 'data/uploads'
        
        # Allowed file extensions
        self.ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        
        # Ensure upload directory exists
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        
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
                    }), 500
                
            except Exception as e:
                logger.error(f"Registration API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/video_register', methods=['POST'])
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
                    }), 500
                
            except Exception as e:
                logger.error(f"Video registration API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recognize', methods=['POST'])
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
                
                result = {
                    'person_name': person_name,
                    'confidence': float(confidence),
                    'face_detected': face_info is not None,
                    'face_bbox': face_bbox
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Recognition API error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recognize_base64', methods=['POST'])
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
                
                result = {
                    'person_name': person_name,
                    'confidence': float(confidence),
                    'face_detected': face_info is not None,
                    'face_bbox': face_bbox
                }
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Base64 recognition API error: {e}")
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
        def api_delete_person(person_id):
            """API endpoint for deleting a person"""
            try:
                person = self.face_id_system.database.get_person_by_id(person_id)
                if not person:
                    return jsonify({'error': 'Person not found'}), 404
                
                success = self.face_id_system.database.delete_person(person_id)
                
                if success:
                    # Also remove from recognition manager
                    self.face_id_system.face_recognizer.remove_person(person['name'])
                    
                    return jsonify({
                        'success': True,
                        'message': f'Successfully deleted {person["name"]}'
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
    
    def run(self, debug=False):
        """Run the web interface"""
        try:
            logger.info(f"Starting Face ID Web Interface on {self.host}:{self.port}")
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
