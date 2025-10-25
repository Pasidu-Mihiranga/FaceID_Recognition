"""
Main Face ID System
Integrates all components for complete face recognition system
"""

import cv2
import numpy as np
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import threading
import time

# Import our modules
from src.face_detection import create_face_detector
from src.face_recognition import create_face_recognizer
from src.database import FaceDatabase
from src.continuous_learning import ContinuousLearningManager
from src.face_identity_manager import FaceIdentityManager
from src.advanced_face_processor import AdvancedFaceProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceIDSystem:
    """Main Face ID System integrating all components"""
    
    def __init__(self, 
                 detector_type: str = 'opencv',
                 recognition_model: str = 'deepface',
                     recognition_threshold: float = 0.15,
                 learning_threshold: float = 0.7):
        """
        Initialize the Face ID System
        
        Args:
            detector_type: Face detector type ('mtcnn', 'retinaface', 'opencv', 'dlib')
            recognition_model: Recognition model type ('arcface', 'facenet', 'vggface')
            recognition_threshold: Threshold for face recognition
            learning_threshold: Threshold for continuous learning
        """
        logger.info("Initializing Face ID System...")
        
        # Initialize components
        self.face_detector = create_face_detector(detector_type)
        self.face_recognizer = create_face_recognizer(recognition_model, recognition_threshold)
        self.database = FaceDatabase()
        self.learning_manager = ContinuousLearningManager(
            self.face_recognizer, 
            self.database,
            learning_threshold
        )
        
        # Initialize Face Identity Manager for robust recognition
        self.identity_manager = FaceIdentityManager()
        
        # Initialize Advanced Face Processor
        self.advanced_processor = AdvancedFaceProcessor()
        
        # System state
        self.is_running = False
        self.camera_thread = None
        self.camera = None
        
        # Recognition statistics
        self.recognition_stats = {
            'total_detections': 0,
            'successful_recognitions': 0,
            'unknown_faces': 0,
            'last_recognition_time': None
        }
        
        logger.info("Face ID System initialized successfully")
    
    def register_person(self, image_path: str, person_name: str, 
                      metadata: Optional[Dict] = None) -> bool:
        """
        Register a new person in the system
        
        Args:
            image_path: Path to the person's image
            person_name: Name of the person
            metadata: Additional metadata about the person
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            logger.info(f"Registering person: {person_name}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            if not faces:
                logger.warning("No faces detected in the image - trying with more lenient parameters")
                
                # Try with more lenient OpenCV parameters
                if hasattr(self.face_detector, 'face_cascade'):
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                    faces = self.face_detector.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.01,  # Very lenient
                        minNeighbors=1,    # Very lenient
                        minSize=(10, 10),  # Very small minimum size
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    # Convert to expected format
                    detected_faces = []
                    for (x, y, w, h) in faces:
                        detected_faces.append({
                            'bbox': (x, y, w, h),
                            'landmarks': None,
                            'confidence': 1.0
                        })
                    faces = detected_faces
                
                if not faces:
                    logger.warning("No faces detected even with lenient parameters - using entire image as fallback")
                    # Create a fallback face detection using the entire image
                    h, w = image.shape[:2]
                    faces = [{
                        'bbox': (0, 0, w, h),
                        'landmarks': None,
                        'confidence': 0.5  # Lower confidence for fallback
                    }]
            
            # Use the first detected face
            face_info = faces[0]
            face_image = self.face_detector.extract_face(image, face_info)
            
            # Extract embedding
            embedding = self.face_recognizer.recognizer.extract_embedding(face_image)
            
            # Add person to database
            person_id = self.database.add_person(person_name, metadata)
            
            # Save face image to database
            face_id = self.database.add_face_image(
                person_id=person_id,
                image_path=image_path,
                embedding=embedding,
                face_bbox=face_info['bbox'],
                landmarks=face_info['landmarks'],
                confidence=face_info['confidence']
            )
            
            # Update recognition manager
            self.face_recognizer.register_face(face_image, person_name)
            
            logger.info(f"Successfully registered {person_name} (Person ID: {person_id}, Face ID: {face_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register person {person_name}: {e}")
            return False
    
    def register_person_from_face_image(self, face_image: np.ndarray, person_name: str, 
                                      image_id: str = None) -> bool:
        """
        Register a new person in the system directly from a face image (numpy array).
        This is useful for video-based registration where images are extracted in-memory.
        
        Args:
            face_image: The pre-extracted face image (numpy array).
            person_name: Name of the person.
            image_id: Optional unique ID for the image, useful for tracking.
            
        Returns:
            True if registration successful, False otherwise.
        """
        try:
            logger.info(f"Registering person {person_name} from in-memory face image.")
            
            # Extract embedding
            embedding = self.face_recognizer.recognizer.extract_embedding(face_image)
            
            # Add person to database if not exists
            person_data = self.database.get_person_by_name(person_name)
            if not person_data:
                person_id = self.database.add_person(person_name)
            else:
                person_id = person_data['id']
            
            # Save face image (embedding) to database
            # For in-memory images, we don't have a file path, so we use a placeholder
            face_id = self.database.add_face_image(
                person_id=person_id,
                image_path=f"in_memory_video_frame_{image_id or datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", # Placeholder
                embedding=embedding,
                face_bbox=(0,0,face_image.shape[1], face_image.shape[0]), # Bbox is full image
                landmarks=None,
                confidence=1.0 # Assume high confidence for extracted face
            )
            
            # Update recognition manager's in-memory database
            self.face_recognizer.register_face(face_image, person_name)
            
            logger.info(f"Successfully registered face for {person_name} (Person ID: {person_id}, Face ID: {face_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register person {person_name} from face image: {e}")
            return False
    
    def _enhance_image_for_recognition(self, face_image: np.ndarray) -> np.ndarray:
        """
        Enhance face image for better recognition, especially for phone photos
        
        Args:
            face_image: Input face image
            
        Returns:
            Enhanced face image
        """
        try:
            enhanced = face_image.copy()
            
            # 1. Resolution normalization (ensure consistent size)
            target_size = (160, 160)
            if enhanced.shape[:2] != target_size:
                enhanced = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # 2. Lighting normalization
            enhanced = self._normalize_lighting_for_recognition(enhanced)
            
            # 3. Contrast enhancement
            enhanced = self._enhance_contrast(enhanced)
            
            # 4. Noise reduction
            enhanced = self._reduce_noise(enhanced)
            
            # 5. Sharpening
            enhanced = self._sharpen_image(enhanced)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return face_image
    
    def _normalize_lighting_for_recognition(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize lighting for recognition
        
        Args:
            image: Input image
            
        Returns:
            Lighting normalized image
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Apply CLAHE for adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_normalized = clahe.apply(l_channel)
            
            # Update LAB image
            lab[:, :, 0] = l_normalized
            
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            logger.error(f"Lighting normalization failed: {e}")
            return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast of the image
        
        Args:
            image: Input image
            
        Returns:
            Contrast enhanced image
        """
        try:
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Apply gamma correction
            gamma = 1.1
            l_enhanced = np.power(l_channel / 255.0, 1.0 / gamma) * 255.0
            l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)
            
            lab[:, :, 0] = l_enhanced
            
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        except Exception as e:
            logger.error(f"Contrast enhancement failed: {e}")
            return image
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce noise in the image
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        try:
            # Apply bilateral filter for noise reduction while preserving edges
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            return denoised
            
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            return image
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen the image
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        try:
            # Create sharpening kernel
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            
            # Apply sharpening
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original to avoid over-sharpening
            result = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            
            return result
            
        except Exception as e:
            logger.error(f"Image sharpening failed: {e}")
            return image
    
    def recognize_face(self, image: np.ndarray) -> Tuple[Optional[str], float, Dict]:
        """
        Recognize a face in the given image
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (person_name, confidence, face_info)
        """
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            if not faces:
                return None, 0.0, {}
            
            # Use the first detected face
            face_info = faces[0]
            
            # Process face with advanced pipeline
            processed_data = self.advanced_processor.process_face_for_recognition(image, face_info['bbox'])
            processed_face = processed_data['processed_face']
            quality_metrics = processed_data['quality_metrics']
            lighting_condition = processed_data['lighting_condition']
            image_quality = processed_data['image_quality']
            
            # Extract embedding for identity matching
            try:
                # Enhanced preprocessing for phone photos
                enhanced_face_image = self._enhance_image_for_recognition(processed_face)
                embedding = self.face_recognizer.recognizer.extract_embedding(enhanced_face_image)
                
                # Normalize features
                normalized_embedding = self.advanced_processor.normalize_features(embedding)
                
            except Exception as e:
                logger.warning(f"Failed to extract embedding: {e}")
                # Fallback to traditional recognition
                person_name, confidence = self.face_recognizer.recognize_face(processed_face)
                return person_name, confidence, face_info
            
            # Try identity-based recognition first
            identity_match, identity_confidence, match_info = self.identity_manager.match_identity(normalized_embedding)
            
            # Apply adaptive thresholding
            adaptive_threshold, threshold_reason = self.advanced_processor.adaptive_thresholding(
                identity_confidence, quality_metrics
            )
            
            if identity_match and identity_confidence >= adaptive_threshold:
                logger.info(f"Identity match: {identity_match} (confidence: {identity_confidence:.3f}, threshold: {adaptive_threshold:.3f})")
                person_name = identity_match
                confidence = identity_confidence
                face_info['recognition_method'] = 'identity_adaptive'
                face_info['identity_quality'] = match_info.get('identity_quality', 0.0)
                face_info['adaptive_threshold'] = adaptive_threshold
                face_info['threshold_reason'] = threshold_reason
                face_info['lighting_condition'] = lighting_condition
                face_info['image_quality'] = image_quality
            else:
                # Fallback to traditional recognition
                person_name, confidence = self.face_recognizer.recognize_face(processed_face)
                face_info['recognition_method'] = 'traditional_fallback'
                face_info['adaptive_threshold'] = adaptive_threshold
                face_info['threshold_reason'] = threshold_reason
                face_info['lighting_condition'] = lighting_condition
                face_info['image_quality'] = image_quality
                if identity_match:
                    face_info['identity_confidence'] = identity_confidence
            
            # Update performance statistics
            recognition_result = {
                'success': person_name is not None,
                'confidence': confidence,
                'lighting_condition': lighting_condition,
                'image_quality': image_quality,
                'recognition_method': face_info['recognition_method']
            }
            self.advanced_processor.update_performance_stats(recognition_result)
            
            # Update statistics
            self.recognition_stats['total_detections'] += 1
            if person_name:
                self.recognition_stats['successful_recognitions'] += 1
            else:
                self.recognition_stats['unknown_faces'] += 1
            self.recognition_stats['last_recognition_time'] = datetime.now()
            
            # Log recognition
            try:
                person_id = None
                if person_name:
                    person_data = self.database.get_person_by_name(person_name)
                    if person_data:
                        person_id = person_data['id']
                
                self.database.log_recognition(
                    person_id=person_id,
                    image_path="",  # No image path for live recognition
                    confidence=confidence,
                    is_unknown=(person_name is None)
                )
            except Exception as e:
                logger.warning(f"Failed to log recognition: {e}")
            
            # Continuous learning enabled - auto training during recognition
            if person_name and confidence > self.learning_manager.learning_threshold:
                self.learning_manager.process_recognition_result(processed_face, person_name, confidence)
            
            # Convert numpy int32 to Python int for JSON serialization
            if face_info and 'bbox' in face_info:
                bbox = face_info['bbox']
                face_info['bbox'] = tuple(int(x) for x in bbox)
            
            return person_name, confidence, face_info
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return None, 0.0, {}
    
    def start_camera_recognition(self, camera_index: int = 0, 
                               display_window: bool = True) -> bool:
        """
        Start real-time camera recognition
        
        Args:
            camera_index: Camera index (usually 0 for default camera)
            display_window: Whether to display the camera feed
            
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.is_running:
                logger.warning("Camera recognition is already running")
                return False
            
            # Initialize camera
            self.camera = cv2.VideoCapture(camera_index)
            if not self.camera.isOpened():
                logger.error(f"Could not open camera {camera_index}")
                return False
            
            self.is_running = True
            
            # Start camera thread
            self.camera_thread = threading.Thread(
                target=self._camera_recognition_loop,
                args=(display_window,)
            )
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
            logger.info("Camera recognition started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera recognition: {e}")
            return False
    
    def stop_camera_recognition(self):
        """Stop real-time camera recognition"""
        try:
            self.is_running = False
            
            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=5)
            
            if self.camera:
                self.camera.release()
            
            cv2.destroyAllWindows()
            logger.info("Camera recognition stopped")
            
        except Exception as e:
            logger.error(f"Error stopping camera recognition: {e}")
    
    def _camera_recognition_loop(self, display_window: bool = True):
        """Optimized camera recognition loop with frame skipping and caching"""
        try:
            frame_count = 0
            last_recognition_time = 0
            recognition_cooldown = 0.5  # Process every 0.5 seconds
            cached_result = None
            
            while self.is_running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Skip frames for faster processing (process every 3rd frame)
                if frame_count % 3 != 0:
                    # Use cached result if available
                    if cached_result:
                        person_name, confidence, face_info = cached_result
                    else:
                        person_name, confidence, face_info = None, 0.0, {}
                else:
                    # Process frame for recognition
                    if current_time - last_recognition_time >= recognition_cooldown:
                        person_name, confidence, face_info = self.recognize_face(frame)
                        cached_result = (person_name, confidence, face_info)
                        last_recognition_time = current_time
                    else:
                        # Use cached result
                        if cached_result:
                            person_name, confidence, face_info = cached_result
                        else:
                            person_name, confidence, face_info = None, 0.0, {}
                
                # Draw results on frame
                if face_info:
                    frame = self._draw_recognition_results(frame, person_name, confidence, face_info)
                
                # Display frame
                if display_window:
                    cv2.imshow('Face ID System', frame)
                    
                    # Check for exit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        self.is_running = False
                        break
                
                # Reduced delay for better responsiveness
                time.sleep(0.016)  # ~60 FPS display, but recognition at ~2 FPS
                
        except Exception as e:
            logger.error(f"Camera recognition loop error: {e}")
        finally:
            self.is_running = False
    
    def _draw_recognition_results(self, frame: np.ndarray, person_name: Optional[str], 
                                confidence: float, face_info: Dict) -> np.ndarray:
        """Draw recognition results on the frame"""
        try:
            x, y, w, h = face_info['bbox']
            
            # Draw bounding box
            color = (0, 255, 0) if person_name else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = person_name if person_name else "Unknown"
            label_text = f"{label} ({confidence:.3f})"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                        (x + text_size[0], y), color, -1)
            
            # Draw text
            cv2.putText(frame, label_text, (x, y - 5), 
                       font, font_scale, (255, 255, 255), thickness)
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to draw recognition results: {e}")
            return frame
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            # Get recognition stats
            recognition_stats = self.recognition_stats.copy()
            
            # Get learning stats
            learning_stats = self.learning_manager.get_learning_stats()
            
            # Get database stats
            all_persons = self.database.get_all_persons()
            total_persons = len(all_persons)
            total_images = sum(person['total_images'] for person in all_persons)
            
            # Get recent recognition stats
            recent_stats = self.database.get_recognition_stats(days=7)
            
            return {
                'recognition_stats': recognition_stats,
                'learning_stats': learning_stats,
                'database_stats': {
                    'total_persons': total_persons,
                    'total_images': total_images,
                    'persons': [{'name': p['name'], 'images': p['total_images']} for p in all_persons]
                },
                'recent_stats': recent_stats,
                'system_status': {
                    'is_running': self.is_running,
                    'detector_type': self.face_detector.detector_type,
                    'recognition_model': self.face_recognizer.model_type,
                    'recognition_threshold': self.face_recognizer.threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def export_system_data(self, export_path: str):
        """Export system data for backup/analysis"""
        try:
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # Export learning data
            learning_export_path = os.path.join(os.path.dirname(export_path), "learning_data.json")
            self.learning_manager.export_learning_data(learning_export_path)
            
            # Export database backup
            db_backup_path = os.path.join(os.path.dirname(export_path), "database_backup.db")
            self.database.backup_database(db_backup_path)
            
            # Export system stats
            stats = self.get_system_stats()
            stats_export_path = os.path.join(os.path.dirname(export_path), "system_stats.json")
            
            import json
            with open(stats_export_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            logger.info(f"System data exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export system data: {e}")
    
    def cleanup_system(self):
        """Clean up system resources"""
        try:
            # Stop camera recognition
            self.stop_camera_recognition()
            
            # Stop learning manager
            self.learning_manager.stop_background_learning()
            
            # Cleanup old logs
            self.database.cleanup_old_logs(days=90)
            
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"System cleanup failed: {e}")

def main():
    """Main function for testing the Face ID System"""
    try:
        # Initialize system
        face_id = FaceIDSystem(
            detector_type='opencv',  # Use OpenCV for better compatibility
            recognition_model='arcface',
            recognition_threshold=0.7
        )
        
        print("Face ID System initialized successfully!")
        print("Available commands:")
        print("1. Register a person: face_id.register_person('image_path', 'person_name')")
        print("2. Start camera recognition: face_id.start_camera_recognition()")
        print("3. Get system stats: face_id.get_system_stats()")
        print("4. Exit: Press 'q' in camera window or Ctrl+C")
        
        # Example usage
        # face_id.register_person("path/to/image.jpg", "John Doe")
        # face_id.start_camera_recognition()
        
        # Keep the system running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down Face ID System...")
            face_id.cleanup_system()
            
    except Exception as e:
        logger.error(f"Face ID System failed to start: {e}")

if __name__ == "__main__":
    main()
