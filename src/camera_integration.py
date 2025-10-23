"""
Camera Integration Module
Handles real-time camera feed and face recognition
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any
from queue import Queue
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraManager:
    """Manages camera operations and real-time processing"""
    
    def __init__(self, camera_index: int = 0, 
                 frame_width: int = 640, 
                 frame_height: int = 480,
                 fps: int = 30):
        """
        Initialize camera manager
        
        Args:
            camera_index: Camera device index
            frame_width: Desired frame width
            frame_height: Desired frame height
            fps: Desired frames per second
        """
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        self.camera = None
        self.is_running = False
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=5)  # Buffer for frames
        
        # Callbacks
        self.frame_callback = None
        self.recognition_callback = None
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'fps_actual': 0,
            'last_frame_time': None
        }
    
    def initialize_camera(self) -> bool:
        """Initialize the camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                logger.error(f"Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual properties
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def start_capture(self) -> bool:
        """Start camera capture"""
        try:
            if not self.camera:
                if not self.initialize_camera():
                    return False
            
            if self.is_running:
                logger.warning("Camera capture is already running")
                return False
            
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            logger.info("Camera capture started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera capture: {e}")
            return False
    
    def stop_capture(self):
        """Stop camera capture"""
        try:
            self.is_running = False
            
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=5)
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            logger.info("Camera capture stopped")
            
        except Exception as e:
            logger.error(f"Error stopping camera capture: {e}")
    
    def _capture_loop(self):
        """Main camera capture loop"""
        try:
            frame_time = 1.0 / self.fps
            last_time = time.time()
            
            while self.is_running:
                current_time = time.time()
                
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Update statistics
                self.stats['frames_captured'] += 1
                self.stats['last_frame_time'] = current_time
                
                # Calculate actual FPS
                if self.stats['frames_captured'] % 30 == 0:  # Update every 30 frames
                    elapsed = current_time - last_time
                    self.stats['fps_actual'] = 30 / elapsed if elapsed > 0 else 0
                    last_time = current_time
                
                # Add frame to queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                # Call frame callback if set
                if self.frame_callback:
                    try:
                        self.frame_callback(frame)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")
                
                # Maintain frame rate
                elapsed = time.time() - current_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Camera capture loop error: {e}")
        finally:
            self.is_running = False
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the queue"""
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
            return None
        except:
            return None
    
    def set_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Set callback function for new frames"""
        self.frame_callback = callback
    
    def set_recognition_callback(self, callback: Callable[[np.ndarray, Dict], None]):
        """Set callback function for recognition results"""
        self.recognition_callback = callback
    
    def get_camera_stats(self) -> Dict[str, Any]:
        """Get camera statistics"""
        return self.stats.copy()
    
    def save_frame(self, frame: np.ndarray, filename: str) -> bool:
        """Save a frame to file"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            success = cv2.imwrite(filename, frame)
            if success:
                logger.info(f"Frame saved to {filename}")
            return success
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return False

class RealTimeFaceRecognition:
    """Real-time face recognition with camera integration"""
    
    def __init__(self, face_id_system, camera_index: int = 0):
        """
        Initialize real-time face recognition
        
        Args:
            face_id_system: FaceIDSystem instance
            camera_index: Camera device index
        """
        self.face_id_system = face_id_system
        self.camera_manager = CameraManager(camera_index)
        
        self.is_running = False
        self.display_window = True
        self.save_unknown_faces = False
        self.unknown_faces_dir = "data/unknown_faces"
        
        # Recognition settings
        self.min_confidence = 0.5
        self.recognition_cooldown = 2.0  # seconds
        self.last_recognition_time = 0
        
        # Statistics
        self.session_stats = {
            'recognitions': 0,
            'unknown_faces': 0,
            'session_start': None
        }
    
    def start_recognition(self, display_window: bool = True, 
                         save_unknown: bool = False) -> bool:
        """
        Start real-time face recognition
        
        Args:
            display_window: Whether to display camera feed
            save_unknown: Whether to save unknown faces
            
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.is_running:
                logger.warning("Real-time recognition is already running")
                return False
            
            self.display_window = display_window
            self.save_unknown_faces = save_unknown
            
            # Create unknown faces directory if needed
            if save_unknown:
                os.makedirs(self.unknown_faces_dir, exist_ok=True)
            
            # Set up camera callbacks
            self.camera_manager.set_frame_callback(self._process_frame)
            
            # Start camera capture
            if not self.camera_manager.start_capture():
                return False
            
            self.is_running = True
            self.session_stats['session_start'] = time.time()
            
            logger.info("Real-time face recognition started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start real-time recognition: {e}")
            return False
    
    def stop_recognition(self):
        """Stop real-time face recognition"""
        try:
            self.is_running = False
            self.camera_manager.stop_capture()
            
            if self.display_window:
                cv2.destroyAllWindows()
            
            logger.info("Real-time face recognition stopped")
            
        except Exception as e:
            logger.error(f"Error stopping real-time recognition: {e}")
    
    def _process_frame(self, frame: np.ndarray):
        """Process a captured frame"""
        try:
            if not self.is_running:
                return
            
            # Check cooldown
            current_time = time.time()
            if current_time - self.last_recognition_time < self.recognition_cooldown:
                if self.display_window:
                    self._display_frame(frame, "Processing...")
                return
            
            # Recognize faces
            person_name, confidence, face_info = self.face_id_system.recognize_face(frame)
            
            # Update statistics
            if person_name:
                self.session_stats['recognitions'] += 1
            else:
                self.session_stats['unknown_faces'] += 1
                
                # Save unknown face if enabled
                if self.save_unknown_faces and face_info:
                    self._save_unknown_face(frame, face_info)
            
            # Draw results
            if face_info:
                frame = self._draw_recognition_results(frame, person_name, confidence, face_info)
            
            # Display frame
            if self.display_window:
                self._display_frame(frame, person_name or "Unknown")
            
            # Update last recognition time
            self.last_recognition_time = current_time
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
    
    def _draw_recognition_results(self, frame: np.ndarray, person_name: Optional[str], 
                                confidence: float, face_info: Dict) -> np.ndarray:
        """Draw recognition results on frame"""
        try:
            x, y, w, h = face_info['bbox']
            
            # Choose color based on recognition result
            if person_name:
                color = (0, 255, 0)  # Green for recognized
                label = f"{person_name} ({confidence:.3f})"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown ({confidence:.3f})"
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                        (x + text_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Draw session stats
            stats_text = f"Recognitions: {self.session_stats['recognitions']} | Unknown: {self.session_stats['unknown_faces']}"
            cv2.putText(frame, stats_text, (10, 30), 
                       font, 0.5, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to draw recognition results: {e}")
            return frame
    
    def _display_frame(self, frame: np.ndarray, status: str):
        """Display frame in window"""
        try:
            # Add status text
            cv2.putText(frame, f"Status: {status}", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Face ID System - Real Time', frame)
            
            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                self.stop_recognition()
                
        except Exception as e:
            logger.error(f"Display error: {e}")
    
    def _save_unknown_face(self, frame: np.ndarray, face_info: Dict):
        """Save unknown face image"""
        try:
            x, y, w, h = face_info['bbox']
            face_crop = frame[y:y+h, x:x+w]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.unknown_faces_dir, f"unknown_{timestamp}.jpg")
            
            cv2.imwrite(filename, face_crop)
            logger.info(f"Unknown face saved to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save unknown face: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        session_duration = 0
        if self.session_stats['session_start']:
            session_duration = time.time() - self.session_stats['session_start']
        
        return {
            'session_stats': self.session_stats.copy(),
            'session_duration': session_duration,
            'camera_stats': self.camera_manager.get_camera_stats(),
            'is_running': self.is_running
        }
    
    def register_unknown_face(self, face_image: np.ndarray, person_name: str) -> bool:
        """Register an unknown face as a new person"""
        try:
            # Save the face image temporarily
            temp_path = f"data/temp_face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(temp_path, face_image)
            
            # Register the person
            success = self.face_id_system.register_person(temp_path, person_name)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if success:
                logger.info(f"Successfully registered unknown face as {person_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to register unknown face: {e}")
            return False

if __name__ == "__main__":
    # Test camera integration
    print("Camera Integration Module Test")
    
    # This would normally be initialized with a FaceIDSystem
    # For testing, we'll just show the structure
    camera_manager = CameraManager()
    
    if camera_manager.initialize_camera():
        print("Camera initialized successfully")
        camera_manager.stop_capture()
    else:
        print("Camera initialization failed")
    
    print("Camera integration module test completed")
