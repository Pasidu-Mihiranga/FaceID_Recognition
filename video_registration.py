"""
Video Registration Module
Captures user videos during registration and extracts multiple training images
"""

import cv2
import numpy as np
import os
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import threading
from main import FaceIDSystem
from src.face_identity_manager import FaceIdentityManager
from src.face_data_augmentation import FaceDataAugmentation
from src.advanced_face_processor import AdvancedFaceProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoRegistrationSystem:
    """Video-based registration system for capturing multiple face angles"""
    
    def __init__(self, face_id_system: FaceIDSystem):
        """
        Initialize video registration system
        
        Args:
            face_id_system: Initialized FaceIDSystem instance
        """
        self.face_id_system = face_id_system
        self.camera = None
        self.is_recording = False
        self.video_frames = []
        self.face_detections = []
        
        # Initialize Face Identity Manager
        self.identity_manager = FaceIdentityManager()
        
        # Initialize Data Augmentation System
        self.data_augmentation = FaceDataAugmentation(augmentation_factor=4)  # 4x augmentation
        
        # Initialize Advanced Face Processor
        self.advanced_processor = AdvancedFaceProcessor()
        
        # Video settings
        self.video_duration = 15  # seconds
        self.fps = 30
        self.frame_width = 640
        self.frame_height = 480
        
        # Frame quality settings
        self.min_face_size = 100  # minimum face size in pixels
        self.max_frames_to_extract = 50  # Increased from 20 to 50
        self.quality_threshold = 0.7
        
        # Enhanced user guidance for more diverse angles
        self.guidance_phases = [
            {"text": "Look straight at the camera", "duration": 2},
            {"text": "Slowly turn your head to the left (45 degrees)", "duration": 3},
            {"text": "Look straight again", "duration": 1},
            {"text": "Turn your head further left (90 degrees)", "duration": 2},
            {"text": "Look straight again", "duration": 1},
            {"text": "Slowly turn your head to the right (45 degrees)", "duration": 3},
            {"text": "Look straight again", "duration": 1},
            {"text": "Turn your head further right (90 degrees)", "duration": 2},
            {"text": "Look straight and smile", "duration": 2},
            {"text": "Look up slightly (30 degrees)", "duration": 2},
            {"text": "Look down slightly (30 degrees)", "duration": 2},
            {"text": "Tilt your head to the left", "duration": 2},
            {"text": "Tilt your head to the right", "duration": 2},
            {"text": "Look straight - final capture", "duration": 1}
        ]
        
        self.current_phase = 0
        self.phase_start_time = 0
        
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """
        Initialize camera for video capture
        
        Args:
            camera_index: Camera device index
            
        Returns:
            True if camera initialized successfully
        """
        try:
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                logger.error(f"Could not open camera {camera_index}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    def start_video_registration(self, person_name: str, display_window: bool = True) -> bool:
        """
        Start video registration process
        
        Args:
            person_name: Name of the person being registered
            display_window: Whether to show camera feed
            
        Returns:
            True if registration successful
        """
        try:
            if not self.camera:
                if not self.initialize_camera():
                    return False
            
            logger.info(f"Starting video registration for {person_name}")
            
            # Reset state
            self.video_frames = []
            self.face_detections = []
            self.current_phase = 0
            self.phase_start_time = time.time()
            
            # Start recording
            self.is_recording = True
            start_time = time.time()
            
            print(f"\n🎥 Video Registration for {person_name}")
            print("Follow the on-screen instructions:")
            print("Press 'q' to quit, 's' to skip current phase")
            
            while self.is_recording and (time.time() - start_time) < self.video_duration:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Process frame
                processed_frame = self._process_registration_frame(frame, person_name)
                
                # Display frame
                if display_window:
                    cv2.imshow('Video Registration - Follow Instructions', processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.is_recording = False
                        break
                    elif key == ord('s'):
                        self._next_guidance_phase()
                
                # Add frame to collection
                self.video_frames.append(frame.copy())
                
                # Update guidance phase
                self._update_guidance_phase()
            
            # Stop recording
            self.is_recording = False
            
            if display_window:
                cv2.destroyAllWindows()
            
            # Process captured video
            if len(self.video_frames) > 0:
                return self._process_registration_video(person_name)
            else:
                logger.error("No frames captured during registration")
                return False
                
        except Exception as e:
            logger.error(f"Video registration failed: {e}")
            return False
    
    def _process_registration_frame(self, frame: np.ndarray, person_name: str) -> np.ndarray:
        """
        Process a single frame during registration
        
        Args:
            frame: Input frame
            person_name: Name of person being registered
            
        Returns:
            Processed frame with overlays
        """
        try:
            # Detect faces in frame
            faces = self.face_id_system.face_detector.detect_faces(frame)
            
            # Draw face detection
            if faces:
                face_info = faces[0]  # Use first detected face
                x, y, w, h = face_info['bbox']
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Store face detection info
                self.face_detections.append({
                    'bbox': face_info['bbox'],
                    'confidence': face_info['confidence'],
                    'frame_index': len(self.video_frames)
                })
                
                # Draw face size indicator
                face_size = w * h
                cv2.putText(frame, f"Face Size: {face_size}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw guidance text
            guidance_text = self._get_current_guidance_text()
            cv2.putText(frame, guidance_text, (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw progress bar
            progress = len(self.video_frames) / (self.video_duration * self.fps)
            bar_width = int(progress * frame.shape[1])
            cv2.rectangle(frame, (0, frame.shape[0] - 10), (bar_width, frame.shape[0]), (0, 255, 0), -1)
            
            # Draw frame count
            cv2.putText(frame, f"Frames: {len(self.video_frames)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame
    
    def _get_current_guidance_text(self) -> str:
        """Get current guidance text"""
        if self.current_phase < len(self.guidance_phases):
            return self.guidance_phases[self.current_phase]["text"]
        return "Registration complete!"
    
    def _update_guidance_phase(self):
        """Update guidance phase based on time"""
        if self.current_phase < len(self.guidance_phases):
            phase_duration = self.guidance_phases[self.current_phase]["duration"]
            if time.time() - self.phase_start_time >= phase_duration:
                self._next_guidance_phase()
    
    def _next_guidance_phase(self):
        """Move to next guidance phase"""
        self.current_phase += 1
        self.phase_start_time = time.time()
        
        if self.current_phase < len(self.guidance_phases):
            print(f"Phase {self.current_phase + 1}: {self.guidance_phases[self.current_phase]['text']}")
    
    def _process_registration_video(self, person_name: str) -> bool:
        """
        Process captured video and create a robust face identity
        
        Args:
            person_name: Name of person being registered
            
        Returns:
            True if processing successful
        """
        try:
            logger.info(f"Processing video registration for {person_name}")
            logger.info(f"Captured {len(self.video_frames)} frames")
            
            # Extract best frames
            best_frames = self._extract_best_frames()
            
            if not best_frames:
                logger.error("No suitable frames found for training")
                return False
            
            logger.info(f"Extracted {len(best_frames)} high-quality frames")
            
            # Extract face images and embeddings
            face_images = []
            face_embeddings = []
            
            for i, frame_info in enumerate(best_frames):
                frame = frame_info['frame']
                face_bbox = frame_info['face_bbox']
                
                # Process face with advanced pipeline
                processed_data = self.advanced_processor.process_face_for_recognition(frame, face_bbox)
                processed_face = processed_data['processed_face']
                face_images.append(processed_face)
                
                # Extract embedding
                try:
                    embedding = self.face_id_system.face_recognizer.recognizer.extract_embedding(processed_face)
                    face_embeddings.append(embedding)
                    logger.info(f"  ✅ Frame {i+1} embedding extracted")
                except Exception as e:
                    logger.warning(f"  ❌ Frame {i+1} embedding extraction failed: {e}")
            
            if len(face_embeddings) == 0:
                logger.error("No embeddings extracted from frames")
                return False
            
            # Apply data augmentation to increase training data
            logger.info(f"Applying data augmentation to {len(face_images)} original images")
            augmented_face_images = self.data_augmentation.augment_face_images(face_images, person_name)
            
            # Extract embeddings for augmented images
            augmented_embeddings = []
            logger.info(f"Extracting embeddings for {len(augmented_face_images)} augmented images")
            
            for i, augmented_image in enumerate(augmented_face_images):
                try:
                    embedding = self.face_id_system.face_recognizer.recognizer.extract_embedding(augmented_image)
                    augmented_embeddings.append(embedding)
                    if i % 20 == 0:  # Log progress every 20 images
                        logger.info(f"  Processed {i+1}/{len(augmented_face_images)} augmented images")
                except Exception as e:
                    logger.warning(f"  Failed to extract embedding for augmented image {i+1}: {e}")
            
            logger.info(f"Successfully extracted {len(augmented_embeddings)} embeddings from {len(augmented_face_images)} augmented images")
            
            # Get augmentation statistics
            aug_stats = self.data_augmentation.get_augmentation_stats(len(face_images), len(augmented_face_images))
            logger.info(f"Data augmentation stats: {aug_stats}")
            
            # Create robust face identity with augmented data
            logger.info(f"Creating face identity for {person_name} from {len(augmented_embeddings)} embeddings")
            identity = self.identity_manager.create_identity(person_name, augmented_face_images, augmented_embeddings)
            
            if identity:
                logger.info(f"✅ Face identity created for {person_name}")
                logger.info(f"   Identity quality: {identity['identity_quality']:.3f}")
                logger.info(f"   Confidence score: {identity['confidence_score']:.3f}")
                
                # Also register in the traditional database for compatibility (using augmented data)
                self._register_traditional_way(person_name, augmented_face_images, augmented_embeddings)
                
                # Save video frames for reference
                self._save_video_frames(person_name, best_frames)
                
                return True
            else:
                logger.error(f"❌ Failed to create face identity for {person_name}")
                return False
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return False
    
    def _register_traditional_way(self, person_name: str, face_images: List[np.ndarray], 
                                 face_embeddings: List[np.ndarray]):
        """
        Register person in traditional database for compatibility
        
        Args:
            person_name: Name of the person
            face_images: List of face images
            face_embeddings: List of face embeddings
        """
        try:
            logger.info(f"Registering {person_name} in traditional database for compatibility")
            
            # Register each face image
            for i, (face_image, embedding) in enumerate(zip(face_images, face_embeddings)):
                try:
                    success = self.face_id_system.register_person_from_face_image(
                        face_image, person_name, f"video_frame_{i}"
                    )
                    if success:
                        logger.info(f"  ✅ Traditional registration {i+1} successful")
                    else:
                        logger.warning(f"  ❌ Traditional registration {i+1} failed")
                except Exception as e:
                    logger.error(f"  ❌ Traditional registration {i+1} error: {e}")
                    
        except Exception as e:
            logger.error(f"Traditional registration failed: {e}")
    
    def _extract_best_frames(self) -> List[Dict[str, Any]]:
        """
        Extract best frames from captured video
        
        Returns:
            List of best frame information
        """
        try:
            best_frames = []
            
            # Score each frame with face detection
            frame_scores = []
            
            for i, frame in enumerate(self.video_frames):
                # Find corresponding face detection
                face_detection = None
                for detection in self.face_detections:
                    if detection['frame_index'] == i:
                        face_detection = detection
                        break
                
                if face_detection:
                    # Calculate frame quality score
                    score = self._calculate_frame_quality(frame, face_detection)
                    frame_scores.append({
                        'frame_index': i,
                        'frame': frame,
                        'face_bbox': face_detection['bbox'],
                        'score': score
                    })
            
            # Sort by score (highest first)
            frame_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Select diverse frames
            selected_frames = []
            used_angles = set()
            
            for frame_info in frame_scores:
                if len(selected_frames) >= self.max_frames_to_extract:
                    break
                
                # Check if this frame adds diversity
                face_bbox = frame_info['face_bbox']
                x, y, w, h = face_bbox
                
                # Calculate face angle (simplified)
                face_center_x = x + w // 2
                frame_center_x = frame_info['frame'].shape[1] // 2
                angle_category = "center"
                
                if face_center_x < frame_center_x - 50:
                    angle_category = "left"
                elif face_center_x > frame_center_x + 50:
                    angle_category = "right"
                
                # Add frame if it's reasonably good quality (lowered threshold for more frames)
                if frame_info['score'] >= (self.quality_threshold * 0.7):  # Lower threshold from 0.7 to 0.49
                    
                    selected_frames.append({
                        'frame': frame_info['frame'],
                        'face_bbox': frame_info['face_bbox'],
                        'score': frame_info['score'],
                        'angle': angle_category
                    })
                    used_angles.add(angle_category)
            
            # If we don't have enough frames, add best remaining frames (lowered threshold)
            if len(selected_frames) < 10:  # Increased from 5 to 10 minimum frames
                for frame_info in frame_scores:
                    if len(selected_frames) >= self.max_frames_to_extract:
                        break
                    
                    if frame_info['score'] >= (self.quality_threshold * 0.6):  # Lower threshold to 0.42
                        # Check if already selected
                        already_selected = any(
                            f['face_bbox'] == frame_info['face_bbox'] 
                            for f in selected_frames
                        )
                        
                        if not already_selected:
                            selected_frames.append({
                                'frame': frame_info['frame'],
                                'face_bbox': frame_info['face_bbox'],
                                'score': frame_info['score'],
                                'angle': 'additional'
                            })
            
            # If still not enough frames, add more with even lower threshold
            if len(selected_frames) < 30:  # Target at least 30 frames
                logger.info(f"Adding more frames to reach target count. Current: {len(selected_frames)}")
                for frame_info in frame_scores:
                    if len(selected_frames) >= self.max_frames_to_extract:
                        break
                    
                    if frame_info['score'] >= (self.quality_threshold * 0.5):  # Even lower threshold to 0.35
                        # Check if already selected
                        already_selected = any(
                            f['face_bbox'] == frame_info['face_bbox'] 
                            for f in selected_frames
                        )
                        
                        if not already_selected:
                            selected_frames.append({
                                'frame': frame_info['frame'],
                                'face_bbox': frame_info['face_bbox'],
                                'score': frame_info['score'],
                                'angle': 'extra'
                            })
            
            logger.info(f"Selected {len(selected_frames)} best frames from {len(frame_scores)} candidates")
            return selected_frames
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []
    
    def _calculate_frame_quality(self, frame: np.ndarray, face_detection: Dict) -> float:
        """
        Enhanced frame quality calculation with multiple factors
        
        Args:
            frame: Input frame
            face_detection: Face detection information
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            score = 0.0
            
            # Face detection confidence
            score += face_detection['confidence'] * 0.25
            
            # Enhanced face size score (larger faces are better, but not too large)
            x, y, w, h = face_detection['bbox']
            face_area = w * h
            ideal_area = (self.frame_width * self.frame_height) * 0.15  # 15% of frame
            size_ratio = face_area / ideal_area
            if size_ratio < 0.5:  # Too small
                size_score = size_ratio * 0.5
            elif size_ratio > 2.0:  # Too large
                size_score = 1.0 - (size_ratio - 2.0) * 0.2
            else:  # Good size
                size_score = 1.0
            score += size_score * 0.2
            
            # Face position score (centered faces are better)
            center_x = x + w // 2
            center_y = y + h // 2
            frame_center_x = self.frame_width // 2
            frame_center_y = self.frame_height // 2
            
            distance_from_center = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
            max_distance = np.sqrt(self.frame_width**2 + self.frame_height**2) // 2
            position_score = 1.0 - (distance_from_center / max_distance)
            score += position_score * 0.15
            
            # Enhanced image sharpness (Laplacian variance)
            face_region = frame[y:y+h, x:x+w]
            if face_region.size > 0:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                sharpness_score = min(1.0, sharpness / 1000.0)  # Normalize
                score += sharpness_score * 0.15
            
            # Enhanced brightness score (avoid too dark or too bright)
            if face_region.size > 0:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray_face)
                if brightness < 50:  # Too dark
                    brightness_score = brightness / 50.0
                elif brightness > 200:  # Too bright
                    brightness_score = 1.0 - (brightness - 200) / 55.0
                else:  # Good brightness
                    brightness_score = 1.0
                score += brightness_score * 0.1
            
            # Face aspect ratio score (faces should be roughly square)
            aspect_ratio = w / h
            ideal_ratio = 1.0
            ratio_score = 1.0 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
            ratio_score = max(0.0, ratio_score)
            score += ratio_score * 0.1
            
            # Face angle diversity score (reward different angles)
            # This is a simple heuristic - in practice, you'd track actual head angles
            angle_diversity_score = 0.05  # Small bonus for diversity
            score += angle_diversity_score
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Enhanced quality calculation failed: {e}")
            return 0.0
    
    def _save_video_frames(self, person_name: str, best_frames: List[Dict]):
        """
        Save extracted frames for reference
        
        Args:
            person_name: Name of person
            best_frames: List of best frames
        """
        try:
            # Create directory for this person's video frames
            frames_dir = f"data/video_frames/{person_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(frames_dir, exist_ok=True)
            
            # Save each frame
            for i, frame_info in enumerate(best_frames):
                frame = frame_info['frame']
                face_bbox = frame_info['face_bbox']
                
                # Extract and save face
                x, y, w, h = face_bbox
                face_image = frame[y:y+h, x:x+w]
                
                filename = os.path.join(frames_dir, f"frame_{i+1}_{frame_info['angle']}.jpg")
                cv2.imwrite(filename, face_image)
            
            logger.info(f"Saved {len(best_frames)} frames to {frames_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save video frames: {e}")
    
    def process_video_file(self, video_path: str, person_name: str) -> bool:
        """
        Process a video file for registration
        
        Args:
            video_path: Path to the video file
            person_name: Name of person being registered
            
        Returns:
            True if processing successful
        """
        try:
            logger.info(f"Processing video file for {person_name}: {video_path}")
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video properties: {total_frames} frames, {fps} FPS, {duration:.1f}s duration")
            logger.info(f"Video file exists: {os.path.exists(video_path)}")
            logger.info(f"Video file size: {os.path.getsize(video_path)} bytes")
            
            # Extract frames
            self.video_frames = []
            self.face_detections = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 2nd frame for more training data (increased from every 5th)
                if frame_count % 2 == 0:
                    # Detect faces in frame
                    faces = self.face_id_system.face_detector.detect_faces(frame)
                    
                    if faces:
                        face_info = faces[0]  # Use first detected face
                        self.video_frames.append(frame.copy())
                        self.face_detections.append({
                            'bbox': face_info['bbox'],
                            'confidence': face_info['confidence'],
                            'frame_index': len(self.video_frames) - 1
                        })
                
                frame_count += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(self.video_frames)} frames with faces")
            
            if len(self.video_frames) == 0:
                logger.error("No frames with faces found in video")
                return False
            
            # Process the extracted frames
            return self._process_registration_video(person_name)
            
        except Exception as e:
            logger.error(f"Video file processing failed: {e}")
            return False

    def cleanup(self):
        """Clean up resources"""
        try:
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            logger.info("Video registration system cleaned up")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Add method to FaceIDSystem for video registration
def register_person_from_face_image(self, face_image: np.ndarray, person_name: str, 
                                  image_id: str = None) -> bool:
    """
    Register a person from a pre-extracted face image
    
    Args:
        face_image: Pre-extracted face image
        person_name: Name of the person
        image_id: Optional identifier for the image
        
    Returns:
        True if registration successful
    """
    try:
        # Extract embedding
        embedding = self.face_recognizer.recognizer.extract_embedding(face_image)
        
        # Add person to database
        person_id = self.database.add_person(person_name, {"source": "video_registration"})
        
        # Save face image to database
        face_id = self.database.add_face_image(
            person_id=person_id,
            image_path=f"video_frame_{image_id}" if image_id else "video_frame",
            embedding=embedding,
            face_bbox=(0, 0, face_image.shape[1], face_image.shape[0]),
            landmarks=None,
            confidence=1.0
        )
        
        # Update recognition manager
        self.face_recognizer.register_face(face_image, person_name)
        
        logger.info(f"Successfully registered {person_name} from face image (Face ID: {face_id})")
        return True
        
    except Exception as e:
        logger.error(f"Failed to register person from face image: {e}")
        return False

# Add the method to FaceIDSystem class
FaceIDSystem.register_person_from_face_image = register_person_from_face_image

if __name__ == "__main__":
    # Test video registration
    print("Video Registration System Test")
    
    # Initialize Face ID System
    face_id = FaceIDSystem()
    
    # Initialize video registration
    video_reg = VideoRegistrationSystem(face_id)
    
    # Test registration
    person_name = input("Enter person name for video registration: ")
    
    if video_reg.start_video_registration(person_name):
        print(f"✅ Video registration successful for {person_name}")
    else:
        print(f"❌ Video registration failed for {person_name}")
    
    # Cleanup
    video_reg.cleanup()
