"""
Face Detection Module
Supports multiple face detection backends: MTCNN, RetinaFace, OpenCV, Dlib
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector(ABC):
    """Abstract base class for face detectors"""
    
    @abstractmethod
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries containing face information:
            - 'bbox': bounding box coordinates (x, y, w, h)
            - 'landmarks': facial landmarks (if available)
            - 'confidence': detection confidence score
        """
        pass

class MTCNNDetector(FaceDetector):
    """MTCNN face detector implementation"""
    
    def __init__(self):
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            logger.info("MTCNN detector initialized successfully")
        except ImportError:
            logger.error("MTCNN not available. Please install: pip install mtcnn")
            raise ImportError("MTCNN not available. Install with: pip install mtcnn")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using MTCNN"""
        try:
            # MTCNN expects RGB images
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            results = self.detector.detect_faces(rgb_image)
            faces = []
            
            for result in results:
                if result['confidence'] > 0.9:  # Confidence threshold
                    bbox = result['box']
                    landmarks = result['keypoints']
                    
                    faces.append({
                        'bbox': (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                        'landmarks': landmarks,
                        'confidence': result['confidence']
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"MTCNN detection failed: {e}")
            return []

class RetinaFaceDetector(FaceDetector):
    """RetinaFace detector using InsightFace"""
    
    def __init__(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("RetinaFace detector initialized successfully")
        except ImportError:
            logger.error("InsightFace not available. Please install: pip install insightface")
            raise ImportError("InsightFace not available. Install with: pip install insightface")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using RetinaFace"""
        try:
            faces = self.app.get(image)
            detected_faces = []
            
            for face in faces:
                bbox = face.bbox.astype(int)
                landmarks = face.kps
                
                detected_faces.append({
                    'bbox': (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
                    'landmarks': landmarks,
                    'confidence': face.det_score
                })
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"RetinaFace detection failed: {e}")
            return []

class OpenCVDetector(FaceDetector):
    """Enhanced OpenCV Haar Cascade face detector with improved accuracy"""
    
    def __init__(self):
        try:
            # Load multiple Haar cascade classifiers for better accuracy
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Load additional cascade for profile faces
            profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            self.profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
            
            if self.face_cascade.empty():
                raise ValueError("Failed to load Haar cascade classifier")
            
            logger.info("Enhanced OpenCV Haar Cascade detector initialized successfully")
        except Exception as e:
            logger.error(f"OpenCV detector initialization failed: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using enhanced OpenCV Haar Cascade with validation"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Apply histogram equalization for better detection
            gray = cv2.equalizeHist(gray)
            
            # Detect frontal faces with stricter parameters
            frontal_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,      # Stricter scaling
                minNeighbors=5,       # Require more neighbors for validation
                minSize=(30, 30),     # Reasonable minimum face size
                maxSize=(300, 300),   # Maximum face size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Detect profile faces
            profile_faces = []
            if not self.profile_cascade.empty():
                profile_faces = self.profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    maxSize=(300, 300),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            
            # Combine and validate faces
            all_faces = []
            
            # Process frontal faces
            for (x, y, w, h) in frontal_faces:
                if self._validate_face_region(image, x, y, w, h):
                    all_faces.append({
                        'bbox': (x, y, w, h),
                        'landmarks': None,
                        'confidence': self._calculate_face_confidence(gray, x, y, w, h)
                    })
            
            # Process profile faces
            for (x, y, w, h) in profile_faces:
                if self._validate_face_region(image, x, y, w, h):
                    all_faces.append({
                        'bbox': (x, y, w, h),
                        'landmarks': None,
                        'confidence': self._calculate_face_confidence(gray, x, y, w, h) * 0.8  # Lower confidence for profile
                    })
            
            # Remove overlapping faces (keep the one with higher confidence)
            all_faces = self._remove_overlapping_faces(all_faces)
            
            # Sort by confidence and return top faces
            all_faces.sort(key=lambda x: x['confidence'], reverse=True)
            
            return all_faces[:3]  # Return maximum 3 faces
            
        except Exception as e:
            logger.error(f"OpenCV detection failed: {e}")
            return []
    
    def _validate_face_region(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        """Validate if the detected region is likely a human face"""
        try:
            # Check aspect ratio (faces are roughly square to slightly rectangular)
            aspect_ratio = w / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.4:
                return False
            
            # Check size constraints
            if w < 20 or h < 20 or w > 400 or h > 400:
                return False
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            if face_region.size == 0:
                return False
            
            # Check for sufficient contrast (faces should have good contrast)
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
            contrast = np.std(gray_face)
            if contrast < 20:  # Too low contrast
                return False
            
            # Check for reasonable brightness distribution
            mean_brightness = np.mean(gray_face)
            if mean_brightness < 30 or mean_brightness > 200:  # Too dark or too bright
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Face validation failed: {e}")
            return False
    
    def _calculate_face_confidence(self, gray_image: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """Calculate confidence score for detected face"""
        try:
            face_region = gray_image[y:y+h, x:x+w]
            
            # Calculate various confidence factors
            contrast_score = min(np.std(face_region) / 50.0, 1.0)  # Normalize contrast
            
            # Check for symmetry (faces are roughly symmetric)
            left_half = face_region[:, :w//2]
            right_half = face_region[:, w//2:]
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Resize to same dimensions for comparison
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            
            symmetry_score = 1.0 - np.mean(np.abs(left_half.astype(float) - right_half_flipped.astype(float))) / 255.0
            
            # Combine scores
            confidence = (contrast_score * 0.6 + symmetry_score * 0.4)
            
            return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5  # Default confidence
    
    def _remove_overlapping_faces(self, faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping face detections, keeping the one with higher confidence"""
        if len(faces) <= 1:
            return faces
        
        # Sort by confidence (highest first)
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_faces = []
        for face in faces:
            x1, y1, w1, h1 = face['bbox']
            
            # Check overlap with already selected faces
            overlaps = False
            for selected_face in filtered_faces:
                x2, y2, w2, h2 = selected_face['bbox']
                
                # Calculate intersection area
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                intersection_area = x_overlap * y_overlap
                
                # Calculate union area
                area1 = w1 * h1
                area2 = w2 * h2
                union_area = area1 + area2 - intersection_area
                
                # Calculate overlap ratio
                overlap_ratio = intersection_area / union_area if union_area > 0 else 0
                
                if overlap_ratio > 0.3:  # 30% overlap threshold
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_faces.append(face)
        
        return filtered_faces

class DlibDetector(FaceDetector):
    """Dlib HOG face detector"""
    
    def __init__(self):
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            logger.info("Dlib detector initialized successfully")
        except ImportError:
            logger.error("Dlib not available. Please install: pip install dlib")
            raise ImportError("Dlib not available. Install with: pip install dlib")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using Dlib"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            faces = self.detector(gray)
            detected_faces = []
            
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                detected_faces.append({
                    'bbox': (x, y, w, h),
                    'landmarks': None,
                    'confidence': face.confidence() if hasattr(face, 'confidence') else 1.0
                })
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"Dlib detection failed: {e}")
            return []

class FaceDetectionManager:
    """Manager class for face detection with multiple backends"""
    
    def __init__(self, detector_type: str = 'mtcnn'):
        """
        Initialize face detection manager
        
        Args:
            detector_type: Type of detector ('mtcnn', 'retinaface', 'opencv', 'dlib')
        """
        self.detector_type = detector_type
        self.detector = self._initialize_detector(detector_type)
    
    def _initialize_detector(self, detector_type: str) -> FaceDetector:
        """Initialize the specified detector"""
        detector_map = {
            'mtcnn': MTCNNDetector,
            'retinaface': RetinaFaceDetector,
            'opencv': OpenCVDetector,
            'dlib': DlibDetector
        }
        
        if detector_type not in detector_map:
            raise ValueError(f"Unsupported detector type: {detector_type}")
        
        return detector_map[detector_type]()
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in the image"""
        return self.detector.detect_faces(image)
    
    def extract_face(self, image: np.ndarray, face_info: Dict[str, Any], 
                    target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
        """
        Extract and align face from image
        
        Args:
            image: Input image
            face_info: Face information from detection
            target_size: Target size for extracted face
            
        Returns:
            Extracted and aligned face image
        """
        x, y, w, h = face_info['bbox']
        
        # Extract face region
        face = image[y:y+h, x:x+w]
        
        # Resize to target size
        face_resized = cv2.resize(face, target_size)
        
        # Apply alignment if landmarks are available
        if face_info['landmarks'] is not None:
            face_resized = self._align_face(face_resized, face_info['landmarks'])
        
        return face_resized
    
    def _align_face(self, face: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align face using landmarks"""
        # Simple alignment - can be enhanced with more sophisticated methods
        return face
    
    def draw_faces(self, image: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes around detected faces"""
        result_image = image.copy()
        
        for face in faces:
            x, y, w, h = face['bbox']
            confidence = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            cv2.putText(result_image, f'{confidence:.2f}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result_image

# Factory function for easy detector creation
def create_face_detector(detector_type: str = 'mtcnn') -> FaceDetectionManager:
    """Create a face detector instance"""
    return FaceDetectionManager(detector_type)

if __name__ == "__main__":
    # Test the face detection
    import os
    
    # Create a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "Test Image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    
    # Test different detectors
    detectors = ['opencv', 'mtcnn', 'retinaface', 'dlib']
    
    for detector_type in detectors:
        try:
            detector = create_face_detector(detector_type)
            faces = detector.detect_faces(test_image)
            print(f"{detector_type}: Found {len(faces)} faces")
        except Exception as e:
            print(f"{detector_type}: Failed - {e}")
