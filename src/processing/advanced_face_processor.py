"""
Advanced Face Processor
Provides face preprocessing, feature normalization, adaptive thresholding,
and performance tracking for the Face ID recognition pipeline.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)


class AdvancedFaceProcessor:
    """Advanced face processing pipeline for recognition"""
    
    def __init__(self, target_size: Tuple[int, int] = (160, 160)):
        """
        Initialize the advanced face processor.
        
        Args:
            target_size: Target face image size for recognition models
        """
        self.target_size = target_size
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.performance_stats = {
            'total_recognitions': 0,
            'successful_recognitions': 0,
            'avg_confidence': 0.0,
            'lighting_distribution': {},
            'quality_distribution': {},
            'method_distribution': {}
        }
        
        logger.info("AdvancedFaceProcessor initialized")
    
    def process_face_for_recognition(self, image: np.ndarray, 
                                      bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        Process a face region for recognition with quality assessment
        and lighting normalization.
        
        Args:
            image: Full input image (BGR format)
            bbox: Face bounding box as (x, y, w, h)
            
        Returns:
            Dictionary containing:
                - processed_face: Preprocessed face image
                - quality_metrics: Quality assessment scores
                - lighting_condition: Detected lighting condition string
                - image_quality: Overall quality score (0.0 to 1.0)
        """
        try:
            x, y, w, h = bbox
            
            # Ensure valid bbox within image bounds
            img_h, img_w = image.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            if w <= 0 or h <= 0:
                return self._empty_result(image)
            
            # Add margin around face (20% padding)
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img_w, x + w + margin_x)
            y2 = min(img_h, y + h + margin_y)
            
            face_region = image[y1:y2, x1:x2].copy()
            
            if face_region.size == 0:
                return self._empty_result(image)
            
            # Assess quality metrics
            quality_metrics = self._assess_quality(face_region)
            
            # Detect lighting condition
            lighting_condition = self._detect_lighting(face_region)
            
            # Apply lighting normalization
            normalized_face = self._normalize_lighting(face_region, lighting_condition)
            
            # Resize to target size
            processed_face = cv2.resize(normalized_face, self.target_size, 
                                        interpolation=cv2.INTER_AREA)
            
            # Calculate overall quality score
            image_quality = self._calculate_overall_quality(quality_metrics)
            
            return {
                'processed_face': processed_face,
                'quality_metrics': quality_metrics,
                'lighting_condition': lighting_condition,
                'image_quality': image_quality
            }
            
        except Exception as e:
            logger.warning(f"Face processing failed: {e}")
            return self._empty_result(image)
    
    def normalize_features(self, embedding: np.ndarray) -> np.ndarray:
        """
        L2-normalize a face embedding vector.
        
        Args:
            embedding: Raw face embedding vector
            
        Returns:
            L2-normalized embedding vector
        """
        if embedding is None or len(embedding) == 0:
            return embedding
            
        embedding = np.array(embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(embedding)
        
        if norm > 0:
            return embedding / norm
        return embedding
    
    def adaptive_thresholding(self, confidence: float, 
                               quality_metrics: Dict[str, float]) -> Tuple[float, str]:
        """
        Calculate an adaptive recognition threshold based on image quality.
        
        Higher quality images get a lower (stricter) threshold because
        we expect more reliable embeddings. Lower quality images get a
        higher (more lenient) threshold.
        
        Args:
            confidence: Raw recognition confidence score
            quality_metrics: Quality assessment dictionary
            
        Returns:
            Tuple of (adaptive_threshold, reason_string)
        """
        base_threshold = 0.45
        
        # Extract quality factors
        sharpness = quality_metrics.get('sharpness', 0.5)
        brightness = quality_metrics.get('brightness', 0.5)
        contrast = quality_metrics.get('contrast', 0.5)
        
        # Calculate quality-based adjustment
        avg_quality = (sharpness + brightness + contrast) / 3.0
        
        if avg_quality > 0.7:
            # High quality: stricter threshold
            threshold = base_threshold - 0.05
            reason = "high_quality_strict"
        elif avg_quality > 0.4:
            # Medium quality: standard threshold
            threshold = base_threshold
            reason = "medium_quality_standard"
        else:
            # Low quality: lenient threshold
            threshold = base_threshold + 0.10
            reason = "low_quality_lenient"
        
        # Clamp threshold
        threshold = max(0.2, min(0.8, threshold))
        
        return threshold, reason
    
    def update_performance_stats(self, recognition_result: Dict[str, Any]) -> None:
        """
        Update running performance statistics from a recognition result.
        
        Args:
            recognition_result: Dictionary with keys:
                - success: bool
                - confidence: float
                - lighting_condition: str
                - image_quality: float
                - recognition_method: str
        """
        self.performance_history.append(recognition_result)
        self.performance_stats['total_recognitions'] += 1
        
        if recognition_result.get('success', False):
            self.performance_stats['successful_recognitions'] += 1
        
        # Update running average confidence
        confidence = recognition_result.get('confidence', 0.0)
        total = self.performance_stats['total_recognitions']
        prev_avg = self.performance_stats['avg_confidence']
        self.performance_stats['avg_confidence'] = (
            (prev_avg * (total - 1) + confidence) / total
        )
        
        # Track lighting distribution
        lighting = recognition_result.get('lighting_condition', 'unknown')
        self.performance_stats['lighting_distribution'][lighting] = (
            self.performance_stats['lighting_distribution'].get(lighting, 0) + 1
        )
        
        # Track quality distribution
        quality = recognition_result.get('image_quality', 0.0)
        quality_bucket = 'high' if quality > 0.7 else ('medium' if quality > 0.4 else 'low')
        self.performance_stats['quality_distribution'][quality_bucket] = (
            self.performance_stats['quality_distribution'].get(quality_bucket, 0) + 1
        )
        
        # Track method distribution
        method = recognition_result.get('recognition_method', 'unknown')
        self.performance_stats['method_distribution'][method] = (
            self.performance_stats['method_distribution'].get(method, 0) + 1
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Return current performance statistics."""
        stats = dict(self.performance_stats)
        total = stats['total_recognitions']
        if total > 0:
            stats['success_rate'] = stats['successful_recognitions'] / total
        else:
            stats['success_rate'] = 0.0
        return stats
    
    # --- Private Methods ---
    
    def _assess_quality(self, face_image: np.ndarray) -> Dict[str, float]:
        """Assess face image quality across multiple dimensions."""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(1.0, laplacian_var / 500.0)
        
        # Brightness (mean intensity normalized)
        mean_brightness = np.mean(gray) / 255.0
        # Penalize both too dark and too bright
        brightness = 1.0 - abs(mean_brightness - 0.5) * 2.0
        brightness = max(0.0, brightness)
        
        # Contrast (standard deviation of intensity)
        contrast = min(1.0, np.std(gray) / 80.0)
        
        # Face size (larger is better for recognition)
        h, w = gray.shape[:2]
        size_score = min(1.0, (h * w) / (160 * 160))
        
        return {
            'sharpness': round(sharpness, 3),
            'brightness': round(brightness, 3),
            'contrast': round(contrast, 3),
            'size_score': round(size_score, 3)
        }
    
    def _detect_lighting(self, face_image: np.ndarray) -> str:
        """Detect the lighting condition of a face image."""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        if mean_val < 60:
            return 'very_dark'
        elif mean_val < 100:
            return 'dark'
        elif mean_val > 200:
            return 'overexposed'
        elif mean_val > 160:
            return 'bright'
        elif std_val < 30:
            return 'flat_lighting'
        else:
            return 'normal'
    
    def _normalize_lighting(self, face_image: np.ndarray, 
                             lighting_condition: str) -> np.ndarray:
        """Apply lighting normalization based on detected condition."""
        if lighting_condition in ('very_dark', 'dark'):
            # Apply CLAHE for contrast enhancement
            lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(l_channel)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        elif lighting_condition == 'overexposed':
            # Reduce brightness
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2].astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif lighting_condition == 'flat_lighting':
            # Apply mild CLAHE
            lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(l_channel)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return face_image
    
    def _calculate_overall_quality(self, quality_metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'sharpness': 0.35,
            'brightness': 0.25,
            'contrast': 0.25,
            'size_score': 0.15
        }
        
        score = sum(
            quality_metrics.get(key, 0.0) * weight 
            for key, weight in weights.items()
        )
        return round(min(1.0, max(0.0, score)), 3)
    
    def _empty_result(self, image: np.ndarray) -> Dict[str, Any]:
        """Return a fallback result when face processing fails."""
        h, w = image.shape[:2]
        face = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        return {
            'processed_face': face,
            'quality_metrics': {
                'sharpness': 0.0,
                'brightness': 0.0,
                'contrast': 0.0,
                'size_score': 0.0
            },
            'lighting_condition': 'unknown',
            'image_quality': 0.0
        }
