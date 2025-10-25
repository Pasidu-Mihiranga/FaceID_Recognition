"""
Data Augmentation System for Face Recognition
Generates multiple variations from original face images for better training
"""

import cv2
import numpy as np
import random
import logging
from typing import List, Tuple, Dict, Any
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class FaceDataAugmentation:
    """
    Comprehensive data augmentation system for face images
    """
    
    def __init__(self, augmentation_factor: int = 4):
        """
        Initialize data augmentation system
        
        Args:
            augmentation_factor: How many variations to generate per original image
        """
        self.augmentation_factor = augmentation_factor
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        logger.info(f"Face Data Augmentation initialized with factor: {augmentation_factor}")
    
    def augment_face_images(self, face_images: List[np.ndarray], 
                           person_name: str) -> List[np.ndarray]:
        """
        Generate augmented variations of face images
        
        Args:
            face_images: List of original face images
            person_name: Name of the person for logging
            
        Returns:
            List of augmented face images
        """
        try:
            logger.info(f"Starting data augmentation for {person_name} with {len(face_images)} original images")
            
            augmented_images = []
            
            # Keep original images
            augmented_images.extend(face_images)
            
            # Generate augmented variations
            for i, original_image in enumerate(face_images):
                logger.info(f"Augmenting image {i+1}/{len(face_images)}")
                
                # Generate multiple variations for each original image
                variations = self._generate_image_variations(original_image, self.augmentation_factor)
                augmented_images.extend(variations)
            
            logger.info(f"Data augmentation complete: {len(face_images)} → {len(augmented_images)} images")
            return augmented_images
            
        except Exception as e:
            logger.error(f"Data augmentation failed for {person_name}: {e}")
            return face_images  # Return original images if augmentation fails
    
    def _generate_image_variations(self, image: np.ndarray, num_variations: int) -> List[np.ndarray]:
        """
        Generate multiple variations of a single image
        
        Args:
            image: Original image
            num_variations: Number of variations to generate
            
        Returns:
            List of augmented images
        """
        try:
            variations = []
            
            for i in range(num_variations):
                # Create a copy of the original image
                augmented = image.copy()
                
                # Apply random augmentation techniques
                augmented = self._apply_random_augmentation(augmented, variation_id=i)
                
                variations.append(augmented)
            
            return variations
            
        except Exception as e:
            logger.error(f"Failed to generate variations: {e}")
            return []
    
    def _apply_random_augmentation(self, image: np.ndarray, variation_id: int) -> np.ndarray:
        """
        Apply random augmentation techniques to an image
        
        Args:
            image: Input image
            variation_id: ID of the variation (for consistent randomization)
            
        Returns:
            Augmented image
        """
        try:
            augmented = image.copy()
            
            # Set random seed based on variation ID for consistency
            random.seed(self.random_seed + variation_id)
            np.random.seed(self.random_seed + variation_id)
            
            # 1. Lighting Variations (always apply)
            augmented = self._apply_lighting_variations(augmented)
            
            # 2. Image Filters (70% chance)
            if random.random() < 0.7:
                augmented = self._apply_image_filters(augmented)
            
            # 3. Geometric Transformations (50% chance)
            if random.random() < 0.5:
                augmented = self._apply_geometric_transforms(augmented)
            
            # 4. Quality Variations (40% chance)
            if random.random() < 0.4:
                augmented = self._apply_quality_variations(augmented)
            
            # 5. Color Variations (60% chance)
            if random.random() < 0.6:
                augmented = self._apply_color_variations(augmented)
            
            return augmented
            
        except Exception as e:
            logger.error(f"Random augmentation failed: {e}")
            return image
    
    def _apply_lighting_variations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply lighting variations to simulate different lighting conditions
        
        Args:
            image: Input image
            
        Returns:
            Image with lighting variations
        """
        try:
            # Random brightness adjustment (-30 to +30)
            brightness = random.uniform(-30, 30)
            augmented = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness)
            
            # Random contrast adjustment (0.8 to 1.2)
            contrast = random.uniform(0.8, 1.2)
            augmented = cv2.convertScaleAbs(augmented, alpha=contrast, beta=0)
            
            # Random gamma correction (0.8 to 1.3)
            gamma = random.uniform(0.8, 1.3)
            lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
            augmented = cv2.LUT(augmented, lookup_table)
            
            return augmented
            
        except Exception as e:
            logger.error(f"Lighting variations failed: {e}")
            return image
    
    def _apply_image_filters(self, image: np.ndarray) -> np.ndarray:
        """
        Apply image filters to simulate different camera effects
        
        Args:
            image: Input image
            
        Returns:
            Image with filters applied
        """
        try:
            filter_type = random.choice(['blur', 'sharpen', 'noise', 'smooth'])
            
            if filter_type == 'blur':
                # Simulate slight blur (phone camera effect)
                kernel_size = random.choice([3, 5])
                augmented = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
                
            elif filter_type == 'sharpen':
                # Apply sharpening
                kernel = np.array([[-1, -1, -1],
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
                sharpened = cv2.filter2D(image, -1, kernel)
                alpha = random.uniform(0.1, 0.3)
                augmented = cv2.addWeighted(image, 1-alpha, sharpened, alpha, 0)
                
            elif filter_type == 'noise':
                # Add noise to simulate compression artifacts
                noise = np.random.normal(0, random.uniform(5, 15), image.shape).astype(np.uint8)
                augmented = cv2.add(image, noise)
                
            elif filter_type == 'smooth':
                # Apply smoothing filter
                augmented = cv2.bilateralFilter(image, 9, 75, 75)
            
            return augmented
            
        except Exception as e:
            logger.error(f"Image filters failed: {e}")
            return image
    
    def _apply_geometric_transforms(self, image: np.ndarray) -> np.ndarray:
        """
        Apply geometric transformations
        
        Args:
            image: Input image
            
        Returns:
            Image with geometric transformations
        """
        try:
            h, w = image.shape[:2]
            
            # Random rotation (-5 to +5 degrees)
            angle = random.uniform(-5, 5)
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(image, rotation_matrix, (w, h))
            
            # Random scaling (0.95 to 1.05)
            scale = random.uniform(0.95, 1.05)
            if scale != 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                augmented = cv2.resize(augmented, (new_w, new_h))
                
                # Crop or pad to original size
                if scale > 1.0:
                    # Crop from center
                    start_x = (new_w - w) // 2
                    start_y = (new_h - h) // 2
                    augmented = augmented[start_y:start_y+h, start_x:start_x+w]
                else:
                    # Pad with black borders
                    pad_x = (w - new_w) // 2
                    pad_y = (h - new_h) // 2
                    augmented = cv2.copyMakeBorder(augmented, pad_y, pad_y, pad_x, pad_x, 
                                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            return augmented
            
        except Exception as e:
            logger.error(f"Geometric transforms failed: {e}")
            return image
    
    def _apply_quality_variations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply quality variations to simulate different image qualities
        
        Args:
            image: Input image
            
        Returns:
            Image with quality variations
        """
        try:
            quality_type = random.choice(['compression', 'resolution', 'artifacts'])
            
            if quality_type == 'compression':
                # Simulate JPEG compression artifacts
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(70, 95)]
                _, encoded_img = cv2.imencode('.jpg', image, encode_param)
                augmented = cv2.imdecode(encoded_img, 1)
                
            elif quality_type == 'resolution':
                # Simulate lower resolution
                scale_factor = random.uniform(0.7, 0.9)
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                augmented = cv2.resize(image, (new_w, new_h))
                augmented = cv2.resize(augmented, (w, h), interpolation=cv2.INTER_LINEAR)
                
            elif quality_type == 'artifacts':
                # Add compression-like artifacts
                augmented = image.copy()
                # Add slight block artifacts
                block_size = random.randint(4, 8)
                for y in range(0, image.shape[0], block_size):
                    for x in range(0, image.shape[1], block_size):
                        if random.random() < 0.1:  # 10% chance per block
                            block = augmented[y:y+block_size, x:x+block_size]
                            mean_color = np.mean(block, axis=(0, 1))
                            augmented[y:y+block_size, x:x+block_size] = mean_color
            
            return augmented
            
        except Exception as e:
            logger.error(f"Quality variations failed: {e}")
            return image
    
    def _apply_color_variations(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color variations to simulate different camera characteristics
        
        Args:
            image: Input image
            
        Returns:
            Image with color variations
        """
        try:
            # Convert to HSV for easier color manipulation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Random hue shift (-10 to +10)
            hue_shift = random.randint(-10, 10)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            
            # Random saturation adjustment (0.8 to 1.2)
            sat_factor = random.uniform(0.8, 1.2)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
            
            # Random value (brightness) adjustment (0.9 to 1.1)
            val_factor = random.uniform(0.9, 1.1)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_factor, 0, 255)
            
            # Convert back to BGR
            augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Random color temperature adjustment
            if random.random() < 0.5:
                # Warm or cool tone
                temp_factor = random.uniform(0.9, 1.1)
                if temp_factor > 1.0:  # Warm
                    augmented[:, :, 2] = np.clip(augmented[:, :, 2] * temp_factor, 0, 255)  # Red
                    augmented[:, :, 0] = np.clip(augmented[:, :, 0] / temp_factor, 0, 255)  # Blue
                else:  # Cool
                    augmented[:, :, 0] = np.clip(augmented[:, :, 0] / temp_factor, 0, 255)  # Blue
                    augmented[:, :, 2] = np.clip(augmented[:, :, 2] * temp_factor, 0, 255)  # Red
            
            return augmented
            
        except Exception as e:
            logger.error(f"Color variations failed: {e}")
            return image
    
    def save_augmented_images(self, augmented_images: List[np.ndarray], 
                             person_name: str, save_dir: str = "data/augmented_images"):
        """
        Save augmented images for inspection/debugging
        
        Args:
            augmented_images: List of augmented images
            person_name: Name of the person
            save_dir: Directory to save images
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            person_dir = os.path.join(save_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            for i, image in enumerate(augmented_images):
                filename = f"{person_name}_augmented_{i:03d}.jpg"
                filepath = os.path.join(person_dir, filename)
                cv2.imwrite(filepath, image)
            
            logger.info(f"Saved {len(augmented_images)} augmented images to {person_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save augmented images: {e}")
    
    def get_augmentation_stats(self, original_count: int, augmented_count: int) -> Dict[str, Any]:
        """
        Get statistics about the augmentation process
        
        Args:
            original_count: Number of original images
            augmented_count: Number of augmented images
            
        Returns:
            Dictionary with augmentation statistics
        """
        return {
            'original_images': original_count,
            'augmented_images': augmented_count,
            'augmentation_factor': augmented_count / original_count if original_count > 0 else 0,
            'additional_images': augmented_count - original_count,
            'improvement_percentage': ((augmented_count - original_count) / original_count * 100) if original_count > 0 else 0
        }
