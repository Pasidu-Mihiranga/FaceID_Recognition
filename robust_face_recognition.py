#!/usr/bin/env python3
"""
Create a more robust face recognition system with better consistency
"""

import sys
sys.path.append('src')

import cv2
import numpy as np
import os
import pickle
import glob
from typing import List, Tuple, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustFaceRecognizer:
    """More robust face recognizer with multiple features"""
    
    def __init__(self, threshold: float = 0.7):
        """
        Initialize robust face recognizer
        
        Args:
            threshold: Similarity threshold for face matching
        """
        self.threshold = threshold
        self.face_database = {}  # Dictionary to store face features
        self.database_path = "data/embeddings/robust_database.pkl"
        
        logger.info("Robust face recognizer initialized")
        self.load_database()
    
    def extract_features(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract multiple features from face image for better recognition
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Convert to grayscale for some features
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) if len(face_image.shape) == 3 else face_image
            
            # Resize to standard size for consistency
            standard_size = (128, 128)
            resized = cv2.resize(gray, standard_size)
            
            features = {}
            
            # 1. Histogram features (normalized)
            hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
            features['histogram'] = hist.flatten() / np.sum(hist)  # Normalize
            
            # 2. LBP (Local Binary Pattern) features
            lbp = self._compute_lbp(resized)
            features['lbp'] = lbp.flatten()
            
            # 3. Edge features
            edges = cv2.Canny(resized, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
            features['edges'] = edge_hist.flatten() / np.sum(edge_hist)  # Normalize
            
            # 4. Texture features using Gabor filters
            gabor_features = self._compute_gabor_features(resized)
            features['gabor'] = gabor_features
            
            # 5. Color histogram (if color image)
            if len(face_image.shape) == 3:
                color_hist = []
                for i in range(3):  # BGR channels
                    hist = cv2.calcHist([face_image], [i], None, [64], [0, 256])
                    color_hist.extend(hist.flatten())
                features['color'] = np.array(color_hist) / np.sum(color_hist)  # Normalize
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise
    
    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """Compute Local Binary Pattern"""
        try:
            # Simple LBP implementation
            lbp = np.zeros_like(image)
            
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    center = image[i, j]
                    binary_string = ""
                    
                    # 8-neighborhood
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += "1" if neighbor >= center else "0"
                    
                    lbp[i, j] = int(binary_string, 2)
            
            return lbp
            
        except Exception as e:
            logger.error(f"LBP computation failed: {e}")
            return np.zeros_like(image)
    
    def _compute_gabor_features(self, image: np.ndarray) -> np.ndarray:
        """Compute Gabor filter features"""
        try:
            features = []
            
            # Different orientations and frequencies
            orientations = [0, 45, 90, 135]
            frequencies = [0.1, 0.3, 0.5]
            
            for freq in frequencies:
                for orient in orientations:
                    # Create Gabor kernel
                    kernel = cv2.getGaborKernel(
                        (21, 21), 5, np.radians(orient), 2*np.pi*freq, 0.5, 0, ktype=cv2.CV_32F
                    )
                    
                    # Apply filter
                    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                    
                    # Compute mean and std
                    features.extend([np.mean(filtered), np.std(filtered)])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Gabor features computation failed: {e}")
            return np.zeros(12)  # Default size
    
    def compare_features(self, features1: Dict[str, np.ndarray], features2: Dict[str, np.ndarray]) -> float:
        """
        Compare two feature sets using weighted combination
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            similarities = []
            weights = []
            
            # Compare each feature type
            for feature_type in features1.keys():
                if feature_type in features2:
                    feat1 = features1[feature_type]
                    feat2 = features2[feature_type]
                    
                    # Normalize features
                    feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
                    feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
                    
                    # Compute cosine similarity
                    similarity = np.dot(feat1_norm, feat2_norm)
                    similarities.append(similarity)
                    
                    # Assign weights based on feature type
                    if feature_type == 'histogram':
                        weights.append(0.3)  # High weight for histogram
                    elif feature_type == 'lbp':
                        weights.append(0.25)  # High weight for LBP
                    elif feature_type == 'edges':
                        weights.append(0.2)   # Medium weight for edges
                    elif feature_type == 'gabor':
                        weights.append(0.15) # Medium weight for Gabor
                    elif feature_type == 'color':
                        weights.append(0.1)  # Lower weight for color
            
            # Weighted average of similarities
            if similarities:
                weighted_similarity = np.average(similarities, weights=weights)
                # Convert to 0-1 range
                return max(0, min(1, (weighted_similarity + 1) / 2))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Feature comparison failed: {e}")
            return 0.0
    
    def register_face(self, face_image: np.ndarray, person_name: str) -> bool:
        """
        Register a new face in the database
        
        Args:
            face_image: Face image to register
            person_name: Name of the person
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Extract features
            features = self.extract_features(face_image)
            
            # Add to database
            if person_name not in self.face_database:
                self.face_database[person_name] = []
            
            self.face_database[person_name].append(features)
            
            # Save database
            self.save_database()
            
            logger.info(f"Registered face for {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Face registration failed for {person_name}: {e}")
            return False
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face from the database
        
        Args:
            face_image: Face image to recognize
            
        Returns:
            Tuple of (person_name, confidence_score)
        """
        try:
            # Extract features
            features = self.extract_features(face_image)
            
            best_match = None
            best_score = 0.0
            
            # Compare with all registered faces
            for person_name, feature_sets in self.face_database.items():
                for stored_features in feature_sets:
                    similarity = self.compare_features(features, stored_features)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = person_name
            
            # Check if similarity exceeds threshold
            if best_score >= self.threshold:
                return best_match, best_score
            else:
                return None, best_score
                
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return None, 0.0
    
    def save_database(self):
        """Save face database to file"""
        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.face_database, f)
            logger.info(f"Database saved to {self.database_path}")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def load_database(self):
        """Load face database from file"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'rb') as f:
                    self.face_database = pickle.load(f)
                logger.info(f"Database loaded from {self.database_path}")
            else:
                logger.info("No existing database found, starting fresh")
                
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            self.face_database = {}

def migrate_to_robust_system():
    """Migrate existing face database to robust system"""
    
    print("=== Migrating to Robust Face Recognition System ===")
    
    # Initialize robust recognizer
    robust_recognizer = RobustFaceRecognizer(threshold=0.7)
    
    # Load existing database
    from main import FaceIDSystem
    system = FaceIDSystem()
    
    # Get all registered persons
    persons = system.database.get_all_persons()
    print(f"Found {len(persons)} registered persons to migrate:")
    
    for person in persons:
        print(f"- {person['name']} (ID: {person['id']})")
    
    # Migrate each person's embeddings
    migrated_count = 0
    
    for person in persons:
        person_id = person['id']
        person_name = person['name']
        
        print(f"\nMigrating {person_name} (ID: {person_id}):")
        
        # Find all embedding files for this person
        embedding_files = glob.glob(f"data/embeddings/face_{person_id}_*.pkl")
        
        for embedding_file in embedding_files:
            try:
                # Get the original image path from database
                face_images = system.database.get_face_images_by_person(person_id)
                
                for face_image in face_images:
                    image_path = face_image['image_path']
                    
                    if os.path.exists(image_path):
                        # Load original image
                        image = cv2.imread(image_path)
                        
                        if image is not None:
                            # Extract face from image using stored bbox
                            face_bbox = face_image.get('face_bbox')
                            if face_bbox:
                                x, y, w, h = face_bbox
                                face_image_crop = image[y:y+h, x:x+w]
                            else:
                                face_image_crop = image
                            
                            # Register with robust system
                            success = robust_recognizer.register_face(face_image_crop, person_name)
                            
                            if success:
                                migrated_count += 1
                                print(f"  Migrated: {image_path}")
                            else:
                                print(f"  Failed to migrate: {image_path}")
                        else:
                            print(f"  Could not load image: {image_path}")
                    else:
                        print(f"  Image not found: {image_path}")
                        
            except Exception as e:
                print(f"  Error migrating {embedding_file}: {e}")
    
    print(f"\nMigration complete! Migrated {migrated_count} face images.")
    
    # Test the robust system
    print(f"\nTesting robust recognition system:")
    
    # Test consistency with multiple runs
    test_images = [
        "data/simple_face_test.jpg",
        "data/test_person1.jpg", 
        "data/test_person2.jpg"
    ]
    
    for test_image_path in test_images:
        if os.path.exists(test_image_path):
            print(f"\nTesting consistency with: {test_image_path}")
            
            # Test multiple times to check consistency
            results = []
            for i in range(5):  # Test 5 times
                try:
                    image = cv2.imread(test_image_path)
                    if image is not None:
                        person_name, confidence = robust_recognizer.recognize_face(image)
                        results.append((person_name, confidence))
                        print(f"  Test {i+1}: {person_name} (confidence: {confidence:.4f})")
                    else:
                        print(f"  Test {i+1}: Could not load image")
                except Exception as e:
                    print(f"  Test {i+1}: Error - {e}")
            
            # Check consistency
            if len(results) > 1:
                person_names = [r[0] for r in results]
                confidences = [r[1] for r in results]
                
                if len(set(person_names)) == 1:
                    print(f"  CONSISTENT: All results show {person_names[0]}")
                    print(f"  Confidence range: {min(confidences):.4f} - {max(confidences):.4f}")
                    print(f"  Confidence std: {np.std(confidences):.4f}")
                else:
                    print(f"  INCONSISTENT: Results vary - {set(person_names)}")
        else:
            print(f"\nImage not found: {test_image_path}")
    
    print(f"\n=== Robust System Migration Complete ===")
    return robust_recognizer

if __name__ == "__main__":
    migrate_to_robust_system()
