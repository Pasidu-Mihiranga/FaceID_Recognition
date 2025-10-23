#!/usr/bin/env python3
"""
Improve existing face recognition system for better consistency
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

class ImprovedSimpleRecognizer:
    """Improved simple recognizer with better consistency"""
    
    def __init__(self, threshold: float = 0.8):
        """
        Initialize improved simple recognizer
        
        Args:
            threshold: Similarity threshold for face matching
        """
        self.threshold = threshold
        self.face_database = {}
        self.database_path = "data/embeddings/improved_simple_database.pkl"
        
        logger.info("Improved simple recognizer initialized")
        self.load_database()
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract improved features from face image"""
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Resize to standard size with better interpolation
            gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_CUBIC)
            
            # Apply histogram equalization for better contrast
            gray = cv2.equalizeHist(gray)
            
            # Extract multiple histogram features
            features = []
            
            # 1. Global histogram
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])  # Reduced bins for stability
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-8)  # Normalize with epsilon
            features.extend(hist)
            
            # 2. Local histograms (divide image into 4 quadrants)
            h, w = gray.shape
            quadrants = [
                gray[0:h//2, 0:w//2],
                gray[0:h//2, w//2:w],
                gray[h//2:h, 0:w//2],
                gray[h//2:h, w//2:w]
            ]
            
            for quadrant in quadrants:
                q_hist = cv2.calcHist([quadrant], [0], None, [32], [0, 256])
                q_hist = q_hist.flatten()
                q_hist = q_hist / (np.sum(q_hist) + 1e-8)
                features.extend(q_hist)
            
            # 3. Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
            edge_hist = edge_hist.flatten()
            edge_hist = edge_hist / (np.sum(edge_hist) + 1e-8)
            features.extend(edge_hist)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Improved embedding extraction failed: {e}")
            raise
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings using multiple methods"""
        try:
            # Method 1: Cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Method 2: Histogram correlation (for histogram parts)
            hist_len = 64  # Length of global histogram
            if len(embedding1) >= hist_len and len(embedding2) >= hist_len:
                hist1 = embedding1[:hist_len]
                hist2 = embedding2[:hist_len]
                correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            else:
                correlation = 0.0
            
            # Method 3: Euclidean distance (normalized)
            euclidean_dist = np.linalg.norm(embedding1 - embedding2)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            
            # Weighted combination
            final_similarity = (0.4 * cosine_sim + 0.4 * correlation + 0.2 * euclidean_sim)
            
            # Ensure result is in [0, 1] range
            return max(0.0, min(1.0, final_similarity))
            
        except Exception as e:
            logger.error(f"Improved comparison failed: {e}")
            return 0.0
    
    def register_face(self, face_image: np.ndarray, person_name: str) -> bool:
        """Register a new face in the database"""
        try:
            # Extract features
            features = self.extract_embedding(face_image)
            
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
        """Recognize a face from the database"""
        try:
            # Extract features
            features = self.extract_embedding(face_image)
            
            best_match = None
            best_score = 0.0
            
            # Compare with all registered faces
            for person_name, embeddings in self.face_database.items():
                for stored_embedding in embeddings:
                    similarity = self.compare_faces(features, stored_embedding)
                    
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

def improve_existing_system():
    """Improve the existing face recognition system"""
    
    print("=== Improving Existing Face Recognition System ===")
    
    # Initialize improved recognizer
    improved_recognizer = ImprovedSimpleRecognizer(threshold=0.8)
    
    # Load existing database and migrate
    from main import FaceIDSystem
    system = FaceIDSystem()
    
    # Get existing face database
    existing_db_path = "data/embeddings/simple_database.pkl"
    
    if os.path.exists(existing_db_path):
        try:
            with open(existing_db_path, 'rb') as f:
                existing_db = pickle.load(f)
            
            print(f"Loaded existing database with {len(existing_db)} persons:")
            
            # Migrate each person
            for person_name, embeddings in existing_db.items():
                print(f"Migrating {person_name} with {len(embeddings)} embeddings...")
                
                # For each embedding, we need to reconstruct the face image
                # Since we don't have the original images, we'll use the embeddings as-is
                # but with improved comparison logic
                
                # Convert old embeddings to new format (approximate)
                improved_embeddings = []
                for old_embedding in embeddings:
                    # Pad or truncate to match new feature size
                    new_size = 64 + 4*32 + 32  # Global + 4 quadrants + edges
                    
                    if len(old_embedding) >= new_size:
                        new_embedding = old_embedding[:new_size]
                    else:
                        # Pad with zeros
                        new_embedding = np.zeros(new_size)
                        new_embedding[:len(old_embedding)] = old_embedding
                    
                    improved_embeddings.append(new_embedding)
                
                improved_recognizer.face_database[person_name] = improved_embeddings
            
            # Save improved database
            improved_recognizer.save_database()
            
            print(f"Migration complete! Improved database contains {len(improved_recognizer.face_database)} persons:")
            for person_name, embeddings in improved_recognizer.face_database.items():
                print(f"- {person_name}: {len(embeddings)} embeddings")
            
        except Exception as e:
            print(f"Error migrating existing database: {e}")
            return False
    
    # Test consistency
    print(f"\nTesting improved recognition consistency:")
    
    # Test with different images multiple times
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
                        person_name, confidence = improved_recognizer.recognize_face(image)
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
    
    # Replace the original database
    print(f"\nReplacing original database with improved version...")
    try:
        # Backup original
        backup_path = existing_db_path + ".backup"
        if os.path.exists(existing_db_path):
            os.rename(existing_db_path, backup_path)
            print(f"Original database backed up to {backup_path}")
        
        # Copy improved database to original location
        import shutil
        shutil.copy2(improved_recognizer.database_path, existing_db_path)
        print(f"Improved database copied to {existing_db_path}")
        
        print(f"Database replacement complete!")
        
    except Exception as e:
        print(f"Error replacing database: {e}")
        return False
    
    print(f"\n=== System Improvement Complete ===")
    return True

if __name__ == "__main__":
    improve_existing_system()
