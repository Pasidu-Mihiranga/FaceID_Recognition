#!/usr/bin/env python3
"""
DeepFace-based face recognizer for improved accuracy
"""

import sys
sys.path.append('src')

import cv2
import numpy as np
import os
import pickle
import logging
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace available - implementing advanced face recognition")
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.error("DeepFace not available")

class DeepFaceRecognizer:
    """DeepFace-based face recognizer for high accuracy"""
    
    def __init__(self, model_name: str = 'VGG-Face', threshold: float = 0.6):
        """
        Initialize DeepFace recognizer
        
        Args:
            model_name: DeepFace model name ('VGG-Face', 'Facenet', 'ArcFace', etc.)
            threshold: Similarity threshold for face matching
        """
        self.model_name = model_name
        self.threshold = threshold
        self.face_database = {}  # Dictionary to store face embeddings
        self.database_path = f"data/embeddings/deepface_{model_name.lower().replace('-', '_')}_database.pkl"
        
        if not DEEPFACE_AVAILABLE:
            raise ImportError("DeepFace not available")
        
        logger.info(f"DeepFace recognizer initialized with {model_name}")
        self.load_database()
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding using DeepFace
        
        Args:
            face_image: Face image as numpy array
            
        Returns:
            Face embedding vector
        """
        try:
            # DeepFace expects RGB images
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Extract embedding using DeepFace
            embedding = DeepFace.represent(
                img_path=rgb_image,
                model_name=self.model_name,
                enforce_detection=False,  # Don't fail if no face detected
                detector_backend='opencv'  # Use OpenCV for face detection
            )
            
            # DeepFace returns a list, get the first embedding
            if isinstance(embedding, list) and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            else:
                return np.array(embedding)
                
        except Exception as e:
            logger.error(f"DeepFace embedding extraction failed: {e}")
            raise
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            # Normalize embeddings
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
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
            # Extract embedding
            embedding = self.extract_embedding(face_image)
            
            # Add to database
            if person_name not in self.face_database:
                self.face_database[person_name] = []
            
            self.face_database[person_name].append(embedding)
            
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
            # Extract embedding
            embedding = self.extract_embedding(face_image)
            
            best_match = None
            best_score = 0.0
            
            # Compare with all registered faces
            for person_name, embeddings in self.face_database.items():
                for stored_embedding in embeddings:
                    similarity = self.compare_faces(embedding, stored_embedding)
                    
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

def migrate_to_deepface():
    """Migrate existing face database to DeepFace system"""
    
    print("=== Migrating to DeepFace Face Recognition ===")
    
    if not DEEPFACE_AVAILABLE:
        print("DeepFace not available - cannot migrate")
        return False
    
    # Initialize DeepFace recognizer
    deepface_recognizer = DeepFaceRecognizer(model_name='VGG-Face', threshold=0.6)
    
    # Load existing system
    from main import FaceIDSystem
    system = FaceIDSystem()
    
    # Get all registered persons
    persons = system.database.get_all_persons()
    print(f"Found {len(persons)} registered persons to migrate:")
    
    for person in persons:
        print(f"- {person['name']} (ID: {person['id']})")
    
    # Migrate each person's face images
    migrated_count = 0
    
    for person in persons:
        person_id = person['id']
        person_name = person['name']
        
        print(f"\nMigrating {person_name} (ID: {person_id}):")
        
        # Get face images for this person from database
        try:
            # Query face images for this person
            import sqlite3
            db_path = "data/face_database.db"
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT image_path, face_bbox FROM face_images 
                    WHERE person_id = ?
                ''', (person_id,))
                
                face_images = cursor.fetchall()
            
            for image_path, face_bbox_json in face_images:
                if os.path.exists(image_path):
                    try:
                        # Load original image
                        image = cv2.imread(image_path)
                        
                        if image is not None:
                            # Extract face from image using stored bbox
                            if face_bbox_json:
                                import json
                                face_bbox = json.loads(face_bbox_json)
                                x, y, w, h = face_bbox
                                face_image_crop = image[y:y+h, x:x+w]
                            else:
                                face_image_crop = image
                            
                            # Register with DeepFace system
                            success = deepface_recognizer.register_face(face_image_crop, person_name)
                            
                            if success:
                                migrated_count += 1
                                print(f"  Migrated: {image_path}")
                            else:
                                print(f"  Failed to migrate: {image_path}")
                        else:
                            print(f"  Could not load image: {image_path}")
                    except Exception as e:
                        print(f"  Error migrating {image_path}: {e}")
                else:
                    print(f"  Image not found: {image_path}")
                    
        except Exception as e:
            print(f"  Error accessing database for {person_name}: {e}")
    
    print(f"\nMigration complete! Migrated {migrated_count} face images.")
    
    # Test the DeepFace system
    print(f"\nTesting DeepFace recognition system:")
    
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
            for i in range(3):
                try:
                    image = cv2.imread(test_image_path)
                    if image is not None:
                        person_name, confidence = deepface_recognizer.recognize_face(image)
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
    
    print(f"\n=== DeepFace Migration Complete ===")
    return deepface_recognizer

if __name__ == "__main__":
    migrate_to_deepface()
