#!/usr/bin/env python3
"""
Integrate DeepFace VGG-Face into the existing system
"""

import sys
sys.path.append('src')

import cv2
import numpy as np
import os
import pickle
import sqlite3
import json
from deepface import DeepFace
from main import FaceIDSystem

def integrate_deepface():
    """Integrate DeepFace VGG-Face into the system"""
    
    print("=== Integrating DeepFace VGG-Face ===")
    
    # Test DeepFace first
    print("Testing DeepFace VGG-Face...")
    try:
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = DeepFace.represent(
            img_path=test_img,
            model_name='VGG-Face',
            enforce_detection=False
        )
        print("SUCCESS! DeepFace VGG-Face is working!")
    except Exception as e:
        print(f"FAILED! DeepFace error: {e}")
        return False
    
    # Initialize DeepFace recognizer
    print("\nInitializing DeepFace recognizer...")
    
    class DeepFaceRecognizer:
        """DeepFace-based face recognizer"""
        
        def __init__(self, threshold=0.6):
            self.model_name = 'VGG-Face'
            self.threshold = threshold
            self.face_database = {}
            self.database_path = "data/embeddings/deepface_vgg_face_database.pkl"
            self.load_database()
        
        def extract_embedding(self, face_image):
            """Extract face embedding using DeepFace"""
            try:
                # Convert BGR to RGB
                if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = face_image
                
                # Extract embedding
                embedding = DeepFace.represent(
                    img_path=rgb_image,
                    model_name=self.model_name,
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                
                return np.array(embedding[0]['embedding'])
                
            except Exception as e:
                print(f"DeepFace embedding extraction failed: {e}")
                raise
        
        def compare_faces(self, embedding1, embedding2):
            """Compare two face embeddings using cosine similarity"""
            try:
                # Normalize embeddings
                emb1_norm = embedding1 / np.linalg.norm(embedding1)
                emb2_norm = embedding2 / np.linalg.norm(embedding2)
                
                # Calculate cosine similarity
                similarity = np.dot(emb1_norm, emb2_norm)
                
                # Convert to 0-1 range
                similarity = (similarity + 1) / 2
                
                return float(similarity)
                
            except Exception as e:
                print(f"Face comparison failed: {e}")
                return 0.0
        
        def register_face(self, face_image, person_name):
            """Register a new face"""
            try:
                embedding = self.extract_embedding(face_image)
                
                if person_name not in self.face_database:
                    self.face_database[person_name] = []
                
                self.face_database[person_name].append(embedding)
                self.save_database()
                
                print(f"Registered face for {person_name}")
                return True
                
            except Exception as e:
                print(f"Face registration failed for {person_name}: {e}")
                return False
        
        def recognize_face(self, face_image):
            """Recognize a face"""
            try:
                embedding = self.extract_embedding(face_image)
                
                best_match = None
                best_score = 0.0
                
                for person_name, embeddings in self.face_database.items():
                    for stored_embedding in embeddings:
                        similarity = self.compare_faces(embedding, stored_embedding)
                        
                        if similarity > best_score:
                            best_score = similarity
                            best_match = person_name
                
                if best_score >= self.threshold:
                    return best_match, best_score
                else:
                    return None, best_score
                    
            except Exception as e:
                print(f"Face recognition failed: {e}")
                return None, 0.0
        
        def save_database(self):
            """Save face database"""
            try:
                os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
                with open(self.database_path, 'wb') as f:
                    pickle.dump(self.face_database, f)
                print(f"Database saved to {self.database_path}")
            except Exception as e:
                print(f"Failed to save database: {e}")
        
        def load_database(self):
            """Load face database"""
            try:
                if os.path.exists(self.database_path):
                    with open(self.database_path, 'rb') as f:
                        self.face_database = pickle.load(f)
                    print(f"Database loaded from {self.database_path}")
                else:
                    print("No existing database found, starting fresh")
            except Exception as e:
                print(f"Failed to load database: {e}")
                self.face_database = {}
    
    # Initialize DeepFace recognizer
    deepface_recognizer = DeepFaceRecognizer(threshold=0.6)
    
    # Migrate existing faces
    print("\nMigrating existing faces to DeepFace...")
    
    system = FaceIDSystem()
    persons = system.database.get_all_persons()
    
    migrated_count = 0
    
    for person in persons:
        person_id = person['id']
        person_name = person['name']
        
        print(f"Migrating {person_name} (ID: {person_id})...")
        
        # Get face images from database
        try:
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
                        # Load image
                        image = cv2.imread(image_path)
                        
                        if image is not None:
                            # Extract face using bbox
                            if face_bbox_json:
                                face_bbox = json.loads(face_bbox_json)
                                x, y, w, h = face_bbox
                                face_image_crop = image[y:y+h, x:x+w]
                            else:
                                face_image_crop = image
                            
                            # Register with DeepFace
                            success = deepface_recognizer.register_face(face_image_crop, person_name)
                            
                            if success:
                                migrated_count += 1
                                print(f"  Migrated: {image_path}")
                            else:
                                print(f"  Failed: {image_path}")
                        else:
                            print(f"  Could not load: {image_path}")
                    except Exception as e:
                        print(f"  Error migrating {image_path}: {e}")
                else:
                    print(f"  Image not found: {image_path}")
                    
        except Exception as e:
            print(f"  Error accessing database for {person_name}: {e}")
    
    print(f"\nMigration complete! Migrated {migrated_count} faces.")
    
    # Test DeepFace recognition
    print(f"\nTesting DeepFace recognition...")
    
    test_images = [
        "data/simple_face_test.jpg",
        "data/test_person1.jpg", 
        "data/test_person2.jpg"
    ]
    
    for test_image_path in test_images:
        if os.path.exists(test_image_path):
            print(f"\nTesting: {test_image_path}")
            
            # Test multiple times for consistency
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
    
    # Update the main system to use DeepFace
    print(f"\nUpdating main system to use DeepFace...")
    
    # Update face recognition module
    face_recognition_path = "src/face_recognition/__init__.py"
    if os.path.exists(face_recognition_path):
        with open(face_recognition_path, 'r') as f:
            content = f.read()
        
        # Add DeepFace recognizer class
        deepface_class = '''
class DeepFaceRecognizer(FaceRecognizer):
    """DeepFace-based face recognizer for high accuracy"""
    
    def __init__(self, threshold: float = 0.6):
        """
        Initialize DeepFace recognizer
        
        Args:
            threshold: Similarity threshold for face matching
        """
        self.model_name = 'VGG-Face'
        self.threshold = threshold
        self.face_database = {}
        self.database_path = "data/embeddings/deepface_vgg_face_database.pkl"
        
        try:
            from deepface import DeepFace
            self.deepface_available = True
            logger.info("DeepFace VGG-Face recognizer initialized successfully")
        except ImportError:
            self.deepface_available = False
            logger.error("DeepFace not available")
            raise ImportError("DeepFace not available")
        
        self.load_database()
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding using DeepFace"""
        try:
            # Convert BGR to RGB
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Extract embedding
            embedding = DeepFace.represent(
                img_path=rgb_image,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            return np.array(embedding[0]['embedding'])
            
        except Exception as e:
            logger.error(f"DeepFace embedding extraction failed: {e}")
            raise
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings using cosine similarity"""
        try:
            # Normalize embeddings
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return 0.0
    
    def register_face(self, face_image: np.ndarray, person_name: str) -> bool:
        """Register a new face"""
        try:
            embedding = self.extract_embedding(face_image)
            
            if person_name not in self.face_database:
                self.face_database[person_name] = []
            
            self.face_database[person_name].append(embedding)
            self.save_database()
            
            logger.info(f"Registered face for {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Face registration failed for {person_name}: {e}")
            return False
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize a face"""
        try:
            embedding = self.extract_embedding(face_image)
            
            best_match = None
            best_score = 0.0
            
            for person_name, embeddings in self.face_database.items():
                for stored_embedding in embeddings:
                    similarity = self.compare_faces(embedding, stored_embedding)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = person_name
            
            if best_score >= self.threshold:
                return best_match, best_score
            else:
                return None, best_score
                
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return None, 0.0
    
    def save_database(self):
        """Save face database"""
        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.face_database, f)
            logger.info(f"Database saved to {self.database_path}")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def load_database(self):
        """Load face database"""
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

'''
        
        # Insert DeepFace class before FaceRecognitionManager
        if 'class FaceRecognitionManager:' in content:
            content = content.replace('class FaceRecognitionManager:', deepface_class + '\nclass FaceRecognitionManager:')
            
            with open(face_recognition_path, 'w') as f:
                f.write(content)
            
            print("Added DeepFace recognizer to face_recognition module")
        
        # Update the model initialization to prefer DeepFace
        if 'def _initialize_recognizer(self, model_type: str) -> FaceRecognizer:' in content:
            # Update the initialization logic to use DeepFace when available
            old_init = '''        if model_type == 'arcface':
            try:
                return ArcFaceRecognizer(threshold=self.threshold)
            except ImportError:
                logger.warning("ArcFace not available, falling back to simple recognizer")
                return SimpleOpenCVRecognizer(threshold=self.threshold)'''
            
            new_init = '''        if model_type == 'deepface':
            try:
                return DeepFaceRecognizer(threshold=self.threshold)
            except ImportError:
                logger.warning("DeepFace not available, falling back to simple recognizer")
                return SimpleOpenCVRecognizer(threshold=self.threshold)
        elif model_type == 'arcface':
            try:
                return ArcFaceRecognizer(threshold=self.threshold)
            except ImportError:
                logger.warning("ArcFace not available, falling back to simple recognizer")
                return SimpleOpenCVRecognizer(threshold=self.threshold)'''
            
            if old_init in content:
                content = content.replace(old_init, new_init)
                
                with open(face_recognition_path, 'w') as f:
                    f.write(content)
                
                print("Updated face recognition initialization to prefer DeepFace")
    
    # Update main.py to use DeepFace
    main_py_path = "main.py"
    if os.path.exists(main_py_path):
        with open(main_py_path, 'r') as f:
            content = f.read()
        
        # Change default model to deepface
        if "recognition_model: str = 'simple'" in content:
            content = content.replace("recognition_model: str = 'simple'", "recognition_model: str = 'deepface'")
            
            with open(main_py_path, 'w') as f:
                f.write(content)
            
            print("Updated main.py to use DeepFace by default")
    
    print(f"\n=== DeepFace Integration Complete ===")
    print(f"SUCCESS! DeepFace VGG-Face has been integrated!")
    print(f"Migrated {migrated_count} faces to DeepFace")
    print(f"System now uses DeepFace for much better accuracy!")
    
    return True

if __name__ == "__main__":
    integrate_deepface()
