#!/usr/bin/env python3
"""
Ultimate VGG-Face Optimization for Maximum Accuracy
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

def implement_ultimate_vgg_optimization():
    """Implement ultimate VGG-Face optimization for maximum accuracy"""
    
    print("=== Implementing Ultimate VGG-Face Optimization ===")
    
    # Test VGG-Face model
    print("1. Testing VGG-Face model...")
    
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    try:
        result = DeepFace.represent(
            img_path=test_img,
            model_name='VGG-Face',
            enforce_detection=False
        )
        print("   SUCCESS: VGG-Face is available and working")
        embedding_size = len(result[0]['embedding'])
        print(f"   Embedding size: {embedding_size}")
    except Exception as e:
        print(f"   FAILED: VGG-Face error - {e}")
        return False
    
    # 2. Create Ultimate VGG-Face Recognizer
    print(f"\n2. Creating Ultimate VGG-Face Recognizer...")
    
    class UltimateVGGRecognizer:
        """Ultimate VGG-Face recognizer with advanced optimization"""
        
        def __init__(self, threshold=0.2):  # Very lenient for maximum sensitivity
            self.model_name = 'VGG-Face'
            self.threshold = threshold
            self.face_database = {}
            self.database_path = "data/embeddings/ultimate_vgg_database.pkl"
            
            try:
                from deepface import DeepFace
                self.DeepFace = DeepFace
                print(f"   Ultimate VGG recognizer initialized")
            except ImportError:
                raise ImportError("DeepFace not available")
            
            self.load_database()
        
        def preprocess_face_ultimate(self, face_image):
            """Ultimate face preprocessing with multiple variations"""
            try:
                # Convert to RGB if needed
                if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = face_image
                
                # Resize to optimal size for VGG-Face
                rgb_image = cv2.resize(rgb_image, (224, 224))
                
                # Create multiple preprocessing variations
                variations = []
                
                # 1. Original
                variations.append(rgb_image.copy())
                
                # 2. Brightness enhanced
                bright_image = cv2.convertScaleAbs(rgb_image, alpha=1.3, beta=30)
                variations.append(bright_image)
                
                # 3. Contrast enhanced
                contrast_image = cv2.convertScaleAbs(rgb_image, alpha=1.4, beta=0)
                variations.append(contrast_image)
                
                # 4. Histogram equalization
                if len(rgb_image.shape) == 3:
                    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    equalized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    variations.append(equalized_image)
                
                # 5. Gaussian blur for noise reduction
                blurred_image = cv2.GaussianBlur(rgb_image, (3, 3), 0)
                variations.append(blurred_image)
                
                # 6. Sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened_image = cv2.filter2D(rgb_image, -1, kernel)
                variations.append(sharpened_image)
                
                # 7. Gamma correction
                gamma = 1.2
                gamma_corrected = np.power(rgb_image / 255.0, gamma) * 255.0
                gamma_corrected = np.uint8(gamma_corrected)
                variations.append(gamma_corrected)
                
                return variations
                
            except Exception as e:
                print(f"   Ultimate preprocessing failed: {e}")
                return [face_image]
        
        def extract_embedding_ultimate(self, face_image):
            """Extract embedding with ultimate preprocessing"""
            try:
                # Get all preprocessing variations
                variations = self.preprocess_face_ultimate(face_image)
                
                embeddings = []
                
                for variation in variations:
                    try:
                        embedding = self.DeepFace.represent(
                            img_path=variation,
                            model_name=self.model_name,
                            enforce_detection=False,
                            detector_backend='opencv'
                        )
                        embeddings.append(np.array(embedding[0]['embedding']))
                    except Exception as e:
                        print(f"   Embedding extraction failed for variation: {e}")
                
                if embeddings:
                    # Return average of all embeddings for robustness
                    avg_embedding = np.mean(embeddings, axis=0)
                    return avg_embedding
                else:
                    raise Exception("No embeddings extracted")
                
            except Exception as e:
                print(f"   Ultimate embedding extraction failed: {e}")
                raise
        
        def compare_faces_ultimate(self, embedding1, embedding2):
            """Ultimate face comparison with multiple metrics"""
            try:
                # Method 1: Cosine similarity
                emb1_norm = embedding1 / np.linalg.norm(embedding1)
                emb2_norm = embedding2 / np.linalg.norm(embedding2)
                cosine_sim = np.dot(emb1_norm, emb2_norm)
                
                # Method 2: Euclidean distance (normalized)
                euclidean_dist = np.linalg.norm(embedding1 - embedding2)
                euclidean_sim = 1.0 / (1.0 + euclidean_dist)
                
                # Method 3: Manhattan distance (normalized)
                manhattan_dist = np.sum(np.abs(embedding1 - embedding2))
                manhattan_sim = 1.0 / (1.0 + manhattan_dist)
                
                # Method 4: Pearson correlation
                correlation = np.corrcoef(embedding1, embedding2)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                correlation_sim = (correlation + 1) / 2
                
                # Method 5: Dot product (normalized)
                dot_product = np.dot(embedding1, embedding2)
                dot_sim = dot_product / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                
                # Weighted ensemble for maximum accuracy
                final_similarity = (
                    0.3 * cosine_sim + 
                    0.25 * euclidean_sim + 
                    0.2 * manhattan_sim + 
                    0.15 * correlation_sim + 
                    0.1 * dot_sim
                )
                
                # Convert to 0-1 range
                final_similarity = (final_similarity + 1) / 2
                
                return float(max(0.0, min(1.0, final_similarity)))
                
            except Exception as e:
                print(f"   Ultimate face comparison failed: {e}")
                return 0.0
        
        def register_face_ultimate(self, face_image, person_name):
            """Ultimate face registration with multiple embeddings"""
            try:
                # Extract ultimate embedding
                embedding = self.extract_embedding_ultimate(face_image)
                
                # Add to database
                if person_name not in self.face_database:
                    self.face_database[person_name] = []
                
                self.face_database[person_name].append(embedding)
                self.save_database()
                
                print(f"   Registered ultimate embedding for {person_name}")
                return True
                
            except Exception as e:
                print(f"   Ultimate face registration failed for {person_name}: {e}")
                return False
        
        def recognize_face_ultimate(self, face_image):
            """Ultimate face recognition with ensemble method"""
            try:
                # Extract ultimate embedding
                embedding = self.extract_embedding_ultimate(face_image)
                
                best_match = None
                best_score = 0.0
                all_scores = {}
                
                # Compare with all registered faces
                for person_name, embeddings in self.face_database.items():
                    scores = []
                    
                    for stored_embedding in embeddings:
                        similarity = self.compare_faces_ultimate(embedding, stored_embedding)
                        scores.append(similarity)
                    
                    if scores:
                        # Use maximum score for this person (most similar match)
                        max_score = np.max(scores)
                        all_scores[person_name] = max_score
                        
                        if max_score > best_score:
                            best_score = max_score
                            best_match = person_name
                
                # Debug information
                print(f"   Ultimate recognition scores: {all_scores}")
                
                if best_score >= self.threshold:
                    return best_match, best_score
                else:
                    return None, best_score
                    
            except Exception as e:
                print(f"   Ultimate face recognition failed: {e}")
                return None, 0.0
        
        def save_database(self):
            """Save face database"""
            try:
                os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
                with open(self.database_path, 'wb') as f:
                    pickle.dump(self.face_database, f)
                print(f"   Ultimate VGG database saved to {self.database_path}")
            except Exception as e:
                print(f"   Failed to save ultimate VGG database: {e}")
        
        def load_database(self):
            """Load face database"""
            try:
                if os.path.exists(self.database_path):
                    with open(self.database_path, 'rb') as f:
                        self.face_database = pickle.load(f)
                    print(f"   Ultimate VGG database loaded from {self.database_path}")
                else:
                    print("   No existing ultimate VGG database found, starting fresh")
            except Exception as e:
                print(f"   Failed to load ultimate VGG database: {e}")
                self.face_database = {}
    
    # Initialize ultimate VGG recognizer
    ultimate_vgg_recognizer = UltimateVGGRecognizer(threshold=0.2)
    
    # 3. Migrate existing faces with ultimate preprocessing
    print(f"\n3. Migrating faces with ultimate VGG preprocessing...")
    
    system = FaceIDSystem()
    persons = system.database.get_all_persons()
    
    migrated_count = 0
    
    for person in persons:
        person_id = person['id']
        person_name = person['name']
        
        print(f"   Migrating {person_name} (ID: {person_id}):")
        
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
                        # Load original image
                        image = cv2.imread(image_path)
                        
                        if image is not None:
                            # Use face crop if available, otherwise full image
                            if face_bbox_json:
                                face_bbox = json.loads(face_bbox_json)
                                x, y, w, h = face_bbox
                                face_image_crop = image[y:y+h, x:x+w]
                            else:
                                face_image_crop = image
                            
                            # Register with ultimate VGG recognizer
                            success = ultimate_vgg_recognizer.register_face_ultimate(face_image_crop, person_name)
                            
                            if success:
                                migrated_count += 1
                                print(f"     Migrated: {image_path}")
                            else:
                                print(f"     Failed: {image_path}")
                        else:
                            print(f"     Could not load: {image_path}")
                    except Exception as e:
                        print(f"     Error migrating {image_path}: {e}")
                else:
                    print(f"     Image not found: {image_path}")
                    
        except Exception as e:
            print(f"   Error accessing database for {person_name}: {e}")
    
    print(f"\nUltimate VGG migration complete! Migrated {migrated_count} faces.")
    
    # 4. Test ultimate VGG recognition
    print(f"\n4. Testing ultimate VGG recognition accuracy...")
    
    test_images = [
        "data/simple_face_test.jpg",
        "data/test_person1.jpg", 
        "data/test_person2.jpg"
    ]
    
    for test_image_path in test_images:
        if os.path.exists(test_image_path):
            print(f"\n   Testing: {test_image_path}")
            
            # Test multiple times for consistency
            results = []
            for i in range(3):
                try:
                    image = cv2.imread(test_image_path)
                    if image is not None:
                        person_name, confidence = ultimate_vgg_recognizer.recognize_face_ultimate(image)
                        results.append((person_name, confidence))
                        print(f"     Test {i+1}: {person_name} (confidence: {confidence:.4f})")
                    else:
                        print(f"     Test {i+1}: Could not load image")
                except Exception as e:
                    print(f"     Test {i+1}: Error - {e}")
            
            # Check consistency
            if len(results) > 1:
                person_names = [r[0] for r in results]
                confidences = [r[1] for r in results]
                
                if len(set(person_names)) == 1:
                    print(f"     CONSISTENT: All results show {person_names[0]}")
                    print(f"     Confidence range: {min(confidences):.4f} - {max(confidences):.4f}")
                    print(f"     Confidence std: {np.std(confidences):.4f}")
                else:
                    print(f"     INCONSISTENT: Results vary - {set(person_names)}")
        else:
            print(f"\n   Image not found: {test_image_path}")
    
    # 5. Update system configuration
    print(f"\n5. Updating system configuration...")
    
    # Update main.py threshold
    main_py_path = "main.py"
    if os.path.exists(main_py_path):
        with open(main_py_path, 'r') as f:
            content = f.read()
        
        # Update threshold to ultimate value
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'recognition_threshold: float =' in line:
                lines[i] = f"                     recognition_threshold: float = 0.2,"
                break
        
        content = '\n'.join(lines)
        
        with open(main_py_path, 'w') as f:
            f.write(content)
        
        print(f"   Updated main.py threshold to 0.2")
    
    # Update face recognition module threshold
    face_recognition_path = "src/face_recognition/__init__.py"
    if os.path.exists(face_recognition_path):
        with open(face_recognition_path, 'r') as f:
            content = f.read()
        
        # Update threshold
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'threshold: float =' in line and 'FaceRecognitionManager' in lines[max(0, i-5):i]:
                lines[i] = f"        threshold: float = 0.2):"
                break
        
        content = '\n'.join(lines)
        
        with open(face_recognition_path, 'w') as f:
            f.write(content)
        
        print(f"   Updated face_recognition module threshold to 0.2")
    
    print(f"\n=== Ultimate VGG-Face Optimization Complete ===")
    print(f"SUCCESS! Ultimate VGG-Face optimization implemented!")
    print(f"Key improvements:")
    print(f"- Ultimate preprocessing: 7 variations per image")
    print(f"- Ensemble comparison: 5 different similarity metrics")
    print(f"- Maximum sensitivity: Threshold 0.2")
    print(f"- Robust embeddings: Average of multiple preprocessing")
    print(f"- Migrated {migrated_count} faces with ultimate preprocessing")
    
    return True

if __name__ == "__main__":
    implement_ultimate_vgg_optimization()
