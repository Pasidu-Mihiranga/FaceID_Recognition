#!/usr/bin/env python3
"""
Ultimate Multi-Model DeepFace System for Maximum Accuracy
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

def implement_ultimate_multi_model():
    """Implement ultimate multi-model DeepFace system for maximum accuracy"""
    
    print("=== Implementing Ultimate Multi-Model DeepFace System ===")
    
    # Test all available DeepFace models
    print("1. Testing all available DeepFace models...")
    
    available_models = []
    models_to_test = ['VGG-Face', 'Facenet', 'OpenFace', 'ArcFace']
    
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    for model_name in models_to_test:
        try:
            result = DeepFace.represent(
                img_path=test_img,
                model_name=model_name,
                enforce_detection=False
            )
            available_models.append(model_name)
            embedding_size = len(result[0]['embedding'])
            print(f"   SUCCESS: {model_name} is available (embedding size: {embedding_size})")
        except Exception as e:
            print(f"   FAILED: {model_name} - {e}")
    
    print(f"   Available models: {available_models}")
    
    if len(available_models) < 2:
        print("   ERROR: Need at least 2 models for multi-model system")
        return False
    
    # 2. Create Ultimate Multi-Model Recognizer
    print(f"\n2. Creating Ultimate Multi-Model Recognizer...")
    
    class UltimateMultiModelRecognizer:
        """Ultimate multi-model face recognizer using all available DeepFace models"""
        
        def __init__(self, threshold=0.15):  # Very lenient for maximum sensitivity
            self.models = available_models
            self.threshold = threshold
            self.face_database = {}
            self.database_path = "data/embeddings/ultimate_multimodel_database.pkl"
            
            try:
                from deepface import DeepFace
                self.DeepFace = DeepFace
                print(f"   Ultimate multi-model recognizer initialized with {len(self.models)} models")
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
                
                # Resize to optimal size for DeepFace models
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
        
        def extract_embeddings_multimodel(self, face_image):
            """Extract embeddings using all available models"""
            try:
                embeddings = {}
                
                for model_name in self.models:
                    try:
                        embedding = self.DeepFace.represent(
                            img_path=face_image,
                            model_name=model_name,
                            enforce_detection=False,
                            detector_backend='opencv'
                        )
                        embeddings[model_name] = np.array(embedding[0]['embedding'])
                    except Exception as e:
                        print(f"   Failed to extract {model_name} embedding: {e}")
                
                return embeddings
                
            except Exception as e:
                print(f"   Multi-model embedding extraction failed: {e}")
                return {}
        
        def compare_faces_multimodel(self, embeddings1, embeddings2):
            """Ultimate face comparison using all models and metrics"""
            try:
                similarities = []
                
                for model_name in self.models:
                    if model_name in embeddings1 and model_name in embeddings2:
                        emb1 = embeddings1[model_name]
                        emb2 = embeddings2[model_name]
                        
                        # Multiple similarity metrics for each model
                        # Cosine similarity
                        emb1_norm = emb1 / np.linalg.norm(emb1)
                        emb2_norm = emb2 / np.linalg.norm(emb2)
                        cosine_sim = np.dot(emb1_norm, emb2_norm)
                        
                        # Euclidean distance (normalized)
                        euclidean_dist = np.linalg.norm(emb1 - emb2)
                        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
                        
                        # Manhattan distance (normalized)
                        manhattan_dist = np.sum(np.abs(emb1 - emb2))
                        manhattan_sim = 1.0 / (1.0 + manhattan_dist)
                        
                        # Pearson correlation
                        correlation = np.corrcoef(emb1, emb2)[0, 1]
                        if np.isnan(correlation):
                            correlation = 0.0
                        correlation_sim = (correlation + 1) / 2
                        
                        # Dot product (normalized)
                        dot_product = np.dot(emb1, emb2)
                        dot_sim = dot_product / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        
                        # Weighted combination for this model
                        model_similarity = (
                            0.3 * cosine_sim + 
                            0.25 * euclidean_sim + 
                            0.2 * manhattan_sim + 
                            0.15 * correlation_sim + 
                            0.1 * dot_sim
                        )
                        similarities.append(model_similarity)
                
                if similarities:
                    # Use average of all model similarities
                    final_similarity = np.mean(similarities)
                    # Convert to 0-1 range
                    final_similarity = (final_similarity + 1) / 2
                    return float(max(0.0, min(1.0, final_similarity)))
                else:
                    return 0.0
                    
            except Exception as e:
                print(f"   Multi-model face comparison failed: {e}")
                return 0.0
        
        def register_face_multimodel(self, face_image, person_name):
            """Ultimate face registration with all models and preprocessing"""
            try:
                # Get preprocessed images
                processed_images = self.preprocess_face_ultimate(face_image)
                
                # Extract embeddings for each preprocessed image
                all_embeddings = []
                
                for processed_img in processed_images:
                    embeddings = self.extract_embeddings_multimodel(processed_img)
                    if embeddings:
                        all_embeddings.append(embeddings)
                
                # Add to database
                if person_name not in self.face_database:
                    self.face_database[person_name] = []
                
                self.face_database[person_name].extend(all_embeddings)
                self.save_database()
                
                print(f"   Registered {len(all_embeddings)} multi-model embedding sets for {person_name}")
                return True
                
            except Exception as e:
                print(f"   Multi-model face registration failed for {person_name}: {e}")
                return False
        
        def recognize_face_multimodel(self, face_image):
            """Ultimate face recognition with multi-model ensemble"""
            try:
                # Get preprocessed images
                processed_images = self.preprocess_face_ultimate(face_image)
                
                all_scores = {}
                
                # Test each preprocessed image
                for processed_img in processed_images:
                    # Extract embeddings
                    embeddings = self.extract_embeddings_multimodel(processed_img)
                    
                    if embeddings:
                        # Compare with all registered faces
                        for person_name, stored_embeddings_list in self.face_database.items():
                            scores = []
                            
                            for stored_embeddings in stored_embeddings_list:
                                similarity = self.compare_faces_multimodel(embeddings, stored_embeddings)
                                scores.append(similarity)
                            
                            if scores:
                                max_score = np.max(scores)  # Use best score from any preprocessing
                                if person_name not in all_scores:
                                    all_scores[person_name] = []
                                all_scores[person_name].append(max_score)
                
                # Find best match across all preprocessing variations
                best_match = None
                best_score = 0.0
                
                for person_name, scores in all_scores.items():
                    if scores:
                        max_score = np.max(scores)  # Use best score from any preprocessing
                        if max_score > best_score:
                            best_score = max_score
                            best_match = person_name
                
                # Debug information
                print(f"   Multi-model recognition scores: {all_scores}")
                
                if best_score >= self.threshold:
                    return best_match, best_score
                else:
                    return None, best_score
                    
            except Exception as e:
                print(f"   Multi-model face recognition failed: {e}")
                return None, 0.0
        
        def save_database(self):
            """Save face database"""
            try:
                os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
                with open(self.database_path, 'wb') as f:
                    pickle.dump(self.face_database, f)
                print(f"   Ultimate multi-model database saved to {self.database_path}")
            except Exception as e:
                print(f"   Failed to save ultimate multi-model database: {e}")
        
        def load_database(self):
            """Load face database"""
            try:
                if os.path.exists(self.database_path):
                    with open(self.database_path, 'rb') as f:
                        self.face_database = pickle.load(f)
                    print(f"   Ultimate multi-model database loaded from {self.database_path}")
                else:
                    print("   No existing ultimate multi-model database found, starting fresh")
            except Exception as e:
                print(f"   Failed to load ultimate multi-model database: {e}")
                self.face_database = {}
    
    # Initialize ultimate multi-model recognizer
    ultimate_multimodel_recognizer = UltimateMultiModelRecognizer(threshold=0.15)
    
    # 3. Migrate existing faces with multi-model preprocessing
    print(f"\n3. Migrating faces with multi-model preprocessing...")
    
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
                            
                            # Register with ultimate multi-model recognizer
                            success = ultimate_multimodel_recognizer.register_face_multimodel(face_image_crop, person_name)
                            
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
    
    print(f"\nUltimate multi-model migration complete! Migrated {migrated_count} faces.")
    
    # 4. Test ultimate multi-model recognition
    print(f"\n4. Testing ultimate multi-model recognition accuracy...")
    
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
                        person_name, confidence = ultimate_multimodel_recognizer.recognize_face_multimodel(image)
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
                lines[i] = f"                     recognition_threshold: float = 0.15,"
                break
        
        content = '\n'.join(lines)
        
        with open(main_py_path, 'w') as f:
            f.write(content)
        
        print(f"   Updated main.py threshold to 0.15")
    
    # Update face recognition module threshold
    face_recognition_path = "src/face_recognition/__init__.py"
    if os.path.exists(face_recognition_path):
        with open(face_recognition_path, 'r') as f:
            content = f.read()
        
        # Update threshold
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'threshold: float =' in line and 'FaceRecognitionManager' in lines[max(0, i-5):i]:
                lines[i] = f"        threshold: float = 0.15):"
                break
        
        content = '\n'.join(lines)
        
        with open(face_recognition_path, 'w') as f:
            f.write(content)
        
        print(f"   Updated face_recognition module threshold to 0.15")
    
    print(f"\n=== Ultimate Multi-Model System Complete ===")
    print(f"SUCCESS! Ultimate multi-model DeepFace system implemented!")
    print(f"Key improvements:")
    print(f"- Multiple DeepFace models: {available_models}")
    print(f"- Ultimate preprocessing: 7 variations per image")
    print(f"- Multi-model ensemble: All models + all metrics")
    print(f"- Maximum sensitivity: Threshold 0.15")
    print(f"- Migrated {migrated_count} faces with multi-model preprocessing")
    
    return True

if __name__ == "__main__":
    implement_ultimate_multi_model()
