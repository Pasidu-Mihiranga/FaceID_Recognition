#!/usr/bin/env python3
"""
Improve face recognition consistency by rebuilding database with better logic
"""

import sys
sys.path.append('src')

from main import FaceIDSystem
import cv2
import numpy as np
import os
import pickle
import glob

def improve_recognition_consistency():
    """Improve face recognition consistency"""
    
    print("=== Improving Face Recognition Consistency ===")
    
    # Initialize system
    system = FaceIDSystem()
    
    # Get all registered persons
    persons = system.database.get_all_persons()
    print(f"Found {len(persons)} registered persons:")
    
    for person in persons:
        print(f"- {person['name']} (ID: {person['id']}) - {person['total_images']} images")
    
    # Clear existing face database
    system.face_recognizer.face_database = {}
    
    # Load embeddings for each person with consistency improvements
    for person in persons:
        person_id = person['id']
        person_name = person['name']
        
        print(f"\nProcessing {person_name} (ID: {person_id}):")
        
        # Find all embedding files for this person
        embedding_files = glob.glob(f"data/embeddings/face_{person_id}_*.pkl")
        
        embeddings_loaded = 0
        valid_embeddings = []
        
        for embedding_file in embedding_files:
            try:
                with open(embedding_file, 'rb') as f:
                    embedding = pickle.load(f)
                
                # Validate embedding
                if isinstance(embedding, np.ndarray) and embedding.size > 0:
                    valid_embeddings.append(embedding)
                    embeddings_loaded += 1
                else:
                    print(f"  Skipping invalid embedding: {embedding_file}")
                
            except Exception as e:
                print(f"  Error loading {embedding_file}: {e}")
        
        # Use only the best embeddings (most consistent ones)
        if valid_embeddings:
            # Calculate consistency between embeddings
            if len(valid_embeddings) > 1:
                consistency_scores = []
                for i, emb1 in enumerate(valid_embeddings):
                    scores = []
                    for j, emb2 in enumerate(valid_embeddings):
                        if i != j:
                            similarity = system.face_recognizer.recognizer.compare_faces(emb1, emb2)
                            scores.append(similarity)
                    consistency_scores.append(np.mean(scores))
                
                # Select the most consistent embeddings (top 50%)
                sorted_indices = np.argsort(consistency_scores)[::-1]
                num_to_keep = max(1, len(valid_embeddings) // 2)
                best_embeddings = [valid_embeddings[i] for i in sorted_indices[:num_to_keep]]
                
                print(f"  Selected {len(best_embeddings)} most consistent embeddings out of {len(valid_embeddings)}")
            else:
                best_embeddings = valid_embeddings
            
            # Add to face database
            system.face_recognizer.face_database[person_name] = best_embeddings
        
        print(f"  Loaded {embeddings_loaded} embeddings")
    
    # Increase threshold for better accuracy
    original_threshold = system.face_recognizer.threshold
    system.face_recognizer.threshold = 0.8  # Higher threshold for better accuracy
    print(f"\nIncreased recognition threshold from {original_threshold} to {system.face_recognizer.threshold}")
    
    # Save the improved database
    db_path = "data/embeddings/simple_database.pkl"
    try:
        with open(db_path, 'wb') as f:
            pickle.dump(system.face_recognizer.face_database, f)
        print(f"Improved database saved to {db_path}")
    except Exception as e:
        print(f"Error saving database: {e}")
        return False
    
    # Verify the improved database
    print(f"\nImproved database contains {len(system.face_recognizer.face_database)} persons:")
    for person_name, embeddings in system.face_recognizer.face_database.items():
        print(f"- {person_name}: {len(embeddings)} embeddings")
        
        # Test consistency within person
        if len(embeddings) > 1:
            consistency_scores = []
            for i, emb1 in enumerate(embeddings):
                for j, emb2 in enumerate(embeddings):
                    if i != j:
                        similarity = system.face_recognizer.recognizer.compare_faces(emb1, emb2)
                        consistency_scores.append(similarity)
            
            avg_consistency = np.mean(consistency_scores)
            print(f"  Average consistency: {avg_consistency:.4f}")
    
    # Test recognition consistency
    print(f"\nTesting recognition consistency:")
    
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
            for i in range(3):
                try:
                    image = cv2.imread(test_image_path)
                    if image is not None:
                        person_name, confidence, face_info = system.recognize_face(image)
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
                else:
                    print(f"  INCONSISTENT: Results vary - {set(person_names)}")
        else:
            print(f"\nImage not found: {test_image_path}")
    
    print(f"\n=== Consistency Improvement Complete ===")
    return True

if __name__ == "__main__":
    improve_recognition_consistency()
