#!/usr/bin/env python3
"""
Fix face recognition database by rebuilding from individual embedding files
"""

import sys
sys.path.append('src')

from main import FaceIDSystem
import os
import pickle
import glob

def rebuild_face_database():
    """Rebuild the face recognition database from individual embedding files"""
    
    print("=== Rebuilding Face Recognition Database ===")
    
    # Initialize system
    system = FaceIDSystem()
    
    # Get all registered persons
    persons = system.database.get_all_persons()
    print(f"Found {len(persons)} registered persons:")
    
    for person in persons:
        print(f"- {person['name']} (ID: {person['id']}) - {person['total_images']} images")
    
    # Clear existing face database
    system.face_recognizer.face_database = {}
    
    # Load embeddings for each person
    for person in persons:
        person_id = person['id']
        person_name = person['name']
        
        print(f"\nLoading embeddings for {person_name} (ID: {person_id}):")
        
        # Find all embedding files for this person
        embedding_files = glob.glob(f"data/embeddings/face_{person_id}_*.pkl")
        
        embeddings_loaded = 0
        for embedding_file in embedding_files:
            try:
                with open(embedding_file, 'rb') as f:
                    embedding = pickle.load(f)
                
                # Add to face database
                if person_name not in system.face_recognizer.face_database:
                    system.face_recognizer.face_database[person_name] = []
                
                system.face_recognizer.face_database[person_name].append(embedding)
                embeddings_loaded += 1
                
            except Exception as e:
                print(f"  Error loading {embedding_file}: {e}")
        
        print(f"  Loaded {embeddings_loaded} embeddings")
    
    # Save the rebuilt database
    db_path = "data/embeddings/simple_database.pkl"
    try:
        with open(db_path, 'wb') as f:
            pickle.dump(system.face_recognizer.face_database, f)
        print(f"\nDatabase saved to {db_path}")
    except Exception as e:
        print(f"Error saving database: {e}")
        return False
    
    # Verify the rebuilt database
    print(f"\nRebuilt database contains {len(system.face_recognizer.face_database)} persons:")
    for person_name, embeddings in system.face_recognizer.face_database.items():
        print(f"- {person_name}: {len(embeddings)} embeddings")
    
    # Test recognition with the rebuilt database
    print(f"\nTesting recognition with rebuilt database:")
    
    # Test with a simple image
    test_image_path = "data/simple_face_test.jpg"
    try:
        import cv2
        image = cv2.imread(test_image_path)
        if image is not None:
            person_name, confidence, face_info = system.recognize_face(image)
            print(f"Test recognition: {person_name} (confidence: {confidence:.4f})")
        else:
            print("No test image available")
    except Exception as e:
        print(f"Test recognition error: {e}")
    
    print(f"\n=== Database Rebuild Complete ===")
    return True

if __name__ == "__main__":
    rebuild_face_database()
