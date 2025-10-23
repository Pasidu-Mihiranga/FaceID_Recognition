#!/usr/bin/env python3
"""
Debug face recognition accuracy
"""

import sys
sys.path.append('src')

from main import FaceIDSystem
import cv2
import numpy as np

def debug_recognition_accuracy():
    """Debug face recognition accuracy issues"""
    
    print("=== Debugging Face Recognition Accuracy ===")
    
    # Initialize system
    system = FaceIDSystem()
    
    # Check registered persons
    persons = system.database.get_all_persons()
    print(f"\nRegistered persons: {len(persons)}")
    
    for person in persons:
        print(f"- {person['name']} (ID: {person['id']}) - {person['total_images']} images")
    
    # Check face database
    print(f"\nFace database size: {len(system.face_recognizer.face_database)}")
    
    for person_name, embeddings in system.face_recognizer.face_database.items():
        print(f"- {person_name}: {len(embeddings)} embeddings")
        
        # Check embedding similarity within same person
        if len(embeddings) > 1:
            print(f"  Testing similarity within {person_name}:")
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    similarity = system.face_recognizer.recognizer.compare_faces(embeddings[i], embeddings[j])
                    print(f"    Embedding {i+1} vs {j+1}: {similarity:.4f}")
    
    # Test cross-person similarity
    print(f"\nTesting cross-person similarity:")
    person_names = list(system.face_recognizer.face_database.keys())
    
    for i in range(len(person_names)):
        for j in range(i+1, len(person_names)):
            person1 = person_names[i]
            person2 = person_names[j]
            
            print(f"\n{person1} vs {person2}:")
            
            # Compare first embeddings of each person
            emb1 = system.face_recognizer.face_database[person1][0]
            emb2 = system.face_recognizer.face_database[person2][0]
            
            similarity = system.face_recognizer.recognizer.compare_faces(emb1, emb2)
            print(f"  Similarity: {similarity:.4f}")
            
            # Check if similarity is too high (problematic)
            if similarity > 0.5:
                print(f"  WARNING: HIGH SIMILARITY - May cause misidentification!")
            elif similarity > 0.3:
                print(f"  WARNING: MODERATE SIMILARITY - Could be problematic")
            else:
                print(f"  OK: LOW SIMILARITY - Good separation")
    
    # Check recognition threshold
    print(f"\nRecognition threshold: {system.face_recognizer.threshold}")
    
    # Test with different thresholds
    print(f"\nTesting different thresholds:")
    original_threshold = system.face_recognizer.threshold
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        system.face_recognizer.threshold = threshold
        print(f"  Threshold {threshold}: ", end="")
        
        # Test with a simple image
        test_image_path = "data/simple_face_test.jpg"
        try:
            image = cv2.imread(test_image_path)
            if image is not None:
                person_name, confidence, face_info = system.recognize_face(image)
                print(f"Recognized as: {person_name} (confidence: {confidence:.4f})")
            else:
                print("No test image available")
        except Exception as e:
            print(f"Error: {e}")
    
    # Restore original threshold
    system.face_recognizer.threshold = original_threshold
    
    print(f"\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_recognition_accuracy()
