#!/usr/bin/env python3
"""
Test recognition accuracy after database rebuild
"""

import sys
sys.path.append('src')

from main import FaceIDSystem
import cv2
import numpy as np

def test_recognition_accuracy():
    """Test recognition accuracy after database rebuild"""
    
    print("=== Testing Recognition Accuracy ===")
    
    # Initialize system with rebuilt database
    system = FaceIDSystem()
    
    # Check face database
    print(f"Face database size: {len(system.face_recognizer.face_database)}")
    
    for person_name, embeddings in system.face_recognizer.face_database.items():
        print(f"- {person_name}: {len(embeddings)} embeddings")
    
    # Test cross-person similarity
    print(f"\nTesting cross-person similarity:")
    person_names = list(system.face_recognizer.face_database.keys())
    
    # Focus on pasidu mihi and kavinu
    target_persons = ['pasidu mihi', 'kavinu']
    available_persons = [p for p in target_persons if p in person_names]
    
    if len(available_persons) >= 2:
        person1 = available_persons[0]
        person2 = available_persons[1]
        
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
    
    # Test recognition with different images
    print(f"\nTesting recognition with different images:")
    
    test_images = [
        "data/simple_face_test.jpg",
        "data/test_person1.jpg", 
        "data/test_person2.jpg"
    ]
    
    for test_image_path in test_images:
        if os.path.exists(test_image_path):
            print(f"\nTesting with: {test_image_path}")
            try:
                image = cv2.imread(test_image_path)
                if image is not None:
                    person_name, confidence, face_info = system.recognize_face(image)
                    print(f"  Recognized as: {person_name} (confidence: {confidence:.4f})")
                else:
                    print(f"  Could not load image")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"\nImage not found: {test_image_path}")
    
    print(f"\n=== Accuracy Test Complete ===")

if __name__ == "__main__":
    import os
    test_recognition_accuracy()
