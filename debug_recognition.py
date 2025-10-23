#!/usr/bin/env python3
"""
Debug recognition threshold and database
"""

import sys
sys.path.append('src')

from main import FaceIDSystem
import cv2
import numpy as np

def debug_recognition():
    """Debug recognition threshold and database"""
    
    print("=== Debugging Recognition System ===")
    
    # Initialize system
    system = FaceIDSystem()
    
    # Check threshold
    print(f"Recognition threshold: {system.face_recognizer.threshold}")
    
    # Check face database
    print(f"Face database size: {len(system.face_recognizer.face_database)}")
    
    for person_name, embeddings in system.face_recognizer.face_database.items():
        print(f"- {person_name}: {len(embeddings)} embeddings")
    
    # Test with a simple image
    test_image_path = "data/simple_face_test.jpg"
    try:
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"❌ Could not load test image: {test_image_path}")
            return
            
        print(f"\nTesting recognition with: {test_image_path}")
        
        # Detect faces
        faces = system.face_detector.detect_faces(image)
        print(f"Faces detected: {len(faces)}")
        
        if faces:
            face_info = faces[0]
            face_image = system.face_detector.extract_face(image, face_info)
            print(f"Face image shape: {face_image.shape}")
            
            # Extract embedding
            embedding = system.face_recognizer.recognizer.extract_embedding(face_image)
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding sample: {embedding[:5]}")
            
            # Test recognition with detailed output
            person_name, confidence, face_info = system.recognize_face(image)
            
            print(f"\nRecognition result:")
            print(f"  Person: {person_name}")
            print(f"  Confidence: {confidence}")
            print(f"  Threshold: {system.face_recognizer.threshold}")
            print(f"  Above threshold: {confidence >= system.face_recognizer.threshold}")
            
            # Test with lower threshold
            print(f"\nTesting with lower threshold (0.3):")
            original_threshold = system.face_recognizer.threshold
            system.face_recognizer.threshold = 0.3
            
            person_name2, confidence2, face_info2 = system.recognize_face(image)
            print(f"  Person: {person_name2}")
            print(f"  Confidence: {confidence2}")
            
            # Restore original threshold
            system.face_recognizer.threshold = original_threshold
            
    except Exception as e:
        print(f"❌ Recognition test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_recognition()
