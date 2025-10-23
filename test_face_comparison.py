#!/usr/bin/env python3
"""
Test face comparison directly
"""

import sys
sys.path.append('src')

from main import FaceIDSystem
from src.face_recognition import SimpleOpenCVRecognizer
import cv2
import numpy as np

def test_face_comparison():
    """Test face comparison directly"""
    
    print("=== Testing Face Comparison ===")
    
    # Initialize system
    system = FaceIDSystem()
    
    # Test with a simple image
    test_image_path = "data/simple_face_test.jpg"
    try:
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"❌ Could not load test image: {test_image_path}")
            return
            
        print(f"Testing with: {test_image_path}")
        
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
            print(f"Embedding sum: {np.sum(embedding)}")
            print(f"Embedding min/max: {np.min(embedding)}/{np.max(embedding)}")
            
            # Test comparison with stored embeddings
            print(f"\nTesting comparisons:")
            for person_name, embeddings in system.face_recognizer.face_database.items():
                print(f"\nPerson: {person_name}")
                for i, stored_embedding in enumerate(embeddings):
                    similarity = system.face_recognizer.recognizer.compare_faces(embedding, stored_embedding)
                    print(f"  Embedding {i+1}: similarity = {similarity}")
                    
                    # Check if stored embedding is valid
                    print(f"    Stored embedding shape: {stored_embedding.shape}")
                    print(f"    Stored embedding sum: {np.sum(stored_embedding)}")
                    print(f"    Stored embedding min/max: {np.min(stored_embedding)}/{np.max(stored_embedding)}")
            
            # Test with a simple correlation
            print(f"\nTesting simple correlation:")
            for person_name, embeddings in system.face_recognizer.face_database.items():
                for i, stored_embedding in enumerate(embeddings):
                    # Direct correlation
                    correlation = np.corrcoef(embedding, stored_embedding)[0, 1]
                    print(f"  {person_name} embedding {i+1}: correlation = {correlation}")
                    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_face_comparison()
