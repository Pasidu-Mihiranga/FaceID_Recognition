#!/usr/bin/env python3
"""
Test recognition API
"""

import sys
sys.path.append('src')

from main import FaceIDSystem
import cv2
import numpy as np

def test_recognition():
    """Test the recognition functionality"""
    
    print("=== Testing Recognition System ===")
    
    # Initialize system
    system = FaceIDSystem()
    
    # Check registered persons
    persons = system.database.get_all_persons()
    print(f"Registered persons: {len(persons)}")
    
    for person in persons:
        print(f"- {person['name']} (ID: {person['id']})")
    
    if len(persons) == 0:
        print("❌ No persons registered - cannot test recognition")
        return
    
    # Test with a simple image
    test_image_path = "data/simple_face_test.jpg"
    try:
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"❌ Could not load test image: {test_image_path}")
            return
            
        print(f"\nTesting recognition with: {test_image_path}")
        print(f"Image shape: {image.shape}")
        
        # Test recognition
        person_name, confidence, face_info = system.recognize_face(image)
        
        print(f"Recognition result:")
        print(f"  Person: {person_name}")
        print(f"  Confidence: {confidence}")
        print(f"  Face detected: {face_info is not None}")
        
        if face_info:
            print(f"  Face bbox: {face_info.get('bbox', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Recognition test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recognition()
