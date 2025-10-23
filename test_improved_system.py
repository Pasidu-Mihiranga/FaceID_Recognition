#!/usr/bin/env python3
"""
Test the improved face recognition system with proper threshold
"""

import sys
sys.path.append('src')

import cv2
import numpy as np
import os
import pickle
from main import FaceIDSystem

def test_improved_system():
    """Test the improved face recognition system"""
    
    print("=== Testing Improved Face Recognition System ===")
    
    # Initialize system
    system = FaceIDSystem()
    
    # Check current database
    print(f"Current database contains {len(system.face_recognizer.face_database)} persons:")
    for person_name, embeddings in system.face_recognizer.face_database.items():
        print(f"- {person_name}: {len(embeddings)} embeddings")
    
    # Test with different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8]
    
    test_images = [
        "data/simple_face_test.jpg",
        "data/test_person1.jpg", 
        "data/test_person2.jpg"
    ]
    
    for threshold in thresholds:
        print(f"\n=== Testing with threshold: {threshold} ===")
        system.face_recognizer.threshold = threshold
        
        for test_image_path in test_images:
            if os.path.exists(test_image_path):
                print(f"\nTesting: {test_image_path}")
                
                # Test multiple times to check consistency
                results = []
                for i in range(3):
                    try:
                        image = cv2.imread(test_image_path)
                        if image is not None:
                            person_name, confidence, face_info = system.recognize_face(image)
                            results.append((person_name, confidence))
                        else:
                            results.append((None, 0.0))
                    except Exception as e:
                        results.append((None, 0.0))
                
                # Show results
                person_names = [r[0] for r in results]
                confidences = [r[1] for r in results]
                
                if len(set(person_names)) == 1:
                    print(f"  Result: {person_names[0]} (confidence: {confidences[0]:.4f}) - CONSISTENT")
                else:
                    print(f"  Results vary: {set(person_names)} - INCONSISTENT")
            else:
                print(f"\nImage not found: {test_image_path}")
    
    # Find the best threshold
    print(f"\n=== Finding Optimal Threshold ===")
    
    # Test with registered persons' images
    persons = system.database.get_all_persons()
    
    for person in persons:
        person_id = person['id']
        person_name = person['name']
        
        print(f"\nTesting with {person_name}:")
        
        # Find face images for this person
        face_images = []
        for face_image in system.database.get_all_face_images():
            if face_image['person_id'] == person_id:
                face_images.append(face_image)
        
        if face_images:
            for face_image in face_images:
                image_path = face_image['image_path']
                if os.path.exists(image_path):
                    print(f"  Testing with: {image_path}")
                    
                    # Test with different thresholds
                    for threshold in [0.5, 0.6, 0.7, 0.8]:
                        system.face_recognizer.threshold = threshold
                        
                        try:
                            image = cv2.imread(image_path)
                            if image is not None:
                                person_name_result, confidence, face_info = system.recognize_face(image)
                                
                                if person_name_result == person_name:
                                    print(f"    Threshold {threshold}: CORRECT - {person_name_result} (confidence: {confidence:.4f})")
                                else:
                                    print(f"    Threshold {threshold}: WRONG - {person_name_result} (confidence: {confidence:.4f})")
                        except Exception as e:
                            print(f"    Threshold {threshold}: ERROR - {e}")
        else:
            print(f"  No face images found for {person_name}")
    
    # Set optimal threshold
    optimal_threshold = 0.6  # Based on testing
    system.face_recognizer.threshold = optimal_threshold
    print(f"\nSetting optimal threshold to: {optimal_threshold}")
    
    # Final consistency test
    print(f"\n=== Final Consistency Test ===")
    
    for test_image_path in test_images:
        if os.path.exists(test_image_path):
            print(f"\nTesting consistency with: {test_image_path}")
            
            # Test multiple times
            results = []
            for i in range(5):
                try:
                    image = cv2.imread(test_image_path)
                    if image is not None:
                        person_name, confidence, face_info = system.recognize_face(image)
                        results.append((person_name, confidence))
                        print(f"  Test {i+1}: {person_name} (confidence: {confidence:.4f})")
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
    
    print(f"\n=== Testing Complete ===")
    return True

if __name__ == "__main__":
    test_improved_system()
