#!/usr/bin/env python3
"""
Debug face detection issue
"""

import cv2
import numpy as np
import os
import sys

# Add src to path
sys.path.append('src')

def debug_face_detection():
    """Debug face detection step by step"""
    
    print("=== Debugging Face Detection ===")
    
    # Test 1: Check cascade file
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    print(f"Cascade path: {cascade_path}")
    print(f"Cascade exists: {os.path.exists(cascade_path)}")
    
    # Test 2: Load cascade
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print(f"Cascade loaded: {not face_cascade.empty()}")
    
    if face_cascade.empty():
        print("ERROR: Cascade failed to load!")
        return
    
    # Test 3: Test with a simple image
    test_image_path = "data/simple_face_test.jpg"
    if os.path.exists(test_image_path):
        print(f"\nTesting with: {test_image_path}")
        
        # Load image
        image = cv2.imread(test_image_path)
        print(f"Image loaded: {image is not None}")
        if image is not None:
            print(f"Image shape: {image.shape}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(f"Grayscale shape: {gray.shape}")
            
            # Try different detection parameters
            print("\nTrying different detection parameters:")
            
            # Standard parameters
            faces1 = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(f"Standard params: {len(faces1)} faces")
            
            # More lenient parameters
            faces2 = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(f"Lenient params: {len(faces2)} faces")
            
            # Very lenient parameters
            faces3 = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.01,
                minNeighbors=1,
                minSize=(10, 10),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(f"Very lenient params: {len(faces3)} faces")
            
            if len(faces3) > 0:
                print(f"Found faces with very lenient params: {faces3}")
                
                # Save image with detected faces
                result_image = image.copy()
                for (x, y, w, h) in faces3:
                    cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.imwrite("debug_face_detection.jpg", result_image)
                print("Saved debug image: debug_face_detection.jpg")
            else:
                print("No faces detected even with very lenient parameters")
                
                # Try with different image sizes
                print("\nTrying different image sizes:")
                for scale in [0.5, 1.0, 2.0]:
                    scaled_image = cv2.resize(gray, None, fx=scale, fy=scale)
                    faces_scaled = face_cascade.detectMultiScale(
                        scaled_image,
                        scaleFactor=1.01,
                        minNeighbors=1,
                        minSize=(10, 10),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    print(f"Scale {scale}: {len(faces_scaled)} faces")
    else:
        print(f"Test image not found: {test_image_path}")
    
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_face_detection()
