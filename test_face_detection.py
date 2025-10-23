#!/usr/bin/env python3
"""
Test face detection on user's specific image
"""

import cv2
import os
import sys
import logging

# Add src to path
sys.path.append('src')

from face_detection import OpenCVDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_face_detection_on_user_image():
    """Test face detection on the user's IMG_9312.jpg"""
    
    # Initialize detector
    detector = OpenCVDetector()
    
    # Test with different images
    test_images = [
        "data/test_person1.jpg",
        "data/test_person2.jpg", 
        "data/simple_face_test.jpg"
    ]
    
    print("=== Testing Face Detection ===")
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\nTesting: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Could not load image")
                continue
                
            print(f"  Image size: {image.shape}")
            
            # Detect faces
            faces = detector.detect_faces(image)
            print(f"  Faces detected: {len(faces)}")
            
            if faces:
                for i, face in enumerate(faces):
                    bbox = face['bbox']
                    print(f"    Face {i+1}: {bbox}")
            else:
                print("  No faces detected - trying lenient parameters...")
                
                # Try with more lenient parameters
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                faces = detector.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,  # More lenient
                    minNeighbors=3,    # More lenient
                    minSize=(20, 20),  # Smaller minimum size
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                print(f"  Lenient detection found: {len(faces)} faces")
                
                if len(faces) == 0:
                    print("  Still no faces - would use fallback (entire image)")
                else:
                    for i, (x, y, w, h) in enumerate(faces):
                        print(f"    Face {i+1}: ({x}, {y}, {w}, {h})")
        else:
            print(f"\nImage not found: {image_path}")
    
    print("\n=== Face Detection Test Complete ===")

if __name__ == "__main__":
    test_face_detection_on_user_image()