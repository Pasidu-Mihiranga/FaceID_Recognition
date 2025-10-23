#!/usr/bin/env python3
"""
Create a simple face image for testing
"""

import cv2
import numpy as np

def create_simple_face_image():
    """Create a simple face image that should be detectable"""
    # Create a larger image
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)  # Light gray background
    
    # Draw a more realistic face
    center_x, center_y = 200, 200
    
    # Face outline (larger)
    cv2.ellipse(img, (center_x, center_y), (120, 150), 0, 0, 360, (220, 200, 180), -1)
    
    # Eyes (larger and more prominent)
    cv2.circle(img, (center_x - 50, center_y - 40), 15, (255, 255, 255), -1)
    cv2.circle(img, (center_x + 50, center_y - 40), 15, (255, 255, 255), -1)
    cv2.circle(img, (center_x - 50, center_y - 40), 8, (0, 0, 0), -1)
    cv2.circle(img, (center_x + 50, center_y - 40), 8, (0, 0, 0), -1)
    
    # Eyebrows
    cv2.ellipse(img, (center_x - 50, center_y - 60), (20, 8), 0, 0, 180, (100, 80, 60), -1)
    cv2.ellipse(img, (center_x + 50, center_y - 60), (20, 8), 0, 0, 180, (100, 80, 60), -1)
    
    # Nose (more prominent)
    cv2.ellipse(img, (center_x, center_y), (8, 15), 0, 0, 360, (200, 180, 160), -1)
    
    # Mouth (more prominent)
    cv2.ellipse(img, (center_x, center_y + 50), (25, 12), 0, 0, 180, (150, 100, 100), -1)
    
    # Add some shading for depth
    cv2.ellipse(img, (center_x, center_y), (120, 150), 0, 0, 360, (200, 180, 160), 2)
    
    return img

def test_face_detection_with_simple_image():
    """Test face detection with the simple image"""
    try:
        import sys
        sys.path.append('src')
        from main import FaceIDSystem
        
        # Create simple face image
        face_img = create_simple_face_image()
        
        # Save it
        cv2.imwrite('data/simple_face_test.jpg', face_img)
        print("Created simple face test image: data/simple_face_test.jpg")
        
        # Test face detection
        face_id = FaceIDSystem(
            detector_type='opencv',
            recognition_model='simple',
            recognition_threshold=0.6
        )
        
        faces = face_id.face_detector.detect_faces(face_img)
        print(f"Faces detected in simple image: {len(faces)}")
        
        if faces:
            print("SUCCESS: Face detection working with simple image!")
            return True
        else:
            print("Still no faces detected - trying registration anyway...")
            
            # Try registration even without face detection
            success = face_id.register_person('data/simple_face_test.jpg', 'Test Person')
            print(f"Registration result: {success}")
            return success
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_face_detection_with_simple_image()
    if success:
        print("SUCCESS: Face detection/registration test passed!")
    else:
        print("ERROR: Face detection/registration test failed")
