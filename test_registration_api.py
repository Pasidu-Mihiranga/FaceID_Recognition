#!/usr/bin/env python3
"""
Test registration API endpoint
"""

import sys
import os
sys.path.append('src')

def test_registration_api():
    """Test the registration API endpoint"""
    try:
        from main import FaceIDSystem
        from src.web_interface import FaceIDWebInterface
        
        print("Testing registration API...")
        
        # Initialize system
        face_id = FaceIDSystem(
            detector_type='opencv',
            recognition_model='simple',
            recognition_threshold=0.6
        )
        
        # Initialize web interface
        web_interface = FaceIDWebInterface(face_id)
        
        # Test registration endpoint
        with web_interface.app.test_client() as client:
            # Test with a simple image file
            test_image_path = 'data/test_person1.jpg'
            
            if os.path.exists(test_image_path):
                with open(test_image_path, 'rb') as f:
                    response = client.post('/api/register', 
                                         data={
                                             'person_name': 'Test Person',
                                             'file': (f, 'test_person1.jpg', 'image/jpeg')
                                         })
                
                print(f"Registration API status: {response.status_code}")
                print(f"Response: {response.get_json()}")
                
                if response.status_code == 200:
                    print("SUCCESS: Registration API is working!")
                    return True
                else:
                    print("ERROR: Registration API failed")
                    return False
            else:
                print(f"ERROR: Test image not found at {test_image_path}")
                return False
        
    except Exception as e:
        print(f"Registration API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_registration_api()
    if success:
        print("SUCCESS: Registration API is working!")
    else:
        print("ERROR: Registration API has issues")
