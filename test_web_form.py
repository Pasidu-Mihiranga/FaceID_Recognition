#!/usr/bin/env python3
"""
Debug web form registration issues
"""

import requests
import os

def test_web_form_registration():
    """Test registration through the web form"""
    try:
        # Test with a simple image
        test_image_path = 'data/simple_face_test.jpg'
        
        if not os.path.exists(test_image_path):
            print(f"Test image not found: {test_image_path}")
            return False
        
        # Prepare form data
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test_image.jpg', f, 'image/jpeg')}
            data = {'person_name': 'Web Test Person'}
            
            # Send POST request to registration endpoint
            response = requests.post('http://localhost:5000/api/register', 
                                   files=files, 
                                   data=data)
        
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("SUCCESS: Web form registration working!")
                return True
            else:
                print(f"ERROR: Registration failed: {result.get('error')}")
                return False
        else:
            print(f"ERROR: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_web_form_registration()
    if success:
        print("SUCCESS: Web form registration is working!")
    else:
        print("ERROR: Web form registration has issues")
