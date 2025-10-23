#!/usr/bin/env python3
"""
Test recognition API directly
"""

import requests
import cv2
import numpy as np
import os

def test_recognition_api():
    """Test the recognition API endpoint"""
    
    print("=== Testing Recognition API ===")
    
    # Test with a simple image
    test_image_path = "data/simple_face_test.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    try:
        # Prepare the request
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            
            print(f"Sending request to /api/recognize...")
            response = requests.post('http://127.0.0.1:5000/api/recognize', files=files)
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"SUCCESS: {result}")
            else:
                print(f"ERROR: {response.status_code}")
                print(f"Response text: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("Connection error - is the server running?")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recognition_api()
