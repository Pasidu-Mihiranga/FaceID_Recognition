#!/usr/bin/env python3
"""
Simple test script to verify web interface
"""

import sys
import os
sys.path.append('src')

def test_web_interface():
    """Test web interface basic functionality"""
    try:
        from main import FaceIDSystem
        from src.web_interface import FaceIDWebInterface
        
        print("Testing web interface...")
        
        # Initialize system
        face_id = FaceIDSystem(
            detector_type='opencv',
            recognition_model='simple',
            recognition_threshold=0.6
        )
        
        # Initialize web interface
        web_interface = FaceIDWebInterface(face_id)
        
        # Test routes
        with web_interface.app.test_client() as client:
            # Test home page
            response = client.get('/')
            print(f"Home page status: {response.status_code}")
            
            # Test register page
            response = client.get('/register')
            print(f"Register page status: {response.status_code}")
            
            # Test recognize page
            response = client.get('/recognize')
            print(f"Recognize page status: {response.status_code}")
            
            # Test dashboard page
            response = client.get('/dashboard')
            print(f"Dashboard page status: {response.status_code}")
        
        print("Web interface test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Web interface test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_web_interface()
    if success:
        print("SUCCESS: Web interface is working!")
    else:
        print("ERROR: Web interface has issues")
