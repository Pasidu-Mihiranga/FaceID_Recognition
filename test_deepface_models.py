#!/usr/bin/env python3
"""
Test DeepFace with OpenFace model (more readily available)
"""

import sys
sys.path.append('src')

import cv2
import numpy as np
import os
from deepface import DeepFace

def test_deepface_models():
    """Test different DeepFace models"""
    
    print("=== Testing DeepFace Models ===")
    
    # Test image
    test_image = "data/simple_face_test.jpg"
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    
    # Load test image
    image = cv2.imread(test_image)
    if image is None:
        print("Could not load test image")
        return
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Test different models
    models_to_test = ['OpenFace', 'Facenet', 'Dlib']
    
    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        
        try:
            # Test embedding extraction
            embedding = DeepFace.represent(
                img_path=rgb_image,
                model_name=model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            print(f"✅ {model_name} - SUCCESS!")
            print(f"   Embedding shape: {np.array(embedding[0]['embedding']).shape}")
            
            # This model works, let's use it
            print(f"🎉 {model_name} is working! Using this model.")
            return model_name
            
        except Exception as e:
            print(f"❌ {model_name} - FAILED: {e}")
    
    print("\n❌ No DeepFace models are working")
    return None

if __name__ == "__main__":
    working_model = test_deepface_models()
    if working_model:
        print(f"\n✅ Recommended model: {working_model}")
    else:
        print(f"\n❌ No working DeepFace models found")
