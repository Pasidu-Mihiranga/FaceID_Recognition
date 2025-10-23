#!/usr/bin/env python3
"""
Test DeepFace with manually downloaded models
"""

import sys
sys.path.append('src')

import cv2
import numpy as np
from deepface import DeepFace

def test_deepface_models():
    """Test DeepFace models with downloaded weights"""
    
    print("=== Testing DeepFace with Downloaded Models ===")
    
    # Create a test image
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    
    # Test VGG-Face model
    print("\n--- Testing VGG-Face ---")
    try:
        result = DeepFace.represent(
            img_path=test_image,
            model_name='VGG-Face',
            enforce_detection=False
        )
        
        embedding = np.array(result[0]['embedding'])
        print(f"SUCCESS! VGG-Face is working!")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding sample: {embedding[:5]}")
        
        return 'VGG-Face'
        
    except Exception as e:
        print(f"FAILED! VGG-Face error: {e}")
    
    # Test other models
    models_to_test = ['Facenet', 'OpenFace', 'ArcFace']
    
    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        
        try:
            result = DeepFace.represent(
                img_path=test_image,
                model_name=model_name,
                enforce_detection=False
            )
            
            embedding = np.array(result[0]['embedding'])
            print(f"SUCCESS! {model_name} is working!")
            print(f"   Embedding shape: {embedding.shape}")
            
            return model_name
            
        except Exception as e:
            print(f"FAILED! {model_name} error: {e}")
    
    print("\nNo DeepFace models are working")
    return None

if __name__ == "__main__":
    working_model = test_deepface_models()
    if working_model:
        print(f"\nSUCCESS! {working_model} is working!")
        print("Ready to integrate DeepFace into your system!")
    else:
        print(f"\nDeepFace models still not working")
        print("Please check if models are downloaded correctly")
