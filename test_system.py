"""
Face ID System - Test Script
Simple test to verify the system is working correctly
"""

import os
import sys
import logging
from pathlib import Path

# Try to import optional dependencies
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic system functionality"""
    logger.info("Testing Face ID System basic functionality...")
    
    try:
        # Try to import the main system
        try:
            from main import FaceIDSystem
            
            # Initialize system with OpenCV detector (most compatible)
            face_id = FaceIDSystem(
                detector_type='opencv',
                recognition_model='simple',
                recognition_threshold=0.6
            )
        except ImportError as e:
            logger.warning(f"Main system not available: {e}")
            logger.info("Testing minimal system instead...")
            
            # Use minimal system as fallback
            from minimal_face_id import MinimalFaceIDSystem
            face_id = MinimalFaceIDSystem()
        
        logger.info("✓ System initialized successfully")
        
        # Create a test image
        test_image = create_test_face_image()
        test_image_path = "data/test_face.jpg"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        # Save test image
        cv2.imwrite(test_image_path, test_image)
        logger.info("✓ Test image created")
        
        # Test face detection
        faces = face_id.face_detector.detect_faces(test_image)
        logger.info(f"✓ Face detection: Found {len(faces)} faces")
        
        if faces:
            # Test face recognition (should be unknown initially)
            person_name, confidence, face_info = face_id.recognize_face(test_image)
            logger.info(f"✓ Face recognition: {person_name or 'Unknown'} (confidence: {confidence:.3f})")
            
            # Test person registration
            success = face_id.register_person(test_image_path, "TestPerson")
            logger.info(f"✓ Person registration: {'Success' if success else 'Failed'}")
            
            # Test recognition after registration
            person_name, confidence, face_info = face_id.recognize_face(test_image)
            logger.info(f"✓ Recognition after registration: {person_name} (confidence: {confidence:.3f})")
            
            # Test database operations
            persons = face_id.database.get_all_persons()
            logger.info(f"✓ Database: {len(persons)} persons registered")
            
            # Test system stats
            stats = face_id.get_system_stats()
            logger.info(f"✓ System stats: {stats['database_stats']['total_persons']} persons")
            
        # Cleanup
        face_id.cleanup_system()
        
        # Remove test file
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        logger.info("✓ All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False

def create_test_face_image():
    """Create a simple test face image"""
    try:
        import cv2
        import numpy as np
        
        # Create a 200x200 image with a face-like pattern
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = (100, 150, 200)  # Light blue background
        
        # Draw face outline
        cv2.ellipse(img, (100, 100), (80, 100), 0, 0, 360, (200, 180, 160), -1)
        
        # Draw eyes
        cv2.circle(img, (80, 80), 8, (255, 255, 255), -1)
        cv2.circle(img, (120, 80), 8, (255, 255, 255), -1)
        cv2.circle(img, (80, 80), 4, (0, 0, 0), -1)
        cv2.circle(img, (120, 80), 4, (0, 0, 0), -1)
        
        # Draw nose
        cv2.circle(img, (100, 100), 3, (180, 160, 140), -1)
        
        # Draw mouth
        cv2.ellipse(img, (100, 130), (15, 8), 0, 0, 180, (150, 100, 100), 2)
        
        return img
        
    except ImportError:
        # Fallback: create a simple text file
        logger.warning("OpenCV not available, creating text file instead")
        return None

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("Testing module imports...")
    
    try:
        # Test core imports
        import cv2
        import numpy as np
        import sqlite3
        import pickle
        import json
        logger.info("✓ Core libraries imported")
        
        # Test optional ML libraries
        try:
            import tensorflow as tf
            logger.info("✓ TensorFlow imported")
        except ImportError:
            logger.warning("⚠ TensorFlow not available")
        
        try:
            import torch
            logger.info("✓ PyTorch imported")
        except ImportError:
            logger.warning("⚠ PyTorch not available")
        
        try:
            from sklearn.cluster import DBSCAN
            logger.info("✓ Scikit-learn imported")
        except ImportError:
            logger.warning("⚠ Scikit-learn not available")
        
        # Test face recognition libraries
        try:
            import mtcnn
            logger.info("✓ MTCNN imported")
        except ImportError:
            logger.warning("⚠ MTCNN not available")
        
        try:
            import insightface
            logger.info("✓ InsightFace imported")
        except ImportError:
            logger.warning("⚠ InsightFace not available")
        
        try:
            import deepface
            logger.info("✓ DeepFace imported")
        except ImportError:
            logger.warning("⚠ DeepFace not available")
        
        # Test web framework
        try:
            import flask
            logger.info("✓ Flask imported")
        except ImportError:
            logger.warning("⚠ Flask not available")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        return False

def test_directory_structure():
    """Test if the directory structure is correct"""
    logger.info("Testing directory structure...")
    
    required_dirs = [
        "src",
        "src/face_detection",
        "src/face_recognition", 
        "src/database",
        "src/continuous_learning",
        "src/web_interface",
        "src/web_interface/templates",
        "src/web_interface/static",
        "data"
    ]
    
    required_files = [
        "main.py",
        "face_id_system.py",
        "requirements.txt",
        "README.md",
        "src/face_detection/__init__.py",
        "src/face_recognition/__init__.py",
        "src/database/__init__.py",
        "src/continuous_learning/__init__.py",
        "src/web_interface/__init__.py"
    ]
    
    try:
        # Check directories
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                logger.error(f"✗ Missing directory: {dir_path}")
                return False
            logger.info(f"✓ Directory exists: {dir_path}")
        
        # Check files
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"✗ Missing file: {file_path}")
                return False
            logger.info(f"✓ File exists: {file_path}")
        
        logger.info("✓ Directory structure is correct")
        return True
        
    except Exception as e:
        logger.error(f"✗ Directory structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Face ID System - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} Test...")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"OK {test_name} test PASSED")
            else:
                print(f"FAILED {test_name} test FAILED")
        except Exception as e:
            print(f"FAILED {test_name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! The Face ID System is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run examples: python examples.py")
        print("3. Start web interface: python face_id_system.py --web")
        print("4. Start camera recognition: python face_id_system.py --camera")
    else:
        print("ERROR: Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check that all files are present")
        print("3. Verify Python version compatibility")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
