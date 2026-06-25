import unittest
import numpy as np
import os
import shutil
from src.recognition import create_face_recognizer, SimpleOpenCVRecognizer

class TestFaceRecognition(unittest.TestCase):
    def setUp(self):
        # Initialize recognizer using the 'simple' mode to avoid loading heavy weights in tests
        self.recognizer = create_face_recognizer('simple', threshold=0.8)
        
        # Use a black image and a white image to ensure completely distinct histograms
        self.face_a = np.zeros((160, 160, 3), dtype=np.uint8)
        self.face_b = np.ones((160, 160, 3), dtype=np.uint8) * 255

    def tearDown(self):
        # Cleanup temporary test database files if created
        db_dir = "data/embeddings"
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir, ignore_errors=True)

    def test_recognizer_creation(self):
        """Test creating face recognizer and checks fallback mechanism"""
        self.assertIsInstance(self.recognizer.recognizer, SimpleOpenCVRecognizer)

    def test_register_face(self):
        """Test registering a face embedding in the database"""
        success = self.recognizer.register_face(self.face_a, "Alice")
        self.assertTrue(success)
        self.assertIn("Alice", self.recognizer.face_database)
        self.assertEqual(len(self.recognizer.face_database["Alice"]), 1)

    def test_recognize_face(self):
        """Test face recognition and matching logic"""
        self.recognizer.register_face(self.face_a, "Alice")
        
        # Test recognition with exact same image (should match with high confidence)
        name, confidence = self.recognizer.recognize_face(self.face_a)
        self.assertEqual(name, "Alice")
        self.assertGreaterEqual(confidence, 0.8)
        
        # Test recognition with completely different image (should not match, returns None)
        name, confidence = self.recognizer.recognize_face(self.face_b)
        self.assertNotEqual(name, "Alice")
        self.assertIsNone(name)

    def test_remove_person(self):
        """Test removing a person and their templates from the recognition database"""
        self.recognizer.register_face(self.face_a, "Alice")
        self.assertIn("Alice", self.recognizer.face_database)
        
        success = self.recognizer.remove_person("Alice")
        self.assertTrue(success)
        self.assertNotIn("Alice", self.recognizer.face_database)

if __name__ == '__main__':
    unittest.main()
