import unittest
import numpy as np
import cv2
from src.detection import create_face_detector, FaceDetectionManager, OpenCVDetector

class TestFaceDetection(unittest.TestCase):
    def setUp(self):
        # Create a mock image
        self.image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(self.image, (150, 150), 50, (255, 255, 255), -1)

    def test_create_detector_default(self):
        """Test creating face detector with default mode"""
        detector = create_face_detector('opencv')
        self.assertIsInstance(detector, FaceDetectionManager)
        self.assertIsInstance(detector.detector, OpenCVDetector)

    def test_opencv_detector_empty_image(self):
        """Test detector behavior on a plain black image (no faces)"""
        detector = OpenCVDetector()
        black_img = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = detector.detect_faces(black_img)
        self.assertEqual(len(faces), 0)

    def test_opencv_detector_invalid_image(self):
        """Test detector handling of invalid image input (returns empty list, doesn't raise exception)"""
        detector = OpenCVDetector()
        faces = detector.detect_faces(None)
        self.assertEqual(faces, [])

    def test_extract_face(self):
        """Test extracting/cropping face from bounding box coordinates"""
        manager = FaceDetectionManager('opencv')
        face_info = {
            'bbox': (50, 50, 100, 100),
            'confidence': 1.0,
            'landmarks': None
        }
        cropped = manager.extract_face(self.image, face_info)
        self.assertEqual(cropped.shape, (160, 160, 3))

if __name__ == '__main__':
    unittest.main()
