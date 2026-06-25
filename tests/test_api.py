import unittest
import json
import base64
import numpy as np
import cv2
from unittest.mock import MagicMock
from flask import Flask
from src.web import FaceIDWebInterface

class TestFaceIDAPI(unittest.TestCase):
    def setUp(self):
        # Create a mock FaceIDSystem
        self.mock_system = MagicMock()
        self.mock_system.database = MagicMock()
        self.mock_system.database.db_path = "mock.db"
        self.mock_system.database.get_all_persons.return_value = [
            {"id": 1, "name": "Alice", "total_images": 1}
        ]
        self.mock_system.database.get_person_by_id.return_value = {
            "id": 1, "name": "Alice", "total_images": 1
        }
        self.mock_system.database.get_person_images.return_value = []
        self.mock_system.get_system_stats.return_value = {
            "recognition_stats": {"total_detections": 10}
        }
        
        # Setup mock recognition return values
        self.mock_system.recognize_face.return_value = ("Alice", 0.95, {"bbox": (0, 0, 10, 10), "is_live": True, "liveness_score": 0.99})
        self.mock_system.recognize_faces.return_value = [
            {"person_name": "Alice", "confidence": 0.95, "bbox": (0, 0, 10, 10), "recognition_method": "test", "is_live": True, "liveness_score": 0.99}
        ]
        
        # Initialize WebInterface with mock system
        self.web_ui = FaceIDWebInterface(self.mock_system)
        self.app = self.web_ui.app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Disable rate limiting / auth decorators for standard unit tests if needed, or pass correct headers
        # Wait, the auth module expects a header X-API-Key or reads from env. Let's make sure we test auth as well!

    def test_api_stats(self):
        """Test the system stats endpoint"""
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('recognition_stats', data)

    def test_api_persons(self):
        """Test getting all registered persons"""
        response = self.client.get('/api/persons')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('persons', data)
        self.assertEqual(len(data['persons']), 1)
        self.assertEqual(data['persons'][0]['name'], 'Alice')

    def test_api_recognize_base64(self):
        """Test face recognition via base64 API"""
        # Create a simple 10x10 mock image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        payload = json.dumps({"image": img_base64})
        response = self.client.post('/api/recognize_base64', 
                                   data=payload, 
                                   content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['person_name'], 'Alice')
        self.assertEqual(data['confidence'], 0.95)
        self.assertTrue(data['is_live'])

    def test_api_recognize_batch(self):
        """Test batch face recognition API endpoint"""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        payload = json.dumps({"images": [img_base64, img_base64]})
        response = self.client.post('/api/recognize_batch', 
                                   data=payload, 
                                   content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 2)
        self.assertEqual(data['results'][0]['person_name'], 'Alice')

if __name__ == '__main__':
    unittest.main()
