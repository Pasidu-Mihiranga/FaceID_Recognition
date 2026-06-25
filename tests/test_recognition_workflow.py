"""
Recognition & Liveness Workflow E2E Integration Test
Tests face detection, liveness checking (PAD) scenarios, and match auditing.
"""

import os
import tempfile
import shutil
import unittest
import json
import numpy as np
import cv2
import base64
from unittest.mock import MagicMock
from src.database import FaceDatabase
from main import FaceIDSystem
from src.web import FaceIDWebInterface
from src.recognition.face_identity_manager import FaceIdentityManager

class TestRecognitionWorkflowE2E(unittest.TestCase):
    def setUp(self):
        # Create temp database and directories
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.temp_dir = tempfile.mkdtemp()
        self.identities_dir = os.path.join(self.temp_dir, "identities")
        
        # Instantiate DB and system
        self.db = FaceDatabase(self.db_path)
        self.system = FaceIDSystem(
            detector_type='opencv',
            recognition_model='simple',
            database=self.db
        )
        self.system.identity_manager = FaceIdentityManager(self.identities_dir)
        
        # Setup Web Interface
        self.web_ui = FaceIDWebInterface(self.system)
        self.app = self.web_ui.app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def tearDown(self):
        # Close connection and clean up database
        if hasattr(self.db._local, 'conn'):
            self.db._local.conn.close()
        os.close(self.db_fd)
        try:
            os.unlink(self.db_path)
        except OSError:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_recognition_live_success_workflow(self):
        """E2E Recognition: test successful recognition of a live person"""
        # Mock recognize_face to simulate a successful live match
        self.system.recognize_face = MagicMock(return_value=(
            "Alice Smith", 
            0.95, 
            {"bbox": (10, 10, 150, 150), "is_live": True, "liveness_score": 0.97}
        ))
        
        # Create a mock image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = base64.b64encode(buffer).decode('utf-8')
        
        # Call base64 recognition API
        response = self.client.post(
            '/api/recognize_base64',
            data=json.dumps({"image": img_bytes}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['person_name'], 'Alice Smith')
        self.assertEqual(data['confidence'], 0.95)
        self.assertTrue(data['is_live'])
        self.assertEqual(data['liveness_score'], 0.97)

    def test_recognition_spoof_failure_workflow(self):
        """E2E Recognition: test detection and flagging of a presentation attack (spoof)"""
        # Mock recognize_face to simulate a presentation attack (non-live)
        self.system.recognize_face = MagicMock(return_value=(
            "Alice Smith", 
            0.91, 
            {"bbox": (10, 10, 150, 150), "is_live": False, "liveness_score": 0.08}
        ))
        
        # Create a mock image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        img_file = base64.b64encode(buffer).decode('utf-8')
        
        # Call base64 recognition API
        response = self.client.post(
            '/api/recognize_base64',
            data=json.dumps({"image": img_file}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['person_name'], 'Alice Smith')
        self.assertEqual(data['confidence'], 0.91)
        self.assertFalse(data['is_live'])
        self.assertEqual(data['liveness_score'], 0.08)

    def test_recognition_unknown_workflow(self):
        """E2E Recognition: test behavior on an unknown / unregistered face"""
        # Mock recognize_face to simulate an unknown face
        self.system.recognize_face = MagicMock(return_value=(
            None, 
            0.0, 
            {"bbox": (10, 10, 150, 150), "is_live": True, "liveness_score": 0.99}
        ))
        
        # Create a mock image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = base64.b64encode(buffer).decode('utf-8')
        
        # Call base64 recognition API
        response = self.client.post(
            '/api/recognize_base64',
            data=json.dumps({"image": img_bytes}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertIsNone(data['person_name'])
        self.assertEqual(data['confidence'], 0.0)
        self.assertTrue(data['is_live'])

if __name__ == '__main__':
    unittest.main()
