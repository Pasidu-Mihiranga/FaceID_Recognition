"""
Enrollment Workflow E2E Integration Test
Tests face enrollment/registration, listing, details query, and comprehensive deletion.
"""

import os
import tempfile
import shutil
import unittest
import json
import numpy as np
import cv2
import io
from unittest.mock import MagicMock
from src.database import FaceDatabase
from main import FaceIDSystem
from src.web import FaceIDWebInterface
from src.recognition.face_identity_manager import FaceIdentityManager

class TestEnrollmentWorkflowE2E(unittest.TestCase):
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
        # Override identity manager path to keep test isolated
        self.system.identity_manager = FaceIdentityManager(self.identities_dir)
        
        # Mock detector to bypass actual Haar-cascade detection which is flaky on geometry
        self.system.face_detector = MagicMock()
        self.system.face_detector.detector_type = 'opencv'
        self.system.face_detector.detect_faces.return_value = [{
            'bbox': (10, 10, 180, 180),
            'landmarks': None,
            'confidence': 1.0
        }]
        # Side effect to return the crop image (just return the sub-region)
        self.system.face_detector.extract_face.side_effect = lambda img, info: img[10:190, 10:190]
        
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

    def test_enrollment_and_deletion(self):
        """E2E Enrollment Workflow: register a person via API, verify listing, and delete comprehensively"""
        # 1. Create a high-quality mock face image in memory
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        # Add high-contrast lines/patterns to easily pass quality sharpness/contrast checks
        cv2.rectangle(img, (20, 20), (180, 180), (0, 0, 0), -1)
        cv2.circle(img, (100, 100), 40, (255, 255, 255), -1)
        for i in range(10):
            cv2.line(img, (20 * i, 0), (20 * i, 200), (0, 0, 0), 1)
            
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = io.BytesIO(buffer)
        
        # 2. Call API to register a new person "John Doe"
        # Since FACEID_API_KEY is not set in env, require_api_key decorator bypasses checks
        response = self.client.post(
            '/api/register',
            data={
                'person_name': 'John Doe',
                'file': (img_bytes, 'john_doe.jpg')
            },
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        reg_data = json.loads(response.data)
        self.assertTrue(reg_data['success'])
        self.assertEqual(reg_data['person_name'], 'John Doe')
        
        # Verify person is in database
        person = self.db.get_person_by_name('John Doe')
        self.assertIsNotNone(person)
        person_id = person['id']
        
        # Verify profile is visible in /api/persons list
        list_response = self.client.get('/api/persons')
        self.assertEqual(list_response.status_code, 200)
        list_data = json.loads(list_response.data)
        person_names = [p['name'] for p in list_data['persons']]
        self.assertIn('John Doe', person_names)
        
        # Verify profile details via details API
        details_response = self.client.get(f'/api/person/{person_id}')
        self.assertEqual(details_response.status_code, 200)
        details_data = json.loads(details_response.data)
        self.assertEqual(details_data['name'], 'John Doe')
        self.assertEqual(len(details_data['images']), 1)
        
        # 3. Call API to delete John Doe comprehensively
        del_response = self.client.delete(f'/api/delete_person/{person_id}')
        self.assertEqual(del_response.status_code, 200)
        del_data = json.loads(del_response.data)
        self.assertTrue(del_data['success'])
        
        # Verify John Doe is removed from database
        self.assertIsNone(self.db.get_person_by_id(person_id))

if __name__ == '__main__':
    unittest.main()
