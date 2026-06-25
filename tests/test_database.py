import os
import unittest
import numpy as np
import tempfile
import threading
from src.database import FaceDatabase

class TestFaceDatabase(unittest.TestCase):
    def setUp(self):
        # Create a temporary file for the database
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.db = FaceDatabase(self.db_path)

    def tearDown(self):
        # Close database connection if cached in main thread
        if hasattr(self.db._local, 'conn'):
            self.db._local.conn.close()
        # Clean up database file
        os.close(self.db_fd)
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_database_initialization(self):
        """Test database tables are initialized properly and user_version is set"""
        # Ensure migration 2 was run (which sets user_version to 2)
        version = self.db._get_user_version()
        self.assertEqual(version, 2)

    def test_add_and_get_person(self):
        """Test adding a person and retrieving them by ID/name"""
        person_name = "Alice Smith"
        metadata = {"role": "admin", "department": "Security"}
        
        person_id = self.db.add_person(person_name, metadata)
        self.assertIsNotNone(person_id)
        
        # Test get by ID
        person_by_id = self.db.get_person_by_id(person_id)
        self.assertEqual(person_by_id['name'], person_name)
        self.assertEqual(person_by_id['metadata']['role'], "admin")
        
        # Test get by name
        person_by_name = self.db.get_person_by_name(person_name)
        self.assertEqual(person_by_name['id'], person_id)

    def test_add_face_image_and_embeddings(self):
        """Test adding face image details and embeddings to a person"""
        person_id = self.db.add_person("Bob Jones")
        embedding = np.random.rand(128).astype(np.float32)
        
        # Add face image (with temporary file mocks)
        face_id = self.db.add_face_image(
            person_id=person_id,
            image_path="test_image.jpg",
            embedding=embedding,
            face_bbox=(10, 10, 100, 100),
            confidence=0.95,
            quality_score=0.85
        )
        self.assertIsNotNone(face_id)
        
        # Check person image list
        images = self.db.get_person_images(person_id)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]['confidence'], 0.95)
        
        # Check embeddings
        embeddings = self.db.get_person_embeddings(person_id)
        self.assertEqual(len(embeddings), 1)
        self.assertTrue(np.allclose(embeddings[0], embedding))

    def test_thread_local_connection(self):
        """Test that database connections are thread-safe and local per thread"""
        main_thread_conn = self.db._get_conn()
        
        connections = []
        errors = []
        
        def thread_task():
            try:
                # Retrieve connection in the background thread
                thread_conn = self.db._get_conn()
                connections.append(thread_conn)
            except Exception as e:
                errors.append(e)
                
        thread = threading.Thread(target=thread_task)
        thread.start()
        thread.join()
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(connections), 1)
        # Ensure the background thread got a different connection descriptor
        self.assertNotEqual(main_thread_conn, connections[0])

    def test_delete_person(self):
        """Test deleting a person and cleaning up references"""
        person_id = self.db.add_person("Charlie Brown")
        embedding = np.random.rand(128).astype(np.float32)
        
        self.db.add_face_image(
            person_id=person_id,
            image_path="test_charlie.jpg",
            embedding=embedding
        )
        
        # Delete person
        success = self.db.delete_person(person_id)
        self.assertTrue(success)
        
        # Assert person is gone
        person = self.db.get_person_by_id(person_id)
        self.assertIsNone(person)

if __name__ == '__main__':
    unittest.main()
