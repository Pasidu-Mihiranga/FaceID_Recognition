"""
Face Database Module
Manages face data storage, retrieval, and database operations
"""

import sqlite3
import json
import pickle
import numpy as np
import cv2
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDatabase:
    """SQLite database for face recognition system"""
    
    def __init__(self, db_path: str = "data/face_database.db"):
        """
        Initialize face database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.ensure_directories()
        self.init_database()
    
    def ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/registered_faces", exist_ok=True)
        os.makedirs("data/embeddings", exist_ok=True)
        os.makedirs("data/thumbnails", exist_ok=True)
    
    def init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create persons table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS persons (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        total_images INTEGER DEFAULT 0,
                        last_seen TIMESTAMP,
                        metadata TEXT
                    )
                ''')
                
                # Add metadata column if it doesn't exist (for existing databases)
                try:
                    cursor.execute('ALTER TABLE persons ADD COLUMN metadata TEXT')
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                # Create face_images table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER NOT NULL,
                        image_path TEXT NOT NULL,
                        thumbnail_path TEXT,
                        embedding_path TEXT NOT NULL,
                        face_bbox TEXT,
                        landmarks TEXT,
                        confidence REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (person_id) REFERENCES persons (id)
                    )
                ''')
                
                # Create recognition_logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recognition_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER,
                        image_path TEXT,
                        confidence REAL,
                        recognized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_unknown BOOLEAN DEFAULT FALSE,
                        FOREIGN KEY (person_id) REFERENCES persons (id)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_face_images_person_id ON face_images(person_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_recognition_logs_person_id ON recognition_logs(person_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_recognition_logs_recognized_at ON recognition_logs(recognized_at)')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def add_person(self, name: str, metadata: Optional[Dict] = None) -> int:
        """
        Add a new person to the database
        
        Args:
            name: Person's name
            metadata: Additional metadata about the person
            
        Returns:
            Person ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                metadata_json = json.dumps(metadata) if metadata else None
                
                cursor.execute('''
                    INSERT OR REPLACE INTO persons (name, metadata, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (name, metadata_json))
                
                person_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Added person: {name} (ID: {person_id})")
                return person_id
                
        except Exception as e:
            logger.error(f"Failed to add person {name}: {e}")
            raise
    
    def add_face_image(self, person_id: int, image_path: str, embedding: np.ndarray,
                       face_bbox: Optional[Tuple] = None, landmarks: Optional[np.ndarray] = None,
                       confidence: float = 1.0) -> int:
        """
        Add a face image to the database
        
        Args:
            person_id: ID of the person
            image_path: Path to the face image
            embedding: Face embedding vector
            face_bbox: Face bounding box coordinates
            landmarks: Facial landmarks
            confidence: Detection confidence
            
        Returns:
            Face image ID
        """
        try:
            # Save embedding
            embedding_path = f"data/embeddings/face_{person_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding, f)
            
            # Create thumbnail
            thumbnail_path = self._create_thumbnail(image_path, person_id)
            
            # Convert data for database storage
            face_bbox_json = json.dumps([int(x) for x in face_bbox]) if face_bbox else None
            landmarks_json = json.dumps(landmarks.tolist()) if landmarks is not None else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO face_images 
                    (person_id, image_path, thumbnail_path, embedding_path, face_bbox, landmarks, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (person_id, image_path, thumbnail_path, embedding_path, face_bbox_json, landmarks_json, confidence))
                
                face_id = cursor.lastrowid
                
                # Update person's image count
                cursor.execute('''
                    UPDATE persons 
                    SET total_images = total_images + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (person_id,))
                
                conn.commit()
                
                logger.info(f"Added face image for person ID {person_id} (Face ID: {face_id})")
                return face_id
                
        except Exception as e:
            logger.error(f"Failed to add face image: {e}")
            raise
    
    def _create_thumbnail(self, image_path: str, person_id: int) -> str:
        """Create a thumbnail of the face image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Resize to thumbnail size
            thumbnail_size = (100, 100)
            thumbnail = cv2.resize(image, thumbnail_size)
            
            # Save thumbnail
            thumbnail_path = f"data/thumbnails/person_{person_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(thumbnail_path, thumbnail)
            
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            return ""
    
    def get_person_by_name(self, name: str) -> Optional[Dict]:
        """Get person information by name"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM persons WHERE name = ?', (name,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'created_at': row[2],
                        'updated_at': row[3],
                        'total_images': row[4],
                        'last_seen': row[5],
                        'metadata': json.loads(row[6]) if row[6] else None
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to get person by name {name}: {e}")
            return None
    
    def get_person_by_id(self, person_id: int) -> Optional[Dict]:
        """Get person information by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM persons WHERE id = ?', (person_id,))
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'created_at': row[2],
                        'updated_at': row[3],
                        'total_images': row[4],
                        'last_seen': row[5],
                        'metadata': json.loads(row[6]) if row[6] else None
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to get person by ID {person_id}: {e}")
            return None
    
    def get_all_persons(self) -> List[Dict]:
        """Get all persons from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM persons ORDER BY name')
                rows = cursor.fetchall()
                
                persons = []
                for row in rows:
                    persons.append({
                        'id': row[0],
                        'name': row[1],
                        'created_at': row[2],
                        'updated_at': row[3],
                        'total_images': row[4],
                        'last_seen': row[5],
                        'metadata': json.loads(row[6]) if row[6] else None
                    })
                
                return persons
                
        except Exception as e:
            logger.error(f"Failed to get all persons: {e}")
            return []
    
    def get_person_embeddings(self, person_id: int) -> List[np.ndarray]:
        """Get all embeddings for a person"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT embedding_path FROM face_images WHERE person_id = ?', (person_id,))
                rows = cursor.fetchall()
                
                embeddings = []
                for row in rows:
                    embedding_path = row[0]
                    if os.path.exists(embedding_path):
                        with open(embedding_path, 'rb') as f:
                            embedding = pickle.load(f)
                            embeddings.append(embedding)
                
                return embeddings
                
        except Exception as e:
            logger.error(f"Failed to get embeddings for person {person_id}: {e}")
            return []
    
    def get_person_images(self, person_id: int) -> List[Dict]:
        """Get all images for a person"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT id, image_path, thumbnail_path, face_bbox, landmarks, confidence, created_at
                    FROM face_images WHERE person_id = ? ORDER BY created_at DESC
                ''', (person_id,))
                rows = cursor.fetchall()
                
                images = []
                for row in rows:
                    images.append({
                        'id': row[0],
                        'image_path': row[1],
                        'thumbnail_path': row[2],
                        'face_bbox': json.loads(row[3]) if row[3] else None,
                        'landmarks': json.loads(row[4]) if row[4] else None,
                        'confidence': row[5],
                        'created_at': row[6]
                    })
                
                return images
                
        except Exception as e:
            logger.error(f"Failed to get images for person {person_id}: {e}")
            return []
    
    def log_recognition(self, person_id: Optional[int], image_path: str, 
                       confidence: float, is_unknown: bool = False):
        """Log a recognition event"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO recognition_logs (person_id, image_path, confidence, is_unknown)
                    VALUES (?, ?, ?, ?)
                ''', (person_id, image_path, confidence, is_unknown))
                
                # Update person's last_seen if not unknown
                if person_id and not is_unknown:
                    cursor.execute('''
                        UPDATE persons SET last_seen = CURRENT_TIMESTAMP WHERE id = ?
                    ''', (person_id,))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log recognition: {e}")
    
    def get_recognition_stats(self, days: int = 30) -> Dict:
        """Get recognition statistics for the last N days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total recognitions
                cursor.execute('''
                    SELECT COUNT(*) FROM recognition_logs 
                    WHERE recognized_at >= datetime('now', '-{} days')
                '''.format(days))
                total_recognitions = cursor.fetchone()[0]
                
                # Successful recognitions
                cursor.execute('''
                    SELECT COUNT(*) FROM recognition_logs 
                    WHERE recognized_at >= datetime('now', '-{} days') AND is_unknown = FALSE
                '''.format(days))
                successful_recognitions = cursor.fetchone()[0]
                
                # Unknown faces
                cursor.execute('''
                    SELECT COUNT(*) FROM recognition_logs 
                    WHERE recognized_at >= datetime('now', '-{} days') AND is_unknown = TRUE
                '''.format(days))
                unknown_faces = cursor.fetchone()[0]
                
                # Most recognized persons
                cursor.execute('''
                    SELECT p.name, COUNT(*) as count 
                    FROM recognition_logs rl
                    JOIN persons p ON rl.person_id = p.id
                    WHERE rl.recognized_at >= datetime('now', '-{} days') AND rl.is_unknown = FALSE
                    GROUP BY p.id, p.name
                    ORDER BY count DESC
                    LIMIT 10
                '''.format(days))
                top_persons = cursor.fetchall()
                
                return {
                    'total_recognitions': total_recognitions,
                    'successful_recognitions': successful_recognitions,
                    'unknown_faces': unknown_faces,
                    'success_rate': successful_recognitions / total_recognitions if total_recognitions > 0 else 0,
                    'top_persons': [{'name': row[0], 'count': row[1]} for row in top_persons]
                }
                
        except Exception as e:
            logger.error(f"Failed to get recognition stats: {e}")
            return {}
    
    def delete_person(self, person_id: int) -> bool:
        """Delete a person and all associated data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get person info
                person = self.get_person_by_id(person_id)
                if not person:
                    return False
                
                # Delete face images and associated files
                cursor.execute('SELECT image_path, thumbnail_path, embedding_path FROM face_images WHERE person_id = ?', (person_id,))
                files_to_delete = cursor.fetchall()
                
                for image_path, thumbnail_path, embedding_path in files_to_delete:
                    # Delete files
                    for file_path in [image_path, thumbnail_path, embedding_path]:
                        if file_path and os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                logger.warning(f"Failed to delete file {file_path}: {e}")
                
                # Delete database records
                cursor.execute('DELETE FROM face_images WHERE person_id = ?', (person_id,))
                cursor.execute('DELETE FROM recognition_logs WHERE person_id = ?', (person_id,))
                cursor.execute('DELETE FROM persons WHERE id = ?', (person_id,))
                
                conn.commit()
                
                logger.info(f"Deleted person: {person['name']} (ID: {person_id})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete person {person_id}: {e}")
            return False
    
    def cleanup_old_logs(self, days: int = 90):
        """Clean up old recognition logs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM recognition_logs 
                    WHERE recognized_at < datetime('now', '-{} days')
                '''.format(days))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old recognition logs")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database"""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to backup database: {e}")

if __name__ == "__main__":
    # Test the database
    db = FaceDatabase()
    
    # Test adding a person
    person_id = db.add_person("Test Person", {"age": 25, "department": "IT"})
    print(f"Added person with ID: {person_id}")
    
    # Test getting person
    person = db.get_person_by_id(person_id)
    print(f"Retrieved person: {person}")
    
    # Test getting all persons
    all_persons = db.get_all_persons()
    print(f"All persons: {all_persons}")
    
    # Test cleanup
    db.cleanup_old_logs(0)  # Clean all logs for testing
