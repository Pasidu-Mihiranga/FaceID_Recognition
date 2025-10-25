"""
Face Recognition Module
Supports multiple face recognition models: ArcFace, FaceNet, VGG-Face
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognizer(ABC):
    """Abstract base class for face recognizers"""
    
    @abstractmethod
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from face image
        
        Args:
            face_image: Preprocessed face image
            
        Returns:
            Face embedding vector
        """
        pass
    
    @abstractmethod
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Similarity score (higher = more similar)
        """
        pass

class ArcFaceRecognizer(FaceRecognizer):
    """ArcFace face recognizer using InsightFace"""
    
    def __init__(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            self.app = FaceAnalysis(name='buffalo_l')
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("ArcFace recognizer initialized successfully")
        except ImportError:
            logger.error("InsightFace not available. Please install: pip install insightface")
            raise ImportError("InsightFace not available. Install with: pip install insightface")
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding using ArcFace"""
        try:
            # Ensure image is in correct format
            if len(face_image.shape) == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            faces = self.app.get(face_image)
            
            if len(faces) == 0:
                raise ValueError("No face detected in image")
            
            # Return the embedding of the first (and should be only) face
            return faces[0].embedding
            
        except Exception as e:
            logger.error(f"ArcFace embedding extraction failed: {e}")
            raise
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings using cosine similarity"""
        try:
            # Normalize embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"ArcFace comparison failed: {e}")
            return 0.0

class FaceNetRecognizer(FaceRecognizer):
    """FaceNet face recognizer using DeepFace"""
    
    def __init__(self):
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            logger.info("FaceNet recognizer initialized successfully")
        except ImportError:
            logger.error("DeepFace not available. Please install: pip install deepface")
            raise ImportError("DeepFace not available. Install with: pip install deepface")
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding using FaceNet"""
        try:
            # DeepFace expects RGB images
            if len(face_image.shape) == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Extract embedding using FaceNet
            embedding_obj = self.deepface.represent(
                img_path=face_image,
                model_name='Facenet',
                detector_backend='opencv',
                enforce_detection=False
            )
            
            return np.array(embedding_obj[0]['embedding'])
            
        except Exception as e:
            logger.error(f"FaceNet embedding extraction failed: {e}")
            raise
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings using cosine similarity"""
        try:
            # Normalize embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"FaceNet comparison failed: {e}")
            return 0.0

class VGGFaceRecognizer(FaceRecognizer):
    """VGG-Face recognizer using DeepFace"""
    
    def __init__(self):
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            logger.info("VGG-Face recognizer initialized successfully")
        except ImportError:
            logger.error("DeepFace not available. Please install: pip install deepface")
            raise ImportError("DeepFace not available. Install with: pip install deepface")
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding using VGG-Face"""
        try:
            # DeepFace expects RGB images
            if len(face_image.shape) == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Extract embedding using VGG-Face
            embedding_obj = self.deepface.represent(
                img_path=face_image,
                model_name='VGG-Face',
                detector_backend='opencv',
                enforce_detection=False
            )
            
            return np.array(embedding_obj[0]['embedding'])
            
        except Exception as e:
            logger.error(f"VGG-Face embedding extraction failed: {e}")
            raise
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings using cosine similarity"""
        try:
            # Normalize embeddings
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"VGG-Face comparison failed: {e}")
            return 0.0

class SimpleOpenCVRecognizer(FaceRecognizer):
    """Simple face recognizer using OpenCV features (fallback)"""
    
    def __init__(self):
        try:
            import cv2
            self.cv2 = cv2
            logger.info("Simple OpenCV recognizer initialized successfully")
        except ImportError:
            logger.error("OpenCV not available")
            raise ImportError("OpenCV not available")
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract simple features from face image"""
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Resize to standard size
            gray = cv2.resize(gray, (64, 64))
            
            # Extract histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Normalize
            hist = hist / np.sum(hist)
            
            return hist
            
        except Exception as e:
            logger.error(f"Simple OpenCV embedding extraction failed: {e}")
            raise
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings using histogram correlation"""
        try:
            # Use histogram correlation
            correlation = cv2.compareHist(embedding1, embedding2, cv2.HISTCMP_CORREL)
            return float(correlation)
            
        except Exception as e:
            logger.error(f"Simple OpenCV comparison failed: {e}")
            return 0.0


class DeepFaceRecognizer(FaceRecognizer):
    """DeepFace-based face recognizer for high accuracy"""
    
    def __init__(self, threshold: float = 0.6):
        """
        Initialize DeepFace recognizer
        
        Args:
            threshold: Similarity threshold for face matching
        """
        self.model_name = 'VGG-Face'
        self.threshold = threshold
        self.face_database = {}
        self.database_path = "data/embeddings/deepface_vgg_face_database.pkl"
        
        try:
            from deepface import DeepFace
            self.deepface_available = True
            self.DeepFace = DeepFace  # Store reference
            logger.info("DeepFace VGG-Face recognizer initialized successfully")
        except ImportError:
            self.deepface_available = False
            logger.error("DeepFace not available")
            raise ImportError("DeepFace not available")
        
        self.load_database()
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding using DeepFace"""
        try:
            # Convert BGR to RGB
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = face_image
            
            # Extract embedding
            embedding = self.DeepFace.represent(
                img_path=rgb_image,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            return np.array(embedding[0]['embedding'])
            
        except Exception as e:
            logger.error(f"DeepFace embedding extraction failed: {e}")
            raise
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings using cosine similarity"""
        try:
            # Normalize embeddings
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return 0.0
    
    def register_face(self, face_image: np.ndarray, person_name: str) -> bool:
        """Register a new face"""
        try:
            embedding = self.extract_embedding(face_image)
            
            if person_name not in self.face_database:
                self.face_database[person_name] = []
            
            self.face_database[person_name].append(embedding)
            self.save_database()
            
            logger.info(f"Registered face for {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Face registration failed for {person_name}: {e}")
            return False
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize a face"""
        try:
            embedding = self.extract_embedding(face_image)
            
            best_match = None
            best_score = 0.0
            
            for person_name, embeddings in self.face_database.items():
                for stored_embedding in embeddings:
                    similarity = self.compare_faces(embedding, stored_embedding)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = person_name
            
            if best_score >= self.threshold:
                return best_match, best_score
            else:
                return None, best_score
                
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return None, 0.0
    
    def save_database(self):
        """Save face database"""
        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.face_database, f)
            logger.info(f"Database saved to {self.database_path}")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def load_database(self):
        """Load face database"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'rb') as f:
                    self.face_database = pickle.load(f)
                logger.info(f"Database loaded from {self.database_path}")
            else:
                logger.info("No existing database found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            self.face_database = {}


class FaceRecognitionManager:
    """Optimized manager class for face recognition with caching and multiple models"""
    
    def __init__(self, model_type: str = 'arcface', threshold: float = 0.5):
        """
        Initialize face recognition manager
        
        Args:
            model_type: Type of recognition model ('arcface', 'facenet', 'vggface')
            threshold: Similarity threshold for face matching
        """
        self.model_type = model_type
        self.threshold = threshold
        self.recognizer = self._initialize_recognizer(model_type)
        self.face_database = {}  # Dictionary to store face embeddings
        self.embedding_cache = {}  # Cache for computed embeddings
        self.cache_max_size = 1000  # Maximum cache size
        self.load_database()
    
    def _initialize_recognizer(self, model_type: str) -> FaceRecognizer:
        """Initialize the specified recognizer"""
        recognizer_map = {
            'deepface': DeepFaceRecognizer,
            'arcface': ArcFaceRecognizer,
            'facenet': FaceNetRecognizer,
            'vggface': VGGFaceRecognizer,
            'simple': SimpleOpenCVRecognizer
        }
        
        if model_type not in recognizer_map:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        try:
            return recognizer_map[model_type]()
        except ImportError as e:
            logger.warning(f"Failed to initialize {model_type}: {e}")
            if model_type != 'simple':
                logger.info("Falling back to simple OpenCV recognizer")
                return SimpleOpenCVRecognizer()
            else:
                raise
    
    def register_face(self, face_image: np.ndarray, person_name: str) -> bool:
        """
        Register a new face in the database
        
        Args:
            face_image: Face image to register
            person_name: Name of the person
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Extract embedding
            embedding = self.recognizer.extract_embedding(face_image)
            
            # Add to database
            if person_name not in self.face_database:
                self.face_database[person_name] = []
            
            self.face_database[person_name].append(embedding)
            
            # Save database
            self.save_database()
            
            logger.info(f"Successfully registered face for {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Face registration failed: {e}")
            return False
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Optimized face recognition with caching and fast matching
        
        Args:
            face_image: Preprocessed face image
            
        Returns:
            Tuple of (person_name, confidence)
        """
        try:
            # Create cache key from image hash
            image_hash = hash(face_image.tobytes())
            
            # Check cache first
            if image_hash in self.embedding_cache:
                embedding = self.embedding_cache[image_hash]
            else:
                # Extract embedding
                embedding = self.recognizer.extract_embedding(face_image)
                
                # Cache the embedding
                if len(self.embedding_cache) < self.cache_max_size:
                    self.embedding_cache[image_hash] = embedding
                else:
                    # Remove oldest cache entries
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]
                    self.embedding_cache[image_hash] = embedding
            
            # Fast matching using precomputed similarities
            best_match = None
            best_similarity = 0.0
            
            # Use optimized comparison for large databases
            if len(self.face_database) > 10:
                best_match, best_similarity = self._fast_face_matching(embedding)
            else:
                # Standard comparison for small databases
                for person_name, embeddings in self.face_database.items():
                    for stored_embedding in embeddings:
                        similarity = self.recognizer.compare_faces(embedding, stored_embedding)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = person_name
            
            # Check if similarity exceeds threshold
            if best_similarity >= self.threshold:
                return best_match, best_similarity
            else:
                return None, best_similarity
                
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return None, 0.0
    
    def _fast_face_matching(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Fast face matching using vectorized operations"""
        try:
            if not self.face_database:
                return None, 0.0
            
            best_match = None
            best_similarity = 0.0
            
            # Normalize query embedding once
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            for person_name, embeddings in self.face_database.items():
                if not embeddings:
                    continue
                
                # Convert embeddings to numpy array for vectorized operations
                embeddings_array = np.array(embeddings)
                
                # Normalize all embeddings at once
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                normalized_embeddings = embeddings_array / norms
                
                # Calculate similarities using dot product
                similarities = np.dot(normalized_embeddings, query_embedding)
                
                # Find maximum similarity for this person
                max_similarity = np.max(similarities)
                
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_match = person_name
            
            return best_match, float(best_similarity)
            
        except Exception as e:
            logger.error(f"Fast face matching failed: {e}")
            return None, 0.0
    
    def update_face(self, face_image: np.ndarray, person_name: str) -> bool:
        """
        Update existing face with new image (continuous learning)
        
        Args:
            face_image: New face image
            person_name: Name of the person
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if person_name not in self.face_database:
                logger.warning(f"Person {person_name} not found in database")
                return False
            
            # Extract embedding
            embedding = self.recognizer.extract_embedding(face_image)
            
            # Add to existing embeddings
            self.face_database[person_name].append(embedding)
            
            # Limit the number of embeddings per person (prevent memory issues)
            max_embeddings = 10
            if len(self.face_database[person_name]) > max_embeddings:
                self.face_database[person_name] = self.face_database[person_name][-max_embeddings:]
            
            # Save database
            self.save_database()
            
            logger.info(f"Successfully updated face for {person_name}")
            return True
            
        except Exception as e:
            logger.error(f"Face update failed: {e}")
            return False
    
    def get_all_persons(self) -> List[str]:
        """Get list of all registered persons"""
        return list(self.face_database.keys())
    
    def remove_person(self, person_name: str) -> bool:
        """Remove a person from the database"""
        try:
            if person_name in self.face_database:
                del self.face_database[person_name]
                self.save_database()
                logger.info(f"Successfully removed {person_name} from database")
                return True
            else:
                logger.warning(f"Person {person_name} not found in database")
                return False
        except Exception as e:
            logger.error(f"Failed to remove person {person_name}: {e}")
            return False
    
    def save_database(self):
        """Save face database to file"""
        try:
            db_path = f"data/embeddings/{self.model_type}_database.pkl"
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            with open(db_path, 'wb') as f:
                pickle.dump(self.face_database, f)
            
            logger.info(f"Database saved to {db_path}")
            
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    def load_database(self):
        """Load face database from file"""
        try:
            # For DeepFace, use the specific database path
            if self.model_type == 'deepface':
                db_path = "data/embeddings/deepface_vgg_face_database.pkl"
            else:
                db_path = f"data/embeddings/{self.model_type}_database.pkl"

            if os.path.exists(db_path):
                with open(db_path, 'rb') as f:
                    self.face_database = pickle.load(f)
                logger.info(f"Database loaded from {db_path}")
            else:
                logger.info("No existing database found, starting fresh")

        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            self.face_database = {}

# Factory function for easy recognizer creation
def create_face_recognizer(model_type: str = 'arcface', threshold: float = 0.6) -> FaceRecognitionManager:
    """Create a face recognizer instance"""
    return FaceRecognitionManager(model_type, threshold)

if __name__ == "__main__":
    # Test the face recognition
    import os
    
    # Create a test face image
    test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    # Test different recognizers
    models = ['arcface', 'facenet', 'vggface']
    
    for model_type in models:
        try:
            recognizer = create_face_recognizer(model_type)
            
            # Test registration
            success = recognizer.register_face(test_face, "test_person")
            print(f"{model_type}: Registration {'successful' if success else 'failed'}")
            
            # Test recognition
            person, confidence = recognizer.recognize_face(test_face)
            print(f"{model_type}: Recognition result - {person} (confidence: {confidence:.3f})")
            
        except Exception as e:
            print(f"{model_type}: Failed - {e}")
